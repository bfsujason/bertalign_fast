#!/usr/bin/env python3

"""
PyQt5 interface for Bertalign-Fast.
Multilingual Parallel Corpus Alignment GUI.
"""

import os
import sys
import html
import json
import multiprocessing
from collections import defaultdict
from itertools import combinations

import igraph as ig

from bertalign_fast.utils import *
from bertalign_fast.corelib import *
from bertalign_fast.encoder import Encoder

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QFileDialog,
    QMessageBox, QProgressBar, QTextEdit, QComboBox, QCheckBox,
    QGroupBox, QFormLayout, QHeaderView, QAbstractItemView,
    QMenu, QAction, QTabWidget, QLineEdit, QUndoStack, QUndoCommand,
    QStyledItemDelegate, QAbstractItemDelegate, QPlainTextEdit,
    QStyleOptionViewItem, QStyle,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import (
    QFont, QKeySequence, QTextOption, QTextDocument,
    QColor, QBrush, QPalette,
)


# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models/static-similarity-mrl-multilingual-v1/0_StaticEmbedding")

TMX_LANG_MAP = {
    "ar": "ar-SA", "hy": "hu-AM", "bg": "bg-BG", "my": "my-BU",
    "ca": "ca-AD", "zh": "zh-CN", "cs": "cs-CZ", "da": "da-DK",
    "nl": "nl-NL", "en": "en-US", "fi": "fi-FI", "fr": "fr-FR",
    "de": "de-DE", "el": "el-GR", "hi": "hi-IN", "hu": "hu-HU",
    "it": "it-IT", "ja": "ja-JP", "lt": "lt-LT", "lv": "lv-LV",
    "mr": "mr-IN", "fa": "fa-IR", "pl": "pl-PL", "pt": "pt-BR",
    "ro": "ro-RO", "ru": "ru-RU", "sk": "sk-SK", "sl": "sl-SI",
    "es": "es-ES", "sv": "sv-SE", "tr": "tr-TR", "ur": "ur-PK",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _as_list(value):
    """Normalize a scalar-or-list value to a list."""
    if isinstance(value, int):
        return [value]
    return value


# ── Alignment Engine ─────────────────────────────────────────────────────────

class Aligner:
    """Wraps the Bertalign-Fast encoder and two-pass alignment algorithm."""

    def __init__(
        self,
        model_path,
        max_align=8,
        embedding_dim=512,
        mean_center=True,
        top_k=3,
        window_size=5,
        skip_penalty=-0.1,
        lambda_size=0.02,
        length_penalty=True,
    ):
        self.model_path = model_path
        self.max_align = max_align
        self.embedding_dim = embedding_dim
        self.mean_center = mean_center
        self.top_k = top_k
        self.window_size = window_size
        self.skip_penalty = skip_penalty
        self.lambda_size = lambda_size
        self.length_penalty = length_penalty
        self.encoder = Encoder(model_path)

    def embed(self, sents):
        vecs, lens = self.encoder.transform(
            sents,
            self.max_align - 1,
            embedding_dim=self.embedding_dim,
            mean_center=self.mean_center,
        )
        return vecs, lens

    def align(self, src_vecs, tgt_vecs, src_lens, tgt_lens):
        print("Using BertalignFast for alignment\n")
        src_len = src_vecs.shape[1]
        tgt_len = tgt_vecs.shape[1]

        # First pass: extract 1-1 anchor beads
        similarities, top_k_indices = find_top_k_similar(
            src_vecs[0], tgt_vecs[0], k=self.top_k,
        )
        first_window, first_path = find_first_pass_search_path(src_len, tgt_len)
        first_bp = first_pass_align(
            src_len, tgt_len, first_window, first_path,
            similarities, top_k_indices,
        )
        first_alignment = first_pass_backtrack(
            src_len, tgt_len, first_bp, first_path,
        )

        # Second pass: full m-to-n alignment
        first_anchors_dp = [(s + 1, t + 1) for s, t in first_alignment]
        align_types = get_alignment_types(self.max_align)
        second_window, second_path = find_second_pass_search_path(
            first_anchors_dp, self.window_size, src_len, tgt_len,
        )
        second_bp, second_cost = second_pass_align(
            src_vecs, tgt_vecs, src_lens, tgt_lens,
            second_window, second_path, align_types,
            self.skip_penalty, self.lambda_size,
            length_penalty=self.length_penalty,
        )
        return second_pass_backtrack(
            src_len, tgt_len, second_bp, second_cost,
            second_path, align_types,
        )

    # ── Multilingual graph construction ──────────────────────────────────

    def build_source_graph(self, pairwise_alignments, source_len):
        """Build an undirected graph of source-sentence co-occurrence."""
        graph = ig.Graph()
        graph.add_vertices(source_len)
        edges = set()

        for alignments in pairwise_alignments.values():
            for src_indices, _, _ in alignments:
                if not src_indices:
                    continue
                src_indices = _as_list(src_indices)
                for edge in combinations(src_indices, 2):
                    edges.add(tuple(sorted(edge)))

        if edges:
            graph.add_edges(list(edges))
        return graph

    def get_source_components(self, graph):
        """Extract sorted connected components from the source graph."""
        if not isinstance(graph, ig.Graph):
            raise TypeError(
                f"Expected igraph.Graph, got {type(graph).__name__}."
            )
        components = [sorted(c) for c in graph.components()]
        components.sort(key=lambda c: c[0] if c else float("inf"))
        return components

    def _find_insertion_position(
        self, target_idx, lang_code, lang_alignments, source_components,
    ):
        """Determine where a null-source alignment should be inserted.

        Returns a fractional position value for sorting among regular
        alignment units.
        """
        before_src = None
        after_src = None

        for src_indices, tgt_indices, _ in lang_alignments:
            if not src_indices:
                continue
            tgt_indices = _as_list(tgt_indices)
            src_indices = _as_list(src_indices)
            min_src = min(src_indices)

            if max(tgt_indices) < target_idx:
                if before_src is None or min_src > before_src:
                    before_src = min_src
            if min(tgt_indices) > target_idx:
                if after_src is None or min_src < after_src:
                    after_src = min_src

        if before_src is not None:
            return before_src + 0.5
        if after_src is not None:
            return after_src - 0.5
        return -0.5

    def build_multilingual_alignments(
        self, source_components, pairwise_alignments,
    ):
        """Merge source components and pairwise alignments into unified rows.

        Returns a list of dicts, each mapping column names ('Source',
        target names) to lists of sentence indices.
        """
        alignment_units = []
        used_targets = defaultdict(set)

        # Step 1: regular alignments (source → targets)
        for component in source_components:
            component_set = set(component)
            unit = {"Source": component, "position": min(component), "type": "regular"}

            for lang_code, alignments in pairwise_alignments.items():
                tgt_for_component = set()
                for src_idx, tgt_idx, _ in alignments:
                    if not src_idx:
                        continue
                    src_idx = _as_list(src_idx)
                    tgt_idx = _as_list(tgt_idx)
                    if any(s in component_set for s in src_idx):
                        tgt_for_component.update(tgt_idx)
                        used_targets[lang_code].update(tgt_idx)
                if tgt_for_component:
                    unit[lang_code] = sorted(tgt_for_component)

            alignment_units.append(unit)

        # Step 2: null-source insertions
        insertion_units = []
        for lang_code, alignments in pairwise_alignments.items():
            for src_idx, tgt_idx, _ in alignments:
                if not src_idx and tgt_idx:
                    tgt_idx = _as_list(tgt_idx)
                    unused = [i for i in tgt_idx if i not in used_targets[lang_code]]
                    if unused:
                        pos = self._find_insertion_position(
                            min(unused), lang_code,
                            pairwise_alignments[lang_code],
                            source_components,
                        )
                        insertion_units.append({
                            lang_code: sorted(unused),
                            "position": pos,
                            "type": "insertion",
                        })
                        used_targets[lang_code].update(unused)

        # Step 3: merge, sort, strip metadata
        all_units = alignment_units + insertion_units
        all_units.sort(key=lambda u: u["position"])
        return [
            {k: v for k, v in u.items() if k not in ("position", "type")}
            for u in all_units
        ]


# ── Background Worker ────────────────────────────────────────────────────────

class AlignmentWorker(QThread):
    """Runs the alignment pipeline in a background thread."""

    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    alignment_completed = pyqtSignal(list, dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, src_info, tgt_infos, embed_model):
        super().__init__()
        self.src_info = src_info
        self.tgt_infos = tgt_infos
        self.embed_model = embed_model

    def run(self):
        try:
            self.status_updated.emit("Preparing for alignment...")
            self.progress_updated.emit(10)

            src_lang = self.src_info["lang"]
            src_text = self.src_info["sents"]
            self.status_updated.emit(
                f"Source: {SUPPORTED_LANGUAGES.get(src_lang, src_lang)}, "
                f"{len(src_text)} sentences"
            )
            self.progress_updated.emit(20)

            tgt_texts = {}
            for i, info in enumerate(self.tgt_infos):
                lang = info["lang"]
                name = f"Target {i + 1}"
                tgt_texts[name] = info["sents"]
                self.status_updated.emit(
                    f"{name}: {SUPPORTED_LANGUAGES.get(lang, lang)}, "
                    f"{len(info['sents'])} sentences"
                )
            self.progress_updated.emit(40)

            self.status_updated.emit(
                f"Initializing aligner with model: "
                f"{os.path.basename(self.embed_model)}..."
            )
            self.progress_updated.emit(50)
            aligner = Aligner(self.embed_model)

            self.status_updated.emit("Generating embeddings...")
            self.progress_updated.emit(60)
            src_vecs, src_lens = aligner.embed(src_text)

            pairwise = {}
            for i, (name, tgt) in enumerate(tgt_texts.items()):
                self.status_updated.emit(f"Aligning with {name}...")
                tgt_vecs, tgt_lens = aligner.embed(tgt)
                pairwise[name] = aligner.align(
                    src_vecs, tgt_vecs, src_lens, tgt_lens,
                )
                self.progress_updated.emit(60 + (i + 1) * 30 // len(tgt_texts))

            # Build final alignment structure
            if len(self.tgt_infos) > 1:
                self.status_updated.emit("Building multilingual alignments...")
                graph = aligner.build_source_graph(pairwise, len(src_text))
                components = aligner.get_source_components(graph)
                result = aligner.build_multilingual_alignments(components, pairwise)
            else:
                self.status_updated.emit("Formatting single-target alignment...")
                name = next(iter(pairwise))
                result = [
                    {"Source": list(src), name: list(tgt)}
                    for src, tgt, _ in pairwise[name]
                ]

            self.progress_updated.emit(100)
            self.status_updated.emit("Alignment completed!")
            self.alignment_completed.emit(result, {"Source": src_text, **tgt_texts})

        except Exception as e:
            self.error_occurred.emit(str(e))


# ── Undo / Redo Commands ────────────────────────────────────────────────────

class ChangeCellCommand(QUndoCommand):
    def __init__(self, table, row, col, old_text, new_text):
        super().__init__(f"Edit cell ({row + 1}, {col + 1})")
        self.table, self.row, self.col = table, row, col
        self.old_text, self.new_text = old_text, new_text

    def undo(self):
        self.table.set_item_text_guarded(self.row, self.col, self.old_text)
        self.table.resizeRowToContents(self.row)

    def redo(self):
        self.table.set_item_text_guarded(self.row, self.col, self.new_text)
        self.table.resizeRowToContents(self.row)


class SplitCommand(QUndoCommand):
    def __init__(self, table, row, col, original_text, first_part, second_part):
        super().__init__("Split cell")
        self.table, self.row, self.col = table, row, col
        self.original_text = original_text
        self.first_part, self.second_part = first_part, second_part
        self.was_marked = self.table.is_row_marked(self.row)

    def undo(self):
        self.table.is_undoing_or_redoing = True
        self.table.removeRow(self.row + 1)
        self.table.setItem(self.row, self.col, QTableWidgetItem(self.original_text))
        self.table.set_row_marked(self.row, self.was_marked, update_visuals=False)
        for c in range(self.table.columnCount()):
            if c != self.col:
                self.table.setItem(
                    self.row, c,
                    QTableWidgetItem(self.table.item(self.row, c).text()),
                )
                self.table.item(self.row, c).setData(Qt.UserRole, self.was_marked)
        self.table.update_row_visuals(self.row)
        self.table.resizeRowToContents(self.row)
        self.table.is_undoing_or_redoing = False
        self.table.update_statistics()

    def redo(self):
        self.table.is_undoing_or_redoing = True
        self.table.setItem(self.row, self.col, QTableWidgetItem(self.first_part))
        self.table.insertRow(self.row + 1)
        for c in range(self.table.columnCount()):
            text = self.second_part if c == self.col else ""
            self.table.setItem(self.row + 1, c, QTableWidgetItem(text))
        self.table.set_row_marked(self.row + 1, self.was_marked)
        self.table.resizeRowToContents(self.row)
        self.table.resizeRowToContents(self.row + 1)
        self.table.is_undoing_or_redoing = False
        self.table.update_statistics()


class MergeCommand(QUndoCommand):
    def __init__(self, table, selected_rows, original_data, merged_data):
        super().__init__("Merge rows")
        self.table = table
        self.selected_rows = selected_rows
        self.original_data = original_data
        self.merged_data = merged_data
        self.new_row_is_marked = any(d["marked"] for d in original_data.values())

    def undo(self):
        self.table.is_undoing_or_redoing = True
        first = self.selected_rows[0]
        for col, text in enumerate(self.original_data[first]["text"]):
            self.table.setItem(first, col, QTableWidgetItem(text))
        self.table.set_row_marked(first, self.original_data[first]["marked"])

        for row_idx in reversed(self.selected_rows[1:]):
            self.table.insertRow(row_idx)
            for col, text in enumerate(self.original_data[row_idx]["text"]):
                self.table.setItem(row_idx, col, QTableWidgetItem(text))
            self.table.set_row_marked(row_idx, self.original_data[row_idx]["marked"])

        for row_idx in self.selected_rows:
            self.table.resizeRowToContents(row_idx)
        self.table.is_undoing_or_redoing = False
        self.table.update_statistics()

    def redo(self):
        self.table.is_undoing_or_redoing = True
        first = self.selected_rows[0]
        for col, text in enumerate(self.merged_data):
            self.table.setItem(first, col, QTableWidgetItem(text))
        self.table.set_row_marked(first, self.new_row_is_marked)
        for row in reversed(self.selected_rows[1:]):
            self.table.removeRow(row)
        self.table.resizeRowToContents(first)
        self.table.is_undoing_or_redoing = False
        self.table.update_statistics()


class DeleteCommand(QUndoCommand):
    def __init__(self, table, row_data):
        super().__init__(f"Delete {len(row_data)} row(s)")
        self.table = table
        self.row_data = row_data

    def undo(self):
        self.table.is_undoing_or_redoing = True
        for row_idx, data in sorted(self.row_data.items()):
            self.table.insertRow(row_idx)
            for col, text in enumerate(data["text"]):
                self.table.setItem(row_idx, col, QTableWidgetItem(text))
            self.table.set_row_marked(row_idx, data["marked"])
            self.table.resizeRowToContents(row_idx)
        self.table.is_undoing_or_redoing = False
        self.table.update_statistics()

    def redo(self):
        self.table.is_undoing_or_redoing = True
        for row_idx in sorted(self.row_data.keys(), reverse=True):
            self.table.removeRow(row_idx)
        self.table.is_undoing_or_redoing = False
        self.table.update_statistics()


class AddRowCommand(QUndoCommand):
    def __init__(self, table, row_idx):
        super().__init__("Add new row")
        self.table, self.row_idx = table, row_idx

    def undo(self):
        self.table.is_undoing_or_redoing = True
        self.table.removeRow(self.row_idx)
        self.table.is_undoing_or_redoing = False
        self.table.update_statistics()

    def redo(self):
        self.table.is_undoing_or_redoing = True
        self.table.insertRow(self.row_idx)
        for col in range(self.table.columnCount()):
            self.table.setItem(self.row_idx, col, QTableWidgetItem(""))
        self.table.set_row_marked(self.row_idx, False)
        self.table.resizeRowToContents(self.row_idx)
        self.table.is_undoing_or_redoing = False
        self.table.update_statistics()


class MoveCellTextCommand(QUndoCommand):
    def __init__(self, table, from_row, to_row, col):
        super().__init__(f"Move cell text from row {from_row + 1} to {to_row + 1}")
        self.table, self.from_row, self.to_row, self.col = table, from_row, to_row, col
        self.text_from = self.table.item(from_row, col).text()
        item_to = self.table.item(to_row, col)
        self.text_to = item_to.text() if item_to else ""

    def _swap(self, from_text, to_text):
        self.table.set_item_text_guarded(self.from_row, self.col, from_text)
        self.table.set_item_text_guarded(self.to_row, self.col, to_text)
        self.table.resizeRowToContents(self.from_row)
        self.table.resizeRowToContents(self.to_row)

    def undo(self):
        self._swap(self.text_from, self.text_to)

    def redo(self):
        self._swap(self.text_to, self.text_from)


class MarkRowsCommand(QUndoCommand):
    def __init__(self, table, rows_to_toggle):
        super().__init__("Mark/Unmark rows")
        self.table = table
        self.rows = rows_to_toggle
        self.original_states = {r: self.table.is_row_marked(r) for r in self.rows}

    def _apply(self, state_fn):
        for row in self.rows:
            self.table.set_row_marked(row, state_fn(row))
        self.table.viewport().update()
        self.table.clearSelection()

    def undo(self):
        self._apply(lambda r: self.original_states[r])

    def redo(self):
        self._apply(lambda r: not self.table.is_row_marked(r))


# ── Custom Editor & Delegate ────────────────────────────────────────────────

class CustomTextEditor(QPlainTextEdit):
    """In-cell text editor with split and move context actions."""

    splitRequested = pyqtSignal(int)
    moveUpRequested = pyqtSignal()
    moveDownRequested = pyqtSignal()

    def __init__(self, table_widget, row, col, parent=None):
        super().__init__(parent)
        self.table_widget = table_widget
        self.row, self.col = row, col
        self.setWordWrapMode(QTextOption.WordWrap)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setContentsMargins(0, 0, 0, 0)
        self.document().setDocumentMargin(0)

    def _has_empty_cell(self, direction):
        """Check whether an empty cell exists above (direction=-1) or below (+1)."""
        if direction == -1:
            scan = range(self.row - 1, -1, -1)
        else:
            scan = range(self.row + 1, self.table_widget.rowCount())
        for r in scan:
            item = self.table_widget.item(r, self.col)
            if item is None or not item.text().strip():
                return True
        return False

    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        menu.addSeparator()

        split_action = QAction("Split at cursor", self)
        split_action.triggered.connect(
            lambda: self.splitRequested.emit(self.textCursor().position())
        )
        menu.addAction(split_action)
        menu.addSeparator()

        move_up = QAction("Move Up", self)
        move_up.setEnabled(self._has_empty_cell(-1))
        move_up.triggered.connect(self.moveUpRequested.emit)
        menu.addAction(move_up)

        move_down = QAction("Move Down", self)
        move_down.setEnabled(self._has_empty_cell(+1))
        move_down.triggered.connect(self.moveDownRequested.emit)
        menu.addAction(move_down)

        menu.exec_(event.globalPos())

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            e.ignore()
            return
        super().keyPressEvent(e)


class EditingDelegate(QStyledItemDelegate):
    """Item delegate that uses CustomTextEditor and adds vertical padding."""

    PADDING = 12

    def paint(self, painter, option, index):
        table = self.parent()
        if isinstance(table, EditableTableWidget) and table.is_row_marked(index.row()):
            if option.state & QStyle.State_Selected:
                opt = QStyleOptionViewItem(option)
                opt.palette.setBrush(QPalette.Highlight, QBrush(table.MARK_COLOR))
                opt.palette.setColor(QPalette.HighlightedText, Qt.black)
                super().paint(painter, opt, index)
                return
        super().paint(painter, option, index)

    def _measure_text(self, text, font, width):
        """Return the document height for the given text at the given width."""
        doc = QTextDocument()
        doc.setPlainText(text or "")
        doc.setDefaultFont(font)
        doc.setTextWidth(width)
        return doc.size().height()

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        text = index.model().data(index, Qt.DisplayRole)
        height = self._measure_text(text, option.font, option.rect.width())
        return QSize(size.width(), int(height + self.PADDING))

    def createEditor(self, parent, option, index):
        table = self.parent()
        row, col = index.row(), index.column()

        if isinstance(table, QTableWidget):
            text = index.model().data(index, Qt.EditRole)
            required = self._measure_text(text, option.font, option.rect.width())
            final = max(required + self.PADDING + 5, 40)
            if final > table.rowHeight(row):
                table.setRowHeight(row, int(final))

        editor = CustomTextEditor(table, row, col, parent)
        editor.setFont(option.font)
        editor.splitRequested.connect(
            lambda pos: self._commit_and_split(editor, index, pos)
        )
        editor.moveUpRequested.connect(lambda: table.handle_move(row, col, -1))
        editor.moveDownRequested.connect(lambda: table.handle_move(row, col, +1))
        return editor

    def setEditorData(self, editor, index):
        editor.setPlainText(index.model().data(index, Qt.EditRole))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.toPlainText(), Qt.EditRole)

    def _commit_and_split(self, editor, index, position):
        table = self.parent()
        table.perform_split(index.row(), index.column(), position)
        table.closeEditor(editor, QAbstractItemDelegate.NoHint)


# ── Editable Results Table ───────────────────────────────────────────────────

class EditableTableWidget(QTableWidget):
    """Table widget with undo/redo, row marking, split/merge/move actions."""

    MARK_COLOR = QColor(255, 255, 180)

    def __init__(self, stats_label_updater, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.setWordWrap(True)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setItemDelegate(EditingDelegate(self))

        self.undo_stack = QUndoStack(self)
        self.is_undoing_or_redoing = False
        self.stats_label_updater = stats_label_updater

        self._edit_old_text = None
        self._edit_address = None
        self.cellActivated.connect(self._on_cell_activated)
        self.itemChanged.connect(self._on_item_changed)

    # ── Statistics ───────────────────────────────────────────────────────

    def update_statistics(self):
        self.stats_label_updater()

    # ── Cell-edit tracking ───────────────────────────────────────────────

    def _on_cell_activated(self, row, col):
        if self.is_undoing_or_redoing:
            return
        item = self.item(row, col)
        self._edit_old_text = item.text() if item else ""
        self._edit_address = (row, col)

    def _on_item_changed(self, item):
        if self.is_undoing_or_redoing:
            return
        row, col = item.row(), item.column()
        if self._edit_address == (row, col):
            new_text = item.text()
            if new_text != self._edit_old_text:
                self.undo_stack.push(
                    ChangeCellCommand(self, row, col, self._edit_old_text, new_text)
                )
        self.resizeRowToContents(row)
        self._edit_address = None
        self._edit_old_text = None

    def set_item_text_guarded(self, row, col, text):
        """Set cell text without triggering undo recording."""
        self.is_undoing_or_redoing = True
        if self.item(row, col) is None:
            self.setItem(row, col, QTableWidgetItem(text))
        else:
            self.item(row, col).setText(text)
        self.is_undoing_or_redoing = False

    # ── Row marking ──────────────────────────────────────────────────────

    def is_row_marked(self, row):
        item = self.item(row, 0)
        return bool(item and item.data(Qt.UserRole))

    def update_row_visuals(self, row):
        marked = self.is_row_marked(row)
        brush = self.MARK_COLOR if marked else QBrush()
        for col in range(self.columnCount()):
            item = self.item(row, col)
            if item:
                item.setBackground(brush)

    def set_row_marked(self, row, marked, update_visuals=True):
        for col in range(self.columnCount()):
            item = self.item(row, col)
            if not item:
                item = QTableWidgetItem()
                self.setItem(row, col, item)
            item.setData(Qt.UserRole, marked)
        if update_visuals:
            self.update_row_visuals(row)

    def mark_rows_by_index(self, indices):
        for row in indices:
            if 0 <= row < self.rowCount():
                self.set_row_marked(row, True)

    # ── Table actions ────────────────────────────────────────────────────

    def perform_split(self, row, col, position):
        item = self.item(row, col)
        if not item:
            return
        text = item.text()
        if 0 < position < len(text):
            self.undo_stack.push(
                SplitCommand(self, row, col, text, text[:position], text[position:])
            )

    def _selected_rows(self):
        return sorted(set(item.row() for item in self.selectedItems()))

    def _show_context_menu(self, position):
        menu = QMenu(self)
        rows = self._selected_rows()
        count = len(rows)

        mark = QAction("Mark/Unmark Selected Rows", self)
        mark.setEnabled(count > 0)
        mark.triggered.connect(self._toggle_mark_selected)
        menu.addAction(mark)
        menu.addSeparator()

        merge = QAction("Merge Selected Rows", self)
        merge.setEnabled(count >= 2)
        merge.triggered.connect(self._merge_selected)
        menu.addAction(merge)

        delete = QAction("Delete Selected Rows", self)
        delete.setEnabled(count > 0)
        delete.triggered.connect(self._delete_selected)
        menu.addAction(delete)
        menu.addSeparator()

        add = QAction("Add New Row After", self)
        add.triggered.connect(self._add_new_row)
        menu.addAction(add)

        menu.exec_(self.mapToGlobal(position))

    def _toggle_mark_selected(self):
        rows = self._selected_rows()
        if rows:
            self.undo_stack.push(MarkRowsCommand(self, rows))

    def _row_snapshot(self, rows):
        """Capture text + marked state for the given rows."""
        return {
            row: {
                "text": [
                    self.item(row, c).text() if self.item(row, c) else ""
                    for c in range(self.columnCount())
                ],
                "marked": self.is_row_marked(row),
            }
            for row in rows
        }

    def _merge_selected(self):
        rows = self._selected_rows()
        if len(rows) < 2:
            return
        original = self._row_snapshot(rows)
        merged = [
            " ".join(
                original[r]["text"][c].strip()
                for r in rows
                if original[r]["text"][c].strip()
            )
            for c in range(self.columnCount())
        ]
        self.undo_stack.push(MergeCommand(self, rows, original, merged))

    def _delete_selected(self):
        rows = self._selected_rows()
        if not rows:
            return
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete {len(rows)} selected row(s)?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.undo_stack.push(DeleteCommand(self, self._row_snapshot(rows)))

    def _add_new_row(self):
        current = self.currentRow()
        insert_at = current + 1 if current >= 0 else self.rowCount()
        self.undo_stack.push(AddRowCommand(self, insert_at))

    # ── Cell move (unified up/down) ──────────────────────────────────────

    def _find_empty_cell(self, start_row, col, direction):
        """Find the first empty cell in the given direction (-1=up, +1=down)."""
        if direction == -1:
            scan = range(start_row - 1, -1, -1)
        else:
            scan = range(start_row + 1, self.rowCount())
        for r in scan:
            item = self.item(r, col)
            if item is None or not item.text().strip():
                return r
        return -1

    def handle_move(self, row, col, direction):
        target = self._find_empty_cell(row, col, direction)
        if target != -1:
            self.undo_stack.push(MoveCellTextCommand(self, row, target, col))
            self.closeEditor(self.focusWidget(), QAbstractItemDelegate.NoHint)


# ── Main Window ──────────────────────────────────────────────────────────────

class CorpusAlignmentGUI(QMainWindow):
    """Main GUI for the Bertalign-Fast corpus alignment tool."""

    def __init__(self):
        super().__init__()
        self.src_file = None
        self.source_file_widget = None
        self.target_file_widgets = []
        self.alignments = []
        self.texts = {}
        self.column_lang_map = {}
        self.worker = None

        self._resize_timer = QTimer(self)
        self._resize_timer.timeout.connect(self._progressively_resize_rows)
        self._resize_row = 0

        if not os.path.isdir(MODEL_PATH):
            QMessageBox.critical(
                self, "Model Not Found",
                f"The required 'model' directory was not found.\n"
                f"Please ensure it is located in:\n\n{SCRIPT_DIR}",
            )
            QTimer.singleShot(100, self.close)
            return

        self._init_ui()
        self._setup_menu()

    # ── UI Construction ──────────────────────────────────────────────────

    def _init_ui(self):
        self.setWindowTitle("Bertalign-Fast")
        self.setGeometry(100, 100, 1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        self._setup_input_tab()
        self._setup_results_tab()

        self.statusBar().showMessage("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _setup_input_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create align button early (file rows reference it via _update_align_button_state)
        self.align_button = QPushButton("Start Alignment")
        self.align_button.setMinimumHeight(40)
        self.align_button.clicked.connect(self._start_alignment)
        self.align_button.setEnabled(False)

        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QFormLayout(file_group)

        self._src_file_layout = QVBoxLayout()
        self._src_file_layout.setContentsMargins(0, 0, 0, 0)
        src_container = QWidget()
        QVBoxLayout(src_container).addLayout(self._src_file_layout)
        src_container.layout().setContentsMargins(0, 0, 0, 0)
        file_layout.addRow(QLabel("Source File:"))
        file_layout.addRow(src_container)
        self._add_source_file_row()

        self._tgt_files_layout = QVBoxLayout()
        tgt_container = QWidget()
        QVBoxLayout(tgt_container).addLayout(self._tgt_files_layout)
        tgt_container.layout().setContentsMargins(0, 0, 0, 0)
        self.add_tgt_button = QPushButton("Add Target File")
        self.add_tgt_button.clicked.connect(self._add_target_file_row)
        tgt_btn_layout = QHBoxLayout()
        tgt_btn_layout.addWidget(self.add_tgt_button)
        tgt_btn_layout.addStretch()
        file_layout.addRow(QLabel("Target Files:"))
        file_layout.addRow(tgt_container)
        file_layout.addRow("", tgt_btn_layout)
        self._add_target_file_row()

        # Configuration
        config_group = QGroupBox("Configuration")
        config_layout = QFormLayout(config_group)
        self.split_checkbox = QCheckBox(
            "Text is already split into sentences (one per line)"
        )
        config_layout.addRow(self.split_checkbox)

        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Courier", 9))
        log_layout.addWidget(self.log_area)

        layout.addWidget(file_group)
        layout.addWidget(config_group)
        layout.addWidget(self.align_button)
        layout.addWidget(log_group)
        layout.addStretch()
        self.tab_widget.addTab(widget, "Input && Configuration")

    def _setup_results_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        btn_layout = QHBoxLayout()
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self._export_results)
        self.export_button.setEnabled(False)
        self.save_button = QPushButton("Save Project")
        self.save_button.clicked.connect(self.save_project)
        self.save_button.setEnabled(False)
        self.load_button = QPushButton("Load Project")
        self.load_button.clicked.connect(self.load_project)
        btn_layout.addWidget(self.export_button)
        btn_layout.addWidget(self.save_button)
        btn_layout.addWidget(self.load_button)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.stats_label = QLabel("No alignment results")
        self.results_table = EditableTableWidget(self._update_stats_label)
        self.results_table.setMinimumHeight(400)
        layout.addWidget(self.results_table)
        layout.addWidget(self.stats_label)
        self.tab_widget.addTab(widget, "Results && Editing")

    def _setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        self._add_action(file_menu, "New Project", self.new_project, QKeySequence.New)
        file_menu.addSeparator()
        self._add_action(file_menu, "Save Project", self.save_project, QKeySequence.Save)
        self._add_action(file_menu, "Load Project", self.load_project, QKeySequence.Open)
        file_menu.addSeparator()
        self._add_action(file_menu, "Export Results", self._export_results)
        file_menu.addSeparator()
        self._add_action(file_menu, "Exit", self.close, QKeySequence.Quit)

        edit_menu = menubar.addMenu("Edit")
        undo = self.results_table.undo_stack.createUndoAction(self, "Undo")
        undo.setShortcuts(QKeySequence.Undo)
        redo = self.results_table.undo_stack.createRedoAction(self, "Redo")
        redo.setShortcuts(QKeySequence.Redo)
        edit_menu.addAction(undo)
        edit_menu.addAction(redo)

        help_menu = menubar.addMenu("Help")
        self._add_action(help_menu, "About", self._show_about)

    @staticmethod
    def _add_action(menu, text, slot, shortcut=None):
        action = QAction(text, menu)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    # ── Logging ──────────────────────────────────────────────────────────

    def _log(self, message):
        self.log_area.append(message)
        self.log_area.verticalScrollBar().setValue(
            self.log_area.verticalScrollBar().maximum()
        )

    def _status_and_log(self, message):
        self.statusBar().showMessage(message, 4000)
        self._log(f"STATUS: {message}")

    # ── File Row Helpers (shared by source and target) ───────────────────

    def _create_file_row_widgets(self, placeholder, file_path=None, lang_iso=None):
        """Build the common widgets for a file-selection row.

        Returns (container_widget, widget_dict) where widget_dict holds
        references to path_edit, lang_combo, and the container.
        """
        row_layout = QHBoxLayout()

        path_edit = QLineEdit()
        path_edit.setPlaceholderText(placeholder)
        path_edit.setReadOnly(True)
        if file_path:
            path_edit.setText(file_path)

        browse_button = QPushButton("Browse...")

        lang_combo = QComboBox()
        for iso, name in sorted(SUPPORTED_LANGUAGES.items(), key=lambda x: x[1]):
            lang_combo.addItem(f"{name} ({iso})", iso)
        if lang_iso:
            idx = lang_combo.findData(lang_iso)
            if idx != -1:
                lang_combo.setCurrentIndex(idx)

        row_layout.addWidget(path_edit, 5)
        row_layout.addWidget(browse_button, 1)
        row_layout.addWidget(lang_combo, 1)

        container = QWidget()
        container.setLayout(row_layout)

        widget_dict = {
            "container": container,
            "path_edit": path_edit,
            "lang_combo": lang_combo,
            "text": None,
        }
        return container, browse_button, row_layout, widget_dict

    def _browse_and_detect_language(self, widget_dict):
        """Open a file dialog and auto-detect the language of the selected file.

        Shared by both source and target file browsing.
        Returns the chosen file path, or None if cancelled.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return None

        widget_dict["path_edit"].setText(path)
        self._log(f"File selected: {path}")

        try:
            self._status_and_log(f"Detecting language for {os.path.basename(path)}...")
            text = open(path, "rt", encoding="utf-8").read()
            text = clean_text(text)
            lang, _ = detect_lang(text)
            widget_dict["text"] = text

            idx = widget_dict["lang_combo"].findData(lang)
            if idx != -1:
                widget_dict["lang_combo"].setCurrentIndex(idx)
                name = SUPPORTED_LANGUAGES.get(lang, lang)
                self._status_and_log(f"Detected language: {name}")
            else:
                self._status_and_log(
                    f"Could not auto-detect a supported language for "
                    f"{os.path.basename(path)}."
                )
        except Exception as e:
            self._status_and_log(f"Language detection failed: {e}")

        self._update_align_button_state()
        return path

    # ── Source / Target Row Management ────────────────────────────────────

    def _add_source_file_row(self, file_path=None, lang_iso=None):
        if self.source_file_widget:
            return
        container, browse_btn, row_layout, widget_dict = self._create_file_row_widgets(
            "Select a source file...", file_path, lang_iso,
        )
        # Spacer to match target row widths
        row_layout.addWidget(QWidget(), 1)
        self._src_file_layout.addWidget(container)
        self.source_file_widget = widget_dict
        browse_btn.clicked.connect(self._browse_source_file)
        if file_path:
            self.src_file = file_path
        self._update_align_button_state()

    def _add_target_file_row(self, file_path=None, lang_iso=None):
        container, browse_btn, row_layout, widget_dict = self._create_file_row_widgets(
            "Select a target file...", file_path, lang_iso,
        )
        if self.target_file_widgets:
            remove_btn = QPushButton("Remove")
            row_layout.addWidget(remove_btn, 1)
            remove_btn.clicked.connect(lambda: self._remove_target_file_row(widget_dict))
        else:
            row_layout.addWidget(QWidget(), 1)

        self._tgt_files_layout.addWidget(container)
        self.target_file_widgets.append(widget_dict)
        browse_btn.clicked.connect(lambda: self._browse_target_file(widget_dict))
        self._update_align_button_state()

    def _remove_target_file_row(self, widget_dict):
        path = widget_dict["path_edit"].text()
        widget_dict["container"].deleteLater()
        self.target_file_widgets.remove(widget_dict)
        self._update_align_button_state()
        if path:
            self._log(f"Removed target: {os.path.basename(path)}")

    def _browse_source_file(self):
        if not self.source_file_widget:
            return
        path = self._browse_and_detect_language(self.source_file_widget)
        if path:
            self.src_file = path

    def _browse_target_file(self, widget_dict):
        self._browse_and_detect_language(widget_dict)

    def _update_align_button_state(self):
        has_source = (
            self.source_file_widget is not None
            and bool(self.source_file_widget["path_edit"].text())
        )
        has_targets = any(w["path_edit"].text() for w in self.target_file_widgets)
        self.align_button.setEnabled(has_source and has_targets)

    # ── Alignment Pipeline ───────────────────────────────────────────────

    def _read_text_if_needed(self, widget_dict):
        """Ensure the widget has loaded text content; read from disk if not."""
        if widget_dict.get("text") is None:
            path = widget_dict["path_edit"].text()
            self._log(f"Reading file: {os.path.basename(path)}")
            text = open(path, "rt", encoding="utf-8").read()
            widget_dict["text"] = clean_text(text)
        return widget_dict["text"]

    @staticmethod
    def _split_sentences(text, lang, is_split):
        return text.splitlines() if is_split else split_sents(text, lang)

    def _start_alignment(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Warning", "Alignment is already in progress.")
            return
        if not self.source_file_widget or not self.source_file_widget["path_edit"].text():
            QMessageBox.critical(self, "Source Error", "No source file has been selected.")
            return
        if not os.path.isdir(MODEL_PATH):
            QMessageBox.critical(
                self, "Model Error",
                f"The model directory is not valid:\n{MODEL_PATH}",
            )
            return

        tgt_widgets = [w for w in self.target_file_widgets if w["path_edit"].text()]
        if not tgt_widgets:
            QMessageBox.warning(self, "Warning", "No valid target files selected.")
            return

        try:
            self._log("--- Preparing texts for alignment ---")
            is_split = self.split_checkbox.isChecked()

            src_text = self._read_text_if_needed(self.source_file_widget)
            src_lang = self.source_file_widget["lang_combo"].currentData()
            src_sents = self._split_sentences(src_text, src_lang, is_split)
            src_info = {"lang": src_lang, "sents": src_sents}
            self.column_lang_map = {"Source": src_lang}
            self._log(
                f"Processed source: "
                f"{os.path.basename(self.source_file_widget['path_edit'].text())}"
            )

            tgt_infos = []
            for i, w in enumerate(tgt_widgets):
                text = self._read_text_if_needed(w)
                lang = w["lang_combo"].currentData()
                sents = self._split_sentences(text, lang, is_split)
                tgt_infos.append({"lang": lang, "sents": sents, "path": w["path_edit"].text()})
                col_name = f"Target {i + 1}"
                self.column_lang_map[col_name] = lang
                self._log(f"Processed target: {os.path.basename(w['path_edit'].text())}")

        except Exception as e:
            QMessageBox.critical(
                self, "File Processing Error",
                f"Failed to read or process input files:\n{e}",
            )
            self._log(f"ERROR: File processing failed: {e}")
            self.align_button.setEnabled(True)
            return

        self.align_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._log("--- Starting Alignment Process ---")

        self.worker = AlignmentWorker(src_info, tgt_infos, MODEL_PATH)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.status_updated.connect(self._status_and_log)
        self.worker.alignment_completed.connect(self._on_alignment_completed)
        self.worker.error_occurred.connect(self._on_alignment_error)
        self.worker.start()

    def _on_alignment_completed(self, alignments, texts):
        self.alignments = alignments
        self.texts = texts
        self._populate_results_table()
        self.tab_widget.setCurrentIndex(1)
        self.align_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(
            f"Alignment completed: {len(alignments)} units found. Resizing table..."
        )
        self._log(f"SUCCESS: {len(alignments)} alignment units found.")
        self._log("--- Alignment Finished ---")

    def _on_alignment_error(self, error_message):
        QMessageBox.critical(
            self, "Alignment Error",
            f"An error occurred during alignment:\n{error_message}",
        )
        self.align_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Alignment failed")
        self._log(f"ERROR: Alignment failed: {error_message}")
        self._log("--- Alignment Finished with Errors ---")

    # ── Results Table ────────────────────────────────────────────────────

    def _populate_results_table(self):
        if not self.alignments or not self.texts:
            return
        table = self.results_table
        table.clear()
        table.undo_stack.clear()

        languages = list(self.texts.keys())
        table.setColumnCount(len(languages))
        table.setHorizontalHeaderLabels(languages)
        table.horizontalHeader().setStyleSheet(
            "QHeaderView::section { font-weight: bold; }"
        )
        self.statusBar().showMessage("Populating table data...")
        table.setRowCount(len(self.alignments))

        table.blockSignals(True)
        for row, alignment in enumerate(self.alignments):
            for col, lang in enumerate(languages):
                text = " ".join(self.texts[lang][i] for i in alignment.get(lang, []))
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, False)
                table.setItem(row, col, item)
        table.blockSignals(False)

        header = table.horizontalHeader()
        for col in range(len(languages)):
            header.setSectionResizeMode(col, QHeaderView.Stretch)

        if table.rowCount() > 0:
            self._resize_row = 0
            self._resize_timer.start(0)
        self._update_stats_label()

    def _progressively_resize_rows(self):
        table = self.results_table
        total = table.rowCount()
        chunk = 50
        end = min(self._resize_row + chunk, total)

        for row in range(self._resize_row, end):
            table.resizeRowToContents(row)
        self._resize_row = end

        pct = int((self._resize_row / total) * 100)
        self.statusBar().showMessage(f"Formatting table... {pct}% complete")

        if self._resize_row >= total:
            self._resize_timer.stop()
            self.statusBar().showMessage("Ready", 5000)
            self._log("Table formatting complete.")
            self._update_stats_label()

    def _update_stats_label(self):
        count = self.results_table.rowCount()
        self.stats_label.setText(
            f"Total alignment units: {count}" if count else "No alignment results"
        )

    # ── Table Data Access (shared by export & save) ──────────────────────

    def _get_headers(self):
        return [
            self.results_table.horizontalHeaderItem(c).text()
            for c in range(self.results_table.columnCount())
        ]

    def _get_cell_text(self, row, col):
        item = self.results_table.item(row, col)
        return item.text() if item else ""

    def _get_table_data(self):
        return [
            [self._get_cell_text(r, c) for c in range(self.results_table.columnCount())]
            for r in range(self.results_table.rowCount())
        ]

    # ── Export ───────────────────────────────────────────────────────────

    def _export_results(self):
        if self.results_table.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "No results to export.")
            return

        path, file_type = QFileDialog.getSaveFileName(
            self, "Export Results", "",
            "TMX Files (*.tmx);;TSV Files (*.tsv);;JSON Files (*.json)",
        )
        if not path:
            return

        exporters = {
            "tmx": self._export_tmx,
            "tsv": self._export_tsv,
            "json": self._export_json,
        }
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        # Match on filter string first, then fall back to extension
        exporter = None
        for key, fn in exporters.items():
            if key in file_type.lower() or ext == key:
                exporter = fn
                break

        if exporter is None:
            return
        try:
            exporter(path)
            QMessageBox.information(self, "Success", f"Results exported to {path}")
            self._log(f"Results exported successfully to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")
            self._log(f"ERROR: Failed to export results: {e}")

    def _export_tsv(self, file_path):
        headers = self._get_headers()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\t".join(headers) + "\n")
            for row in range(self.results_table.rowCount()):
                cells = [self._get_cell_text(row, c) for c in range(len(headers))]
                f.write("\t".join(cells) + "\n")

    def _export_json(self, file_path):
        headers = self._get_headers()
        data = [
            {h: self._get_cell_text(row, c) for c, h in enumerate(headers)}
            for row in range(self.results_table.rowCount())
        ]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _export_tmx(self, file_path):
        if not self.column_lang_map:
            QMessageBox.warning(
                self, "Export Error",
                "Cannot export to TMX: language information is missing. "
                "Please re-run the alignment or load a compatible project.",
            )
            self._log("ERROR: TMX export failed — missing column language map.")
            return

        headers = self._get_headers()
        if not headers:
            return

        src_header = headers[0]
        try:
            src_tmx = TMX_LANG_MAP.get(
                self.column_lang_map[src_header],
                self.column_lang_map[src_header],
            )
        except KeyError:
            QMessageBox.warning(
                self, "Export Error",
                f"Could not find source language for column '{src_header}'.",
            )
            return

        lines = [
            f'<?xml version="1.0" encoding="UTF-8" ?>',
            f'<tmx version="1.4">',
            f'<header creationtool="Bertalign-Fast" creationtoolversion="1.0" '
            f'segtype="sentence" o-tmf="unknown" adminlang="en-US" '
            f'srclang="{src_tmx}" datatype="plaintext" />',
            f"<body>",
        ]

        tu_tmpl = (
            "<tu>\n"
            '<tuv xml:lang="{src_lang}"><seg>{src}</seg></tuv>\n'
            '<tuv xml:lang="{tgt_lang}"><seg>{tgt}</seg></tuv>\n'
            "</tu>"
        )

        for row in range(self.results_table.rowCount()):
            src_text = self._get_cell_text(row, 0).strip()
            if not src_text:
                continue
            for col in range(1, len(headers)):
                tgt_text = self._get_cell_text(row, col).strip()
                if not tgt_text:
                    continue
                tgt_iso = self.column_lang_map.get(headers[col])
                if tgt_iso is None:
                    self._log(
                        f"WARNING: Skipping column '{headers[col]}' in row {row + 1} "
                        f"— missing language info."
                    )
                    continue
                tgt_tmx = TMX_LANG_MAP.get(tgt_iso, tgt_iso)
                lines.append(tu_tmpl.format(
                    src_lang=src_tmx,
                    tgt_lang=tgt_tmx,
                    src=html.escape(src_text),
                    tgt=html.escape(tgt_text),
                ))

        lines.append("</body>")
        lines.append("</tmx>")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ── Project Save / Load ──────────────────────────────────────────────

    def save_project(self):
        if self.results_table.rowCount() == 0 and not self.source_file_widget:
            QMessageBox.warning(self, "Warning", "No data to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "Project Files (*.alp)",
        )
        if not path:
            return
        try:
            src_info = None
            if self.source_file_widget:
                src_info = {
                    "path": self.source_file_widget["path_edit"].text(),
                    "lang": self.source_file_widget["lang_combo"].currentData(),
                }

            tgt_info = [
                {"path": w["path_edit"].text(), "lang": w["lang_combo"].currentData()}
                for w in self.target_file_widgets
            ]

            marked = [
                r for r in range(self.results_table.rowCount())
                if self.results_table.is_row_marked(r)
            ]

            project = {
                "version": "0.1.0",
                "src_info": src_info,
                "tgt_files_info": tgt_info,
                "is_split": self.split_checkbox.isChecked(),
                "alignments": self.alignments,
                "texts": self.texts,
                "table_data": self._get_table_data(),
                "headers": self._get_headers() if self.results_table.columnCount() > 0 else [],
                "marked_rows": marked,
                "column_lang_map": self.column_lang_map,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(project, f, ensure_ascii=False, indent=2)
            self.results_table.undo_stack.setClean()
            QMessageBox.information(self, "Success", f"Project saved to {path}")
            self._log(f"Project saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")
            self._log(f"ERROR: Failed to save project: {e}")

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", "Project Files (*.alp)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._clear_project_state()
            self._log(f"--- Loading project from {os.path.basename(path)} ---")

            self.column_lang_map = data.get("column_lang_map", {})
            self.alignments = data.get("alignments", [])
            self.texts = data.get("texts", {})

            src_info = data.get("src_info")
            if src_info and src_info.get("path"):
                self.source_file_widget["path_edit"].setText(src_info["path"])
                idx = self.source_file_widget["lang_combo"].findData(src_info.get("lang"))
                if idx != -1:
                    self.source_file_widget["lang_combo"].setCurrentIndex(idx)
                self._log(f"Loaded source file: {src_info['path']}")

            tgt_files = data.get("tgt_files_info", [])
            if tgt_files:
                first = tgt_files[0]
                self.target_file_widgets[0]["path_edit"].setText(first.get("path"))
                idx = self.target_file_widgets[0]["lang_combo"].findData(first.get("lang"))
                if idx != -1:
                    self.target_file_widgets[0]["lang_combo"].setCurrentIndex(idx)
                self._log(f"Loaded target file: {first.get('path')}")
            for info in tgt_files[1:]:
                self._add_target_file_row(
                    file_path=info.get("path"), lang_iso=info.get("lang"),
                )
                self._log(f"Loaded target file: {info.get('path')}")

            self.split_checkbox.setChecked(data.get("is_split", False))
            self._log("Using bundled model from 'model' directory.")

            headers = data.get("headers", [])
            table_data = data.get("table_data", [])
            if not headers and self.texts:
                headers = list(self.texts.keys())
            self._restore_table_data(table_data, headers)

            marked = data.get("marked_rows", [])
            if marked:
                self.results_table.mark_rows_by_index(marked)

            self._update_align_button_state()
            self.export_button.setEnabled(self.results_table.rowCount() > 0)
            self.save_button.setEnabled(True)
            self.results_table.undo_stack.setClean()
            QMessageBox.information(self, "Success", "Project loaded successfully")
            self._log("Project loaded successfully.")
            self.tab_widget.setCurrentIndex(1)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load project:\n{e}")
            self._log(f"ERROR: Failed to load project: {e}")

    def _restore_table_data(self, table_data, headers):
        self.results_table.clear()
        self.results_table.undo_stack.clear()
        if not headers and not table_data:
            return

        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.setRowCount(len(table_data))

        self.results_table.blockSignals(True)
        for row, row_data in enumerate(table_data):
            for col, text in enumerate(row_data):
                if col < self.results_table.columnCount():
                    item = QTableWidgetItem(text)
                    item.setData(Qt.UserRole, False)
                    self.results_table.setItem(row, col, item)
        self.results_table.blockSignals(False)

        header = self.results_table.horizontalHeader()
        for col in range(len(headers)):
            header.setSectionResizeMode(col, QHeaderView.Stretch)
        self._update_stats_label()

    def _clear_project_state(self):
        self.src_file = None
        self.alignments = []
        self.texts = {}
        self.column_lang_map = {}

        if self.source_file_widget:
            self.source_file_widget["path_edit"].clear()
            self.source_file_widget["text"] = None
            self.src_file = None

        for wd in reversed(self.target_file_widgets[1:]):
            self._remove_target_file_row(wd)

        if self.target_file_widgets:
            self.target_file_widgets[0]["path_edit"].clear()
            self.target_file_widgets[0]["text"] = None

        self.log_area.clear()
        self.results_table.clear()
        self.results_table.setRowCount(0)
        self.results_table.setColumnCount(0)
        self.results_table.undo_stack.clear()
        self.stats_label.setText("No alignment results")
        self.align_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.tab_widget.setCurrentIndex(0)
        self.statusBar().showMessage("Ready")

    def new_project(self):
        if not self.results_table.undo_stack.isClean():
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Start a new project anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return
        self._clear_project_state()
        self._log("--- New project started ---")
        self.statusBar().showMessage("New project started")

    # ── About / Close ────────────────────────────────────────────────────

    def _show_about(self):
        about = (
            "<h3>Bertalign-Fast v0.1.0</h3>"
            "<p>A PyQt5 GUI for aligning multilingual parallel corpora.</p>"
            "<h4>Features:</h4><ul>"
            "<li>Inference optimization for embedding texts on CPU</li>"
            "<li>Automatic language detection with user override</li>"
            "<li>Alignment of multiple target texts</li>"
            "<li>Interactive table editing (split, merge, delete, add)</li>"
            "<li>Undo/Redo support (Ctrl+Z / Ctrl+Y)</li>"
            "<li>Mark/Unmark rows with a distinct color</li>"
            "<li>Export to TMX, TSV, and JSON formats</li>"
            "<li>Project save/load (including marked rows)</li>"
            "</ul>"
            "<h4>Editing Actions:</h4><ul>"
            "<li>Edit a cell: Double-click the cell.</li>"
            "<li>Mark rows: Select rows, right-click → Mark/Unmark.</li>"
            "<li>Split text: While editing, right-click → Split at cursor.</li>"
            "<li>Move text: While editing, right-click → Move Up/Down.</li>"
            "<li>Merge rows: Select rows, right-click → Merge.</li>"
            "</ul>"
        )
        msg = QMessageBox()
        msg.setWindowTitle("About")
        msg.setTextFormat(Qt.RichText)
        msg.setText(about)
        msg.exec_()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Exit",
                "Alignment is in progress. Force quit?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.worker.terminate()
                self.worker.wait(2000)
            else:
                event.ignore()
                return

        if not self.results_table.undo_stack.isClean():
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Save before exiting?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self.save_project()
                if not self.results_table.undo_stack.isClean():
                    event.ignore()
                    return
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        event.accept()


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Corpus Alignment Tool")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("BertalignFast")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyle("Fusion")

    window = CorpusAlignmentGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
