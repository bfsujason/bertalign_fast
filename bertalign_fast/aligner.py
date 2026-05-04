import os
import time
import numpy as np

from bertalign_fast.utils import (
    clean_text,
    detect_lang,
    split_sents,
    SUPPORTED_LANGUAGES,
)
from bertalign_fast.corelib import (
    find_top_k_similar,
    get_alignment_types,
    find_first_pass_search_path,
    first_pass_align,
    first_pass_backtrack,
    find_second_pass_search_path,
    second_pass_align,
    second_pass_backtrack,
)
from bertalign_fast.encoder import Encoder

_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_file_dir)
MODEL_PATH = os.path.join(_parent_dir, 'models/static-similarity-mrl-multilingual-v1/0_StaticEmbedding')

class BertalignFast:
    """Two-pass sentence aligner built on static word embeddings.

    The first pass uses only 1-1 beads over a wide diagonal band to find
    anchor points. The second pass searches all m-to-n bead types inside
    a tighter band derived from those anchors.

    Attributes:
        languages:  List of supported languages. 
        src_sents:  List of source sentences (populated after align_sents).
        tgt_sents:  List of target sentences.
        alignment:  List of (src_indices, tgt_indices, bead_score).
        bitext:     List of (src_sent, tgt_sent)
    """

    def __init__(self, model_path=MODEL_PATH):
        self.encoder = Encoder(model_path)
        
        self.languages = list(
            SUPPORTED_LANGUAGES.values()
        )
        self.src_sents = []
        self.tgt_sents = []
        self.alignment = []
        self.bitext = []
        
    def align_sents(
        self,
        src_text,
        tgt_text,
        split=True,
        max_align=8,
        embedding_dim=None,
        mean_center=True,
        top_k=3,
        window_size=5,
        skip_penalty=-0.1,
        lambda_size=0.02,
        length_penalty=True,
    ):
        """Run the two-pass alignment pipeline on a source / target text pair.

        Args:
            src_text:           Raw source document (string).
            tgt_text:           Raw target document (string).
            split:              If True, use the built-in splitter; otherwise
                                treat each line as a sentence.
            max_align:          Maximum bead size: src_count + tgt_count <= max_align.
            embedding_dim:      Matryoshka truncation dimension. None keeps full 1024.
            mean_center:        If True, subtract the per-side centroid from each bead
                                before L2-normalisation.
            top_k:              Top-k 1-1 candidates per source in the first pass.
            window_size:        Half-width added around each anchor when building
                                the second-pass search band.
            skip_penalty:       Fixed score for deletion / insertion beads.
            lambda_size:        Coefficient on the (m + n) size bonus,
                                added before the length penalty. Compensates for
                                the cosine bias against larger beads. 0 disables.
                                Typical useful values are 0.01 - 0.05.
            length_penalty:     If True, multiply each bead score by
                                sqrt(min / max) of source / target byte lengths.
        """
        start_time = time.time()
        
        # --- Preprocessing ---
        src_text = clean_text(src_text)
        tgt_text = clean_text(tgt_text)
        src_lang_code, src_lang = detect_lang(src_text)
        tgt_lang_code, tgt_lang = detect_lang(tgt_text)

        if split:
            src_sents = split_sents(src_text, src_lang_code)
            tgt_sents = split_sents(tgt_text, tgt_lang_code)
        else:
            src_sents = src_text.splitlines()
            tgt_sents = tgt_text.splitlines()
            
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents

        source_length = len(src_sents)
        target_length = len(tgt_sents)

        print(f"Source language: {src_lang}, Number of sentences: {source_length}")
        print(f"Target language: {tgt_lang}, Number of sentences: {target_length}")

        # --- Embedding ---
        print("Embedding source and target text ...")
        src_vecs, src_lengths = self.encoder.transform(
            src_sents,
            max_align - 1,
            embedding_dim=embedding_dim,
            mean_center=mean_center,
        )
        tgt_vecs, tgt_lengths = self.encoder.transform(
            tgt_sents,
            max_align - 1,
            embedding_dim=embedding_dim,
            mean_center=mean_center,
        )

        # --- First pass: extract 1-1 anchor beads ---
        print("Performing first-pass alignment ...")
        similarities, top_k_indices = find_top_k_similar(
            src_vecs[0], tgt_vecs[0], k=top_k,
        )
        first_window, first_path = find_first_pass_search_path(
            source_length, target_length,
        )
        first_backpointers = first_pass_align(
            source_length,
            target_length,
            first_window,
            first_path,
            similarities,
            top_k_indices,
        )
        first_alignment = first_pass_backtrack(
            source_length,
            target_length,
            first_backpointers,
            first_path,
        )

        # --- Second pass: full m-to-n alignment ---
        print("Performing second-pass alignment ...")

        # Convert 0-based sentence indices back to 1-based DP coordinates
        # for the search-path builder, which operates in DP grid space.
        first_anchors_dp = [(s + 1, t + 1) for s, t in first_alignment]

        second_alignment_types = get_alignment_types(max_align)
        second_window, second_path = find_second_pass_search_path(
            first_anchors_dp, window_size, source_length, target_length,
        )
        second_backpointers, second_cost = second_pass_align(
            src_vecs,
            tgt_vecs,
            src_lengths,
            tgt_lengths,
            second_window,
            second_path,
            second_alignment_types,
            skip_penalty,
            lambda_size,
            length_penalty=length_penalty,
        )
        second_alignment = second_pass_backtrack(
            source_length,
            target_length,
            second_backpointers,
            second_cost,
            second_path,
            second_alignment_types,
        )
        
        self.alignment = second_alignment
        self.bitext = self.get_bitext()

        elapsed = time.time() - start_time
        print(
            f"Finished! Aligned {source_length} {src_lang} sentences "
            f"to {target_length} {tgt_lang} sentences."
        )
        print(f"Time spent: {elapsed:.2f} secs\n")
        
    def get_bitext(self):
        bitext = []
        for bead in (self.alignment):
            src_line = _join_sentences(bead[0], self.src_sents)
            tgt_line = _join_sentences(bead[1], self.tgt_sents)
            #print(src_line + "\n" + tgt_line + "\n")
            bitext.append((src_line, tgt_line))
        return bitext

def _join_sentences(indices, sentences):
    """Concatenate the sentences at indices into a single string."""
    if not indices:
        return ""
    return " ".join(sentences[indices[0] : indices[-1] + 1])
    