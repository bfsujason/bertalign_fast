import numpy as np
import numba as nb

from usearch.index import Index

def second_pass_backtrack(
    source_length,
    target_length,
    backpointers,
    cost,
    search_path,
    alignment_types,
):
    """Trace back through the second-pass DP table to extract m-to-n beads.

    Walks from the terminal cell (source_length, target_length) back to the
    origin (0, 0), recording the sentence ranges and bead score for every
    alignment on the optimal path.

    The bead score is the difference between the DP cost at the current cell
    and its predecessor, isolating the contribution of that single bead.

    Args:
        source_length:      Number of source sentences.
        target_length:      Number of target sentences.
        backpointers:       uint8 array, shape (source_length + 1, max_band_width).
                            Each cell stores the alignment_types of the winning transition.
        cost:               float32 array, shape (source_length + 1, max_band_width).
                            Each cell stores the cumulative DP scores.
        search_path:        int array, shape (source_length + 1, 2). Row i holds
                            [band_start, band_end].
        alignment_types:    int array, shape (num_types, 2). Each row is
                            [src_step, tgt_step].

    Returns:
        alignments:         List of (src_indices, tgt_indices, bead_score) tuples.
                            src_indices and tgt_indices are lists of 0-based sentence indices;
                            bead_score is the contribution of this bead to the total DP cost.
    """
    alignments = []
    src_idx = source_length
    tgt_idx = target_length

    while src_idx > 0 or tgt_idx > 0:
        col_offset = tgt_idx - search_path[src_idx][0]
        transition = backpointers[src_idx][col_offset]

        src_step = int(alignment_types[transition][0])
        tgt_step = int(alignment_types[transition][1])

        # 0-based sentence indices covered by this bead, in ascending order.
        src_indices = list(range(src_idx - src_step, src_idx))
        tgt_indices = list(range(tgt_idx - tgt_step, tgt_idx))

        # Isolate this bead's score from the cumulative DP cost.
        prev_src = src_idx - src_step
        prev_tgt = tgt_idx - tgt_step
        prev_col_offset = prev_tgt - search_path[prev_src][0]
        bead_score = float(cost[src_idx][col_offset] - cost[prev_src][prev_col_offset])

        alignments.append((src_indices, tgt_indices, bead_score))

        src_idx = prev_src
        tgt_idx = prev_tgt

    alignments.reverse()
    return alignments

@nb.jit(nopython=True, fastmath=True, cache=True)
def second_pass_align(
    src_vecs,
    tgt_vecs,
    src_lengths,
    tgt_lengths,
    max_band_width,
    search_path,
    alignment_types,
    skip_penalty,
    lambda_size,
    length_penalty=True,
):
    """Perform second-pass DP alignment to extract m-to-n beads.

    Unlike the first pass (which only scores 1-1 beads), this pass evaluates
    all bead types up to the configured maximum alignment size. Each m-to-n
    bead is scored by:

        1. Cosine similarity between the concatenated source and target
           sentence embeddings.
        2. (Optional) A length-ratio penalty that down-weights beads
           whose source and target character counts are poorly balanced.

    Deletions (1, 0) and insertions (0, 1) receive a fixed skip_penalty
    instead of an embedding score.

    Args:
        src_vecs:           float array, shape (max_align−1, num_src, embedding_dim).
                            src_vecs[m−1, i−1] is the embedding for source sentences
                            [i−m .. i−1] concatenated.
        tgt_vecs:           float array, shape (max_align−1, num_tgt, embedding_dim).
                            tgt_vecs[m−1, i−1] is the embedding for target sentences
                            [i−m .. i−1] concatenated.
        src_lengths:        float array, shape (max_align−1, num_src).
                            Cumulative character lengths matching src_vecs.
        tgt_lengths:        float array, shape (max_align−1, num_tgt).
                            Cumulative character lengths matching tgt_vecs.
        max_band_width:     Column count of the compressed DP band.
        search_path:        int array, shape (num_src + 1, 2).
                            Row i holds [band_start, band_end].
        alignment_types:    int array, shape (num_types, 2).
                            Each row is [src_step, tgt_step].
        skip_penalty:       Fixed score added for insertions and deletions.
        lambda_size:        Coefficient on the (m + n - 2) size bonus. Set to
                            0 to disable.
        length_penalty:     If True, multiply embedding score by sqrt(min / max)
                            of source/target byte lengths.

    Returns:
        backpointers:       uint8 array, transition types, shape (num_src + 1, max_band_width).
        cost:               float32 array, DP scores, shape (num_src + 1, max_band_width). 
    """
    source_length = src_vecs.shape[1]
    target_length = tgt_vecs.shape[1]
    num_alignment_types = alignment_types.shape[0]

    # Compressed DP tables: only band columns are stored per source row.
    cost = np.zeros((source_length + 1, max_band_width), dtype=nb.float32)
    backpointers = np.zeros((source_length + 1, max_band_width), dtype=nb.uint8)

    # Global character-length ratio used to normalise target lengths.
    char_ratio = np.sum(src_lengths[0]) / np.sum(tgt_lengths[0])

    for src_idx in range(source_length + 1):
        band_start = search_path[src_idx][0]
        band_end = search_path[src_idx][1]

        for tgt_idx in range(band_start, band_end + 1):
            if src_idx == 0 and tgt_idx == 0:
                continue  # Origin is pre-initialised to zero.

            best_score = -np.inf
            best_type = nb.uint8(0)

            for type_idx in range(num_alignment_types):
                src_step = alignment_types[type_idx][0]
                tgt_step = alignment_types[type_idx][1]

                prev_src = src_idx - src_step
                prev_tgt = tgt_idx - tgt_step
                if prev_src < 0 or prev_tgt < 0:
                    continue

                # Ensure the predecessor falls inside its own band.
                prev_band_start = search_path[prev_src][0]
                prev_band_end = search_path[prev_src][1]
                if prev_tgt < prev_band_start or prev_tgt > prev_band_end:
                    continue

                score = cost[prev_src][prev_tgt - prev_band_start]

                if src_step == 0 or tgt_step == 0:
                    # ---- Deletion / insertion: skip penalty ----
                    bead_score = skip_penalty
                else:
                    # ---- m-to-n bead: embedding similarity ----
                    src_vec = src_vecs[src_step - 1, src_idx - 1]
                    tgt_vec = tgt_vecs[tgt_step - 1, tgt_idx - 1]

                    bead_score = np.dot(src_vec, tgt_vec)
                    
                    # ---- Size compensation: offset the systematic dilution of larger beads ----
                    bead_score += lambda_size * (src_step + tgt_step)

                    # ---- Length-ratio penalty ----
                    if length_penalty:
                        src_char_len = src_lengths[src_step - 1, src_idx - 1]
                        tgt_char_len = tgt_lengths[tgt_step - 1, tgt_idx - 1] * char_ratio

                        short_side = min(src_char_len, tgt_char_len)
                        long_side = max(src_char_len, tgt_char_len)

                        penalty = np.sqrt(short_side / long_side)
                        bead_score *= penalty

                score += bead_score
                if score > best_score:
                    best_score = score
                    best_type = nb.uint8(type_idx)

            col_offset = tgt_idx - band_start
            cost[src_idx][col_offset] = best_score
            backpointers[src_idx][col_offset] = best_type

    return backpointers, cost
    
def find_second_pass_search_path(
    first_pass_alignments,
    window_size,
    source_length,
    target_length,
):
    """Build the diagonal-band search path for the second-pass alignment.

    Takes the 1-1 anchor points from the first pass and expands each pair
    of consecutive anchors into a rectangular band. For source rows between
    two anchors (prev_src, prev_tgt) and (src, tgt), every row gets the
    target range [prev_tgt − window_size, tgt + window_size], clamped to
    [0, target_length].

    Anchor coordinates are in DP-grid space (1-based), matching the rows
    and columns of the second-pass cost matrix.

    Args:
        first_pass_alignments:  List of (source, target) anchor tuples in
                                DP-grid coordinates, as returned by first_pass_backtrack
                                after adding 1 to each index.
        window_size:            Half-width added above/below each anchor band.
        source_length:          Number of source sentences.
        target_length:          Number of target sentences.

    Returns:
        max_band_width:         The widest band (inclusive) across all rows.
        search_path:            int array of shape (source_length + 1, 2).  Row i
                                holds [band_start, band_end] for DP source index i.
    """
    # --- Ensure the anchor list ends at the DP boundary (source_length, target_length) ---
    anchors = list(first_pass_alignments)  # Avoid mutating the caller's list.

    last_src, last_tgt = anchors[-1]
    if last_src == source_length and last_tgt == target_length:
        pass  # Already correct.
    elif last_src != source_length and last_tgt != target_length:
        anchors.append((source_length, target_length))  # Keep as intermediate anchor.
    else:
        anchors[-1] = (source_length, target_length)  # One coord matched — fix the other.

    # --- Build per-row target bands from consecutive anchor pairs ---
    search_path = np.empty((source_length + 1, 2), dtype=np.int64)
    max_band_width = 0

    prev_src, prev_tgt = 0, 0
    for src, tgt in anchors:
        band_start = max(0, prev_tgt - window_size)
        band_end = min(target_length, tgt + window_size)

        for row in range(prev_src + 1, src + 1):
            search_path[row] = [band_start, band_end]

        max_band_width = max(max_band_width, band_end - band_start)
        prev_src, prev_tgt = src, tgt

    # Row 0 inherits the same band as row 1.
    search_path[0] = search_path[1]
    max_band_width += 1  # Convert from width to inclusive column count.

    return max_band_width, search_path
            
def first_pass_backtrack(source_length, target_length, backpointers, search_path):
    """Trace back through the first-pass DP table to extract 1-1 alignments.

    Walks from the terminal cell (source_length, target_length) back to the
    origin (0, 0), collecting every cell where the best transition was a 1-1 bead.

    Transition encoding (matches first_pass_align):
        0 → insertion (0, 1):  step back in target only
        1 → deletion  (1, 0):  step back in source only
        2 → 1-1 bead  (1, 1):  step back in both (this is what we collect)

    Args:
        source_length:  Number of source sentences.
        target_length:  Number of target sentences.
        backpointers:   uint8 array of shape (source_length + 1, 2 * window_size + 1).
                        Each cell stores the winning transition index (0, 1, or 2).
        search_path:    int array of shape (source_length + 1, 2).  Row i holds
                        [band_start, band_end].

    Returns:
        alignments:     List of (source_index, target_index) tuples for every 1-1
                        bead on the optimal path, in reading order.
                        Indices are 0-based sentence positions.
    """
    # Step sizes for each transition type, indexed by backpointer value.
    SRC_STEP = (0, 1, 1)  # insertion, deletion, 1-1
    TGT_STEP = (1, 0, 1)

    BEAD_1_1 = 2

    alignments = []
    src_idx = source_length
    tgt_idx = target_length

    while src_idx > 0 or tgt_idx > 0:
        col_offset = tgt_idx - search_path[src_idx][0]
        transition = backpointers[src_idx][col_offset]

        if transition == BEAD_1_1:
            alignments.append((src_idx - 1, tgt_idx - 1))  # Convert to 0-based.

        src_idx -= SRC_STEP[transition]
        tgt_idx -= TGT_STEP[transition]

    alignments.reverse()
    return alignments
    
@nb.jit(nopython=True, fastmath=True, cache=True)
def first_pass_align(
    source_length,
    target_length,
    window_size,
    search_path,
    similarities,
    top_k_indices,
):
    """Perform first-pass DP alignment, scoring only 1-1 beads.

    Runs Viterbi-style dynamic programming over a diagonal band of the
    (source_length + 1) × (target_length + 1) alignment table.  Three
    transition types are considered at every cell:

        (0, 1)  insertion  — skip a target sentence (score: 0)
        (1, 0)  deletion   — skip a source sentence (score: 0)
        (1, 1)  1-1 bead   — align one source to one target (scored)

    Only 1-1 beads contribute a positive similarity score; deletions and
    insertions serve purely as structural transitions.

    The DP table is stored in compressed form: for each source index i only
    the columns inside the band are materialised, so the j-axis has width
    2 * window_size + 1 and the physical column is j - band_start.

    Args:
        source_length:  Number of source sentences.
        target_length:  Number of target sentences.
        window_size:    Half-width of the diagonal search band.
        search_path:    int array of shape (source_length + 1, 2).  Row i
                        holds [band_start, band_end] — the inclusive range of
                        valid target indices for source index i.
        similarities:   float array of shape (source_length, k).  Pre-computed
                        similarity scores for the top-k target matches of each
                        source sentence.
        top_k_indices:  int array of shape (source_length, k).  Target sentence
                        indices corresponding to similarities.

    Returns:
        backpointers:   uint8 array of shape (source_length + 1, 2 * window_size + 1).
                        Each cell stores the transition type that produced the
                        best score: 0 → insertion, 1 → deletion, 2 → 1-1 bead.
    """
    num_band_cols = 2 * window_size + 1
    top_k = top_k_indices.shape[1]

    # Compressed DP tables: only the band columns are stored per source row.
    cost = np.zeros((source_length + 1, num_band_cols), dtype=nb.float32)
    backpointers = np.zeros((source_length + 1, num_band_cols), dtype=nb.uint8)

    # Fixed transition types: (src_step, tgt_step)
    #   Index 0 → insertion (0, 1)
    #   Index 1 → deletion  (1, 0)
    #   Index 2 → 1-1 bead  (1, 1)
    TRANSITIONS = np.array([[0, 1], [1, 0], [1, 1]], dtype=nb.int32)

    for src_idx in range(source_length + 1):
        band_start = search_path[src_idx][0]
        band_end = search_path[src_idx][1]

        for tgt_idx in range(band_start, band_end + 1):
            if src_idx == 0 and tgt_idx == 0:
                continue  # Origin is pre-initialised to zero.

            best_score = -np.inf
            best_type = nb.uint8(0)

            for t in range(3):
                src_step = TRANSITIONS[t][0]
                tgt_step = TRANSITIONS[t][1]

                prev_src = src_idx - src_step
                prev_tgt = tgt_idx - tgt_step
                if prev_src < 0 or prev_tgt < 0:
                    continue

                # Ensure the predecessor falls inside its own band.
                prev_band_start = search_path[prev_src][0]
                prev_band_end = search_path[prev_src][1]
                if prev_tgt < prev_band_start or prev_tgt > prev_band_end:
                    continue

                score = cost[prev_src][prev_tgt - prev_band_start]

                # Only the 1-1 bead (t == 2) contributes a similarity score.
                if t == 2:
                    src_sent = src_idx - 1
                    tgt_sent = tgt_idx - 1
                    for k in range(top_k):
                        if top_k_indices[src_sent][k] == tgt_sent:
                            score += similarities[src_sent][k]
                            break  # Each target appears at most once.

                if score > best_score:
                    best_score = score
                    best_type = nb.uint8(t)

            col_offset = tgt_idx - band_start
            cost[src_idx][col_offset] = best_score
            backpointers[src_idx][col_offset] = best_type

    return backpointers
    
def find_first_pass_search_path(
    source_length,
    target_length,
    min_window_size=250,
    expansion_ratio=0.06,
):
    """Compute the search path for the first-pass alignment.

    In the DP table, searching every cell is O(source_length × target_length).
    This function restricts the search to a diagonal band: for each source index i,
    only target indices within a window centred on the diagonal position are considered.

    The window size is the larger of min_window_size and a expansion_ratio
    of the longer side, ensuring it scales with document length.

    Args:
        source_length:      Number of source sentences.
        target_length:      Number of target sentences.
        min_window_size:    Floor on the half-width of the diagonal band.
        expansion_ratio:    Fraction of the longer document.

    Returns:
        window_size:        Effective half-width of the diagonal band.
        search_path:        numpy array of shape (source_length + 1, 2).
                            Row i contains [band_start, band_end) — the range of
                            valid target indices for source index i.
    """
    longer_side = max(source_length, target_length)
    window_size = max(min_window_size, int(longer_side * expansion_ratio))

    # Ratio used to project a source index onto the target axis.
    target_source_ratio = target_length / source_length

    search_path = np.empty((source_length + 1, 2), dtype=np.int64)
    for src_idx in range(source_length + 1):
        center = int(target_source_ratio * src_idx)
        search_path[src_idx, 0] = max(0, center - window_size)
        search_path[src_idx, 1] = min(center + window_size, target_length)

    return window_size, search_path
    
def get_alignment_types(max_alignment_size):
    """Generate all valid sentence alignment types.

    An alignment type (m, n) means m consecutive source sentences are aligned
    to n consecutive target sentences.
    
    This includes deletion (1, 0) and insertion (0, 1) as special cases.
    For every other type, both m >= 1 and n >= 1, subject to m + n <= max_alignment_size.

    Args:
        max_alignment_size: Upper bound on the sum of source and target
                            sentence counts in a single alignment bead.

    Returns:
        alignment_types:    numpy array of shape (num_types, 2), where each row
                            is [source_count, target_count].
    """
    # Deletion (source sentence has no target) and insertion (vice versa).
    alignment_types = [[0, 1], [1, 0]]

    # Many-to-many alignment beads: (m, n) with m >= 1, n >= 1.
    for source_count in range(1, max_alignment_size):
        for target_count in range(1, max_alignment_size - source_count + 1):
            alignment_types.append([source_count, target_count])

    return np.array(alignment_types)
    
def find_top_k_similar(source_vectors, target_vectors, k=3):
    """Find the top-k most similar target vectors for each source vector.

    Use USearch to build an in-memory index over target_vectors,
    then perform an exact nearest-neighbour search for every vector
    in source_vectors.

    Args:
        source_vectors: float array, shape (num_source, embedding_dim).
        target_vectors: float array, shape (num_target, embedding_dim).
        k:              Number of most similar target vectors per source vector.

    Returns:
        similarities:   float array, shape (num_source, k).
                        Inner-product similarity scores in descending order.
        target_indices: int array, shape (num_source, k).
                        Corresponding indices into target_vectors.
    """
    embedding_dim = target_vectors.shape[1]

    # Build an exact-search index over the target vectors.
    index = Index(ndim=embedding_dim, metric="ip", dtype="f32")
    index.add(np.arange(len(target_vectors)), target_vectors)

    # Batch query: for every source vector, retrieve the k closest targets.
    matches = index.search(source_vectors, k, exact=True)
    
    similarities = matches.distances
    target_indices = matches.keys

    return similarities, target_indices
