import os
import time

import numpy as np

from tokenizers import Tokenizer
from safetensors.numpy import load_file

# Maximum character length for any single overlap string sent to the encoder,
# preventing degenerate runtime on very long concatenations.
_MAX_OVERLAP_CHARS = 10_000

# Placeholder text for empty lines so the encoder always receives valid input.
_BLANK_PLACEHOLDER = "BLANK_LINE"
_PAD_PLACEHOLDER = "PAD"


class Encoder:
    """Static-embedding multilingual encoder for sentence alignment.

    Wraps a SentenceTransformer-style static model, which encodes
    every sentence and every consecutive group of 2, 3, … num_overlaps
    sentences. This lets the downstream aligner score m-to-n beads by
    comparing the concatenated-sentence embedding on one side against
    the concatenated-sentence embedding on the other.

    Args:
        model_path: Local path to the model directory containing
                    tokenizer.json and model.safetensors.
    """

    def __init__(self, model_path):
        print(f"Loading embedding model '{model_path}' ...")
        start_time = time.time()
        
        tokenizer_path = os.path.join(model_path, 'tokenizer.json')
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        weights_path = os.path.join(model_path, 'model.safetensors')
        weights = load_file(weights_path)
        self.embedding_table = list(weights.values())[0]
        
        elapsed = time.time() - start_time
        print(f"Time spent: {elapsed:.2f} secs\n")

    def transform(self, sentences, num_overlaps, embedding_dim=None, mean_center=False):
        """Encode sentences and their consecutive overlaps.

        For N sentences and num_overlaps = K, the method produces K layers
        of embeddings: layer 0 holds single-sentence vectors, layer 1 holds
        vectors for consecutive pairs, and so on up to layer K−1.

        Args:
            sentences:          List of N sentence strings.
            num_overlaps:       Number of overlap layers (typically max_align − 1).
            embedding_dim:      Matryoshka truncation dimension. None keeps full 1024.
            mean_center:        If True, subtract the per-side centroid (computed over
                                all bead embeddings of every size) from each bead before
                                L2-normalisation.

        Returns:
            embedding_matrix:   float array, shape (num_overlaps, N, embedding_dim).
                                L2-normalised bead embeddings.
            length_matrix:      int array, shape (num_overlaps, N). 
                                UTF-8 byte length of each bead concatenation.
        """
        overlap_strings = list(self._yield_overlaps(sentences, num_overlaps))
        vectors = self.encode(overlap_strings, embedding_dim=embedding_dim)
        
        if mean_center:
            vectors -= np.mean(vectors, axis=0, keepdims=True)  
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-9)
        
        num_sentences = len(sentences)
        embedding_dim = vectors.shape[1]
        embedding_matrix = vectors.reshape(num_overlaps, num_sentences, embedding_dim)

        byte_lengths = np.array(
            [len(s.encode("utf-8")) for s in overlap_strings]
        )
        length_matrix = byte_lengths.reshape(num_overlaps, num_sentences)

        return embedding_matrix, length_matrix
        
    def encode(self, texts, embedding_dim=None):
        """Encode texts into vectors, with optional Matryoshka truncation.

        Tokenizes each input text, looks up token embeddings from the embedding
        table, and produces a single vector per text by mean-pooling over its
        valid token embeddings.

        Args:
            texts:          Input texts to encode.
            embedding_dim:  If true, truncate each output vector
                            to this many leading dimensions.
        Returns:
            emb:            float array, shape (len(texts), D)
                            where D is embedding_dim if provided, otherwise
                            the full embedding dimension.
        """
        out = []
        for text in texts:
            ids = self.tokenizer.encode(text).ids
            ids = [i for i in ids if 0 <= i < len(self.embedding_table)]
            vec = self.embedding_table[ids].mean(axis=0) if ids else np.zeros(self.embedding_table.shape[1])
            out.append(vec.astype(np.float32))
        emb = np.stack(out)
        return emb[:, :embedding_dim] if embedding_dim else emb

    # ------------------------------------------------------------------
    # Overlap generation (based on Vecalign)
    # https://github.com/thompsonb/vecalign
    # ------------------------------------------------------------------

    @staticmethod
    def _yield_overlaps(sentences, num_overlaps):
        """Yield concatenated sentence strings for overlap layers 1 … num_overlaps.

        Layer k concatenates every k consecutive sentences. Positions near
        the start that have fewer than k predecessors are filled with a PAD
        token to keep each layer exactly N elements long.
        """
        cleaned = [
            line.strip() if line.strip() else _BLANK_PLACEHOLDER
            for line in sentences
        ]

        for layer_size in range(1, num_overlaps + 1):
            for text in _build_overlap_layer(cleaned, layer_size):
                yield text[:_MAX_OVERLAP_CHARS]

def _build_overlap_layer(lines, layer_size, separator=" "):
    """Concatenate every layer_size consecutive lines into one string.

    Positions at the start of the list that cannot form a full group are
    filled with a PAD placeholder so the output always has len(lines) elements.

    Args:
        lines:      List of pre-cleaned sentence strings.
        layer_size: Number of consecutive sentences to concatenate.
        separator:  String used to join consecutive sentences.

    Returns:
        List of len(lines) strings.

    Raises:
        ValueError: If layer_size < 1.
    """
    if layer_size < 1:
        raise ValueError(f"layer_size must be >= 1, got {layer_size}")

    num_pad = min(layer_size - 1, len(lines))
    padded = [_PAD_PLACEHOLDER] * num_pad

    for start in range(len(lines) - layer_size + 1):
        padded.append(separator.join(lines[start : start + layer_size]))

    return padded
    