"""What every encoder returns, and the model-independent maths of bge-m3.

bge-m3 runs one transformer pass and reads two things from its last hidden
states (shape [batch, tokens, 1024]):

- **dense**: the hidden state of the first token ([CLS]), L2-normalised, so
  cosine similarity = dot product;
- **sparse** ("lexical weights"): a small linear layer (1024 -> 1, shipped as
  `sparse_linear.pt`) gives each token a score; ReLU keeps positive ones. The
  sparse vector maps *token ids* to weights: when a token occurs several times,
  its highest weight is kept; special tokens are ignored. It behaves like a
  learned BM25: exact terms matter (names, rare words), and Qdrant stores it as
  a sparse vector over the 250k-token vocabulary.

These functions take plain arrays, so they are tested without torch.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

Sparse = tuple[np.ndarray, np.ndarray]  # (indices int32, sorted; values float32)


@dataclass
class Encoded:
    dense: np.ndarray  # [n, dim] float32, L2-normalised
    sparse: list[Sparse]


class Encoder(Protocol):
    name: str  # model id, e.g. "BAAI/bge-m3"
    backend: str  # "manual" or "flag"
    dim: int

    def encode(self, texts: Sequence[str]) -> Encoded: ...


class EncoderUnavailable(RuntimeError):
    def __init__(self, backend: str, model: str, cause: Exception):
        hint = (
            "install it with `uv sync --extra flag`" if backend == "flag"
            else "check `transformers`/`torch` are installed (uv sync)"
        )
        super().__init__(
            f"Cannot load the {backend!r} encoder for {model!r} ({type(cause).__name__}: {cause}). "
            f"Model files: `uv run rag model` downloads or locates them; otherwise {hint}."
        )


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32)


def cls_pooling(hidden: np.ndarray) -> np.ndarray:
    """[batch, tokens, dim] -> [batch, dim]: the first token's hidden state."""
    return hidden[:, 0]


def lexical_weights(
    input_ids: Iterable[Sequence[int]],
    token_weights: Iterable[Sequence[float]],
    ignore_ids: set[int],
) -> list[Sparse]:
    """Token scores -> one sparse vector per text: max weight per token id,
    positive weights only, special tokens ignored. Padding positions must be
    removed (or be ids in `ignore_ids`)."""
    out: list[Sparse] = []
    for ids, weights in zip(input_ids, token_weights):
        best: dict[int, float] = {}
        for token, w in zip(ids, weights):
            token, w = int(token), float(w)
            if token in ignore_ids or w <= 0:
                continue
            if w > best.get(token, 0.0):
                best[token] = w
        indices = np.array(sorted(best), dtype=np.int32)
        values = np.array([best[i] for i in indices], dtype=np.float32)
        out.append((indices, values))
    return out


def sparse_from_dict(weights: dict) -> Sparse:
    """FlagEmbedding format ({"1234": 0.31, ...}) -> (indices, values)."""
    items = sorted((int(k), float(v)) for k, v in weights.items() if float(v) > 0)
    return (np.array([k for k, _ in items], dtype=np.int32),
            np.array([v for _, v in items], dtype=np.float32))


def sparse_dot(a: Sparse, b: Sparse) -> float:
    """Similarity of two sparse vectors (what Qdrant computes for sparse search)."""
    common, ia, ib = np.intersect1d(a[0], b[0], return_indices=True)
    return float(np.dot(a[1][ia], b[1][ib])) if len(common) else 0.0
