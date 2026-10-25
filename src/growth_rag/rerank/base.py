"""Reranking: rescore (question, passage) pairs with a cross-encoder.

Why it helps (README 6, upgrade 2; 8.6): the first search compares two vectors
computed separately (the question's and the chunk's), so the model never sees
them together. A cross-encoder reads the question and the passage *as one
input* and outputs one relevance score; every word of the question can attend
to every word of the passage. Much more accurate, much slower: one model pass
per pair. Hence two stages: a fast search collects candidates (hybrid top N),
the reranker reorders only those N.

    candidates = hybrid search, top N
    score(q, c) = cross_encoder([q ; c])          one number per pair (a logit)
    results = candidates sorted by score, best first

The passage given to the reranker is the same text the chunk was embedded
with: header (title, author, section) + text.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from growth_rag.index.search import Hit


class RerankerUnavailable(RuntimeError):
    def __init__(self, backend: str, model: str, cause: Exception):
        hint = ("install it with `uv sync --extra flag`" if backend == "flag"
                else "check `transformers`/`torch` are installed (uv sync)")
        super().__init__(
            f"Cannot load the {backend!r} reranker for {model!r} ({type(cause).__name__}: {cause}). "
            f"Model files: `uv run rag model --rerank` downloads or locates them; otherwise {hint}."
        )


class Reranker(Protocol):
    name: str
    backend: str
    revision: str | None

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """One relevance score per passage (higher = more relevant)."""
        ...


def passage_text(payload: dict) -> str:
    """The chunk as embedded: header + text (Chunk.embedding_text)."""
    header, text = payload.get("header") or "", payload.get("text") or ""
    return f"{header}\n\n{text}" if header else text


def rerank(reranker: Reranker, query: str, hits: list[Hit]) -> list[Hit]:
    """Candidates reordered by the reranker's score, best first. Each Hit's
    score becomes the reranker's (a logit: comparable within a question only).
    Ties keep the search order (sorted() is stable)."""
    if not hits:
        return []
    scores = reranker.score(query, [passage_text(h.payload) for h in hits])
    order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))
    return [Hit(float(scores[i]), hits[i].payload) for i in order]


def sigmoid(x: float) -> float:
    """Logit -> (0, 1): easier to read, same order. Not a calibrated probability.
    Written so that exp() never overflows, whatever the sign of x."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def length_order(texts: Sequence[str]) -> list[int]:
    """Indexes sorted by length, longest first: batches of similar length waste
    less padding (results are put back in the input order by the caller)."""
    return sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
