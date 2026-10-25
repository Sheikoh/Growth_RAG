"""A deterministic stand-in for the cross-encoder: the score is the number of
question words found in the passage (no torch, no model)."""

from __future__ import annotations

import re

import numpy as np


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


class FakeReranker:
    name = "fake/reranker"
    backend = "fake"
    revision = None

    def __init__(self, bias: float = 0.0):
        self.bias = bias  # shifts every score: same order, different logits
        self.calls: list[int] = []

    def score(self, query, passages):
        self.calls.append(len(passages))
        q = _words(query)
        return np.array([len(q & _words(p)) + self.bias for p in passages], dtype=np.float32)
