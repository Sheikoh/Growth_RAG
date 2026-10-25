"""A deterministic stand-in for bge-m3, so embedding, indexing and search are
tested without torch or a model download. Words are hashed to token ids; the
dense vector is a normalised bag of words, the sparse vector its word counts."""

from __future__ import annotations

import re
import zlib

import numpy as np

from growth_rag.embedding.encoders.base import Encoded, l2_normalize

DIM = 32


def _ids(text: str) -> list[int]:
    return [zlib.crc32(w.encode()) % 5000 + 10 for w in re.findall(r"\w+", text.lower())]


class FakeEncoder:
    name = "fake/bge"
    backend = "fake"
    dim = DIM

    def __init__(self):
        self.batches: list[int] = []  # sizes of the batches received

    def encode(self, texts):
        self.batches.append(len(texts))
        dense = np.zeros((len(texts), DIM), dtype=np.float32)
        sparse = []
        for i, text in enumerate(texts):
            counts: dict[int, float] = {}
            for t in _ids(text):
                dense[i, t % DIM] += 1.0
                counts[t] = counts.get(t, 0.0) + 1.0
            if not counts:
                dense[i, 0] = 1.0
            ids = np.array(sorted(counts), dtype=np.int32)
            sparse.append((ids, np.array([counts[k] for k in ids], dtype=np.float32)))
        return Encoded(dense=l2_normalize(dense), sparse=sparse)
