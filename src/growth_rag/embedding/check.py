"""Compare two encoders on the same texts (`rag embed-check`).

Used to verify that the manual implementation reproduces FlagEmbedding:
- dense: cosine similarity between the two vectors of each text (expect > 0.999
  in fp32; slightly lower in fp16, rounding differs between implementations);
- sparse: overlap of the token sets (Jaccard) and relative difference of the
  sparse self-similarity (the score Qdrant would compute).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from growth_rag.embedding.encoders.base import Encoder, sparse_dot


@dataclass
class Comparison:
    texts: int
    dense_cosine_min: float
    dense_cosine_mean: float
    sparse_jaccard_mean: float
    sparse_score_rel_diff_max: float


def compare_encoders(a: Encoder, b: Encoder, texts: Sequence[str]) -> Comparison:
    ea, eb = a.encode(texts), b.encode(texts)
    cos = np.sum(ea.dense * eb.dense, axis=1)
    jaccard, rel = [], []
    for sa, sb in zip(ea.sparse, eb.sparse):
        ta, tb = set(sa[0].tolist()), set(sb[0].tolist())
        jaccard.append(len(ta & tb) / max(len(ta | tb), 1))
        da, db = sparse_dot(sa, sa), sparse_dot(sb, sb)
        rel.append(abs(da - db) / max(da, db, 1e-9))
    return Comparison(
        texts=len(texts),
        dense_cosine_min=round(float(cos.min()), 5),
        dense_cosine_mean=round(float(cos.mean()), 5),
        sparse_jaccard_mean=round(float(np.mean(jaccard)), 4),
        sparse_score_rel_diff_max=round(float(np.max(rel)), 4),
    )
