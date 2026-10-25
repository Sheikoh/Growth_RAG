"""Compare two rerankers on the same pairs (`rag rerank-check`): the manual
version must reproduce FlagEmbedding.

- max absolute difference of the logits (expect < 0.05 in fp16: rounding
  differs between implementations);
- rank correlation (Spearman) of the two orders (expect > 0.99);
- same best passage for every question (expect all).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from growth_rag.rerank.base import Reranker


@dataclass
class RerankComparison:
    questions: int
    pairs: int
    max_abs_diff: float
    spearman_min: float
    same_top1: int


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation: 1 = same order (no ties expected for float scores)."""
    if len(a) < 2:
        return 1.0
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def compare_rerankers(a: Reranker, b: Reranker, cases: Sequence[tuple[str, Sequence[str]]]) -> RerankComparison:
    diffs, rhos, top = [], [], 0
    for query, passages in cases:
        sa, sb = a.score(query, passages), b.score(query, passages)
        diffs.append(float(np.max(np.abs(sa - sb))))
        rhos.append(spearman(sa, sb))
        top += int(np.argmax(sa) == np.argmax(sb))
    return RerankComparison(
        questions=len(cases), pairs=sum(len(p) for _, p in cases),
        max_abs_diff=round(max(diffs, default=0.0), 4),
        spearman_min=round(min(rhos, default=1.0), 4), same_top1=top,
    )
