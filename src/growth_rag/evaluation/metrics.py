"""Retrieval metrics for one question (pure functions, no I/O).

Vocabulary:
- results: the keys returned by a search, best first (chunk keys, or book
  keys for recommendations);
- entries: the question's relevant entries (one per evidence quote, or per
  book), each satisfied by a set of keys (a quote crossing a chunk boundary
  is satisfied by either chunk) and carrying a grade (2 answers, 1 partial);
- an entry's rank: position (1-based) of the first result that satisfies it.

Answers vs related passages: hit@k, recall@k and RR only count a question's
*best-graded* entries (grade 2 when it has any: real answers); grade-1
entries (partial answers, related books) only add to nDCG. Otherwise adding a
loosely related passage to the ground truth would inflate the hit rate.

Metrics (k = how many results are looked at):
- hit@k:     at least one entry found in the top k. "Did the context
             contain an answer?" The main number for a RAG.
- recall@k:  share of the entries found in the top k. Differs from hit@k for
             questions whose answer is spread over several passages.
- RR:        1 / rank of the first entry found (0 if none): rewards putting
             an answer first. Averaged over questions = MRR.
- nDCG@k:    graded and position-aware: an entry's gain (2^grade - 1) is
             credited at the first result that satisfies it, discounted by
             log2(rank + 1), and divided by the best achievable value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    keys: frozenset[str]  # any of these results satisfies the entry
    grade: int = 2


def entry_ranks(results: list[str], entries: list[Entry]) -> list[int | None]:
    first: dict[str, int] = {}
    for rank, key in enumerate(results, start=1):
        first.setdefault(key, rank)
    return [min((first[k] for k in e.keys if k in first), default=None) for e in entries]


def hit_at(ranks: list[int | None], k: int) -> float:
    return float(any(r is not None and r <= k for r in ranks))


def recall_at(ranks: list[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks) if ranks else 0.0


def reciprocal_rank(ranks: list[int | None]) -> float:
    found = [r for r in ranks if r is not None]
    return 1.0 / min(found) if found else 0.0


def _gain(grade: int) -> float:
    return 2.0 ** grade - 1.0


def ndcg_at(results: list[str], entries: list[Entry], k: int) -> float:
    """One result can satisfy several entries (two quotes in one chunk): it then
    earns all their gains. The ideal ranking is built the same way (greedily,
    best result first), so a perfect retrieval scores 1."""
    if not entries:
        return 0.0
    dcg, covered = 0.0, set()
    for rank, key in enumerate(results[:k], start=1):
        new = [i for i, e in enumerate(entries) if i not in covered and key in e.keys]
        covered.update(new)
        dcg += sum(_gain(entries[i].grade) for i in new) / math.log2(rank + 1)
    ideal = _ideal_dcg(entries, k)
    return min(dcg / ideal, 1.0) if ideal else 0.0  # ideal 0: no relevant result is indexed at all


def _ideal_dcg(entries: list[Entry], k: int) -> float:
    candidates = set().union(*(e.keys for e in entries))
    covered: set[int] = set()
    ideal = 0.0
    for rank in range(1, k + 1):
        gains = {key: sum(_gain(e.grade) for i, e in enumerate(entries) if i not in covered and key in e.keys)
                 for key in candidates}
        best = max(gains, key=gains.get, default=None)
        if best is None or gains[best] == 0:
            break
        covered.update(i for i, e in enumerate(entries) if best in e.keys)
        ideal += gains[best] / math.log2(rank + 1)
    return ideal


KS = (1, 3, 5, 10)


def primary(entries: list[Entry]) -> list[int]:
    """Indexes of the best-graded entries (the real answers)."""
    top = max((e.grade for e in entries), default=2)
    return [i for i, e in enumerate(entries) if e.grade == top]


def score(results: list[str], entries: list[Entry]) -> dict[str, float]:
    """All metrics for one question, as a flat dict (what run files store)."""
    all_ranks = entry_ranks(results, entries)
    ranks = [all_ranks[i] for i in primary(entries)]
    out = {f"hit@{k}": hit_at(ranks, k) for k in KS}
    out["recall@10"] = recall_at(ranks, 10)
    out["rr"] = reciprocal_rank(ranks)
    out["ndcg@10"] = ndcg_at(results, entries, 10)
    return out
