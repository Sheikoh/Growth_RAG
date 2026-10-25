"""Retrieval metrics, on hand-computed cases."""

import math

import pytest

from growth_rag.evaluation.metrics import Entry, entry_ranks, hit_at, ndcg_at, recall_at, reciprocal_rank, score


def E(*keys, grade=2):
    return Entry(frozenset(keys), grade)


def test_rank_is_the_first_result_satisfying_the_entry():
    # an entry satisfied by either of two chunks (quote across a boundary)
    assert entry_ranks(["x", "b", "a"], [E("a", "b"), E("z")]) == [2, None]


def test_hit_recall_and_reciprocal_rank():
    ranks = [3, None]
    assert hit_at(ranks, 2) == 0.0 and hit_at(ranks, 3) == 1.0
    assert recall_at(ranks, 10) == 0.5
    assert reciprocal_rank(ranks) == pytest.approx(1 / 3)
    assert reciprocal_rank([None]) == 0.0


def test_ndcg_perfect_partial_and_zero():
    entries = [E("a"), E("b", grade=1)]
    assert ndcg_at(["a", "b"], entries, 10) == pytest.approx(1.0)
    ideal = 3 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at(["b", "a"], entries, 10) == pytest.approx((1 / math.log2(2) + 3 / math.log2(3)) / ideal)
    assert ndcg_at(["x"], entries, 10) == 0.0


def test_ndcg_two_entries_in_one_chunk_is_still_perfect():
    entries = [E("c5"), E("c5"), E("c4")]  # two quotes in the same chunk
    assert ndcg_at(["c5", "c4"], entries, 10) == pytest.approx(1.0)
    assert ndcg_at(["c4", "c5"], entries, 10) < 1.0


def test_nothing_indexed_scores_zero_instead_of_failing():
    assert ndcg_at(["a"], [Entry(frozenset())], 10) == 0.0  # e.g. a passage in front matter


def test_score_has_every_metric():
    out = score(["x", "a"], [E("a")])
    assert out["hit@1"] == 0 and out["hit@3"] == 1 and out["rr"] == 0.5
    assert set(out) == {"hit@1", "hit@3", "hit@5", "hit@10", "recall@10", "rr", "ndcg@10"}


def test_related_passages_do_not_count_as_hits():
    entries = [E("answer"), E("related", grade=1)]
    out = score(["related", "x", "answer"], entries)
    assert out["hit@1"] == 0 and out["hit@3"] == 1 and out["rr"] == pytest.approx(1 / 3)
    assert out["recall@10"] == 1.0  # over the answers only
    assert 0 < out["ndcg@10"] < 1  # the related passage still earns its (smaller) gain


def test_questions_with_only_related_entries_use_them():
    assert score(["b"], [E("a", grade=1), E("b", grade=1)])["hit@1"] == 1.0  # e.g. "books like X"
