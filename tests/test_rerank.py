"""Reranking: reorder, passage text, comparison of two rerankers, settings."""

import builtins

import numpy as np
import pytest

from fake_reranker import FakeReranker
from growth_rag.config import BGE_RERANKER_REVISION, RerankSettings, Settings
from growth_rag.index.search import Hit
from growth_rag.rerank import RerankerUnavailable, make_reranker, rerank
from growth_rag.rerank.base import length_order, passage_text, sigmoid
from growth_rag.rerank.check import compare_rerankers, spearman


def hit(key, text, header="", score=0.0):
    return Hit(score, {"key": key, "text": text, "header": header})


def test_rerank_reorders_by_score_and_keeps_ties_in_search_order():
    hits = [hit("a", "nothing here"), hit("b", "the white whale"), hit("c", "a whale"), hit("d", "zzz")]
    out = rerank(FakeReranker(), "white whale", hits)
    assert [h.payload["key"] for h in out] == ["b", "c", "a", "d"]  # a and d tie at 0: search order kept
    assert out[0].score == 2.0  # the reranker's score replaces the search score
    assert rerank(FakeReranker(), "x", []) == []


def test_the_reranker_reads_header_and_text():
    assert passage_text({"header": "Moby Dick — CHAPTER 1", "text": "Call me Ishmael."}) == \
        "Moby Dick — CHAPTER 1\n\nCall me Ishmael."
    assert passage_text({"text": "only text"}) == "only text"
    out = rerank(FakeReranker(), "chapter loomings", [hit("a", "x"), hit("b", "x", header="CHAPTER 1. Loomings")])
    assert out[0].payload["key"] == "b"  # matched through the header


def test_helpers():
    assert sigmoid(0.0) == 0.5 and sigmoid(4) > 0.98
    assert sigmoid(-1000.0) == 0.0 and sigmoid(1000.0) == 1.0  # no overflow
    assert length_order(["aa", "a", "aaaa"]) == [2, 0, 1]


def test_comparison_of_two_rerankers():
    cases = [("white whale", ["the white whale", "a whale", "nothing"]), ("sea", ["sea sea", "land"])]
    same = compare_rerankers(FakeReranker(), FakeReranker(bias=0.01), cases)
    assert same.max_abs_diff == pytest.approx(0.01) and same.spearman_min == 1.0 and same.same_top1 == 2
    assert same.pairs == 5
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == pytest.approx(-1.0)


def test_settings_are_pinned_and_overridable(monkeypatch):
    assert RerankSettings().revision == BGE_RERANKER_REVISION
    monkeypatch.setenv("GROWTH_RAG_RERANK_CANDIDATES", "30")
    monkeypatch.setenv("GROWTH_RAG_RERANK_REVISION", "")
    rr = Settings.from_env().rerank
    assert rr.candidates == 30 and rr.revision is None


def test_backends_and_clear_errors(monkeypatch):
    with pytest.raises(ValueError):
        make_reranker(RerankSettings(backend="nope"))
    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name in ("torch", "transformers"):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    with pytest.raises(RerankerUnavailable, match="torch.*rag model --rerank"):
        make_reranker(RerankSettings()).score("q", ["p"])
