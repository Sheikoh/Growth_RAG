"""Second-stage reranking with a cross-encoder (bge-reranker-v2-m3).

    base.py    why and how (Reranker protocol, rerank(), passage text)
    manual.py  transformers, every step visible
    flag.py    FlagEmbedding (reference, optional extra)
    files.py   pinned model files, from the local cache
    check.py   manual vs FlagEmbedding (rag rerank-check)
"""

from growth_rag.config import RerankSettings
from growth_rag.rerank.base import Reranker, RerankerUnavailable, rerank


def make_reranker(settings: RerankSettings) -> Reranker:
    if settings.backend == "manual":
        from growth_rag.rerank.manual import ManualReranker

        return ManualReranker(settings)
    if settings.backend == "flag":
        from growth_rag.rerank.flag import FlagReranker

        return FlagReranker(settings)
    raise ValueError(f"Unknown rerank backend {settings.backend!r} (expected 'manual' or 'flag')")


__all__ = ["Reranker", "RerankerUnavailable", "make_reranker", "rerank"]
