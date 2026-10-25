"""Search, to check the index (`rag search`). Full retrieval (reranking,
neighbour expansion, context assembly) is the next step; this is the baseline.

Modes on the `chunks` collection:
- dense:  semantic similarity (cosine on the dense vectors);
- sparse: lexical matching (bge-m3 token weights: names, rare words);
- hybrid: both, merged with Reciprocal Rank Fusion (rank-based, so the two
          score scales do not need to be comparable).
`books` mode searches the books collection (metadata or centroid vector).
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from growth_rag.config import QdrantSettings
from growth_rag.embedding.encoders.base import Encoder


@dataclass
class Hit:
    score: float
    payload: dict


def build_filter(languages=None, book_keys=None, kinds=None, speaker_labels=None) -> models.Filter | None:
    must = []
    for key, values in (("language", languages), ("book_key", book_keys), ("kind", kinds),
                        ("speaker_labels", speaker_labels)):
        if values:
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(values))))
    return models.Filter(must=must) if must else None


def search_chunks(
    client: QdrantClient, settings: QdrantSettings, encoder: Encoder, query: str,
    mode: str = "hybrid", k: int = 5, query_filter: models.Filter | None = None,
) -> list[Hit]:
    encoded = encoder.encode([query])
    return query_chunks(client, settings, encoded.dense[0], encoded.sparse[0], mode, k, query_filter)


def query_chunks(
    client: QdrantClient, settings: QdrantSettings, dense_vec, sparse_vec,
    mode: str = "hybrid", k: int = 5, query_filter: models.Filter | None = None,
) -> list[Hit]:
    """Search with an already encoded query (dense array, sparse (ids, weights));
    `rag eval` encodes all its questions in one batch, then calls this per mode."""
    dense = [float(x) for x in dense_vec]
    idx, val = sparse_vec
    sparse = models.SparseVector(indices=[int(i) for i in idx], values=[float(v) for v in val])
    if mode == "dense":
        res = client.query_points(settings.chunks_collection, query=dense, using="dense", limit=k,
                                  query_filter=query_filter, with_payload=True)
    elif mode == "sparse":
        res = client.query_points(settings.chunks_collection, query=sparse, using="sparse", limit=k,
                                  query_filter=query_filter, with_payload=True)
    elif mode == "hybrid":
        res = client.query_points(
            settings.chunks_collection,
            prefetch=[
                models.Prefetch(query=dense, using="dense", limit=k * 4, filter=query_filter),
                models.Prefetch(query=sparse, using="sparse", limit=k * 4, filter=query_filter),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF), limit=k, with_payload=True,
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    return [Hit(p.score, p.payload) for p in res.points]


def search_books(
    client: QdrantClient, settings: QdrantSettings, encoder: Encoder, query: str,
    vector: str = "metadata", k: int = 5, languages=None, book_keys=None,
) -> list[Hit]:
    dense = encoder.encode([query]).dense[0]
    return query_books(client, settings, dense, vector, k, languages, book_keys)


def query_books(
    client: QdrantClient, settings: QdrantSettings, dense_vec,
    vector: str = "metadata", k: int = 5, languages=None, book_keys=None,
) -> list[Hit]:
    must = []
    if languages:
        must.append(models.FieldCondition(key="languages", match=models.MatchAny(any=list(languages))))
    if book_keys:
        must.append(models.FieldCondition(key="key", match=models.MatchAny(any=list(book_keys))))
    flt = models.Filter(must=must) if must else None
    res = client.query_points(settings.books_collection, query=[float(x) for x in dense_vec], using=vector,
                              limit=k, query_filter=flt, with_payload=True)
    return [Hit(p.score, p.payload) for p in res.points]
