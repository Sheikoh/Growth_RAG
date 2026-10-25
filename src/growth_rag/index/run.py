"""Index step over the registry: vectors .npz + chunks JSONL -> Qdrant.

Nothing is embedded here: the vectors written by `rag embed` are uploaded as
they are. A book is (re)uploaded when it has never been, when its vectors file
changed (checksum), when the target (Qdrant URL/path, collection names)
changed, or with --force. --recreate rebuilds both collections first.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from growth_rag.chunking.run import display_authors
from growth_rag.chunking.store import iter_chunks
from growth_rag.config import DataPaths, QdrantSettings
from growth_rag.embedding.store import load_vectors
from growth_rag.index.qdrant import (
    delete_book_chunks,
    ensure_collections,
    point_id,
    target_of,
    upsert_batched,
)
from growth_rag.registry import BookRecord, IndexInfo, Registry
from growth_rag.utils.io import utc_now

log = logging.getLogger(__name__)


@dataclass
class IndexSummary:
    indexed: int = 0
    skipped: int = 0
    points: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    interrupted: bool = False


def needs_indexing(record: BookRecord, settings: QdrantSettings, force: bool = False) -> bool:
    emb = record.embedding
    if emb is None or emb.status != "ok":
        return False
    info = record.index
    if force or info is None or info.status == "error":
        return True
    same_target = (info.target, info.chunks_collection, info.books_collection) == (
        target_of(settings), settings.chunks_collection, settings.books_collection)
    return not same_target or info.vectors_sha256 != emb.vectors_sha256


def book_payload(record: BookRecord) -> dict:
    authors = display_authors(record.authors)
    return {
        "key": record.key, "source": record.source, "source_id": record.source_id,
        "title": record.title, "authors_display": authors,
        "authors": [a.name for a in record.authors if a.role is None],
        "languages": record.languages, "subjects": record.subjects,
        "bookshelves": record.bookshelves, "issued": record.issued,
        "chunks": record.chunking.chunks if record.chunking else None,
        "cleaning_status": record.cleaning.status if record.cleaning else None,
    }


def index_record(record: BookRecord, paths: DataPaths, client: QdrantClient, settings: QdrantSettings) -> IndexInfo:
    emb = record.embedding
    base = dict(target=target_of(settings), chunks_collection=settings.chunks_collection,
                books_collection=settings.books_collection, vectors_sha256=emb.vectors_sha256)
    try:
        vectors = load_vectors(paths.resolve(emb.path))
        chunks = list(iter_chunks(paths.resolve(record.chunking.path)))
        if [c.key for c in chunks] != vectors.keys:
            raise ValueError("chunks and vectors are out of sync: run `rag embed` for this book")
        book = book_payload(record)
        shared = {"title": book["title"], "authors_display": book["authors_display"],
                  "source": record.source, "embedding_model": emb.model}

        def chunk_points():
            for chunk, dense, (idx, val) in zip(chunks, vectors.dense, vectors.sparse):
                yield models.PointStruct(
                    id=point_id(chunk.key),
                    vector={"dense": dense.tolist(),
                            "sparse": models.SparseVector(indices=idx.tolist(), values=val.tolist())},
                    payload={**chunk.model_dump(mode="json"), **shared},
                )

        delete_book_chunks(client, settings.chunks_collection, record.key)
        n = upsert_batched(client, settings.chunks_collection, chunk_points())

        book_vectors = {}
        if vectors.book_metadata is not None:
            book_vectors["metadata"] = vectors.book_metadata.tolist()
        if vectors.book_centroid is not None:
            book_vectors["centroid"] = vectors.book_centroid.tolist()
        client.upsert(settings.books_collection, points=[models.PointStruct(
            id=point_id(record.key), vector=book_vectors, payload=book)], wait=True)
        return IndexInfo(status="ok", points=n, indexed_at=utc_now(), **base)
    except Exception as exc:
        log.exception("Indexing failed for %s", record.key)
        return IndexInfo(status="error", indexed_at=utc_now(), error=f"{type(exc).__name__}: {exc}", **base)


def index_books(
    registry: Registry,
    paths: DataPaths,
    client: QdrantClient,
    settings: QdrantSettings,
    keys: Iterable[str] | None = None,
    *,
    force: bool = False,
    recreate: bool = False,
) -> IndexSummary:
    wanted = set(keys) if keys is not None else None
    records = [r for r in registry if (wanted is None or r.key in wanted)
               and r.embedding is not None and r.embedding.status == "ok"]
    dims = {r.embedding.dim for r in records if r.embedding.dim}
    if len(dims) > 1:
        raise ValueError(f"Books were embedded with different dimensions {dims}: re-embed first")
    if records:
        ensure_collections(client, settings, dims.pop() if dims else 1024, recreate=recreate)
    todo = [r for r in records if needs_indexing(r, settings, force or recreate)]
    summary = IndexSummary(skipped=len(records) - len(todo))
    log.info("%d embedded books | %d up to date | %d to upload to %s",
             len(records), summary.skipped, len(todo), target_of(settings))
    try:
        for i, record in enumerate(todo, 1):
            info = index_record(record, paths, client, settings)
            record.index = info
            summary.indexed += 1
            summary.points += info.points or 0
            summary.by_status[info.status] = summary.by_status.get(info.status, 0) + 1
            log.info("[%d/%d] %-16s %-5s points=%-6s %s %s", i, len(todo), record.key, info.status,
                     info.points, record.title[:40], info.error or "")
            if i % 10 == 0:
                registry.save()
    except KeyboardInterrupt:
        summary.interrupted = True
    finally:
        registry.save()
    return summary
