"""Collections and points.

Collection `chunks`: one point per chunk.
    vectors: "dense" (1024, cosine) + sparse "sparse" (bge-m3 lexical weights)
    payload: the chunk (text, offsets, section path, speakers, annotations, ...)
             + book fields used for filtering (title, authors, source)
    indexes: book_key, language, kind, speakers, speaker_labels, section

Collection `books`: one point per book, named vectors declared up front:
    "metadata" (catalog description), "centroid" (mean of chunk vectors),
    "summary" (empty until LLM summaries exist). A point may lack some vectors,
    so adding summaries later needs no rebuild.

Point ids are UUIDv5 of the keys: uploading again updates points in place.
Before uploading a book's chunks, its previous chunk points are deleted, so a
book re-chunked into fewer chunks leaves no stale points.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable

from qdrant_client import QdrantClient, models

from growth_rag.config import QdrantSettings

log = logging.getLogger(__name__)

NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/growth-rag")
CHUNK_VECTORS = ("dense",)
BOOK_VECTORS = ("metadata", "centroid", "summary")
CHUNK_INDEXES = {
    "book_key": models.PayloadSchemaType.KEYWORD,
    "language": models.PayloadSchemaType.KEYWORD,
    "kind": models.PayloadSchemaType.KEYWORD,
    "speakers": models.PayloadSchemaType.KEYWORD,
    "speaker_labels": models.PayloadSchemaType.KEYWORD,
    "section": models.PayloadSchemaType.INTEGER,
}
BOOK_INDEXES = {
    "key": models.PayloadSchemaType.KEYWORD,
    "languages": models.PayloadSchemaType.KEYWORD,
    "authors_display": models.PayloadSchemaType.KEYWORD,
}


class IndexUnavailable(RuntimeError):
    pass


def point_id(key: str) -> str:
    return str(uuid.uuid5(NAMESPACE, key))


def target_of(settings: QdrantSettings) -> str:
    return f"path:{settings.path}" if settings.path else settings.url


def connect(settings: QdrantSettings) -> QdrantClient:
    if settings.path:
        return QdrantClient(path=settings.path)  # embedded, no server
    # The client's own compatibility check runs at construction, before we know
    # whether the server is reachable, and only warns. It is done below instead,
    # once the connection is confirmed.
    client = QdrantClient(url=settings.url, api_key=settings.api_key, timeout=60,
                          check_compatibility=False)
    try:
        client.get_collections()
    except Exception as exc:
        raise IndexUnavailable(
            f"Cannot reach Qdrant at {settings.url} ({type(exc).__name__}: {exc}). "
            "Is the container running? From 03_RAG: docker compose up -d"
        ) from exc
    _check_versions(client)
    return client


def _check_versions(client: QdrantClient) -> None:
    """Warn when client and server differ by more than one minor version (the
    range Qdrant supports), with the fix."""
    from importlib.metadata import version

    try:
        server = client.info().version
        mine = version("qdrant-client")
        s_major, s_minor = (int(x) for x in server.split(".")[:2])
        c_major, c_minor = (int(x) for x in mine.split(".")[:2])
    except Exception:
        return  # version unknown: nothing reliable to say
    if s_major != c_major or abs(s_minor - c_minor) > 1:
        log.warning("Qdrant server %s vs qdrant-client %s: set QDRANT_VERSION=v%s in 03_RAG/.env "
                    "and run `docker compose up -d` (data is kept in the volume).", server, mine, mine)


def ensure_collections(client: QdrantClient, settings: QdrantSettings, dim: int, recreate: bool = False) -> None:
    specs = {
        settings.chunks_collection: dict(
            vectors_config={name: models.VectorParams(size=dim, distance=models.Distance.COSINE)
                            for name in CHUNK_VECTORS},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
            indexes=CHUNK_INDEXES,
        ),
        settings.books_collection: dict(
            vectors_config={name: models.VectorParams(size=dim, distance=models.Distance.COSINE)
                            for name in BOOK_VECTORS},
            sparse_vectors_config=None,
            indexes=BOOK_INDEXES,
        ),
    }
    for name, spec in specs.items():
        exists = client.collection_exists(name)
        if exists and recreate:
            client.delete_collection(name)
            exists = False
        if exists:
            _check_dims(client, name, spec["vectors_config"])
        else:
            client.create_collection(name, vectors_config=spec["vectors_config"],
                                     sparse_vectors_config=spec["sparse_vectors_config"])
        if settings.path:
            continue  # embedded mode: payload indexes have no effect
        # Idempotent: also adds indexes introduced after the collection was created.
        existing = set((client.get_collection(name).payload_schema or {}).keys())
        for field_name, schema in spec["indexes"].items():
            if field_name not in existing:
                client.create_payload_index(name, field_name=field_name, field_schema=schema)


def _check_dims(client: QdrantClient, name: str, expected: dict) -> None:
    current = client.get_collection(name).config.params.vectors
    for vector_name, params in expected.items():
        have = current.get(vector_name) if isinstance(current, dict) else None
        if have is None or have.size != params.size:
            raise IndexUnavailable(
                f"Collection {name!r} exists with a different vector layout "
                f"({vector_name}: {getattr(have, 'size', None)} vs {params.size}). "
                "Re-run with --recreate to rebuild it."
            )


def delete_book_chunks(client: QdrantClient, collection: str, book_key: str) -> None:
    client.delete(collection, points_selector=models.FilterSelector(filter=models.Filter(
        must=[models.FieldCondition(key="book_key", match=models.MatchValue(value=book_key))]
    )), wait=True)


def upsert_batched(client: QdrantClient, collection: str, points: Iterable[models.PointStruct],
                   batch_size: int = 256) -> int:
    batch, total = [], 0
    for point in points:
        batch.append(point)
        if len(batch) == batch_size:
            client.upsert(collection, points=batch, wait=True)
            total += len(batch)
            batch = []
    if batch:
        client.upsert(collection, points=batch, wait=True)
        total += len(batch)
    return total
