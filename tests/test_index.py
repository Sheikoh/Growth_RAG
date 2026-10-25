"""Qdrant upload and search, with Qdrant embedded in the test process."""

import pytest

from fake_encoder import FakeEncoder
from growth_rag.chunking.run import chunk_books
from growth_rag.chunking.store import iter_chunks
from growth_rag.chunking.tokens import ApproxTokenCounter
from growth_rag.config import ChunkingSettings, EmbeddingSettings, QdrantSettings
from growth_rag.embedding.run import embed_books
from growth_rag.index.qdrant import IndexUnavailable, connect, ensure_collections, point_id
from growth_rag.index.run import index_books, needs_indexing
from growth_rag.index.search import build_filter, search_books, search_chunks
from growth_rag.registry import Registry
from growth_rag.sources.gutenberg.clean import clean_books
from test_gutenberg_clean import add_book

PLAY = (
    "Header\n*** START OF THE PROJECT GUTENBERG EBOOK PLAY ***\n\n"
    "ACT I\n\nSCENE I. A platform.\n\nEnter guards.\n\n"
    + "HAMLET.\nTo be, or not to be, that is the question:\nWhether tis nobler in the mind to suffer.\n\n" * 3
    + "OPHELIA.\nGood my lord, how does your honour for this many a day?\nI humbly thank you.\n\n" * 3
    + "\n*** END OF THE PROJECT GUTENBERG EBOOK PLAY ***\n"
).encode("utf-8")
NOVEL = (
    "Header\n*** START OF THE PROJECT GUTENBERG EBOOK NOVEL ***\n\nCHAPTER I.\n\n"
    + "The whale swam under the ship while the sailors watched the stormy sea. " * 12
    + "\n\nCHAPTER II.\n\n" + "In the harbour the captain counted barrels of oil and coins. " * 12
    + "\n\n*** END OF THE PROJECT GUTENBERG EBOOK NOVEL ***\n"
).encode("utf-8")


@pytest.fixture
def qdrant(tmp_path):
    settings = QdrantSettings(path=str(tmp_path / "qdrant"))
    client = connect(settings)
    yield client, settings
    client.close()


def build(paths, qdrant):
    client, settings = qdrant
    registry = Registry(paths.registry)
    add_book(paths, registry, book_id=1524, raw=PLAY, shelves=["Category: Plays/Films/Dramas"])
    add_book(paths, registry, book_id=2701, raw=NOVEL)
    registry.get("gutenberg:2701").title = "Moby Dick"
    clean_books(registry, paths)
    chunk_books(registry, paths, ApproxTokenCounter(),
                ChunkingSettings(tokenizer="approx-4cpt", target_tokens=60, max_tokens=120, min_tokens=20))
    embed_books(registry, paths, FakeEncoder(), EmbeddingSettings(model="fake/bge"))
    summary = index_books(registry, paths, client, settings)
    return registry, summary


def test_index_uploads_every_chunk_and_one_point_per_book(paths, qdrant):
    client, settings = qdrant
    registry, summary = build(paths, qdrant)
    total = sum(r.chunking.chunks for r in registry)
    assert summary.by_status == {"ok": 2} and summary.points == total
    assert client.count(settings.chunks_collection, exact=True).count == total
    assert client.count(settings.books_collection, exact=True).count == 2

    chunk = next(iter_chunks(paths.resolve(registry.get("gutenberg:2701").chunking.path)))
    point = client.retrieve(settings.chunks_collection, ids=[point_id(chunk.key)], with_payload=True)[0]
    assert point.payload["text"] == chunk.text and point.payload["title"] == "Moby Dick"
    book = client.retrieve(settings.books_collection, ids=[point_id("gutenberg:2701")], with_vectors=True)[0]
    assert set(book.vector) == {"metadata", "centroid"}  # "summary" declared, filled later


def test_reindexing_replaces_a_book_without_stale_points(paths, qdrant):
    client, settings = qdrant
    registry, _ = build(paths, qdrant)
    record = registry.get("gutenberg:2701")
    assert not needs_indexing(record, settings)
    assert index_books(registry, paths, client, settings).indexed == 0

    # Re-chunk with bigger chunks -> fewer chunks; the old points must disappear.
    before = record.chunking.chunks
    chunk_books(registry, paths, ApproxTokenCounter(),
                ChunkingSettings(tokenizer="approx-4cpt", target_tokens=350, max_tokens=512, min_tokens=80))
    embed_books(registry, paths, FakeEncoder(), EmbeddingSettings(model="fake/bge"))
    assert needs_indexing(record, settings)
    index_books(registry, paths, client, settings)
    after = record.chunking.chunks
    assert after < before
    total = sum(r.chunking.chunks for r in registry)
    assert client.count(settings.chunks_collection, exact=True).count == total


def test_collection_with_other_dimensions_is_refused(paths, qdrant):
    client, settings = qdrant
    ensure_collections(client, settings, dim=8)
    with pytest.raises(IndexUnavailable, match="--recreate"):
        ensure_collections(client, settings, dim=32)
    ensure_collections(client, settings, dim=32, recreate=True)  # rebuilt


def test_search_modes_and_filters(paths, qdrant):
    client, settings = qdrant
    build(paths, qdrant)
    encoder = FakeEncoder()
    for mode in ("dense", "sparse", "hybrid"):
        hits = search_chunks(client, settings, encoder, "whale ship stormy sea", mode=mode, k=3)
        assert hits and hits[0].payload["book_key"] == "gutenberg:2701", mode

    flt = build_filter(speaker_labels=["OPHELIA"])
    hits = search_chunks(client, settings, encoder, "lord honour", mode="dense", k=10, query_filter=flt)
    assert hits and all("OPHELIA" in h.payload["speaker_labels"] for h in hits)

    hits = search_chunks(client, settings, encoder, "whale", mode="dense", k=10,
                         query_filter=build_filter(book_keys=["gutenberg:1524"]))
    assert all(h.payload["book_key"] == "gutenberg:1524" for h in hits)

    books = search_books(client, settings, encoder, "Moby Dick", k=2)
    assert books[0].payload["key"] == "gutenberg:2701"


def test_unreachable_server_gives_a_clear_error():
    with pytest.raises(IndexUnavailable, match="docker compose up"):
        connect(QdrantSettings(url="http://127.0.0.1:9"))


def test_title_resolution_feeds_the_filter(paths, qdrant):
    from growth_rag.index.resolve import resolve_books

    client, settings = qdrant
    registry, _ = build(paths, qdrant)
    keys = resolve_books(registry, title="moby dick").keys
    assert keys == ["gutenberg:2701"]
    hits = search_chunks(client, settings, FakeEncoder(), "lord honour whale", mode="dense", k=10,
                         query_filter=build_filter(book_keys=keys))
    assert hits and all(h.payload["book_key"] == "gutenberg:2701" for h in hits)
    books = search_books(client, settings, FakeEncoder(), "anything", k=5, book_keys=keys)
    assert [b.payload["key"] for b in books] == ["gutenberg:2701"]
