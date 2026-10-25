"""Chunking wired to the registry: outputs, re-chunk rules, errors."""

import pytest

from growth_rag.chunking import CHUNKING_VERSION
from growth_rag.chunking.run import TokenizerUnavailable, chunk_books, display_authors, needs_chunking
from growth_rag.chunking.store import iter_chunks
from growth_rag.chunking.tokens import ApproxTokenCounter
from growth_rag.config import ChunkingSettings
from growth_rag.registry import Author, Registry
from growth_rag.sources.gutenberg.clean import clean_books
from test_gutenberg_clean import add_book

SETTINGS = ChunkingSettings(tokenizer=ApproxTokenCounter.name)


def cleaned_registry(paths):
    registry = Registry(paths.registry)
    add_book(paths, registry)
    clean_books(registry, paths)
    return registry


def test_chunk_writes_jsonl_and_updates_registry(paths):
    registry = cleaned_registry(paths)
    summary = chunk_books(registry, paths, ApproxTokenCounter(), SETTINGS)
    assert summary.by_status == {"ok": 1} and summary.chunks >= 1

    info = Registry.load(paths.registry).get("gutenberg:1").chunking
    assert info.version == CHUNKING_VERSION and info.tokenizer == "approx-4cpt"
    assert info.path == "chunks/gutenberg/pg1.jsonl"
    chunks = list(iter_chunks(paths.resolve(info.path)))
    assert len(chunks) == info.chunks and chunks[0].book_key == "gutenberg:1"
    assert chunks[0].header.startswith("Test")


def test_rechunk_rules(paths):
    registry = cleaned_registry(paths)
    chunk_books(registry, paths, ApproxTokenCounter(), SETTINGS)
    record = registry.get("gutenberg:1")
    assert not needs_chunking(record, paths, SETTINGS)
    assert chunk_books(registry, paths, ApproxTokenCounter(), SETTINGS).chunked == 0

    other_tokenizer = ChunkingSettings(tokenizer="BAAI/bge-m3")
    assert needs_chunking(record, paths, other_tokenizer)  # sizes depend on the tokenizer
    bigger = ChunkingSettings(tokenizer=ApproxTokenCounter.name, target_tokens=500)
    assert needs_chunking(record, paths, bigger)
    clean_books(registry, paths, force=True)  # re-cleaned, same text -> nothing to redo
    assert not needs_chunking(registry.get("gutenberg:1"), paths, SETTINGS)
    raw = paths.resolve(record.acquisition.path)
    raw.write_bytes(raw.read_bytes().replace(b"Some prose", b"Other prose"))
    record.acquisition.sha256 = "changed"
    clean_books(registry, paths)  # re-cleaned, different text -> re-chunk
    assert needs_chunking(registry.get("gutenberg:1"), paths, SETTINGS)


def test_uncleaned_books_are_not_chunked(paths):
    registry = Registry(paths.registry)
    add_book(paths, registry)  # downloaded, not cleaned
    assert chunk_books(registry, paths, ApproxTokenCounter(), SETTINGS).chunked == 0


def test_unavailable_tokenizer_stops_before_the_loop(paths):
    class Broken:
        name = "broken"

        def count(self, texts):
            raise OSError("no connection")

    registry = cleaned_registry(paths)
    with pytest.raises(TokenizerUnavailable, match="approx-4cpt"):
        chunk_books(registry, paths, Broken(), ChunkingSettings(tokenizer="broken"))
    assert registry.get("gutenberg:1").chunking is None  # nothing half-written


def test_display_authors_skips_contributors():
    authors = [Author(name="Hugo, Victor", raw="x"), Author(name="Smith, John", role="Translator", raw="y"),
               Author(name="Various", raw="z")]
    assert display_authors(authors) == "Victor Hugo, Various"
