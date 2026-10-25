"""bge-m3 maths, vector store, book vectors, embedding run over the registry."""

import numpy as np
import pytest

from fake_encoder import FakeEncoder
from growth_rag.chunking.run import chunk_books
from growth_rag.chunking.tokens import ApproxTokenCounter
from growth_rag.config import ChunkingSettings, EmbeddingSettings
from growth_rag.embedding import EMBEDDING_VERSION
from growth_rag.embedding.books import centroid, metadata_text
from growth_rag.embedding.check import compare_encoders
from growth_rag.embedding.encoders import make_encoder
from growth_rag.embedding.encoders.base import (
    EncoderUnavailable,
    cls_pooling,
    l2_normalize,
    lexical_weights,
    sparse_dot,
    sparse_from_dict,
)
from growth_rag.embedding.run import embed_books, encode_sorted, needs_embedding
from growth_rag.embedding.store import BookVectors, load_vectors, save_vectors
from growth_rag.registry import Registry
from growth_rag.sources.gutenberg.clean import clean_books
from test_gutenberg_clean import add_book

EM = EmbeddingSettings(model="fake/bge", batch_size=4)


# --- the maths of bge-m3 (independent of torch) -----------------------------

def test_cls_pooling_and_normalisation():
    hidden = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)  # [batch, tokens, dim]
    dense = l2_normalize(cls_pooling(hidden))
    assert dense.shape == (2, 4)
    np.testing.assert_allclose(np.linalg.norm(dense, axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(dense[0], hidden[0, 0] / np.linalg.norm(hidden[0, 0]), rtol=1e-6)


def test_lexical_weights_keep_max_per_token_and_skip_special_tokens():
    # ids: 0 = <s> (CLS), 2 = </s>; token 7 appears twice, token 9 has a zero weight
    ids = [[0, 7, 8, 7, 9, 2]]
    weights = [[0.9, 0.2, 0.5, 0.6, 0.0, 0.8]]
    (indices, values), = lexical_weights(ids, weights, ignore_ids={0, 1, 2, 3})
    assert indices.tolist() == [7, 8]
    np.testing.assert_allclose(values, [0.6, 0.5])


def test_flag_format_and_sparse_dot():
    a = sparse_from_dict({"7": 0.6, "8": 0.5, "9": 0.0})
    b = sparse_from_dict({"8": 0.4, "12": 1.0})
    assert a[0].tolist() == [7, 8]
    assert sparse_dot(a, b) == pytest.approx(0.5 * 0.4)
    assert sparse_dot(a, sparse_from_dict({"99": 1.0})) == 0.0


def test_make_encoder_rejects_unknown_backend_and_loads_lazily():
    with pytest.raises(ValueError):
        make_encoder(EmbeddingSettings(backend="nope"))
    encoder = make_encoder(EmbeddingSettings(backend="manual"))  # no download at construction
    assert encoder.backend == "manual" and encoder.name == "BAAI/bge-m3"


def test_manual_encoder_reports_a_clear_error_when_it_cannot_load(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name in ("torch", "transformers"):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    with pytest.raises(EncoderUnavailable, match="torch.*rag model"):
        make_encoder(EmbeddingSettings(backend="manual")).encode(["x"])


# --- batching, storage, book vectors -------------------------------------------

def test_encode_sorted_batches_by_length_and_restores_order():
    texts = ["a", "a much longer text here", "mid size", "x y", "the longest text of all of them"]
    encoder = FakeEncoder()
    out = encode_sorted(encoder, texts, batch_size=2)
    assert encoder.batches == [2, 2, 1]
    for i, text in enumerate(texts):  # same vector as encoding the text alone
        np.testing.assert_allclose(out.dense[i], FakeEncoder().encode([text]).dense[0], rtol=1e-6)


def test_vectors_round_trip(tmp_path):
    enc = FakeEncoder().encode(["one two", "three", ""])
    vectors = BookVectors(keys=["k0", "k1", "k2"], dense=enc.dense, sparse=enc.sparse,
                          book_metadata=enc.dense[0], book_centroid=None, meta={"model": "fake"})
    path = tmp_path / "v.npz"
    save_vectors(path, vectors)
    loaded = load_vectors(path)
    assert loaded.keys == ["k0", "k1", "k2"] and loaded.meta == {"model": "fake"}
    np.testing.assert_allclose(loaded.dense, enc.dense, atol=1e-3)  # float16 on disk
    for (i1, v1), (i2, v2) in zip(loaded.sparse, enc.sparse):
        assert i1.tolist() == i2.tolist()
        np.testing.assert_allclose(v1, v2)
    assert loaded.book_centroid is None and loaded.book_metadata.shape == (32,)


def test_book_metadata_text_and_centroid(paths):
    registry = Registry(paths.registry)
    add_book(paths, registry, shelves=["Category: Novels", "Category: Novels", "Gothic Fiction"])
    record = registry.get("gutenberg:1")
    record.subjects = ["Sea stories"]
    text = metadata_text(record, "Herman Melville")
    assert text.splitlines() == ["Test", "Author: Herman Melville", "Subjects: Sea stories",
                                 "Bookshelves: Novels; Gothic Fiction", "Language: en"]
    c = centroid(l2_normalize(np.eye(3, dtype=np.float32)))
    np.testing.assert_allclose(np.linalg.norm(c), 1.0, rtol=1e-6)
    assert centroid(np.zeros((0, 3), dtype=np.float32)) is None


def test_identical_encoders_compare_perfectly():
    result = compare_encoders(FakeEncoder(), FakeEncoder(), ["a text", "another text here"])
    assert result.dense_cosine_min == pytest.approx(1.0)
    assert result.sparse_jaccard_mean == 1.0 and result.sparse_score_rel_diff_max == 0.0


# --- embedding over the registry ------------------------------------------------

def chunked_registry(paths):
    registry = Registry(paths.registry)
    add_book(paths, registry)
    clean_books(registry, paths)
    chunk_books(registry, paths, ApproxTokenCounter(), ChunkingSettings(tokenizer="approx-4cpt"))
    return registry


def test_embed_writes_vectors_and_updates_registry(paths):
    registry = chunked_registry(paths)
    summary = embed_books(registry, paths, FakeEncoder(), EM)
    assert summary.by_status == {"ok": 1}
    info = Registry.load(paths.registry).get("gutenberg:1").embedding
    assert info.version == EMBEDDING_VERSION and info.model == "fake/bge" and info.dim == 32
    assert info.path == "vectors/gutenberg/pg1.npz" and info.vectors_sha256
    vectors = load_vectors(paths.resolve(info.path))
    assert len(vectors.keys) == info.chunks
    assert vectors.book_metadata is not None and vectors.book_centroid is not None


def test_reembedding_rules(paths):
    registry = chunked_registry(paths)
    embed_books(registry, paths, FakeEncoder(), EM)
    record = registry.get("gutenberg:1")
    assert not needs_embedding(record, paths, EM)
    assert embed_books(registry, paths, FakeEncoder(), EM).embedded == 0
    assert needs_embedding(record, paths, EmbeddingSettings(model="another/model"))
    assert record.embedding.revision is None  # not recorded: not redone for that
    assert not needs_embedding(record, paths, EmbeddingSettings(model="fake/bge", revision="new"))
    record.embedding.revision = "old"
    assert needs_embedding(record, paths, EmbeddingSettings(model="fake/bge", revision="new"))
    record.embedding.revision = None
    paths.resolve(record.chunking.path).write_text("", encoding="utf-8")  # chunks changed
    assert needs_embedding(record, paths, EM)


def test_unloadable_encoder_stops_before_the_first_book(paths):
    class Broken(FakeEncoder):
        def encode(self, texts):
            raise EncoderUnavailable("manual", "x", OSError("offline"))

    registry = chunked_registry(paths)
    with pytest.raises(EncoderUnavailable):
        embed_books(registry, paths, Broken(), EM)
    assert registry.get("gutenberg:1").embedding is None
