"""Embedding step over the registry: chunks JSONL -> vectors .npz (+ book vectors).

Chunks are sorted by length before batching: texts of similar length are padded
less, which is the largest speed gain available. Results are put back in the
chunks' order before saving.

A book is (re)embedded when it has never been embedded, when its chunks file
changed (checksum), when the model or EMBEDDING_VERSION changed, when the
output is missing, or with --force.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import numpy as np

from growth_rag.chunking.run import display_authors
from growth_rag.chunking.store import iter_chunks
from growth_rag.config import DataPaths, EmbeddingSettings
from growth_rag.embedding import EMBEDDING_VERSION
from growth_rag.embedding.books import centroid, metadata_text
from growth_rag.embedding.encoders.base import Encoded, Encoder, EncoderUnavailable
from growth_rag.embedding.store import BookVectors, save_vectors
from growth_rag.registry import BookRecord, EmbeddingInfo, Registry
from growth_rag.utils.io import sha256_file, utc_now

log = logging.getLogger(__name__)


@dataclass
class EmbedSummary:
    embedded: int = 0
    skipped: int = 0
    chunks: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    interrupted: bool = False


def encode_sorted(encoder: Encoder, texts: list[str], batch_size: int) -> Encoded:
    """Encode in batches of similar length, return results in the input order."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    dense: list[np.ndarray | None] = [None] * len(texts)
    sparse: list = [None] * len(texts)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        out = encoder.encode([texts[i] for i in idx])
        for k, i in enumerate(idx):
            dense[i], sparse[i] = out.dense[k], out.sparse[k]
    dim = encoder.dim if not texts else len(dense[0])
    matrix = np.stack(dense) if texts else np.zeros((0, dim), dtype=np.float32)
    return Encoded(dense=matrix.astype(np.float32), sparse=sparse)


def needs_embedding(record: BookRecord, paths: DataPaths, settings: EmbeddingSettings, force: bool = False) -> bool:
    chunking = record.chunking
    if chunking is None or chunking.status != "ok" or not chunking.path:
        return False
    info = record.embedding
    if force or info is None or info.status == "error":
        return True
    if info.version != EMBEDDING_VERSION or info.model != settings.model:
        return True
    # Books embedded before revisions were recorded (None) are not redone for that.
    if info.revision and settings.revision and info.revision != settings.revision:
        return True
    chunks_file = paths.resolve(chunking.path)
    if not chunks_file.exists() or info.chunks_sha256 != sha256_file(chunks_file):
        return True
    return not (info.path and paths.resolve(info.path).exists())


def embed_record(record: BookRecord, paths: DataPaths, encoder: Encoder, settings: EmbeddingSettings) -> EmbeddingInfo:
    """Embed one chunked book. Errors are returned as status 'error', except an
    encoder that cannot be loaded, which would fail for every book."""
    base = dict(version=EMBEDDING_VERSION, model=encoder.name, backend=encoder.backend,
                revision=getattr(encoder, "revision", None))
    started = time.perf_counter()
    try:
        chunks_file = paths.resolve(record.chunking.path)
        chunks = list(iter_chunks(chunks_file))
        encoded = encode_sorted(encoder, [c.embedding_text() for c in chunks], settings.batch_size)
        authors = display_authors(record.authors)
        book_meta = encoder.encode([metadata_text(record, authors)]).dense[0]
        vectors = BookVectors(
            keys=[c.key for c in chunks],
            dense=encoded.dense,
            sparse=encoded.sparse,
            book_metadata=book_meta,
            book_centroid=centroid(encoded.dense),
            meta={"model": encoder.name, "backend": encoder.backend, "version": EMBEDDING_VERSION,
                  "dim": int(encoded.dense.shape[1]), "created_at": utc_now().isoformat()},
        )
        stem = PurePosixPath(record.chunking.path).stem
        path = paths.vectors_dir(record.source) / f"{stem}.npz"
        save_vectors(path, vectors)
        nnz = [len(i) for i, _ in encoded.sparse]
        return EmbeddingInfo(
            status="ok", dim=int(encoded.dense.shape[1]), chunks_sha256=sha256_file(chunks_file),
            vectors_sha256=sha256_file(path), embedded_at=utc_now(), path=paths.relative(path),
            chunks=len(chunks), sparse_terms_mean=round(float(np.mean(nnz)), 1) if nnz else 0.0,
            seconds=round(time.perf_counter() - started, 1), **base,
        )
    except EncoderUnavailable:
        raise
    except Exception as exc:
        log.exception("Embedding failed for %s", record.key)
        return EmbeddingInfo(status="error", embedded_at=utc_now(), error=f"{type(exc).__name__}: {exc}", **base)


def embed_books(
    registry: Registry,
    paths: DataPaths,
    encoder: Encoder,
    settings: EmbeddingSettings,
    keys: Iterable[str] | None = None,
    *,
    force: bool = False,
) -> EmbedSummary:
    wanted = set(keys) if keys is not None else None
    records = [r for r in registry if (wanted is None or r.key in wanted)
               and r.chunking is not None and r.chunking.status == "ok"]
    todo = [r for r in records if needs_embedding(r, paths, settings, force)]
    summary = EmbedSummary(skipped=len(records) - len(todo))
    log.info("%d chunked books | %d up to date | %d to embed (%s, %s backend)",
             len(records), summary.skipped, len(todo), encoder.name, encoder.backend)
    if todo:
        encoder.encode(["warm-up"])  # loads the model once; EncoderUnavailable stops here
    try:
        for i, record in enumerate(todo, 1):
            info = embed_record(record, paths, encoder, settings)
            record.embedding = info
            registry.save()  # embedding is slow: save after every book
            summary.embedded += 1
            summary.chunks += info.chunks or 0
            summary.by_status[info.status] = summary.by_status.get(info.status, 0) + 1
            rate = f"{info.chunks / info.seconds:.0f} chunks/s" if info.chunks and info.seconds else ""
            log.info("[%d/%d] %-16s %-5s chunks=%-5s %-12s %s %s", i, len(todo), record.key, info.status,
                     info.chunks, rate, record.title[:40], info.error or "")
    except KeyboardInterrupt:
        summary.interrupted = True
        registry.save()
    return summary
