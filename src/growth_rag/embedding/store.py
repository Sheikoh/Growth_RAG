"""Vectors of one book, in a single compressed .npz file.

Arrays:
    keys            [n]           chunk keys, same order as the chunks file
    dense           [n, dim]      float16 on disk (half the size; cosine is
                                  unaffected at this precision), float32 in memory
    sparse_indptr   [n + 1]       CSR layout: chunk i's sparse vector is
    sparse_indices  [nnz]         indices[indptr[i]:indptr[i+1]] / values[...]
    sparse_values   [nnz]
    book_metadata   [dim]         book-level vectors (README: Qdrant `books`)
    book_centroid   [dim]
    meta            JSON string   model, backend, version, created_at, ...
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from growth_rag.embedding.encoders.base import Sparse
from growth_rag.utils.io import atomic_write_bytes


@dataclass
class BookVectors:
    keys: list[str]
    dense: np.ndarray  # [n, dim] float32
    sparse: list[Sparse]
    book_metadata: np.ndarray | None
    book_centroid: np.ndarray | None
    meta: dict


def save_vectors(path: Path, vectors: BookVectors) -> None:
    lengths = [len(i) for i, _ in vectors.sparse]
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    empty = np.zeros(0, dtype=np.int32)
    indices = np.concatenate([i for i, _ in vectors.sparse]) if vectors.sparse else empty
    values = np.concatenate([v for _, v in vectors.sparse]) if vectors.sparse else empty.astype(np.float32)
    dim = vectors.dense.shape[1] if vectors.dense.ndim == 2 else 0
    none = np.zeros(0, dtype=np.float16)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        keys=np.array(vectors.keys, dtype=np.str_),
        dense=vectors.dense.astype(np.float16).reshape(len(vectors.keys), dim),
        sparse_indptr=indptr,
        sparse_indices=indices.astype(np.int32),
        sparse_values=values.astype(np.float32),
        book_metadata=none if vectors.book_metadata is None else vectors.book_metadata.astype(np.float16),
        book_centroid=none if vectors.book_centroid is None else vectors.book_centroid.astype(np.float16),
        meta=np.array(json.dumps(vectors.meta, ensure_ascii=False)),
    )
    atomic_write_bytes(path, buf.getvalue())


def load_vectors(path: Path) -> BookVectors:
    with np.load(path, allow_pickle=False) as data:
        indptr, indices, values = data["sparse_indptr"], data["sparse_indices"], data["sparse_values"]
        sparse = [(indices[a:b], values[a:b]) for a, b in zip(indptr[:-1], indptr[1:])]
        metadata, centroid = data["book_metadata"], data["book_centroid"]
        return BookVectors(
            keys=[str(k) for k in data["keys"]],
            dense=data["dense"].astype(np.float32),
            sparse=sparse,
            book_metadata=metadata.astype(np.float32) if metadata.size else None,
            book_centroid=centroid.astype(np.float32) if centroid.size else None,
            meta=json.loads(str(data["meta"])),
        )
