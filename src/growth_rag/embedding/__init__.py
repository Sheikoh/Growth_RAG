"""Embedding: chunks -> dense + sparse vectors (bge-m3), stored on disk per book.

Two interchangeable encoders produce the same vectors:
- `encoders/manual.py`: plain transformers + bge-m3's sparse head, every step
  visible (CLS pooling, normalisation, token weights -> sparse vector);
- `encoders/flag.py`: the reference implementation (FlagEmbedding).
`rag embed-check` compares them on your own chunks.

Vectors are written to `data/vectors/<source>/<stem>.npz` before any upload,
so re-indexing (or moving to a cloud Qdrant) never requires re-embedding.

Bump EMBEDDING_VERSION when the way vectors are computed changes.
"""

EMBEDDING_VERSION = "1.0.0"
