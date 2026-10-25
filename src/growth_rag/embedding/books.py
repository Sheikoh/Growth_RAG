"""Book-level vectors, for recommendations (README: Qdrant `books` collection).

- `metadata`: embedding of a short description built from the catalog: title,
  authors, subjects, bookshelves, language. Available for every book now.
- `centroid`: mean of the book's chunk vectors (re-normalised): what the book
  *is about*, according to its own text.
- `summary` (declared in Qdrant, filled later): embedding of an LLM summary.
"""

from __future__ import annotations

import numpy as np

from growth_rag.embedding.encoders.base import l2_normalize
from growth_rag.registry import BookRecord


def metadata_text(record: BookRecord, authors: str) -> str:
    shelves = [s.removeprefix("Category: ") for s in record.bookshelves]
    lines = [
        record.title,
        f"Author: {authors}" if authors else "",
        f"Subjects: {'; '.join(record.subjects)}" if record.subjects else "",
        f"Bookshelves: {'; '.join(dict.fromkeys(shelves))}" if shelves else "",
        f"Language: {', '.join(record.languages)}" if record.languages else "",
    ]
    return "\n".join(line for line in lines if line)


def centroid(dense: np.ndarray) -> np.ndarray | None:
    if dense.size == 0:
        return None
    return l2_normalize(dense.mean(axis=0, keepdims=True))[0]
