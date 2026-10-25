"""Chunks are stored as JSON Lines: one chunk per line, streamable by the
ingestion step without loading a whole corpus in memory."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from growth_rag.chunking.models import Chunk
from growth_rag.utils.io import atomic_write_bytes


def save_chunks(path: Path, chunks: list[Chunk]) -> None:
    lines = (json.dumps(c.model_dump(mode="json"), ensure_ascii=False) for c in chunks)
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8") if chunks else b"")


def iter_chunks(path: Path) -> Iterator[Chunk]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Chunk.model_validate_json(line)
