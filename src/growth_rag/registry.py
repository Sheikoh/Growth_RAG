"""The registry: one record per book, whatever its source.

It is the contract between pipeline steps. Acquisition writes it; cleaning,
chunking and Qdrant ingestion read it. Its fields will become the Qdrant payload
of the `books` collection and the properties of Book nodes in the graph phase,
so the schema is explicit and validated (Pydantic) rather than an implicit dict.

Records are keyed by "<source>:<source_id>" (e.g. "gutenberg:1342") so books from
several sources (Gutenberg, owned books, ...) can live in one registry.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from growth_rag.utils.io import atomic_write_json, read_json, utc_now

SCHEMA_VERSION = 2


def make_key(source: str, source_id: str | int) -> str:
    return f"{source}:{source_id}"


class Author(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    birth_year: int | None = None  # negative = BCE
    death_year: int | None = None
    active: str | None = None  # floruit text when no life dates, e.g. "12th century"
    role: str | None = None  # None = author; else "Translator", "Editor", ...
    raw: str  # original catalog string, always kept


AcquisitionStatus = Literal["ok", "not_found", "error"]


class Acquisition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: AcquisitionStatus
    source_url: str
    attempted_at: datetime
    http_status: int | None = None
    content_type: str | None = None
    error: str | None = None
    # Present when status == "ok":
    path: str | None = None  # relative to the data root, POSIX separators
    sha256: str | None = None
    size_bytes: int | None = None
    downloaded_at: datetime | None = None
    # Diagnostics for the cleaning step:
    utf8_valid: bool | None = None
    has_start_marker: bool | None = None
    has_end_marker: bool | None = None


CleaningStatus = Literal["ok", "flagged", "error"]


class CleaningInfo(BaseModel):
    """Summary of the cleaning step for one book. The full structure lives in the
    cleaned book's JSON file; this is what is needed to decide whether to re-clean
    and to list flagged books."""

    model_config = ConfigDict(extra="forbid")

    status: CleaningStatus
    version: str  # cleaning rules version; a change triggers re-cleaning
    raw_sha256: str | None = None  # raw file it was computed from
    cleaned_at: datetime
    text_path: str | None = None  # relative to the data root
    text_sha256: str | None = None  # of the cleaned text: decides re-chunking
    structure_path: str | None = None
    flags: list[str] = []
    error: str | None = None
    sections: int | None = None
    paragraphs: int | None = None
    annotations: int | None = None


ChunkingStatus = Literal["ok", "error"]


class ChunkingInfo(BaseModel):
    """Summary of the chunking step for one book (chunks are in the JSONL file)."""

    model_config = ConfigDict(extra="forbid")

    status: ChunkingStatus
    version: str  # chunking rules version
    tokenizer: str  # sizes depend on it: a change triggers re-chunking
    target_tokens: int
    max_tokens: int
    min_tokens: int
    cleaned_text_sha256: str | None = None  # cleaned text it was computed from
    chunked_at: datetime
    path: str | None = None  # relative to the data root
    chunks: int | None = None
    tokens_total: int | None = None
    tokens_median: int | None = None
    tokens_max: int | None = None
    by_kind: dict[str, int] = {}
    error: str | None = None


class EmbeddingInfo(BaseModel):
    """Summary of the embedding step for one book (vectors are in the .npz file)."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error"]
    version: str  # embedding code version
    model: str  # e.g. "BAAI/bge-m3": a change triggers re-embedding
    revision: str | None = None  # model commit; a change triggers re-embedding (None: not recorded)
    backend: str  # "manual" or "flag" (same vectors expected; recorded for traceability)
    dim: int | None = None
    chunks_sha256: str | None = None  # chunks file it was computed from
    vectors_sha256: str | None = None  # decides re-indexing
    embedded_at: datetime
    path: str | None = None
    chunks: int | None = None
    sparse_terms_mean: float | None = None  # average non-zero terms per sparse vector
    seconds: float | None = None
    error: str | None = None


class IndexInfo(BaseModel):
    """Where and from what the book was last uploaded to Qdrant."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error"]
    target: str  # Qdrant URL or local path
    chunks_collection: str
    books_collection: str
    vectors_sha256: str | None = None
    points: int | None = None
    indexed_at: datetime
    error: str | None = None


class BookRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    source: str
    source_id: str
    title: str
    languages: list[str] = []
    authors: list[Author] = []
    subjects: list[str] = []
    bookshelves: list[str] = []  # Gutenberg-specific, empty for other sources
    locc: list[str] = []  # Library of Congress classes, if known
    issued: str | None = None
    media_type: str | None = None
    catalog_snapshot: str | None = None  # which catalog file the metadata came from
    manifests: list[str] = []  # every manifest that requested this book
    acquisition: Acquisition | None = None
    # Added after schema v2 was released: optional, so existing registries stay
    # valid (policy: additive optional fields do not change SCHEMA_VERSION).
    cleaning: CleaningInfo | None = None
    chunking: ChunkingInfo | None = None
    embedding: EmbeddingInfo | None = None
    index: IndexInfo | None = None

    @model_validator(mode="after")
    def _key_matches(self) -> BookRecord:
        if self.key != make_key(self.source, self.source_id):
            raise ValueError(f"key {self.key!r} != {make_key(self.source, self.source_id)!r}")
        return self

    @property
    def primary_language(self) -> str:
        return self.languages[0] if self.languages else "und"


class RegistryFormatError(RuntimeError):
    pass


class Registry:
    def __init__(self, path: Path, books: dict[str, BookRecord] | None = None):
        self.path = path
        self._books: dict[str, BookRecord] = books or {}

    # -- persistence -------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> Registry:
        if not path.exists():
            return cls(path)
        data = read_json(path)
        version = data.get("schema_version") if isinstance(data, dict) else None
        if version != SCHEMA_VERSION:
            raise RegistryFormatError(
                f"{path} has schema version {version!r}, expected {SCHEMA_VERSION}. "
                "If it was written by the old single-file script, run: gutenberg migrate"
            )
        books = {}
        for key, rec in data["books"].items():
            try:
                books[key] = BookRecord.model_validate(rec)
            except ValidationError as exc:
                first = exc.errors()[0]
                field = ".".join(str(x) for x in first["loc"])
                raise RegistryFormatError(
                    f"{path}: record {key!r} does not match the current schema "
                    f"({field}: {first['msg']}). It was probably written by another version "
                    "of the code; re-run the step that wrote this field, or restore a backup."
                ) from exc
        return cls(path, books)

    def save(self) -> None:
        atomic_write_json(
            self.path,
            {
                "schema_version": SCHEMA_VERSION,
                "updated_at": utc_now().isoformat(),
                "books": {
                    key: self._books[key].model_dump(mode="json") for key in sorted(self._books)
                },
            },
        )

    # -- access ------------------------------------------------------------

    def get(self, key: str) -> BookRecord | None:
        return self._books.get(key)

    def upsert(self, record: BookRecord) -> None:
        self._books[record.key] = record

    def __len__(self) -> int:
        return len(self._books)

    def __iter__(self) -> Iterator[BookRecord]:
        return iter(self._books.values())

    def downloaded(self) -> Iterator[BookRecord]:
        return (r for r in self if r.acquisition and r.acquisition.status == "ok")

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict:
        statuses = Counter(r.acquisition.status if r.acquisition else "pending" for r in self)
        ok = list(self.downloaded())
        return {
            "books": len(self),
            "by_status": dict(statuses.most_common()),
            "by_source": dict(Counter(r.source for r in self).most_common()),
            "downloaded_mb": round(sum(r.acquisition.size_bytes or 0 for r in ok) / 1e6, 1),
            "downloaded_by_language": dict(Counter(r.primary_language for r in ok).most_common()),
            "cleaning_flags": {
                "non_utf8": sum(r.acquisition.utf8_valid is False for r in ok),
                "no_start_marker": sum(r.acquisition.has_start_marker is False for r in ok),
                "no_end_marker": sum(r.acquisition.has_end_marker is False for r in ok),
            },
            "cleaning": dict(
                Counter(r.cleaning.status if r.cleaning else "not_cleaned" for r in ok).most_common()
            ),
            "cleaning_issues": dict(
                Counter(f for r in ok if r.cleaning for f in r.cleaning.flags).most_common()
            ),
            "chunking": dict(
                Counter(r.chunking.status if r.chunking else "not_chunked" for r in ok).most_common()
            ),
            "chunks_total": sum(r.chunking.chunks or 0 for r in ok if r.chunking),
            "chunks_by_kind": dict(sum(
                (Counter(r.chunking.by_kind) for r in ok if r.chunking), Counter()
            ).most_common()),
            "embedding": dict(
                Counter(r.embedding.status if r.embedding else "not_embedded" for r in ok).most_common()
            ),
            "index": dict(
                Counter(r.index.status if r.index else "not_indexed" for r in ok).most_common()
            ),
            "points_indexed": sum(r.index.points or 0 for r in ok if r.index and r.index.status == "ok"),
        }
