"""Selection: filter the catalog, stratify by language, write a manifest.

A manifest is the dataset definition: which catalog snapshot, which criteria,
which IDs. It is small, human-readable and versionable; you can also write or
edit one by hand, `download` only needs its `ids`.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from growth_rag.sources.gutenberg import SOURCE
from growth_rag.sources.gutenberg.catalog import CatalogEntry
from growth_rag.utils.io import atomic_write_json, read_json, sha256_file, utc_now
from growth_rag.utils.text_norm import contains_words, normalize


class SelectionCriteria(BaseModel):
    """All text filters are case-insensitive substring matches. Within one filter
    the values are OR-ed; different filters are AND-ed."""

    model_config = ConfigDict(extra="forbid")

    media_type: str = "Text"
    languages: list[str] | None = None  # ISO codes; None = all
    bookshelves: list[str] | None = None
    authors: list[str] | None = None  # matched on the raw author strings
    subjects: list[str] | None = None
    exclude_subjects: list[str] | None = None
    titles: list[str] | None = None  # whole words, accent-insensitive; any of these titles
    ids: list[int] | None = None  # explicit book numbers
    per_language: int | None = None  # cap per language stratum; None = no cap
    seed: int = 42


class Manifest(BaseModel):
    # Criteria are stored as a plain dict so manifests stay readable even if the
    # criteria model evolves (older manifests used different key names).
    name: str
    source: str = SOURCE
    created_at: datetime
    catalog_snapshot: str
    catalog_sha256: str | None = None
    criteria: dict[str, Any] = {}
    counts: dict[str, dict[str, int]] = {}
    ids: list[int]

    @classmethod
    def load(cls, path: Path) -> Manifest:
        return cls.model_validate(read_json(path))

    def save(self, path: Path) -> None:
        atomic_write_json(path, self.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# Filtering and stratification (pure functions)
# --------------------------------------------------------------------------- #

def any_contains(values: Iterable[str], needles: Iterable[str]) -> bool:
    lowered = [v.lower() for v in values]
    return any(n.lower() in v for n in needles for v in lowered)


def matches(entry: CatalogEntry, c: SelectionCriteria) -> bool:
    if entry.media_type != c.media_type:
        return False
    if c.ids and entry.id not in c.ids:
        return False
    if c.titles:
        title = normalize(entry.title)
        if not any(contains_words(title, normalize(t)) for t in c.titles):
            return False
    if c.languages and not set(c.languages) & set(entry.languages):
        return False
    if c.bookshelves and not any_contains(entry.bookshelves, c.bookshelves):
        return False
    if c.authors and not any_contains((a.raw for a in entry.authors), c.authors):
        return False
    if c.subjects and not any_contains(entry.subjects, c.subjects):
        return False
    if c.exclude_subjects and any_contains(entry.subjects, c.exclude_subjects):
        return False
    return True


def stratum_language(entry: CatalogEntry, requested: list[str] | None) -> str:
    """First of the book's languages that was requested (a multilingual book is
    counted once), else its first language."""
    for lang in entry.languages:
        if not requested or lang in requested:
            return lang
    return "und"


def select(
    catalog: Mapping[int, CatalogEntry], criteria: SelectionCriteria
) -> tuple[list[int], dict[str, dict[str, int]]]:
    """Return (selected ids, per-language counts). Deterministic for a given
    catalog and seed: ids are sorted before the seeded shuffle."""
    strata: dict[str, list[int]] = {}
    for entry in catalog.values():
        if matches(entry, criteria):
            strata.setdefault(stratum_language(entry, criteria.languages), []).append(entry.id)

    rng = random.Random(criteria.seed)
    selected: list[int] = []
    counts: dict[str, dict[str, int]] = {}
    for lang in sorted(strata):
        ids = sorted(strata[lang])
        rng.shuffle(ids)
        taken = ids if criteria.per_language is None else ids[: criteria.per_language]
        selected.extend(taken)
        counts[lang] = {"candidates": len(ids), "selected": len(taken)}
    return sorted(selected), counts


def build_manifest(
    name: str,
    catalog: Mapping[int, CatalogEntry],
    snapshot: Path,
    criteria: SelectionCriteria,
) -> Manifest:
    ids, counts = select(catalog, criteria)
    return Manifest(
        name=name,
        created_at=utc_now(),
        catalog_snapshot=snapshot.name,
        catalog_sha256=sha256_file(snapshot),
        criteria=criteria.model_dump(),
        counts=counts,
        ids=ids,
    )
