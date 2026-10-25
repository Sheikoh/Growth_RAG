"""Gutenberg catalog: download dated snapshots and parse pg_catalog.csv.

Parsing is pure (string in, structured data out) so it is tested without files
or network. Columns: Text#, Type, Issued, Title, Language, Authors, Subjects,
LoCC, Bookshelves. Multi-valued fields are separated by '; '.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import requests

from growth_rag.config import GutenbergSettings, HttpSettings
from growth_rag.registry import Author, BookRecord, make_key
from growth_rag.sources.gutenberg import SOURCE
from growth_rag.utils.http import http_get
from growth_rag.utils.io import atomic_write_bytes

log = logging.getLogger(__name__)

SNAPSHOT_GLOB = "pg_catalog_*.csv"


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

ROLE_RE = re.compile(r"\s*\[([^\]]+)\]\s*$")
# Life dates: "1775-1817", "1950-", "-1817", "428? BCE-348? BCE".
YEARS_RE = re.compile(r",\s*(?:(\d{1,4})\??(\s*BCE)?)?\s*-\s*(?:(\d{1,4})\??(\s*BCE)?)?\s*$")
# Floruit instead of life dates: "active 1799-1828", "active 12th century".
ACTIVE_RE = re.compile(r",\s*active\s+(.+)$")


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One catalog row, parsed. Internal to the Gutenberg source (a dataclass,
    not a Pydantic model, because ~80k rows are parsed on every selection)."""

    id: int
    media_type: str
    title: str
    issued: str | None = None
    languages: tuple[str, ...] = ()
    authors: tuple[Author, ...] = ()
    subjects: tuple[str, ...] = ()
    locc: tuple[str, ...] = ()
    bookshelves: tuple[str, ...] = field(default=())


def split_multi(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(";") if part.strip())


def _year(digits: str | None, bce: str | None) -> int | None:
    if not digits:
        return None
    return -int(digits) if bce else int(digits)


def parse_author(raw: str) -> Author:
    """'Austen, Jane, 1775-1817 [Editor]' -> name / birth / death / role.
    Best effort: the catalog has irregular forms ('?', BCE, 'active ...', corporate
    authors, 'Various', 'Anonymous'), so the raw string is always kept."""
    rest, role = raw.strip(), None
    m = ROLE_RE.search(rest)
    if m:
        role, rest = m.group(1).strip(), rest[: m.start()]
    birth = death = active = None
    m = YEARS_RE.search(rest)
    if m:
        birth, death = _year(m.group(1), m.group(2)), _year(m.group(3), m.group(4))
        rest = rest[: m.start()]
    else:
        m = ACTIVE_RE.search(rest)
        if m:
            active, rest = m.group(1).strip(), rest[: m.start()]
    return Author(
        name=rest.strip().rstrip(","),
        birth_year=birth,
        death_year=death,
        active=active,
        role=role,
        raw=raw,
    )


def parse_row(row: dict[str, str]) -> CatalogEntry | None:
    try:
        book_id = int(row["Text#"])
    except (KeyError, ValueError):
        return None
    return CatalogEntry(
        id=book_id,
        media_type=(row.get("Type") or "").strip(),
        title=" ".join((row.get("Title") or "").split()),  # titles may span lines
        issued=(row.get("Issued") or "").strip() or None,
        languages=split_multi(row.get("Language")),
        authors=tuple(parse_author(a) for a in split_multi(row.get("Authors"))),
        subjects=split_multi(row.get("Subjects")),
        locc=split_multi(row.get("LoCC")),
        bookshelves=split_multi(row.get("Bookshelves")),
    )


def read_catalog(path: Path) -> dict[int, CatalogEntry]:
    # newline="" lets the csv module handle quoted fields containing line breaks.
    with path.open(encoding="utf-8", newline="") as f:
        entries = (parse_row(row) for row in csv.DictReader(f))
        return {e.id: e for e in entries if e is not None}


def to_record(entry: CatalogEntry, snapshot_name: str) -> BookRecord:
    """Catalog entry -> registry record (without acquisition info)."""
    return BookRecord(
        key=make_key(SOURCE, entry.id),
        source=SOURCE,
        source_id=str(entry.id),
        title=entry.title,
        languages=list(entry.languages),
        authors=list(entry.authors),
        subjects=list(entry.subjects),
        bookshelves=list(entry.bookshelves),
        locc=list(entry.locc),
        issued=entry.issued,
        media_type=entry.media_type,
        catalog_snapshot=snapshot_name,
    )


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #

def snapshot_path(catalog_dir: Path, day: date) -> Path:
    return catalog_dir / f"pg_catalog_{day.isoformat()}.csv"


def latest_snapshot(catalog_dir: Path) -> Path:
    snapshots = sorted(catalog_dir.glob(SNAPSHOT_GLOB))  # ISO dates sort lexically
    if not snapshots:
        raise FileNotFoundError(f"No catalog snapshot in {catalog_dir}. Run: gutenberg catalog")
    return snapshots[-1]


def fetch_catalog(
    session: requests.Session,
    catalog_dir: Path,
    *,
    pg: GutenbergSettings,
    http: HttpSettings,
    day: date | None = None,
    force: bool = False,
) -> Path:
    """Download today's catalog snapshot unless it already exists."""
    target = snapshot_path(catalog_dir, day or date.today())
    if target.exists() and not force:
        log.info("Snapshot already exists: %s (use --force to re-download)", target)
        return target
    log.info("Downloading catalog from %s ...", pg.catalog_url)
    response = http_get(session, pg.catalog_url, timeout_s=http.timeout_s, max_retries=http.max_retries)
    response.raise_for_status()
    atomic_write_bytes(target, response.content)
    log.info("Saved %s (%.1f MB)", target, len(response.content) / 1e6)
    return target
