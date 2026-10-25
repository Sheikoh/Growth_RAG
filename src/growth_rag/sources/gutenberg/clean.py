"""Cleaning step for Gutenberg books: registry -> raw file -> cleaned outputs.

Gutenberg-specific parts only (reading the raw file, licence markers, the
"Plays" bookshelf hint); everything else is the generic `growth_rag.cleaning`
pipeline. Results are recorded in the registry (`BookRecord.cleaning`).

A book is (re)cleaned when it has never been cleaned, when the cleaning rules
changed (CLEANING_VERSION), when its raw file changed (checksum), when its
outputs are missing, or with --force.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from growth_rag.cleaning import CLEANING_VERSION
from growth_rag.cleaning.pipeline import BookInput, clean_book
from growth_rag.cleaning.store import save_cleaned
from growth_rag.cleaning.text import decode, normalize
from growth_rag.config import DataPaths
from growth_rag.registry import BookRecord, CleaningInfo, Registry
from growth_rag.sources.gutenberg import SOURCE
from growth_rag.sources.gutenberg.boilerplate import extract_body
from growth_rag.utils.io import sha256_bytes, utc_now

log = logging.getLogger(__name__)

PLAY_SHELF_WORDS = ("plays", "drama")


@dataclass
class CleanSummary:
    cleaned: int = 0
    skipped: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    interrupted: bool = False


def needs_cleaning(record: BookRecord, paths: DataPaths, force: bool = False) -> bool:
    acq = record.acquisition
    if acq is None or acq.status != "ok":
        return False  # nothing to clean
    info = record.cleaning
    if force or info is None or info.status == "error":
        return True
    if info.version != CLEANING_VERSION or info.raw_sha256 != acq.sha256:
        return True
    outputs = [info.text_path, info.structure_path]
    return not all(p and paths.resolve(p).exists() for p in outputs)


def _play_hint(record: BookRecord) -> bool:
    return any(word in shelf.casefold() for shelf in record.bookshelves for word in PLAY_SHELF_WORDS)


def clean_record(record: BookRecord, paths: DataPaths) -> CleaningInfo:
    """Clean one downloaded book and write its outputs. Never raises: errors are
    returned as a CleaningInfo with status 'error'."""
    acq = record.acquisition
    try:
        data = paths.resolve(acq.path).read_bytes()
        text, used_fallback = decode(data)
        body = extract_body(normalize(text))
        source_flags = []
        if used_fallback:
            source_flags.append("decode_fallback")
        if not body.has_start:
            source_flags.append("no_start_marker")
        if not body.has_end:
            source_flags.append("no_end_marker")

        cleaned_text, book = clean_book(BookInput(
            key=record.key,
            language=record.primary_language,
            body=body.text,
            body_offset=body.offset,
            play_hint=_play_hint(record),
            title=record.title,
            raw_sha256=acq.sha256,
            source_flags=tuple(source_flags),
        ))
        stem = f"pg{record.source_id}"
        text_path, structure_path = save_cleaned(paths.clean_dir(SOURCE), stem, cleaned_text, book)
        return CleaningInfo(
            status=book.status,
            version=CLEANING_VERSION,
            raw_sha256=acq.sha256,
            cleaned_at=book.cleaned_at,
            text_path=paths.relative(text_path),
            text_sha256=sha256_bytes(cleaned_text.encode("utf-8")),
            structure_path=paths.relative(structure_path),
            flags=book.flags,
            sections=len(book.sections),
            paragraphs=len(book.paragraphs),
            annotations=len(book.annotations),
        )
    except Exception as exc:  # one bad book must not stop the run
        log.exception("Cleaning failed for %s", record.key)
        return CleaningInfo(status="error", version=CLEANING_VERSION, raw_sha256=acq.sha256,
                            cleaned_at=utc_now(), error=f"{type(exc).__name__}: {exc}")


def clean_books(
    registry: Registry,
    paths: DataPaths,
    keys: Iterable[str] | None = None,
    *,
    force: bool = False,
    save_every: int = 10,
) -> CleanSummary:
    """Clean the given books (default: every downloaded Gutenberg book)."""
    wanted = set(keys) if keys is not None else None
    records = [
        r for r in registry
        if r.source == SOURCE and (wanted is None or r.key in wanted)
        and r.acquisition and r.acquisition.status == "ok"
    ]
    todo = [r for r in records if needs_cleaning(r, paths, force)]
    summary = CleanSummary(skipped=len(records) - len(todo))
    log.info("%d downloaded books | %d up to date | %d to clean", len(records), summary.skipped, len(todo))
    try:
        for i, record in enumerate(todo, 1):
            info = clean_record(record, paths)
            record.cleaning = info
            summary.cleaned += 1
            summary.by_status[info.status] = summary.by_status.get(info.status, 0) + 1
            detail = info.error or ", ".join(info.flags) or ""
            log.info("[%d/%d] %-16s %-8s sections=%-4s %s %s", i, len(todo), record.key, info.status,
                     info.sections, record.title[:45], detail)
            if i % save_every == 0:
                registry.save()
    except KeyboardInterrupt:
        summary.interrupted = True
    finally:
        registry.save()
    return summary
