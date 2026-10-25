"""Download the texts listed in a manifest from the Gutenberg mirror.

Idempotent and resumable: books already present (and intact) are skipped, the
registry is checkpointed regularly and saved on interruption, and files are
written atomically. Raw files are stored byte-exact; cleaning happens later.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import requests

from growth_rag.config import DataPaths, HttpSettings
from growth_rag.registry import Acquisition, Registry, make_key
from growth_rag.sources.gutenberg import SOURCE
from growth_rag.sources.gutenberg.catalog import CatalogEntry, to_record
from growth_rag.sources.gutenberg.selection import Manifest
from growth_rag.utils.http import HttpError, http_get
from growth_rag.utils.io import atomic_write_bytes, is_utf8, sha256_bytes, sha256_file, utc_now

log = logging.getLogger(__name__)

START_MARKER = re.compile(rb"\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE)
END_MARKER = re.compile(rb"\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE)


@dataclass
class DownloadPlan:
    todo: list[int] = field(default_factory=list)
    already_present: int = 0
    known_missing: int = 0


@dataclass
class DownloadSummary:
    attempted: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    interrupted: bool = False


def raw_filename(book_id: int) -> str:
    return f"pg{book_id}.txt"


def is_intact(acq: Acquisition | None, paths: DataPaths, verify_hash: bool) -> bool:
    """True if the registry says 'ok' and the file on disk still matches."""
    if acq is None or acq.status != "ok" or not acq.path:
        return False
    path = paths.resolve(acq.path)
    if not path.exists() or path.stat().st_size != acq.size_bytes:
        return False
    return not verify_hash or sha256_file(path) == acq.sha256


def download_book(
    session: requests.Session,
    book_id: int,
    *,
    mirror_url: str,
    paths: DataPaths,
    http: HttpSettings,
) -> Acquisition:
    url = mirror_url.format(id=book_id)
    attempted_at = utc_now()
    try:
        response = http_get(session, url, timeout_s=http.timeout_s, max_retries=http.max_retries)
    except HttpError as exc:
        return Acquisition(status="error", source_url=url, attempted_at=attempted_at, error=str(exc))

    content_type = response.headers.get("Content-Type")
    base = dict(
        source_url=url,
        attempted_at=attempted_at,
        http_status=response.status_code,
        content_type=content_type,
    )
    if response.status_code == 404:
        return Acquisition(status="not_found", **base)  # e.g. no plain-text edition
    if response.status_code != 200:
        return Acquisition(status="error", error=f"HTTP {response.status_code}", **base)
    data = response.content
    if not data:
        return Acquisition(status="error", error="empty body", **base)
    if content_type and "html" in content_type.lower():
        # A 200 carrying an HTML page is an error page, not a book.
        return Acquisition(status="error", error=f"unexpected content type {content_type}", **base)

    path = paths.raw_dir(SOURCE) / raw_filename(book_id)
    atomic_write_bytes(path, data)
    return Acquisition(
        status="ok",
        path=paths.relative(path),
        sha256=sha256_bytes(data),
        size_bytes=len(data),
        downloaded_at=utc_now(),
        utf8_valid=is_utf8(data),
        has_start_marker=bool(START_MARKER.search(data)),
        has_end_marker=bool(END_MARKER.search(data)),
        **base,
    )


def plan_downloads(
    manifest: Manifest,
    registry: Registry,
    paths: DataPaths,
    *,
    retry_missing: bool = False,
    verify_hash: bool = False,
) -> DownloadPlan:
    """Decide what to download. Side effect: books already in the registry are
    tagged with this manifest's name, even when skipped."""
    plan = DownloadPlan()
    for book_id in manifest.ids:
        record = registry.get(make_key(SOURCE, book_id))
        if record is not None:
            if manifest.name not in record.manifests:
                record.manifests.append(manifest.name)
            if is_intact(record.acquisition, paths, verify_hash):
                plan.already_present += 1
                continue
            if record.acquisition and record.acquisition.status == "not_found" and not retry_missing:
                plan.known_missing += 1
                continue
        plan.todo.append(book_id)
    return plan


def download_manifest(
    manifest: Manifest,
    catalog: Mapping[int, CatalogEntry],
    snapshot_name: str,
    registry: Registry,
    paths: DataPaths,
    session: requests.Session,
    *,
    http: HttpSettings,
    mirror_url: str,
    delay_s: float,
    retry_missing: bool = False,
    verify_hash: bool = False,
    save_every: int = 10,
    sleep: Callable[[float], None] = time.sleep,
) -> DownloadSummary:
    plan = plan_downloads(manifest, registry, paths, retry_missing=retry_missing, verify_hash=verify_hash)
    log.info(
        "Manifest '%s': %d books | %d already present | %d known missing | %d to download",
        manifest.name, len(manifest.ids), plan.already_present, plan.known_missing, len(plan.todo),
    )
    if plan.todo:
        log.info("Estimated minimum duration: %.1f min", len(plan.todo) * delay_s / 60)

    summary = DownloadSummary()
    try:
        for i, book_id in enumerate(plan.todo, 1):
            entry = catalog.get(book_id)
            acquisition = download_book(session, book_id, mirror_url=mirror_url, paths=paths, http=http)
            key = make_key(SOURCE, book_id)
            previous = registry.get(key)

            if entry is not None:
                record = to_record(entry, snapshot_name)
            elif previous is not None:
                record = previous.model_copy(deep=True)
            else:
                log.warning("pg%d is not in catalog %s; recording without metadata", book_id, snapshot_name)
                record = to_record(CatalogEntry(id=book_id, media_type="", title=""), snapshot_name)

            record.manifests = sorted(set(previous.manifests if previous else []) | {manifest.name})
            record.acquisition = acquisition
            registry.upsert(record)

            summary.attempted += 1
            summary.by_status[acquisition.status] = summary.by_status.get(acquisition.status, 0) + 1
            detail = f"{acquisition.size_bytes / 1e3:.0f} kB" if acquisition.size_bytes else (acquisition.error or "")
            log.info("[%d/%d] pg%-6d %-9s %-10s %s", i, len(plan.todo), book_id,
                     acquisition.status, detail, record.title[:60])

            if i % save_every == 0:
                registry.save()
            if i < len(plan.todo):
                sleep(delay_s)
    except KeyboardInterrupt:
        summary.interrupted = True
    finally:
        registry.save()
    return summary
