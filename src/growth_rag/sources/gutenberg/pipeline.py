"""Add books end to end: select -> download -> clean -> chunk -> embed -> index.

    uv run gutenberg add --title "moby dick" --languages en --dry-run   # look first
    uv run gutenberg add --title "moby dick" --languages en             # then add
    uv run gutenberg add --author "Verne, Jules" --languages fr --per-language 5
    uv run gutenberg add --id 1342 84 2701 --name eval-extra
    uv run gutenberg add --manifest data/manifests/pilot.json           # re-run / resume
    uv run gutenberg add --title hamlet --until chunk                   # stop after a step

Every run writes (or reuses) a manifest in data/manifests/: the selection is
recorded and reproducible (an evaluation set is a manifest plus questions).
Every step is the same function as its own command (gutenberg download/clean,
rag chunk/embed/index) and is idempotent: books already done are skipped, so
an interrupted run is resumed by running the same command again.

If Qdrant is not running, everything up to the vectors on disk is done and the
run says to start Qdrant and run `rag index`.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from growth_rag.chunking.run import chunk_books
from growth_rag.chunking.tokens import TokenCounter
from growth_rag.config import DataPaths, Settings
from growth_rag.embedding.encoders.base import Encoder
from growth_rag.embedding.run import embed_books
from growth_rag.index.qdrant import IndexUnavailable
from growth_rag.index.run import index_books
from growth_rag.registry import Registry, make_key
from growth_rag.sources.gutenberg import SOURCE
from growth_rag.sources.gutenberg.catalog import CatalogEntry
from growth_rag.sources.gutenberg.clean import clean_books
from growth_rag.sources.gutenberg.download import download_manifest
from growth_rag.sources.gutenberg.selection import Manifest

log = logging.getLogger(__name__)

STEPS = ("download", "clean", "chunk", "embed", "index")
CONFIRM_ABOVE = 50  # books; more than this requires --yes


@dataclass
class PipelineReport:
    steps_run: list[str] = field(default_factory=list)
    stopped_at: str | None = None  # step that could not run (e.g. Qdrant down)
    status: dict[str, dict[str, int]] = field(default_factory=dict)  # stage -> status -> count
    flagged: list[str] = field(default_factory=list)  # "key  title  flags"


def default_manifest_name() -> str:
    return datetime.now().strftime("add-%Y%m%d-%H%M%S")


def describe_selection(manifest: Manifest, catalog: dict[int, CatalogEntry], limit: int = 25) -> list[str]:
    """Lines to show before running: totals by language, then the books."""
    langs = Counter(catalog[i].languages[0] if i in catalog and catalog[i].languages else "?"
                    for i in manifest.ids)
    lines = [f"{len(manifest.ids)} book(s): " + ", ".join(f"{k}={v}" for k, v in langs.most_common())]
    for i in manifest.ids[:limit]:
        e = catalog.get(i)
        if e is None:
            lines.append(f"  pg{i:<6} (not in catalog)")
            continue
        author = next((a.name for a in e.authors if a.role is None), "")
        lines.append(f"  pg{i:<6} [{','.join(e.languages)}] {e.title[:60]} — {author[:30]}")
    if len(manifest.ids) > limit:
        lines.append(f"  ... and {len(manifest.ids) - limit} more (see the manifest file)")
    return lines


def run_pipeline(
    registry: Registry,
    manifest: Manifest,
    catalog: dict[int, CatalogEntry],
    snapshot_name: str,
    paths: DataPaths,
    settings: Settings,
    *,
    until: str = "index",
    session,
    counter: TokenCounter,
    encoder_factory: Callable[[], Encoder],
    client_factory: Callable[[], object],
    delay_s: float | None = None,
    sleep=None,
) -> PipelineReport:
    """Run the steps up to `until` for the manifest's books only. The
    factories delay loading the model / connecting to Qdrant until needed."""
    keys = [make_key(SOURCE, i) for i in manifest.ids]
    last = STEPS.index(until)
    report = PipelineReport()

    def step(name: str) -> bool:
        if STEPS.index(name) > last:
            return False
        log.info("--- %s ---", name)
        report.steps_run.append(name)
        return True

    if step("download"):
        extra = {"sleep": sleep} if sleep is not None else {}
        download_manifest(manifest, catalog, snapshot_name, registry, paths, session,
                          http=settings.http, mirror_url=settings.gutenberg.mirror_url,
                          delay_s=settings.gutenberg.download_delay_s if delay_s is None else delay_s,
                          **extra)
    if step("clean"):
        clean_books(registry, paths, keys)
    if step("chunk"):
        chunk_books(registry, paths, counter, settings.chunking, keys, source=SOURCE)
    if step("embed"):
        embed_books(registry, paths, encoder_factory(), settings.embedding, keys)
    if step("index"):
        try:
            index_books(registry, paths, client_factory(), settings.qdrant, keys)
        except IndexUnavailable as exc:
            report.stopped_at = "index"
            log.warning("%s", exc)
            log.warning("Vectors are on disk: start Qdrant, then run `uv run rag index`.")

    report.status, report.flagged = _status(registry, keys)
    return report


def _status(registry: Registry, keys: list[str]) -> tuple[dict, list[str]]:
    stages = {
        "download": lambda r: r.acquisition and r.acquisition.status,
        "clean": lambda r: r.cleaning and r.cleaning.status,
        "chunk": lambda r: r.chunking and r.chunking.status,
        "embed": lambda r: r.embedding and r.embedding.status,
        "index": lambda r: r.index and r.index.status,
    }
    status: dict[str, dict[str, int]] = {}
    flagged: list[str] = []
    for key in keys:
        record = registry.get(key)
        for stage, get in stages.items():
            value = (get(record) if record else None) or "not done"
            status.setdefault(stage, {}).setdefault(value, 0)
            status[stage][value] += 1
        if record and record.cleaning and record.cleaning.status == "flagged":
            flagged.append(f"{key}  {record.title[:50]}  {', '.join(record.cleaning.flags)}")
    return status, flagged


def log_report(report: PipelineReport) -> None:
    log.info("--- summary ---")
    for stage in STEPS:
        if stage in report.status:
            counts = ", ".join(f"{k}={v}" for k, v in sorted(report.status[stage].items()))
            log.info("  %-8s %s", stage, counts)
    if report.flagged:
        log.info("Flagged by cleaning (usable, worth a look):")
        for line in report.flagged:
            log.info("  %s", line)
    if report.stopped_at:
        log.info("Stopped before '%s' (see above). Re-run the same command to resume.", report.stopped_at)
