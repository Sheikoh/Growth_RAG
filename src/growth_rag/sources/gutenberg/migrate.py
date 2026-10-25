"""One-off migration from the single-file script (gutenberg_acquire.py) layout.

Old layout                        New layout
  catalog/pg_catalog_*.csv   ->     catalog/gutenberg/pg_catalog_*.csv
  raw/pg<id>.txt             ->     raw/gutenberg/pg<id>.txt
  registry.json (v1)         ->     registry.json (schema v2, keys "gutenberg:<id>")
  manifests/*.json                  unchanged (still readable)

Idempotent: running it on an already migrated folder changes nothing. The old
registry is kept as registry.v1.json.bak.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass

from growth_rag.config import DataPaths
from growth_rag.registry import SCHEMA_VERSION, Acquisition, Author, BookRecord, Registry, make_key
from growth_rag.sources.gutenberg import SOURCE
from growth_rag.utils.io import read_json

log = logging.getLogger(__name__)


@dataclass
class MigrationReport:
    catalogs_moved: int = 0
    raw_files_moved: int = 0
    records_converted: int = 0
    registry_already_current: bool = False


def _move(src, dst) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        os.remove(src)  # already migrated copy exists; drop the duplicate
    else:
        shutil.move(src, dst)


def _convert_record(old_key: str, old: dict, paths: DataPaths) -> BookRecord:
    acquisition = None
    if old.get("acquisition"):
        acq = dict(old["acquisition"])
        if acq.get("path"):
            # The file was moved to raw/gutenberg/ by the step before.
            acq["path"] = paths.relative(paths.raw_dir(SOURCE) / os.path.basename(acq["path"]))
        acquisition = Acquisition.model_validate(acq)
    return BookRecord(
        key=make_key(SOURCE, old_key),
        source=SOURCE,
        source_id=str(old_key),
        title=old.get("title", ""),
        languages=old.get("languages", []),
        authors=[Author.model_validate(a) for a in old.get("authors", [])],
        subjects=old.get("subjects", []),
        bookshelves=old.get("bookshelves", []),
        locc=old.get("locc", []),
        issued=old.get("issued"),
        media_type=old.get("type"),
        catalog_snapshot=old.get("catalog_snapshot"),
        manifests=old.get("manifests", []),
        acquisition=acquisition,
    )


def migrate_v1(paths: DataPaths) -> MigrationReport:
    report = MigrationReport()

    # 1. Catalog snapshots
    for src in sorted((paths.root / "catalog").glob("pg_catalog_*.csv")):
        _move(src, paths.catalog_dir(SOURCE) / src.name)
        report.catalogs_moved += 1

    # 2. Raw files
    for src in sorted((paths.root / "raw").glob("pg*.txt")):
        _move(src, paths.raw_dir(SOURCE) / src.name)
        report.raw_files_moved += 1

    # 3. Registry
    if paths.registry.exists():
        data = read_json(paths.registry)
        if isinstance(data, dict) and data.get("schema_version") == SCHEMA_VERSION:
            report.registry_already_current = True
        else:
            registry = Registry(paths.registry)
            for old_key, old in data.items():
                registry.upsert(_convert_record(old_key, old, paths))
                report.records_converted += 1
            shutil.copy2(paths.registry, paths.root / "registry.v1.json.bak")
            registry.save()
    return report
