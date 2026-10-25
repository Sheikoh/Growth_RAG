import json

from growth_rag.registry import Registry
from growth_rag.sources.gutenberg.migrate import migrate_v1

LEGACY_RECORD = {
    "id": 1342, "type": "Text", "issued": "1998-06-01", "title": "Pride and Prejudice",
    "languages": ["en"],
    "authors": [{"name": "Austen, Jane", "birth_year": 1775, "death_year": 1817,
                 "role": None, "raw": "Austen, Jane, 1775-1817"}],
    "subjects": ["Courtship -- Fiction"], "locc": ["PR"], "bookshelves": ["Category: Novels"],
    "manifests": ["pilot"], "catalog_snapshot": "pg_catalog_2026-10-25.csv",
    "acquisition": {
        "source_url": "https://gutenberg.pglaf.org/cache/epub/1342/pg1342.txt",
        "attempted_at": "2026-10-25T12:00:00+00:00", "http_status": 200,
        "content_type": "text/plain; charset=utf-8", "status": "ok",
        "path": "raw/pg1342.txt", "sha256": "abc", "size_bytes": 5,
        "downloaded_at": "2026-10-25T12:00:01+00:00", "utf8_valid": True,
        "has_start_marker": True, "has_end_marker": True,
    },
}


def make_legacy_layout(paths):
    (paths.root / "catalog").mkdir(parents=True)
    (paths.root / "catalog" / "pg_catalog_2026-10-25.csv").write_text("csv", encoding="utf-8")
    (paths.root / "raw").mkdir()
    (paths.root / "raw" / "pg1342.txt").write_bytes(b"hello")
    paths.registry.write_text(json.dumps({"1342": LEGACY_RECORD}), encoding="utf-8")


def test_migrate_moves_files_and_converts_registry(paths):
    make_legacy_layout(paths)
    report = migrate_v1(paths)
    assert (report.catalogs_moved, report.raw_files_moved, report.records_converted) == (1, 1, 1)

    assert (paths.catalog_dir("gutenberg") / "pg_catalog_2026-10-25.csv").exists()
    assert (paths.raw_dir("gutenberg") / "pg1342.txt").read_bytes() == b"hello"
    assert (paths.root / "registry.v1.json.bak").exists()

    record = Registry.load(paths.registry).get("gutenberg:1342")
    assert record.acquisition.path == "raw/gutenberg/pg1342.txt"
    assert record.media_type == "Text" and record.manifests == ["pilot"]


def test_migrate_is_idempotent(paths):
    make_legacy_layout(paths)
    migrate_v1(paths)
    report = migrate_v1(paths)
    assert report.registry_already_current
    assert (report.catalogs_moved, report.raw_files_moved, report.records_converted) == (0, 0, 0)
