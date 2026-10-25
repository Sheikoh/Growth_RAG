from conftest import FakeResponse, FakeSession, book_bytes
from growth_rag.config import HttpSettings
from growth_rag.registry import Registry
from growth_rag.sources.gutenberg.catalog import read_catalog
from growth_rag.sources.gutenberg.download import download_manifest
from growth_rag.sources.gutenberg.selection import SelectionCriteria, build_manifest

MIRROR = "https://mirror.test/{id}/pg{id}.txt"
HTTP = HttpSettings(max_retries=1)


def run(paths, catalog_file, session, manifest, **kwargs):
    registry = Registry.load(paths.registry)
    summary = download_manifest(manifest, read_catalog(catalog_file), catalog_file.name, registry,
                                paths, session, http=HTTP, mirror_url=MIRROR, delay_s=0,
                                sleep=lambda s: None, **kwargs)
    return summary, Registry.load(paths.registry)


def setup(paths, catalog_file, name="pilot"):
    manifest = build_manifest(name, read_catalog(catalog_file), catalog_file,
                              SelectionCriteria(languages=["en", "fr"]))
    routes = {MIRROR.format(id=i): FakeResponse(200, book_bytes(i)) for i in (1342, 84, 17489)}
    routes[MIRROR.format(id=1497)] = FakeResponse(200, b"<html>error</html>", "text/html")
    return manifest, FakeSession(routes)


def test_download_writes_files_and_records(paths, catalog_file):
    manifest, session = setup(paths, catalog_file)  # ids: 84, 1342, 1497, 17489, 99999
    summary, registry = run(paths, catalog_file, session, manifest)

    assert summary.by_status == {"ok": 3, "error": 1, "not_found": 1}
    rec = registry.get("gutenberg:1342")
    assert rec.acquisition.path == "raw/gutenberg/pg1342.txt"
    assert paths.resolve(rec.acquisition.path).read_bytes() == book_bytes(1342)
    assert rec.acquisition.has_start_marker and rec.acquisition.utf8_valid
    assert rec.manifests == ["pilot"]
    assert "content type" in registry.get("gutenberg:1497").acquisition.error  # HTML page rejected


def test_rerun_skips_present_and_known_missing(paths, catalog_file):
    manifest, session = setup(paths, catalog_file)
    run(paths, catalog_file, session, manifest)
    calls_after_first = len(session.calls)

    summary, _ = run(paths, catalog_file, session, manifest)
    # Only the error (1497) is retried; ok books and the 404 are skipped.
    assert summary.attempted == 1
    assert len(session.calls) == calls_after_first + 1

    summary, _ = run(paths, catalog_file, session, manifest, retry_missing=True)
    assert summary.attempted == 2  # 1497 again + 99999


def test_modified_file_is_downloaded_again(paths, catalog_file):
    manifest, session = setup(paths, catalog_file)
    _, registry = run(paths, catalog_file, session, manifest)
    paths.resolve(registry.get("gutenberg:84").acquisition.path).write_bytes(b"truncated")
    summary, _ = run(paths, catalog_file, session, manifest)
    assert summary.by_status.get("ok") == 1  # pg84 re-fetched


def test_second_manifest_tags_existing_records(paths, catalog_file):
    manifest, session = setup(paths, catalog_file)
    run(paths, catalog_file, session, manifest)
    other = manifest.model_copy(update={"name": "other", "ids": [1342]})
    summary, registry = run(paths, catalog_file, session, other)
    assert summary.attempted == 0
    assert registry.get("gutenberg:1342").manifests == ["pilot", "other"]
