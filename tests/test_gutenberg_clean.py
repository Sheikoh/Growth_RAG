"""Cleaning step wired to the registry: outputs, registry update, re-clean rules."""

from growth_rag.cleaning import CLEANING_VERSION
from growth_rag.cleaning.store import load_cleaned
from growth_rag.registry import Acquisition, BookRecord, Registry
from growth_rag.sources.gutenberg.clean import clean_books, needs_cleaning
from growth_rag.utils.io import sha256_bytes, utc_now

RAW = (
    "Header text\r\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\r\n\r\n"
    "CHAPTER I.\r\n\r\n" + "Some prose that is long enough to be a chapter body. " * 8 + "\r\n\r\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\r\nLicence text\r\n"
).encode("utf-8")


def add_book(paths, registry, book_id=1, raw=RAW, shelves=()):
    path = paths.raw_dir("gutenberg") / f"pg{book_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    registry.upsert(BookRecord(
        key=f"gutenberg:{book_id}", source="gutenberg", source_id=str(book_id), title="Test",
        languages=["en"], bookshelves=list(shelves),
        acquisition=Acquisition(status="ok", source_url="x", attempted_at=utc_now(),
                                path=paths.relative(path), sha256=sha256_bytes(raw), size_bytes=len(raw)),
    ))


def test_clean_writes_outputs_and_updates_registry(paths):
    registry = Registry(paths.registry)
    add_book(paths, registry)
    summary = clean_books(registry, paths)
    assert summary.by_status == {"ok": 1}

    info = Registry.load(paths.registry).get("gutenberg:1").cleaning
    assert info.version == CLEANING_VERSION and info.text_path == "clean/gutenberg/pg1.txt"
    text, book = load_cleaned(paths.resolve(info.text_path), paths.resolve(info.structure_path))
    assert text.startswith("CHAPTER I.")
    assert "Licence" not in text and "Header" not in text and "\r" not in text
    assert book.key == "gutenberg:1"


def test_up_to_date_books_are_skipped_and_changes_trigger_recleaning(paths):
    registry = Registry(paths.registry)
    add_book(paths, registry)
    clean_books(registry, paths)
    record = registry.get("gutenberg:1")
    assert not needs_cleaning(record, paths)
    assert clean_books(registry, paths).cleaned == 0

    record.cleaning.version = "0.0.0"  # rules changed
    assert needs_cleaning(record, paths)
    clean_books(registry, paths)

    record.acquisition.sha256 = "different"  # raw file changed
    assert needs_cleaning(record, paths)
    assert needs_cleaning(registry.get("gutenberg:1"), paths, force=True)


def test_missing_markers_are_flagged_not_fatal(paths):
    registry = Registry(paths.registry)
    add_book(paths, registry, raw=b"CHAPTER I.\n\n" + b"Plain text without markers. " * 20)
    clean_books(registry, paths)
    info = registry.get("gutenberg:1").cleaning
    assert info.status == "flagged"
    assert {"no_start_marker", "no_end_marker"} <= set(info.flags)


def test_errors_are_recorded_and_do_not_stop_the_run(paths):
    registry = Registry(paths.registry)
    add_book(paths, registry, book_id=1)
    add_book(paths, registry, book_id=2)
    paths.resolve(registry.get("gutenberg:1").acquisition.path).unlink()  # file vanished
    summary = clean_books(registry, paths)
    assert summary.by_status == {"error": 1, "ok": 1}
    assert "FileNotFoundError" in registry.get("gutenberg:1").cleaning.error


def test_play_hint_comes_from_bookshelves(paths):
    from growth_rag.sources.gutenberg.clean import _play_hint

    registry = Registry(paths.registry)
    add_book(paths, registry, shelves=["Category: Plays/Films/Dramas"])
    assert _play_hint(registry.get("gutenberg:1"))
