import pytest

from growth_rag.sources.gutenberg.catalog import (
    latest_snapshot,
    parse_author,
    read_catalog,
    split_multi,
    to_record,
)


@pytest.mark.parametrize(
    "raw, name, birth, death, active, role",
    [
        ("Austen, Jane, 1775-1817", "Austen, Jane", 1775, 1817, None, None),
        ("Plato, 428? BCE-348? BCE", "Plato", -428, -348, None, None),
        ("Euripides, 481? BCE-407 BCE", "Euripides", -481, -407, None, None),
        ("Nobody, 1900-", "Nobody", 1900, None, None, None),
        ("Smith, John [Editor]", "Smith, John", None, None, None, "Editor"),
        ("Richards, Amy, active 1896-1918 [Illustrator]", "Richards, Amy", None, None, "1896-1918", "Illustrator"),
        ("Sunzi, active 6th century B.C.", "Sunzi", None, None, "6th century B.C.", None),
        ("Kennedy, John F. (John Fitzgerald), 1917-1963", "Kennedy, John F. (John Fitzgerald)", 1917, 1963, None, None),
        ("Various", "Various", None, None, None, None),
    ],
)
def test_parse_author(raw, name, birth, death, active, role):
    a = parse_author(raw)
    assert (a.name, a.birth_year, a.death_year, a.active, a.role) == (name, birth, death, active, role)
    assert a.raw == raw  # the original string is always kept


def test_split_multi():
    assert split_multi("en; fr ;") == ("en", "fr")
    assert split_multi("") == ()
    assert split_multi(None) == ()


def test_read_catalog_handles_multiline_titles_and_lists(catalog_file):
    catalog = read_catalog(catalog_file)
    assert len(catalog) == 7
    assert catalog[17489].title == "Les misérables Tome I"
    assert catalog[1497].languages == ("en", "fr")
    assert [a.role for a in catalog[2229].authors] == [None, "Editor"]
    assert catalog[5000].media_type == "Sound"


def test_to_record_builds_source_qualified_key(catalog_file):
    record = to_record(read_catalog(catalog_file)[1342], catalog_file.name)
    assert record.key == "gutenberg:1342"
    assert record.source_id == "1342"
    assert record.catalog_snapshot == catalog_file.name
    assert record.acquisition is None


def test_latest_snapshot(paths, catalog_file):
    older = catalog_file.with_name("pg_catalog_2026-01-01.csv")
    older.write_text("x", encoding="utf-8")
    assert latest_snapshot(catalog_file.parent) == catalog_file
    with pytest.raises(FileNotFoundError):
        latest_snapshot(paths.root / "nowhere")
