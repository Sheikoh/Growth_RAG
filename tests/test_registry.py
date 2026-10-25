import json

import pytest
from pydantic import ValidationError

from growth_rag.registry import BookRecord, Registry, RegistryFormatError


def record(book_id="1", **kwargs) -> BookRecord:
    return BookRecord(key=f"gutenberg:{book_id}", source="gutenberg", source_id=book_id,
                      title=f"Book {book_id}", **kwargs)


def test_round_trip(paths):
    registry = Registry(paths.registry)
    registry.upsert(record("2", languages=["fr"]))
    registry.upsert(record("1"))
    registry.save()
    loaded = Registry.load(paths.registry)
    assert len(loaded) == 2
    assert loaded.get("gutenberg:2").primary_language == "fr"
    assert loaded.get("gutenberg:1").primary_language == "und"


def test_missing_file_gives_empty_registry(paths):
    assert len(Registry.load(paths.registry)) == 0


def test_legacy_format_is_rejected_with_a_hint(paths):
    paths.registry.parent.mkdir(parents=True)
    paths.registry.write_text(json.dumps({"1342": {"id": 1342}}), encoding="utf-8")
    with pytest.raises(RegistryFormatError, match="gutenberg migrate"):
        Registry.load(paths.registry)


def test_key_must_match_source_and_id():
    with pytest.raises(ValidationError):
        BookRecord(key="gutenberg:1", source="gutenberg", source_id="2", title="x")


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        record(titel="typo")


def test_record_from_another_code_version_gives_a_clear_error(paths):
    registry = Registry(paths.registry)
    registry.upsert(record("1"))
    registry.save()
    data = json.loads(paths.registry.read_text(encoding="utf-8"))
    data["books"]["gutenberg:1"]["obsolete_field"] = 1
    paths.registry.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RegistryFormatError, match="gutenberg:1.*obsolete_field"):
        Registry.load(paths.registry)
