"""Acceptance tests on the 11 sample books (manifest `cleaning_samples`).

The raw files are not in git (10 MB). The tests run when they are found in
GROWTH_RAG_SAMPLES_DIR, or in 03_RAG/data/raw/gutenberg, and are skipped
otherwise. Expected counts were checked against the books' actual structure;
a change in these numbers means a change in behaviour, intended or not.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest

from growth_rag.cleaning.pipeline import BookInput, clean_book
from growth_rag.cleaning.text import decode, normalize
from growth_rag.sources.gutenberg.boilerplate import extract_body

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = Path(os.environ.get("GROWTH_RAG_SAMPLES_DIR", ROOT / "03_RAG" / "data" / "raw" / "gutenberg"))

# book id: (language, play, expected sections by label prefix, extra checks)
EXPECTED = {
    76: ("en", False, {"CHAPTER": 43}),  # 42 + "Chapter the Last"
    1524: ("en", True, {"ACT": 5, "SCENE": 20}),
    2000: ("es", False, {"Capítulo": 126, "PRÓLOGO": 1}),
    2229: ("de", True, {"": 28 + 1}),  # 28 scenes from the TOC + front matter
    2701: ("en", False, {"CHAPTER": 135, "Epilogue": 1, "ETYMOLOGY": 1, "EXTRACTS": 1}),
    11940: ("fi", False, {"": 14 + 1}),  # 14 "… LUKU" + front matter
    17489: ("fr", False, {"Livre": 8, "Chapitre": 70}),
    24264: ("zh", False, {"第": 120}),
    29062: ("el", False, {"ΚΕΦΑΛΑΙΟΝ": 10}),
    45334: ("it", False, {"CAPITOLO": 38, "INTRODUZIONE": 1}),
}


def _clean(book_id: int, language: str, play: bool):
    path = SAMPLES / f"pg{book_id}.txt"
    if not path.exists():
        pytest.skip(f"sample {path.name} not available")
    body = extract_body(normalize(decode(path.read_bytes())[0]))
    return clean_book(BookInput(key=f"gutenberg:{book_id}", language=language,
                                body=body.text, play_hint=play))


@pytest.mark.parametrize("book_id", sorted(EXPECTED))
def test_section_counts(book_id):
    language, play, expected = EXPECTED[book_id]
    _, book = _clean(book_id, language, play)
    assert book.status == "ok", book.flags
    for prefix, count in expected.items():
        found = sum(1 for s in book.sections if s.label.startswith(prefix))
        assert found == count, f"pg{book_id}: {prefix!r} {found} != {count}"


def test_hamlet_speakers():
    _, book = _clean(1524, "en", True)
    cast = {c.label: c.speeches for c in book.cast}
    assert cast["HAMLET"] == 358 and cast["OPHELIA"] == 58
    assert "PROLOGUE" in cast  # a character, not a section


def test_faust_is_verse_and_has_its_cast():
    _, book = _clean(2229, "de", True)
    assert book.metrics.verse_share > 0.6
    assert {"FAUST", "MEPHISTOPHELES", "MARGARETE", "GRETCHEN"} <= {c.label for c in book.cast}


def test_italian_footnotes_and_illustrations():
    _, book = _clean(45334, "it", False)
    m = book.metrics
    assert (m.footnote_references, m.footnote_bodies, m.footnotes_matched) == (45, 44, 44)
    assert Counter(a.type for a in book.annotations)["illustration"] == 46


def test_fleurs_du_mal_poems_keep_their_lines():
    text, book = _clean(6099, "fr", False)
    albatros = next(s for s in book.sections if s.label == "L'ALBATROS")
    assert albatros.path == ["SPLEEN ET IDÉAL", "L'ALBATROS"]
    assert "d'équipage\nPrennent" in text[albatros.start:albatros.end]
    assert book.metrics.verse_share > 0.9


# --- Chunking on the samples ---------------------------------------------------

def _chunks(book_id: int, language: str, play: bool):
    from growth_rag.chunking.chunker import BookMeta, chunk_book
    from growth_rag.chunking.tokens import ApproxTokenCounter

    text, book = _clean(book_id, language, play)
    return text, book, chunk_book(text, book, BookMeta(book.key, "T", "A"), ApproxTokenCounter())


@pytest.mark.parametrize("book_id", sorted(EXPECTED))
def test_chunks_are_exact_bounded_and_link_every_annotation(book_id):
    language, play, _ = EXPECTED[book_id]
    text, book, chunks = _chunks(book_id, language, play)
    assert all(c.text == text[c.start:c.end] for c in chunks)
    assert max(c.tokens for c in chunks) <= 512
    linked = {k for c in chunks for k in c.annotations}
    indexed = {s.index for s in book.sections if s.role == "main_text"}
    expected = {a.key for a in book.annotations if a.anchor is not None and a.section in indexed}
    assert expected <= linked


def test_hamlet_speeches_are_split_only_when_longer_than_a_chunk():
    from growth_rag.chunking.tokens import ApproxTokenCounter

    text, book, chunks = _chunks(1524, "en", True)
    spans = [(c.start, c.end) for c in chunks]
    split = [p for p in book.paragraphs if p.kind == "speech"
             and not any(a <= p.start and p.end <= b for a, b in spans)]
    # Only "O, what a rogue and peasant slave am I" (611 tokens > 512).
    assert all(n > 512 for n in ApproxTokenCounter().count([text[p.start:p.end] for p in split]))
    assert len(split) == 1


def test_fleurs_du_mal_poems_are_mostly_whole():
    _, _, chunks = _chunks(6099, "fr", False)
    poems = [c for c in chunks if c.kind == "poem"]
    whole = [c for c in poems if c.part is None]
    assert len(poems) > 130 and len(whole) / len(poems) > 0.75
