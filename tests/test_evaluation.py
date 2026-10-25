"""Evaluation sets: schema, quote location, `rag eval-check`."""

import json

import pytest

from growth_rag.chunking.run import chunk_books
from growth_rag.chunking.tokens import ApproxTokenCounter
from growth_rag.config import ChunkingSettings
from growth_rag.evaluation.check import check_questions, coverage, find_all
from growth_rag.evaluation.dataset import EvalFormatError, Question, dump_questions, load_questions
from growth_rag.registry import Registry
from growth_rag.sources.gutenberg.clean import clean_books
from test_cleaning_pipeline import PROSE
from test_gutenberg_clean import add_book

BODY = (
    "CHAPTER I.\n\nIs it not curious that the whale’s eye is so small?[1]\n" + PROSE
    + "\n[1] Compare Herschel’s great telescope.\n\n"
    "CHAPTER II.\n\nJackson’s Island is good enough for me, nobody ever comes there.\n" + PROSE
)
RAW = ("*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n\n" + BODY
       + "\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n").encode("utf-8")


def q(**fields) -> Question:
    base = {"id": "q", "task": "passage", "query": "?", "language": "en"}
    return Question.model_validate({**base, **fields})


def ev(evidence, book="gutenberg:1", **extra):
    return {"book": book, "evidence": evidence, **extra}


# --- locating quotes ---------------------------------------------------------

def test_quotes_ignore_case_whitespace_and_quote_marks():
    text = "He said: “It’s a\n  long Line.” Then left."
    [(start, end)] = find_all('it\'s a long line."', text)
    assert text[start:end] == "It’s a\n  long Line.”"


def test_quotes_match_across_line_breaks_without_spaces():
    text = "兩邊又有一幅對聯，道是：\n\n假作真時真亦假"
    [(start, end)] = find_all("道是：假作真時", text)
    assert text[start:end] == "道是：\n\n假作真時"


def test_every_occurrence_is_returned():
    assert len(find_all("ab", "ab xab ab")) == 3
    assert find_all("zz", "ab") == []


# --- schema --------------------------------------------------------------------

def test_passage_needs_evidence_and_book_takes_none():
    with pytest.raises(ValueError):
        q(relevant=[{"book": "gutenberg:1"}])
    with pytest.raises(ValueError):
        q(task="book", relevant=[ev("x")])
    assert not q().answerable  # no relevant entry: the corpus has no answer


def test_load_errors_name_the_line(tmp_path):
    path = tmp_path / "questions.jsonl"
    good = json.dumps({"id": "a", "task": "passage", "query": "?", "language": "en"})
    path.write_text(good + "\n\n" + good + "\n", encoding="utf-8")
    with pytest.raises(EvalFormatError, match="line 3.*already used on line 1"):
        load_questions(path)
    path.write_text(good + "\n{not json\n", encoding="utf-8")
    with pytest.raises(EvalFormatError, match="line 2"):
        load_questions(path)


def test_dump_leaves_defaults_out_and_round_trips(tmp_path):
    path = tmp_path / "questions.jsonl"
    questions = [q(relevant=[ev("x", **{"in": "annotation"}), ev("y", grade=1)])]
    dump_questions(path, questions)
    line = json.loads(path.read_text(encoding="utf-8"))
    assert "filters" not in line and "grade" not in line["relevant"][0]
    assert line["relevant"][0]["in"] == "annotation"
    assert load_questions(path) == questions


# --- checking against cleaned and chunked books --------------------------------

def cleaned_and_chunked(paths) -> Registry:
    """A registry with one book (BODY), cleaned and chunked."""
    reg = Registry(paths.registry)
    add_book(paths, reg, raw=RAW)
    clean_books(reg, paths)
    chunk_books(reg, paths, ApproxTokenCounter(), ChunkingSettings(tokenizer=ApproxTokenCounter.name))
    return reg


@pytest.fixture
def registry(paths):
    return cleaned_and_chunked(paths)


def test_text_and_footnote_evidence_resolve_to_chunks(paths, registry):
    questions = [
        q(id="text", relevant=[ev("Jackson's Island is good enough for me")]),
        q(id="note", relevant=[ev("Herschel's great telescope", **{"in": "annotation"})]),
        q(id="books", task="book", relevant=[{"book": "gutenberg:1"}]),
    ]
    report = check_questions(questions, registry, paths)
    assert report.problems == []
    [text] = report.targets["text"]
    assert text.span and len(text.chunks) == 1
    [note] = report.targets["note"]
    assert note.annotation and note.chunks  # the chunk holding the reference
    assert report.targets["books"][0].chunks == []


def test_dataset_bugs_are_errors(paths, registry):
    questions = [
        q(id="twice", relevant=[ev("long enough to count as the real text")]),  # PROSE is in both chapters
        q(id="typo", relevant=[ev("Jackson's Isle")]),
        q(id="no-book", relevant=[ev("x", book="gutenberg:999")]),
        q(id="filter", relevant=[ev("Jackson's Island")], filters={"title": "Hamlet"}),
        q(id="speaker", relevant=[ev("Jackson's Island")], filters={"speaker": ["OPHELIA"]}),
    ]
    report = check_questions(questions, registry, paths, manifest_ids={"gutenberg:1"})
    by_id = {p.question: p for p in report.problems}
    assert set(by_id) == {"twice", "typo", "no-book", "filter", "speaker"}
    assert all(p.level == "error" for p in report.problems if p.question != "no-book")
    assert "found 2 times" in by_id["twice"].message and "not found" in by_id["typo"].message
    assert any("not in the manifest" in p.message for p in report.problems if p.question == "no-book")


def test_declared_gaps_are_not_errors(paths, registry):
    gap = q(relevant=[ev("Jackson's Island")], filters={"speaker": ["OPHELIA"]}, known_gap="no speakers")
    report = check_questions([gap], registry, paths)
    assert [p.level for p in report.problems] == ["expected"]


def test_coverage_counts_cross_lingual_questions(paths, registry):
    questions = [q(id="en", relevant=[ev("Jackson's Island")]),
                 q(id="fr", language="fr", relevant=[ev("Jackson's Island")])]
    counts = coverage(questions, registry)
    assert counts["cross-lingual"] == {"no": 1, "yes": 1}


def test_cli_eval_check_and_show(paths, registry, tmp_path, capsys):
    from growth_rag.cli import main

    folder = tmp_path / "eval-test"
    folder.mkdir()
    dump_questions(folder / "questions.jsonl",
                   [q(id="island", relevant=[ev("Jackson's Island is good enough for me")])])
    main(["--data-dir", str(paths.root), "eval-check", "--set", str(folder)])
    assert "every evidence quote was found exactly once" in capsys.readouterr().err  # log on stderr
    main(["--data-dir", str(paths.root), "eval-check", "--set", str(folder), "--show", "island"])
    out = capsys.readouterr().out
    assert "gutenberg:1/s" in out and "Jackson’s Island" in out
