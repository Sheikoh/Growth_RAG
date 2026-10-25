"""Locate every evidence quote in the cleaned books: `rag eval-check`.

For each relevant entry of a passage question:
- the quote must occur exactly once in the book (normalised: case,
  whitespace ignored, curly vs straight quote marks, NFKC). Zero is an error (typo,
  other edition, cleaning changed); several is an error too (quote more);
- the chunks overlapping the quote are the relevant chunks. None means the
  passage is not indexed (front matter, apparatus): a warning, unless the
  question declares it as a known gap;
- for "in": "annotation", the quote must occur in exactly one annotation;
  the relevant chunks are those carrying that annotation's key.

Filters are checked too: a filter that excludes the relevant book, or a
speaker filter that matches none of the relevant chunks, is a dataset bug.

The result (`Targets`) is what a retrieval evaluation scores against.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import cache

from growth_rag.chunking.models import Chunk
from growth_rag.chunking.store import iter_chunks
from growth_rag.cleaning.models import CleanedBook
from growth_rag.cleaning.store import load_cleaned
from growth_rag.config import DataPaths
from growth_rag.evaluation.dataset import Question, Relevant
from growth_rag.index.resolve import resolve_books
from growth_rag.registry import Registry

_MARKS = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'", "“": '"', "”": '"', "«": '"', "»": '"', "„": '"'})


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalised text plus, for each of its characters, the offset of the
    original character it comes from (to map a match back to the book).
    Whitespace is dropped, not collapsed: a quote matches whether or not it
    crosses a line break, also in scripts without spaces (Chinese)."""
    out: list[str] = []
    origin: list[int] = []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch).casefold().translate(_MARKS):
            if not c.isspace():
                out.append(c)
                origin.append(i)
    return "".join(out), origin


def normalize_quote(quote: str) -> str:
    return normalize_with_map(quote)[0]


def find_all(quote: str, text: str) -> list[tuple[int, int]]:
    """(start, end) offsets in `text` of every occurrence of `quote`."""
    norm, origin = _normalized(text)
    needle = normalize_quote(quote)
    spans, pos = [], norm.find(needle)
    while needle and pos != -1:
        spans.append((origin[pos], origin[pos + len(needle) - 1] + 1))
        pos = norm.find(needle, pos + 1)
    return spans


@cache
def _normalized(text: str) -> tuple[str, list[int]]:
    return normalize_with_map(text)


@dataclass
class Target:
    """One relevant entry, resolved."""

    book: str
    grade: int
    span: tuple[int, int] | None = None  # offsets in the cleaned text (text evidence)
    annotation: str | None = None  # annotation key (footnote evidence)
    chunks: list[str] = field(default_factory=list)  # relevant chunk keys


@dataclass
class Problem:
    level: str  # "error" (dataset bug), "warning", "expected" (declared known gap)
    question: str
    message: str


@dataclass
class CheckReport:
    targets: dict[str, list[Target]] = field(default_factory=dict)  # question id -> targets
    problems: list[Problem] = field(default_factory=list)

    @property
    def errors(self) -> list[Problem]:
        return [p for p in self.problems if p.level == "error"]


class BookStore:
    """Cleaned text, structure and chunks of registry books, loaded once each."""

    def __init__(self, registry: Registry, paths: DataPaths):
        self.registry, self.paths = registry, paths
        self._cache: dict[str, tuple[str, CleanedBook, list[Chunk]] | None] = {}

    def get(self, key: str) -> tuple[str, CleanedBook, list[Chunk]] | None:
        if key not in self._cache:
            self._cache[key] = self._load(key)
        return self._cache[key]

    def _load(self, key: str):
        record = self.registry.get(key)
        if record is None or not record.cleaning or not record.cleaning.text_path:
            return None
        text, book = load_cleaned(self.paths.resolve(record.cleaning.text_path),
                                  self.paths.resolve(record.cleaning.structure_path))
        chunk_path = record.chunking and record.chunking.path and self.paths.resolve(record.chunking.path)
        chunks = list(iter_chunks(chunk_path)) if chunk_path and chunk_path.exists() else []
        return text, book, chunks

    def why_missing(self, key: str) -> str:
        record = self.registry.get(key)
        if record is None:
            return "not in the registry (add the eval manifest: gutenberg add --manifest ...)"
        return "not cleaned yet (gutenberg clean)"


def check_questions(questions: list[Question], registry: Registry, paths: DataPaths,
                    manifest_ids: set[str] | None = None) -> CheckReport:
    store = BookStore(registry, paths)
    report = CheckReport()
    for q in questions:
        report.targets[q.id] = _check_question(q, store, registry, manifest_ids, report.problems)
    return report


def _check_question(q: Question, store: BookStore, registry: Registry,
                    manifest_ids: set[str] | None, problems: list[Problem]) -> list[Target]:
    def problem(message: str, level: str = "error") -> None:
        problems.append(Problem(level, q.id, message))

    targets: list[Target] = []
    for key in [r.book for r in q.relevant] + q.seed_books:
        if manifest_ids is not None and key not in manifest_ids:
            problem(f"{key} is not in the manifest")
    for r in q.relevant:
        if q.task == "book":
            if registry.get(r.book) is None:
                problem(f"{r.book}: {store.why_missing(r.book)}")
            targets.append(Target(r.book, r.grade))
            continue
        loaded = store.get(r.book)
        if loaded is None:
            problem(f"{r.book}: {store.why_missing(r.book)}")
            continue
        target = _locate(r, *loaded, problem)
        if target is None:
            continue
        if not target.chunks:
            problem(f"{r.book}: evidence found but in no chunk (not indexed)",
                    "expected" if q.known_gap else "warning")
        targets.append(target)
    _check_filters(q, targets, store, registry, problem)
    return targets


def _locate(r: Relevant, text: str, book: CleanedBook, chunks: list[Chunk], problem) -> Target | None:
    short = r.evidence if len(r.evidence) <= 50 else r.evidence[:47] + "..."
    if r.in_ == "annotation":
        hits = [a for a in book.annotations if a.text and find_all(r.evidence, a.text)]
        if len(hits) != 1:
            problem(f"{r.book}: annotation evidence {short!r} found in {len(hits)} annotations (need 1)")
            return None
        key = hits[0].key
        return Target(r.book, r.grade, annotation=key, chunks=[c.key for c in chunks if key in c.annotations])
    spans = find_all(r.evidence, text)
    if len(spans) != 1:
        what = "not found" if not spans else f"found {len(spans)} times, quote more"
        problem(f"{r.book}: evidence {short!r} {what}")
        return None
    start, end = spans[0]
    return Target(r.book, r.grade, span=(start, end),
                  chunks=[c.key for c in chunks if c.start < end and start < c.end])


def _check_filters(q: Question, targets: list[Target], store: BookStore, registry: Registry, problem) -> None:
    f = q.filters
    if f is None or not targets:
        return
    allowed: set[str] | None = None
    if f.title or f.author:
        allowed = set(resolve_books(registry, title=f.title, author=f.author, languages=f.language).keys)
    if f.book:
        allowed = set(f.book) if allowed is None else allowed & set(f.book)
    if allowed is not None:
        for t in targets:
            if t.book not in allowed:
                problem(f"filters {f.model_dump(exclude_none=True)} exclude the relevant book {t.book}",
                        "expected" if q.known_gap else "error")
    if f.speaker:
        wanted = {s.upper() for s in f.speaker}
        by_key = {c.key: c for t in targets if store.get(t.book) for c in store.get(t.book)[2]}
        if not any(wanted & set(by_key[k].speaker_labels) for t in targets for k in t.chunks if k in by_key):
            problem(f"speaker filter {sorted(wanted)} matches none of the relevant chunks",
                    "expected" if q.known_gap else "error")


def coverage(questions: list[Question], registry: Registry) -> dict[str, Counter]:
    """What the set covers: tasks, tags, query languages, book languages,
    cross-lingual questions (query language not among the books'), filters."""
    out = {name: Counter() for name in ("task", "tags", "query language", "book language",
                                         "cross-lingual", "filtered", "answerable")}
    for q in questions:
        out["task"][q.task] += 1
        out["tags"].update(q.tags)
        out["query language"][q.language] += 1
        langs = {lang for b in q.books if (rec := registry.get(b)) for lang in rec.languages}
        out["book language"].update(langs)
        if langs:
            out["cross-lingual"]["yes" if q.language not in langs else "no"] += 1
        out["filtered"]["yes" if q.filters else "no"] += 1
        out["answerable"]["yes" if q.answerable else "no (corpus has no answer)"] += 1
    return out
