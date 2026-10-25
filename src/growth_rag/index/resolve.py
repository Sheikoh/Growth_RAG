"""Find books by title / author in the registry (the names users actually say).

Matching rules:
- case- and accent-insensitive ("miserables" finds "Les Misérables");
- whole words only, in any order ("Faust" does not match "Faustus");
- every word of the query must appear;
- *all* matching books are returned: volumes (Tome I, II, ...) and editions are
  legitimately several books, and Qdrant filters on a list of books anyway.

The same resolver will serve the LLM tool later, where users name books by
title, never by Gutenberg number.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from dataclasses import dataclass, field

from growth_rag.registry import BookRecord, Registry
from growth_rag.utils.text_norm import contains_words as _contains_words
from growth_rag.utils.text_norm import normalize


@dataclass
class Resolution:
    books: list[BookRecord] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)  # closest titles when nothing matched

    @property
    def keys(self) -> list[str]:
        return [b.key for b in self.books]


def resolve_books(
    records: Iterable[BookRecord] | Registry,
    title: str | None = None,
    author: str | None = None,
    languages: Iterable[str] | None = None,
) -> Resolution:
    records = list(records)
    q_title = normalize(title) if title else ""
    q_author = normalize(author) if author else ""
    langs = set(languages or [])

    matches = []
    for r in records:
        if q_title and not _contains_words(normalize(r.title), q_title):
            continue
        if q_author and not any(
            _contains_words(normalize(a.name), q_author) for a in r.authors if a.role is None
        ):
            continue
        if langs and not langs & set(r.languages):
            continue
        matches.append(r)

    resolution = Resolution(books=sorted(matches, key=lambda r: (r.title, r.key)))
    if not matches and q_title:
        titles = {normalize(r.title): r.title for r in records}
        close = difflib.get_close_matches(q_title, list(titles), n=5, cutoff=0.4)
        # also titles where the query is a prefix of a word ("miser" -> "miserables")
        partial = [t for t in titles if any(w.startswith(q_title.split()[0]) for w in t.split())]
        resolution.suggestions = list(dict.fromkeys(titles[t] for t in close + partial))[:5]
    return resolution
