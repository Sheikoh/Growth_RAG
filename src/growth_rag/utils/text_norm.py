"""Normalising names for matching: case, accents, punctuation."""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """'Les Misérables, Tome I' -> 'les miserables tome i'."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    no_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(re.findall(r"\w+", no_accents))


def contains_words(normalized_text: str, normalized_query: str) -> bool:
    """Every word of the query is a whole word of the text, in any order."""
    words = set(normalized_text.split())
    return all(w in words for w in normalized_query.split())
