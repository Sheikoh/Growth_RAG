"""Gutenberg-specific stripping: licence header and footer around the book."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Line-anchored; covers "THE"/"THIS", "EBOOK"/"E-BOOK"/"ETEXT" and missing spaces.
START_RE = re.compile(
    r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG (?:E-?BOOK|ETEXT)[^\n]*$", re.I | re.M
)
END_RE = re.compile(
    r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG (?:E-?BOOK|ETEXT)[^\n]*$", re.I | re.M
)
# Older files sometimes end with this line instead of a proper END marker.
OLD_END_RE = re.compile(r"^End of (?:the )?Project Gutenberg'?s? .*$", re.I | re.M)


@dataclass
class Body:
    text: str
    offset: int  # position of the body in the normalised raw text
    has_start: bool
    has_end: bool


def extract_body(text: str) -> Body:
    """Text between the START and END markers. Missing markers are reported,
    never guessed: the whole text is kept on that side."""
    start = START_RE.search(text)
    begin = start.end() + 1 if start else 0
    end = END_RE.search(text, begin)
    if end is None:
        end = OLD_END_RE.search(text, begin)
        finish, has_end = (end.start(), False) if end else (len(text), False)
    else:
        finish, has_end = end.start(), True
    return Body(text[begin:finish], begin, start is not None, has_end)
