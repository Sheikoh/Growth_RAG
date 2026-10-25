"""Schema of a cleaned book: the contract between cleaning and chunking.

Stored as two files per book: `<name>.txt` (the cleaned text, readable as-is)
and `<name>.json` (this model). All offsets are **character positions in the
Python string** of the cleaned text (not bytes): `text[start:end]` gives the
span. Every later step (chunking, Qdrant payload, graph) uses the same unit.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

SectionRole = Literal["front_matter", "main_text", "apparatus"]
HeadingSource = Literal["pattern", "toc", "pattern+toc", "front", "fallback"]
ParagraphKind = Literal["heading", "text", "toc", "speech", "stage_direction"]
AnnotationType = Literal[
    "footnote", "illustration", "transcriber_note", "credit", "scene_break"
]
AnnotationStatus = Literal["matched", "unmatched_reference", "unmatched_body", "standalone"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Section(_Model):
    index: int  # sequential within the book (s0, s1, ...): the stable identity
    level: int  # 1 = top level; front matter and fallback sections are level 1
    label: str  # heading as printed, e.g. "Chapitre I" or "CHAPTER 1. Loomings."
    title: str | None = None  # title printed on the line after a bare heading
    path: list[str]  # labels from the top level down, e.g. ["Livre premier--Un juste", "Chapitre I"]
    role: SectionRole
    heading_source: HeadingSource
    start: int
    end: int


class Paragraph(_Model):
    index: int
    section: int
    kind: ParagraphKind
    verse: bool = False  # line breaks are meaningful and were kept
    speaker: str | None = None  # label as printed, e.g. "HAMLET"
    speaker_key: str | None = None  # book-scoped, e.g. "gutenberg:1524/speaker/hamlet"
    start: int
    end: int


class Annotation(_Model):
    key: str  # book-scoped and sequential, e.g. "gutenberg:45334/a12"
    type: AnnotationType
    label: str | None = None  # printed marker: "[1]", "[Illustration: ...]"
    text: str | None = None  # footnote body, caption, note text
    anchor: int | None = None  # offset in the cleaned text where it occurred; None = book-level
    section: int | None = None
    paragraph: int | None = None  # paragraph containing the anchor
    raw_start: int | None = None  # offsets in the normalised raw text (LF line endings)
    raw_end: int | None = None
    status: AnnotationStatus = "standalone"


class CastMember(_Model):
    key: str  # book-scoped: the same label in two books gives two different keys
    label: str
    speeches: int


class Metrics(_Model):
    characters: int
    sections: int
    headings_by_source: dict[str, int]
    toc_entries: int
    toc_entries_matched: int
    front_matter_share: float
    section_chars_median: int
    section_chars_max: int
    paragraphs: int
    verse_share: float
    is_play: bool
    speeches: int
    footnote_references: int
    footnote_bodies: int
    footnotes_matched: int


class CleanedBook(_Model):
    key: str  # registry key, e.g. "gutenberg:1342"
    language: str
    cleaning_version: str
    cleaned_at: datetime
    raw_sha256: str | None = None
    status: Literal["ok", "flagged"]
    flags: list[str] = []
    sections: list[Section]
    paragraphs: list[Paragraph]
    annotations: list[Annotation]
    cast: list[CastMember] = []
    metrics: Metrics
