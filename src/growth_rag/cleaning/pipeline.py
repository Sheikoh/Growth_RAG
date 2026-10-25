"""Generic cleaning pipeline: body text -> (cleaned text, CleanedBook).

Steps, in order:
1. split into blocks; recognise annotation blocks (footnote bodies,
   illustrations, separators, credits, transcriber's notes);
2. detect headings (patterns + TOC dictionary) and build sections;
3. detect plays and speakers;
4. walk the blocks once, building the cleaned text and recording every offset
   (sections, paragraphs, annotation anchors) *as the text is built*;
5. match footnote references to bodies, compute metrics and flags.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from growth_rag.cleaning import CLEANING_VERSION
from growth_rag.cleaning import annotations as ann
from growth_rag.cleaning import headings as hd
from growth_rag.cleaning import speakers as sp
from growth_rag.cleaning.models import (
    Annotation,
    CastMember,
    CleanedBook,
    Metrics,
    Paragraph,
    Section,
)
from growth_rag.cleaning.text import Block, is_separator, is_verse, join_lines, split_blocks, strip_markup, wrap_width
from growth_rag.utils.io import utc_now

SEPARATOR = "\n\n"  # between paragraphs in the cleaned text
POETRY_SHARE = 0.15  # share of verse blocks above which poem titles are detected
# Flags that make a book "flagged" (kept and usable, but to review).
SEVERE_FLAGS = {
    "no_headings", "toc_heading_mismatch", "large_front_matter", "no_start_marker",
    "no_end_marker", "decode_fallback", "single_dominant_section",
}


@dataclass
class BookInput:
    key: str  # registry key, used to build annotation and speaker keys
    language: str
    body: str  # text after source-specific boilerplate removal
    body_offset: int = 0  # position of body in the normalised raw text
    play_hint: bool = False  # e.g. Gutenberg bookshelf "Plays/Films/Dramas"
    title: str | None = None  # catalog title; "A; B" names the works of a collection
    raw_sha256: str | None = None
    source_flags: tuple[str, ...] = ()  # flags raised by the source adapter


class _Builder:
    """Accumulates the cleaned text and records offsets as it grows."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.pos = 0
        self.paragraphs: list[Paragraph] = []

    def add(self, text: str, section: int, kind: str, **extra) -> Paragraph:
        if self.parts:
            self.pos += len(SEPARATOR)
        para = Paragraph(index=len(self.paragraphs), section=section, kind=kind,
                         start=self.pos, end=self.pos + len(text), **extra)
        self.parts.append(text)
        self.pos += len(text)
        self.paragraphs.append(para)
        return para

    @property
    def text(self) -> str:
        return SEPARATOR.join(self.parts)


def clean_book(book: BookInput) -> tuple[str, CleanedBook]:
    blocks = split_blocks(book.body, standalone=hd.is_standalone_line)
    width = wrap_width(blocks)

    # 1. Annotation blocks. Credits/transcriber's notes only count before the
    #    first heading, so a first heading pass is needed to know where that is.
    first_pass, _ = hd.detect_headings(blocks, excluded=set())
    front_end = first_pass[0].block if first_pass else 0
    kinds: dict[int, str] = {}
    for i, block in enumerate(blocks):
        kind = ann.block_kind(block, front=i < front_end)
        if kind:
            kinds[i] = kind
        elif is_separator(block):
            kinds[i] = "separator"

    # 2. Headings, excluding annotation blocks and, in plays, speaker blocks
    #    ("PROLOGUE." is a character in Hamlet, not a section).
    play = sp.is_play(blocks, book.play_hint)
    speaker_blocks = {i for i, b in enumerate(blocks) if play and sp.speaker_label(b)}
    multi_line = [b for b in blocks if len(b.lines) >= 2]
    poetry = not play and sum(is_verse(b, width) for b in multi_line) >= POETRY_SHARE * max(len(multi_line), 1)
    heads, tocs = hd.detect_headings(blocks, excluded=set(kinds) | speaker_blocks,
                                     not_text=set(kinds), poetry=poetry,
                                     extra_entries=_works_in_title(book.title))
    levels = hd.assign_levels(heads)
    head_at = {h.block: h for h in heads}
    title_blocks = {h.title_block for h in heads if h.title_block is not None}
    title_blocks |= {b for h in heads for b in h.extra_blocks}
    toc_blocks = {b for r in tocs for b in range(r.header, r.end)}

    # 4. Single walk over the blocks.
    out = _Builder()
    sections: list[Section] = []
    notes: list[Annotation] = []
    pending_refs: list[Annotation] = []  # references waiting for their body
    path: list[tuple[int, str]] = []  # (level, label) stack
    cast: dict[str, int] = {}
    current_speaker: str | None = None
    apparatus_level: int | None = None

    def new_annotation(type_: str, block: Block | None, **fields) -> Annotation:
        a = Annotation(
            key=f"{book.key}/a{len(notes)}", type=type_,
            raw_start=book.body_offset + block.start if block else None,
            raw_end=book.body_offset + block.end if block else None,
            **fields,
        )
        notes.append(a)
        return a

    def open_section(level: int, label: str, title: str | None, role: str, source: str) -> int:
        nonlocal path
        if sections:
            sections[-1].end = out.pos
        path = [p for p in path if p[0] < level] + [(level, label)]
        sections.append(Section(
            index=len(sections), level=level, label=label, title=title,
            path=[p[1] for p in path], role=role, heading_source=source,
            start=out.pos + (len(SEPARATOR) if out.parts else 0), end=out.pos,
        ))
        return sections[-1].index

    if heads:
        section = open_section(1, "Front matter", None, "front_matter", "front")
    else:
        section = open_section(1, "Main text", None, "main_text", "fallback")

    for i, block in enumerate(blocks):
        kind = kinds.get(i)
        if i in title_blocks:
            continue  # emitted together with its heading

        if i in head_at:
            h = head_at[i]
            level = levels[h.rank]
            label = " ".join(block.stripped)
            title = blocks[h.title_block].first if h.title_block is not None else None
            decoration = "\n".join(blocks[b].first for b in h.extra_blocks)
            if apparatus_level is not None and level <= apparatus_level:
                apparatus_level = None
            if _is_apparatus(label, title):
                apparatus_level = level
            role = "apparatus" if apparatus_level is not None else "main_text"
            section = open_section(level, label, title, role, h.source)
            heading_text = "\n".join(x for x in (label, decoration, title) if x)
            out.add(heading_text, section, "heading")
            current_speaker = None
            continue

        if kind == "separator":
            # A decoration line right after a heading is not a scene break.
            if out.paragraphs and out.paragraphs[-1].kind != "heading":
                new_annotation("scene_break", block, label=block.first, anchor=out.pos,
                               section=section, paragraph=out.paragraphs[-1].index)
            continue
        if kind == "illustration":
            new_annotation("illustration", block, label=block.first[:80],
                           text=ann.illustration_caption(block) or None, anchor=out.pos,
                           section=section,
                           paragraph=out.paragraphs[-1].index if out.paragraphs else None)
            continue
        if kind in ("credit", "transcriber_note"):
            new_annotation(kind, block, text=join_lines(block.lines, verse=False), section=section)
            continue
        if kind == "footnote_body":
            label, text = ann.footnote_body(block)
            ref = _take_reference(pending_refs, label, section)
            if ref is not None:
                ref.text, ref.status = text, "matched"
                ref.raw_start = book.body_offset + block.start
                ref.raw_end = book.body_offset + block.end
            else:
                new_annotation("footnote", block, label=f"[{label}]", text=text,
                               section=section, status="unmatched_body")
            continue

        # Ordinary text block.
        para_kind, speaker, labelled = ("toc" if i in toc_blocks else "text"), None, False
        if play and para_kind == "text" and sections[section].role == "main_text":
            label = sp.speaker_label(block)
            if label is not None:
                current_speaker, speaker, labelled = label, label, True
                para_kind = "speech"
            elif current_speaker is None or sp.is_stage_direction(block):
                # Before the first speech of a section, or explicit direction.
                # The current speaker is kept: verse speeches often continue
                # after a bracketed direction without repeating the label.
                para_kind = "stage_direction"
            else:
                para_kind, speaker = "speech", current_speaker

        if labelled:
            speech = Block(block.lines[1:], block.start, block.end)
            verse = is_verse(speech, width)
            # The label stays in the text, on its own line: it helps the LLM
            # (and lexical search) know who speaks.
            text = speaker + "\n" + strip_markup(join_lines(speech.lines, verse))
        else:
            verse = is_verse(block, width)
            text = strip_markup(join_lines(block.lines, verse))
        text, refs = ann.extract_references(text)
        if not text.strip():
            continue
        extra = {}
        if speaker:
            extra = {"speaker": speaker, "speaker_key": sp.speaker_key(book.key, speaker)}
            if labelled:
                cast[speaker] = cast.get(speaker, 0) + 1
        para = out.add(text, section, para_kind, verse=verse, **extra)
        for label, offset in refs:
            pending_refs.append(new_annotation(
                "footnote", None, label=f"[{label}]", anchor=para.start + offset,
                section=section, paragraph=para.index, status="unmatched_reference",
            ))

    sections[-1].end = out.pos
    text = out.text

    # Drop an empty front-matter section (book starting directly with a heading).
    if heads and sections[0].role == "front_matter" and sections[0].end <= sections[0].start:
        sections, notes, out.paragraphs = _drop_first_section(sections, notes, out.paragraphs)

    # 5. Metrics and flags.
    toc_entries = sum(hd.heading_like_entries(r) for r in tocs)
    toc_matched = sum(1 for h in heads if "toc" in h.source)
    metrics = _metrics(text, sections, out.paragraphs, notes, heads, toc_entries, toc_matched,
                       play, sum(cast.values()))
    flags = list(book.source_flags) + _flags(metrics, heads, tocs)
    cleaned = CleanedBook(
        key=book.key, language=book.language, cleaning_version=CLEANING_VERSION,
        cleaned_at=utc_now(), raw_sha256=book.raw_sha256,
        status="flagged" if SEVERE_FLAGS & set(flags) else "ok", flags=flags,
        sections=sections, paragraphs=out.paragraphs, annotations=notes,
        cast=[CastMember(key=sp.speaker_key(book.key, label), label=label, speeches=n)
              for label, n in sorted(cast.items(), key=lambda kv: -kv[1])],
        metrics=metrics,
    )
    return text, cleaned


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_APPARATUS_WORDS = (
    "notes", "footnotes", "endnotes", "appendix", "index", "glossary", "errata",
    "bibliography", "contents", "table des matières", "notes du traducteur", "anmerkungen",
    "indice", "índice", "note", "inhalt", "sisällys", "περιεχόμενα",
)


_ALTERNATIVE_TITLE = re.compile(r"^(?:or|oder|ou|o|ossia|ovvero|osia|eller|tai|of)\b", re.I)


def _works_in_title(title: str | None) -> list[str] | None:
    """'Gladius Dei; Schwere Stunde' -> two works to look for as headings.
    '…; Or, The Missing Pilot' is an alternative title, not a second work."""
    if not title or ";" not in title:
        return None
    parts = [p.strip() for p in title.split(";")]
    works = [p for p in parts if p and not _ALTERNATIVE_TITLE.match(p)]
    return works if len(works) >= 2 else None


def _is_apparatus(label: str, title: str | None) -> bool:
    words = (title or label).casefold().strip(" .:")
    return words in _APPARATUS_WORDS


def _take_reference(pending: list[Annotation], label: str, section: int) -> Annotation | None:
    """Nearest preceding unmatched reference with the same label, in the same
    section first, then anywhere before (notes collected at the end)."""
    for same_section in (True, False):
        for k in range(len(pending) - 1, -1, -1):
            ref = pending[k]
            if ref.label == f"[{label}]" and (ref.section == section or not same_section):
                return pending.pop(k)
    return None


def _drop_first_section(sections, notes, paragraphs):
    sections = sections[1:]
    for s in sections:
        s.index -= 1
    for p in paragraphs:
        p.section -= 1
    for a in notes:
        if a.section is not None:
            a.section = max(a.section - 1, 0)
    return sections, notes, paragraphs


def _metrics(text, sections, paragraphs, notes, heads, toc_entries, toc_matched, play, speeches) -> Metrics:
    lengths = [s.end - s.start for s in sections] or [0]
    front = sum(s.end - s.start for s in sections if s.role == "front_matter")
    by_source: dict[str, int] = {}
    for h in heads:
        by_source[h.source] = by_source.get(h.source, 0) + 1
    footnotes = [a for a in notes if a.type == "footnote"]
    text_paras = [p for p in paragraphs if p.kind in ("text", "speech")]
    return Metrics(
        characters=len(text),
        sections=len(sections),
        headings_by_source=by_source,
        toc_entries=toc_entries,
        toc_entries_matched=toc_matched,
        front_matter_share=round(front / max(len(text), 1), 3),
        section_chars_median=int(statistics.median(lengths)),
        section_chars_max=max(lengths),
        paragraphs=len(paragraphs),
        verse_share=round(sum(p.verse for p in text_paras) / max(len(text_paras), 1), 3),
        is_play=play,
        speeches=speeches,
        footnote_references=sum(1 for a in footnotes if a.anchor is not None),
        footnote_bodies=sum(1 for a in footnotes if a.text is not None),
        footnotes_matched=sum(1 for a in footnotes if a.status == "matched"),
    )


SHORT_WORK_CHARS = 60_000  # below this, a text without chapters is normal


def _flags(m: Metrics, heads, tocs) -> list[str]:
    flags = []
    if not heads:
        # A short story without chapters is normal; a long novel without any
        # detected heading needs review.
        flags.append("no_headings" if m.characters >= SHORT_WORK_CHARS else "no_headings_short_work")
    if m.front_matter_share > 0.3:
        flags.append("large_front_matter")
    if m.sections > 2 and m.section_chars_max > 0.6 * m.characters:
        flags.append("single_dominant_section")
    if tocs and m.toc_entries >= 5 and max(m.toc_entries_matched, len(heads)) < 0.5 * m.toc_entries:
        # The table lists many more entries than headings were found, by the
        # dictionary or by patterns (a TOC written "I. Title" while the body
        # says "CHAPTER I" is fine: the patterns found the chapters).
        flags.append("toc_heading_mismatch")
    if m.footnote_references >= 5 and m.footnotes_matched < 0.8 * m.footnote_references:
        flags.append("unmatched_footnotes")
    return flags
