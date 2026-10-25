"""Speakers in plays (README 2.8b).

Supported convention (both plays in the samples): the speaker's name in capitals
with a period, alone on the first line of a block, followed by the speech.
Stage directions are blocks before the first speech of a section, or blocks in
brackets/parentheses, or starting with Enter/Exit/Exeunt.

Speaker keys include the book key: "HAMLET" in two different books gives two
different keys (homonyms across books are never merged here; that is a
graph-stage decision).
"""

from __future__ import annotations

import re
import unicodedata

from growth_rag.cleaning.text import Block

UPPER = r"A-ZÀ-ÖØ-ÞĀ-ŽΑ-ΩΆ-Ώ"
SPEAKER_LABEL = re.compile(rf"^([{UPPER}][{UPPER}'’ .&\-]{{0,38}}[{UPPER}])\.\s*$")
STAGE_DIRECTION = re.compile(r"^(?:[\[(]|_?(?:Enter|Exit|Exeunt|Re-enter|Manet|Flourish|Alarum)\b)")
MIN_LABELS = 30
MIN_LABEL_SHARE = 0.1


def speaker_label(block: Block) -> str | None:
    if len(block.lines) < 2:  # the label must be followed by the speech
        return None
    m = SPEAKER_LABEL.match(block.first)
    return m.group(1).strip() if m else None


def is_play(blocks: list[Block], hint: bool) -> bool:
    """Bookshelf hint, or enough speaker labels in the text."""
    labels = sum(1 for b in blocks if speaker_label(b))
    if hint:
        return labels >= 5
    return labels >= MIN_LABELS and labels / max(len(blocks), 1) >= MIN_LABEL_SHARE


def is_stage_direction(block: Block) -> bool:
    return bool(STAGE_DIRECTION.match(block.first))


def speaker_key(book_key: str, label: str) -> str:
    """'FIRST CLOWN' -> '<book_key>/speaker/first-clown'. Non-Latin labels keep
    their script (Greek, Cyrillic), so they never collapse to the same key."""
    slug = re.sub(r"[\W_]+", "-", unicodedata.normalize("NFC", label).casefold()).strip("-")
    return f"{book_key}/speaker/{slug or 'speaker'}"
