"""Recognising annotation blocks and inline footnote references.

Annotations are removed from the running text but kept as records with an
anchor (see README 2.8). This module only recognises them; the pipeline
assigns anchors while it builds the cleaned text.
"""

from __future__ import annotations

import re

from growth_rag.cleaning.text import Block, join_lines, strip_markup

ILLUSTRATION = re.compile(r"^\[(?:Illustration|ILLUSTRATION|Ilustración|Illustrazione|Abbildung)\b")
# Footnote bodies: "[1] text", "[A] text", "[*] text", "[Footnote 1: text]".
FOOTNOTE_BODY = re.compile(r"^\[(?:Footnote\s+)?(\d{1,3}|[A-Z]|\*)(?:\]\s*|\s*:\s*)")
# Inline references: "word[1]", "word[A]". Not at the very start of a paragraph.
FOOTNOTE_REF = re.compile(r"\[(\d{1,3}|[A-Z]|\*)\]")
CREDIT = re.compile(
    r"^(?:Produced by|This e-?(?:book|text) was produced by|E-?text prepared by|"
    r"Transcribed (?:from|by)|Digitized by|Scanned by|Prepared by)\b",
    re.I,
)
TRANSCRIBER_NOTE = re.compile(
    r"^(?:\[?Transcriber'?s?['’]? ?Notes?|Note:|NOTE:|NOTE DEL TRASCRITTORE|Nota del trascrittore|"
    r"Note du transcripteur|NOTE DU TRANSCRIPTEUR|Nota del transcriptor|Anmerkungen? zur Transkription|"
    r"Anmerkung des Transkribierenden|Σημείωση:)",
    re.I,
)


def block_kind(block: Block, front: bool) -> str | None:
    """Annotation type of a whole block, or None for ordinary text.
    Credits and transcriber's notes are only recognised in the front region
    (before the first heading), where they are reliable."""
    first = block.first
    if ILLUSTRATION.match(first):
        return "illustration"
    if FOOTNOTE_BODY.match(first):
        return "footnote_body"
    if TRANSCRIBER_NOTE.match(first):
        return "transcriber_note"
    if front and CREDIT.match(first):
        return "credit"
    return None


def footnote_body(block: Block) -> tuple[str, str]:
    """(label, text) of a footnote body block."""
    text = strip_markup(join_lines(block.lines, verse=False))
    m = FOOTNOTE_BODY.match(text)
    label = m.group(1) if m else "?"
    body = text[m.end():] if m else text
    if text.startswith("[Footnote") and body.endswith("]"):
        body = body[:-1]
    return label, body.strip()


def illustration_caption(block: Block) -> str:
    text = strip_markup(join_lines(block.lines, verse=False)).strip()
    caption = re.sub(r"^\[\w+\s*:?\s*", "", text)
    return caption[:-1].strip() if caption.endswith("]") else caption.strip()


def extract_references(text: str) -> tuple[str, list[tuple[str, int]]]:
    """Remove inline footnote references; return (new text, [(label, offset)]),
    offsets being positions in the *new* text where each reference was."""
    refs: list[tuple[str, int]] = []
    out, last = [], 0
    for m in FOOTNOTE_REF.finditer(text):
        if m.start() == 0:
            continue
        out.append(text[last:m.start()])
        refs.append((m.group(1), sum(len(p) for p in out)))
        last = m.end()
    out.append(text[last:])
    return "".join(out), refs
