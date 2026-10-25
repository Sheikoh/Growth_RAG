"""Rule-based sentence boundaries, used only to split paragraphs that are too
long for one chunk (rare). No dependency; covers Western punctuation and the
Chinese/Japanese full stops. A neural multilingual segmenter can replace it
later if evaluation shows bad cuts.

Returns character offsets so that chunk offsets stay exact.
"""

from __future__ import annotations

import re

# End of sentence: . ! ? … (possibly repeated), optional closing quotes or
# brackets, then whitespace, then something that can start a sentence.
_WESTERN = re.compile(
    r"[.!?…]+[\"'”’»)\]]*(?=\s+[\"'“‘«(\[—–\-]?[A-ZÀ-ÖØ-ÞΑ-ΩА-Я0-9¿¡])"
)
# Chinese/Japanese: 。！？ end a sentence even without a following space.
_CJK = re.compile(r"[。！？][」』”’）]*")
# Common abbreviations that end with a period but do not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "mt", "vs", "etc", "e.g", "i.e", "cf", "no", "vol", "ch",
    "m", "mm", "mme", "mlle", "sr", "sra", "srta", "sig", "dott", "prof", "hr", "fr", "nr", "z.b",
}


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """[(start, end)] covering the whole text, cut after sentence ends."""
    cuts: list[int] = []
    for m in _WESTERN.finditer(text):
        word = re.search(r"(\S+?)[.!?…]+[\"'”’»)\]]*$", text[: m.end()])
        if m.group(0).startswith(".") and word and word.group(1).lower().strip("(\"'“«") in _ABBREVIATIONS:
            continue
        if word and len(word.group(1)) == 1 and word.group(1).isupper():
            continue  # initials: "J. R. R. Tolkien"
        cuts.append(m.end())
    cuts.extend(m.end() for m in _CJK.finditer(text))
    cuts = sorted(c for c in set(cuts) if 0 < c < len(text))

    spans, start = [], 0
    for cut in cuts:
        # include the following whitespace in the previous sentence
        while cut < len(text) and text[cut] in " \t\n":
            cut += 1
        if cut > start:
            spans.append((start, cut))
            start = cut
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def word_spans(text: str) -> list[tuple[int, int]]:
    """Fallback for a single sentence longer than a chunk: words (or single
    characters for scripts without spaces)."""
    if " " not in text.strip():
        return [(i, i + 1) for i in range(len(text))]
    return [(m.start(), m.end()) for m in re.finditer(r"\S+\s*", text)]
