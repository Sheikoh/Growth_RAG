"""Cleaned book -> chunks.

Within each section (sections are never crossed):

1. *Pieces* are built from the paragraphs: one piece per paragraph, or several
   if a paragraph alone exceeds `max_tokens` (cut at sentences for prose, at
   lines for verse). In plays, a speech continuing over several paragraphs
   (stanzas) stays one piece when it fits.
2. The section's *mode* decides how pieces are packed:
   - poem  (≥ 80 % verse paragraphs, not a play): the whole poem is one chunk
     if it fits in `max_tokens`, whatever its size; otherwise stanzas are
     packed up to `target_tokens`.
   - prose / play: pieces are packed up to `target_tokens`, preferring to cut
     at scene breaks; a short tail is merged into the previous chunk.
3. Chunks get their exact token count, speakers, annotations, header, keys and
   previous/next links.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from growth_rag.chunking.models import Chunk
from growth_rag.chunking.sentences import sentence_spans, word_spans
from growth_rag.chunking.tokens import TokenCounter
from growth_rag.cleaning.models import CleanedBook, Paragraph, Section


@dataclass(frozen=True)
class ChunkSettings:
    target_tokens: int = 350
    max_tokens: int = 512
    min_tokens: int = 80
    skip_roles: tuple[str, ...] = ("front_matter", "apparatus")
    skip_kinds: tuple[str, ...] = ("heading", "toc")
    poem_verse_share: float = 0.8


@dataclass
class BookMeta:
    key: str
    title: str
    authors: str  # display form, e.g. "Victor Hugo"


@dataclass
class _Piece:
    start: int
    end: int
    tokens: int
    first: int  # paragraph indices covered
    last: int
    break_after: bool = False  # a scene break follows: preferred cut


@dataclass
class _Draft:
    section: Section
    kind: str
    pieces: list[_Piece] = field(default_factory=list)
    part: str | None = None

    @property
    def tokens(self) -> int:
        return sum(p.tokens for p in self.pieces)


def chunk_book(
    text: str,
    book: CleanedBook,
    meta: BookMeta,
    counter: TokenCounter,
    settings: ChunkSettings = ChunkSettings(),
) -> list[Chunk]:
    paragraphs = [p for p in book.paragraphs if p.kind not in settings.skip_kinds]
    counts = dict(zip((p.index for p in paragraphs),
                      counter.count([text[p.start:p.end] for p in paragraphs])))
    breaks = {a.paragraph for a in book.annotations if a.type == "scene_break" and a.paragraph is not None}

    by_section: dict[int, list[Paragraph]] = {}
    for p in paragraphs:
        by_section.setdefault(p.section, []).append(p)

    drafts: list[_Draft] = []
    for section in book.sections:
        paras = by_section.get(section.index)
        if section.role in settings.skip_roles or not paras:
            continue
        mode = _mode(paras, settings)
        pieces = _pieces(paras, mode, text, counts, breaks, counter, settings)
        if mode == "poem":
            drafts.extend(_pack_poem(section, pieces, settings))
        else:
            drafts.extend(_pack(section, mode, pieces, settings))

    return _finalise(drafts, text, book, meta, counter)


# --------------------------------------------------------------------------- #
# Modes and pieces
# --------------------------------------------------------------------------- #

def _mode(paras: list[Paragraph], settings: ChunkSettings) -> str:
    if any(p.kind in ("speech", "stage_direction") for p in paras):
        return "play"
    verse = sum(p.verse for p in paras)
    return "poem" if verse >= settings.poem_verse_share * len(paras) else "prose"


def _is_continuation(prev: Paragraph, para: Paragraph, text: str) -> bool:
    """A speech paragraph without its own label, by the same speaker."""
    return (
        prev.kind == "speech" and para.kind == "speech" and prev.speaker == para.speaker
        and not text[para.start:para.end].startswith(f"{para.speaker}\n")
    )


def _pieces(paras, mode, text, counts, breaks, counter, settings) -> list[_Piece]:
    pieces: list[_Piece] = []
    prev: Paragraph | None = None
    for para in paras:
        tokens = counts[para.index]
        if tokens > settings.max_tokens:
            pieces.extend(_split_paragraph(para, text, counter, settings))
        elif (
            mode == "play" and prev is not None and pieces and _is_continuation(prev, para, text)
            and pieces[-1].last == prev.index and pieces[-1].tokens + tokens <= settings.max_tokens
        ):
            last = pieces[-1]  # keep a speech together across its stanzas
            last.end, last.last, last.tokens = para.end, para.index, last.tokens + tokens
        else:
            pieces.append(_Piece(para.start, para.end, tokens, para.index, para.index))
        if para.index in breaks:
            pieces[-1].break_after = True
        prev = para
    return pieces


def _split_paragraph(para: Paragraph, text: str, counter: TokenCounter, settings: ChunkSettings) -> list[_Piece]:
    """Cut an oversized paragraph at lines (verse) or sentences (prose), then
    pack the parts up to the target. A single sentence above the maximum is cut
    at words (or characters for scripts without spaces)."""
    body = text[para.start:para.end]
    if para.verse:
        spans, pos = [], 0
        for line in body.split("\n"):
            spans.append((pos, pos + len(line) + 1))
            pos += len(line) + 1
        spans[-1] = (spans[-1][0], len(body))
    else:
        spans = sentence_spans(body)
    counts = counter.count([body[a:b] for a, b in spans])

    units: list[tuple[int, int, int]] = []  # (start, end, tokens) in body coordinates
    for (a, b), n in zip(spans, counts):
        if n <= settings.max_tokens:
            units.append((a, b, n))
            continue
        words = word_spans(body[a:b])
        for (wa, wb), wn in zip(words, counter.count([body[a + x:a + y] for x, y in words])):
            units.append((a + wa, a + wb, wn))

    # Balanced parts: a 700-token paragraph becomes 2 x 350, not 512 + 188, and
    # never leaves a tiny last part.
    total = sum(n for _, _, n in units)
    n_parts = max(1, -(-total // settings.target_tokens))  # ceiling division
    part_target = min(settings.max_tokens, -(-total // n_parts))
    pieces: list[_Piece] = []
    start = end = tokens = None
    for a, b, n in units:
        if start is not None and (
            (tokens + n > part_target and tokens >= settings.min_tokens)
            or tokens + n > settings.max_tokens
        ):
            pieces.append(_Piece(para.start + start, para.start + end, tokens, para.index, para.index))
            start = None
        if start is None:
            start, tokens = a, 0
        end, tokens = b, tokens + n
    if start is not None:
        pieces.append(_Piece(para.start + start, para.start + end, tokens, para.index, para.index))
    for piece in pieces:  # do not end a chunk on whitespace
        while piece.end > piece.start and text[piece.end - 1] in " \n":
            piece.end -= 1
    return pieces


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #

def _pack(section: Section, mode: str, pieces: list[_Piece], s: ChunkSettings) -> list[_Draft]:
    kind = "play" if mode == "play" else "prose"
    drafts: list[_Draft] = []
    current = _Draft(section, kind)
    for piece in pieces:
        over_target = current.tokens + piece.tokens > s.target_tokens
        over_max = current.tokens + piece.tokens > s.max_tokens
        # A skipped paragraph (e.g. a TOC) in between: the text is not contiguous.
        gap = bool(current.pieces) and piece.first > current.pieces[-1].last + 1
        if current.pieces and ((over_target and current.tokens >= s.min_tokens) or over_max or gap):
            drafts.append(current)
            current = _Draft(section, kind)
        current.pieces.append(piece)
        if piece.break_after and current.tokens >= s.min_tokens:
            drafts.append(current)
            current = _Draft(section, kind)
    if current.pieces:
        drafts.append(current)
    # A short tail is merged into the previous chunk when it fits.
    if len(drafts) >= 2 and drafts[-1].tokens < s.min_tokens and drafts[-2].tokens + drafts[-1].tokens <= s.max_tokens:
        drafts[-2].pieces.extend(drafts.pop().pieces)
    return drafts


def _pack_poem(section: Section, pieces: list[_Piece], s: ChunkSettings) -> list[_Draft]:
    """Whole poem if it fits, even if short; otherwise stanzas packed up to the
    target, never cutting inside a stanza (unless a stanza alone is too long)."""
    whole = _Draft(section, "poem", list(pieces))
    if whole.tokens <= s.max_tokens:
        return [whole]
    drafts = _pack(section, "poem", pieces, s)
    for i, d in enumerate(drafts, 1):
        d.kind, d.part = "poem", f"{i}/{len(drafts)}"
    return drafts


# --------------------------------------------------------------------------- #
# Final chunks
# --------------------------------------------------------------------------- #

def _header(meta: BookMeta, section: Section) -> str:
    book_line = " — ".join(x for x in (meta.title, meta.authors) if x)
    path = " / ".join(section.path)
    if section.title:
        path = f"{path}: {section.title}" if path else section.title
    return "\n".join(x for x in (book_line, path) if x)


def _finalise(drafts: list[_Draft], text: str, book: CleanedBook, meta: BookMeta,
              counter: TokenCounter) -> list[Chunk]:
    paragraphs = {p.index: p for p in book.paragraphs}
    anchors = sorted((a.anchor, a.key) for a in book.annotations if a.anchor is not None)
    exact = counter.count([text[d.pieces[0].start:d.pieces[-1].end] for d in drafts])

    chunks: list[Chunk] = []
    per_section: dict[int, int] = {}
    for d, tokens in zip(drafts, exact):
        start, end = d.pieces[0].start, d.pieces[-1].end
        first, last = d.pieces[0].first, d.pieces[-1].last
        n = per_section.get(d.section.index, 0)
        per_section[d.section.index] = n + 1
        speakers: list[str] = []
        labels: list[str] = []
        for i in range(first, last + 1):
            p = paragraphs[i]
            if p.speaker_key and p.speaker_key not in speakers and p.end > start and p.start < end:
                speakers.append(p.speaker_key)
                labels.append(p.speaker)
        chunks.append(Chunk(
            key=f"{book.key}/s{d.section.index}/c{n}",
            book_key=book.key, index=len(chunks), section=d.section.index, index_in_section=n,
            section_path=d.section.path, section_title=d.section.title, kind=d.kind,
            language=book.language, start=start, end=end, text=text[start:end],
            header=_header(meta, d.section), tokens=tokens, tokenizer=counter.name,
            paragraphs=[first, last], part=d.part, speakers=speakers, speaker_labels=labels,
        ))

    _label_split_paragraphs(chunks, paragraphs)
    _link_annotations(chunks, anchors)
    for a, b in zip(chunks, chunks[1:]):
        a.next, b.previous = b.key, a.key
    return chunks


def _label_split_paragraphs(chunks: list[Chunk], paragraphs: dict[int, Paragraph]) -> None:
    """Chunks that cover only part of one paragraph get part = 'k/n'."""
    groups: dict[int, list[Chunk]] = {}
    for c in chunks:
        p = paragraphs[c.paragraphs[0]]
        if c.paragraphs[0] == c.paragraphs[1] and (c.start > p.start or c.end < p.end):
            groups.setdefault(p.index, []).append(c)
    for group in groups.values():
        for k, c in enumerate(group, 1):
            c.part = f"{k}/{len(group)}"


def _link_annotations(chunks: list[Chunk], anchors: Sequence[tuple[int, str]]) -> None:
    """Each anchored annotation goes to the first chunk whose span contains it
    (an anchor right at the end of a chunk belongs to that chunk)."""
    if not chunks:
        return
    c = 0
    for anchor, key in anchors:
        while c < len(chunks) and chunks[c].end < anchor:
            c += 1
        if c == len(chunks):
            break
        if chunks[c].start <= anchor <= chunks[c].end:
            chunks[c].annotations.append(key)
