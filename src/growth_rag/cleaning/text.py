"""Text-level helpers: normalisation, block splitting, line joining, markup.

A *block* is a run of non-blank lines (blocks are separated by blank lines).
Chinese-style texts are an exception: paragraphs are marked by a full-width
indentation instead of blank lines, so such blocks are split further.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

FULLWIDTH_SPACE = "　"
_CJK = re.compile(r"[⺀-鿿豈-﫿＀-￯　-〿]")
_MULTISPACE = re.compile(r"[ \t]+")
_ITALIC = re.compile(r"(?<![\w_])_(?=\S)([^_\n]*?\S)_(?![\w_])")
_BOLD = re.compile(r"(?<![\w=])=(?=\S)([^=\n]*?\S)=(?![\w=])")
_SEPARATOR = re.compile(r"^[\s\-_*=~·•.]*$")


def normalize(text: str) -> str:
    """BOM removal, LF line endings, Unicode NFC."""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def decode(data: bytes) -> tuple[str, bool]:
    """Decode bytes; returns (text, used_fallback). UTF-8 first, then cp1252,
    then latin-1 (which never fails)."""
    try:
        return data.decode("utf-8"), False
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252"), True
    except UnicodeDecodeError:
        return data.decode("latin-1"), True


@dataclass
class Block:
    lines: list[str]  # raw lines, trailing whitespace removed
    start: int  # offset of the first character in the body text
    end: int  # offset just after the last line

    @property
    def first(self) -> str:
        return self.lines[0].strip()

    @property
    def stripped(self) -> list[str]:
        return [line.strip() for line in self.lines]

    @property
    def chars(self) -> int:
        return sum(len(line.strip()) for line in self.lines)


def is_separator_line(line: str) -> bool:
    stripped = line.strip()
    return len(stripped.replace(" ", "")) >= 3 and bool(_SEPARATOR.match(stripped))


def split_blocks(body: str, standalone: Callable[[str], bool] | None = None) -> list[Block]:
    """Split on blank lines. Lines for which `standalone(line)` is true (separator
    lines, Chinese chapter headings glued to the text) always form their own block."""
    breaker = standalone or is_separator_line
    blocks: list[Block] = []
    current: list[str] = []
    start = pos = 0

    def flush(end: int) -> None:
        nonlocal current
        if current:
            blocks.append(Block(current, start, end))
            current = []

    for line in body.split("\n"):
        if not line.strip():
            flush(pos - 1)
        elif breaker(line):
            flush(pos - 1)
            blocks.append(Block([line.rstrip()], pos, pos + len(line.rstrip())))
        else:
            if not current:
                start = pos
            current.append(line.rstrip())
        pos += len(line) + 1
    flush(pos - 1)
    return _split_indented(blocks)


def _split_indented(blocks: list[Block]) -> list[Block]:
    """Split blocks at lines starting with a full-width indentation (Chinese
    paragraph convention). Western blocks are left untouched."""
    out: list[Block] = []
    for block in blocks:
        cuts = [i for i, line in enumerate(block.lines) if i > 0 and line.startswith(FULLWIDTH_SPACE)]
        if not cuts:
            out.append(block)
            continue
        offsets, pos = [], block.start
        for line in block.lines:
            offsets.append(pos)
            pos += len(line) + 1
        bounds = [0, *cuts, len(block.lines)]
        for a, b in zip(bounds, bounds[1:]):
            lines = block.lines[a:b]
            out.append(Block(lines, offsets[a], offsets[b - 1] + len(lines[-1])))
    return out


MIN_WESTERN_WIDTH = 60  # Gutenberg plain text is wrapped at ~70 characters


def wrap_width(blocks: list[Block]) -> int:
    """Width at which the book's prose is wrapped (99th percentile of line
    lengths). Mostly-verse books would give a too small value, so Western texts
    get a floor; Chinese/Japanese texts (wrapped at ~35 wide characters) don't."""
    lines = [line.strip() for b in blocks for line in b.lines if line.strip()]
    if not lines:
        return 70
    lengths = sorted(len(line) for line in lines)
    width = lengths[int(0.99 * (len(lengths) - 1))]
    cjk = sum(1 for line in lines if _CJK.search(line)) > 0.5 * len(lines)
    return width if cjk else max(width, MIN_WESTERN_WIDTH)


def is_verse(block: Block, width: int) -> bool:
    """Verse: line breaks that are not imposed by the width.

    In hard-wrapped prose a line ends because the next word would not fit. In
    verse the next word would have fitted easily, yet the line ends. A block is
    verse when most of its breaks are such voluntary breaks. (Lines without
    spaces, e.g. Chinese, never count as voluntary breaks.)"""
    lines = [line.strip() for line in block.lines if line.strip()]
    if len(lines) < 2:
        return False
    voluntary = 0
    for line, nxt in zip(lines, lines[1:]):
        next_word = nxt.split()[0] if nxt.split() else nxt
        if len(line) + 1 + len(next_word) < 0.92 * width:
            voluntary += 1
    return voluntary >= 0.6 * (len(lines) - 1)


def is_separator(block: Block) -> bool:
    """Lines made only of -, _, *, =, ~ ... (e.g. '* * *' or '------')."""
    return all(is_separator_line(line) for line in block.lines)


def _joiner(left: str, right: str) -> str:
    if not left or not right:
        return ""
    if left.endswith("-") and not left.endswith("--") and left[-2:-1].isalpha():
        return ""  # "fit-" + "il" -> "fit-il", "Deux-" + "Mondes" -> "Deux-Mondes"
    if _CJK.match(left[-1]) or _CJK.match(right[0]):
        return ""  # Chinese/Japanese: no spaces between words
    return " "


def join_lines(lines: list[str], verse: bool) -> str:
    """Verse keeps its line breaks; prose lines are joined (no space for CJK)."""
    parts = [_MULTISPACE.sub(" ", line.strip()) for line in lines if line.strip()]
    if verse:
        return "\n".join(parts)
    out = parts[0] if parts else ""
    for part in parts[1:]:
        out += _joiner(out, part) + part
    return out


def strip_markup(text: str) -> str:
    """Remove Gutenberg/DP emphasis markers, keep the words: _italics_, =bold=."""
    return _BOLD.sub(r"\1", _ITALIC.sub(r"\1", text))
