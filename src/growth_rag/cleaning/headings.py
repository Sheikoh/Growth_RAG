"""Heading detection.

Two complementary sources of headings:

1. **Patterns**: keyword + number ("CHAPTER I.", "Capítulo primero.", "第一回"),
   ordinal + keyword ("ENSIMMÄINEN LUKU"), bare numerals ("IV"). Case-sensitive
   and anchored on the first line of a short block.
2. **TOC dictionary**: when the book has a table of contents, each entry is
   searched again, in order, after the table. This recovers headings no pattern
   can know (Faust's scene names, Moby Dick's "ETYMOLOGY.").

Tables of contents repeat the headings, so a candidate only counts if real text
follows it before the next candidate (or if it is the parent of a kept child,
e.g. "Livre premier" directly followed by "Chapitre I").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from growth_rag.cleaning.text import Block, is_separator, is_separator_line

# A heading is the first line of a short block. Pattern headings may carry a
# long title wrapping over a few lines ("Capítulo VIII. Del buen suceso que …").
MAX_HEADING_LINES = 3
MAX_HEADING_CHARS = 150
MAX_PATTERN_LINES = 4
MAX_PATTERN_CHARS = 320
MAX_HEADING_LINE = 100
MIN_SECTION_TEXT = 200  # characters of text a kept heading must be followed by
MIN_POEM_TEXT = 60  # a poem title may introduce a single short stanza
MIN_BARE_NUMERALS = 3  # "IV" alone counts as a heading only if the book has several

# --- keyword tables (all languages at once: collisions are harmless) ----------
# rank 1 = part level, 2 = chapter level, 3 = bare numerals
RANK1 = (
    "Part|PART|Book|BOOK|Volume|VOLUME|Act|ACT|Partie|PARTIE|Livre|LIVRE|Tome|TOME|"
    "Acte|ACTE|Parte|PARTE|Libro|LIBRO|Acto|ACTO|Atto|ATTO|Teil|TEIL|Buch|BUCH|Akt|AKT|"
    "Aufzug|AUFZUG|Osa|OSA|ΜΕΡΟΣ|Μέρος|ΒΙΒΛΙΟΝ|ΒΙΒΛΙΟ|ΠΡΑΞΙΣ|ΠΡΑΞΗ"
)
RANK2 = (
    "Chapter|CHAPTER|Scene|SCENE|Canto|CANTO|Chapitre|CHAPITRE|Scène|SCÈNE|Capítulo|CAPÍTULO|"
    "Capitulo|CAPITULO|Escena|ESCENA|Capitolo|CAPITOLO|Scena|SCENA|Kapitel|KAPITEL|Szene|SZENE|"
    "Auftritt|AUFTRITT|Luku|LUKU|ΚΕΦΑΛΑΙΟΝ|ΚΕΦΑΛΑΙΟ|Κεφάλαιον|Κεφάλαιο|ΣΚΗΝΗ|"
    # Epistolary works: "LETTRE I.", "Letter XII", "LETTERA PRIMA", "Brief 3"
    "Letter|LETTER|Lettre|LETTRE|Lettera|LETTERA|Carta|CARTA|Brief|BRIEF|Kirje|KIRJE|"
    "Chap\\.|CHAP\\."
)
# Standalone section names (no number).
STANDALONE = (
    "Prologue|PROLOGUE|Epilogue|EPILOGUE|Preface|PREFACE|Introduction|INTRODUCTION|"
    "Conclusion|CONCLUSION|Prologo|PROLOGO|Prólogo|PRÓLOGO|Epílogo|EPÍLOGO|Préface|PRÉFACE|"
    "Épilogue|ÉPILOGUE|Avant-propos|AVANT-PROPOS|Introduzione|INTRODUZIONE|Epilogo|EPILOGO|"
    "Prefazione|PREFAZIONE|Prolog|PROLOG|Epilog|EPILOG|Vorrede|VORREDE|Vorwort|VORWORT|"
    "Einleitung|EINLEITUNG|Nachwort|NACHWORT|ΠΡΟΛΟΓΟΣ|ΕΠΙΛΟΓΟΣ|Πρόλογος|Επίλογος"
)
ORDINALS_ROMANCE = (
    "Primera|Segunda|Tercera|Cuarta|Quinta|Prima|Seconda|Terza|Quarta|Première|Deuxième|"
    "Troisième|Quatrième|Seconde|PREMIÈRE|DEUXIÈME|TROISIÈME|QUATRIÈME|SECONDE|PRIMERA|"
    "SEGUNDA|TERCERA|PRIMA|SECONDA|TERZA"
)

# What may follow a keyword: a numeral or an ordinal word, so that "Book in
# hand, he …" or "Part of the reason …" are not headings.
ORDINAL_WORDS = (
    "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|last|"
    "one|two|three|four|five|six|seven|eight|nine|ten|the|"
    "premier|première|deuxième|second|seconde|troisième|quatrième|cinquième|sixième|septième|"
    "huitième|neuvième|dixième|onzième|douzième|dernier|dernière|unique|"
    "primero|primera|segundo|segunda|tercero|tercera|cuarto|cuarta|quinto|quinta|sexto|séptimo|"
    "octavo|noveno|décimo|último|última|único|"
    "primo|secondo|terzo|quarto|sesto|settimo|ottavo|nono|decimo|ultimo|unico|"
    "erstes|zweites|drittes|viertes|fünftes|erster|zweiter|dritter|letztes"
)
_EN_UNITS = "one|two|three|four|five|six|seven|eight|nine|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth"
_EN_TEENS = (
    "ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    "tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth"
)
_EN_TENS = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|twentieth|thirtieth|fortieth|fiftieth"
NUMBER_WORDS = (
    rf"(?:(?:{_EN_TENS})(?:[-\s](?:{_EN_UNITS}))?|{_EN_TEENS}|"
    r"(?:one\s)?hundred(?:\s(?:and\s)?\w+)?|"
    r"\w+i[èe]me|"  # French ordinals: treizième, vingt-et-unième
    r"\w+esim[oa])"  # Italian ordinals: undicesimo, ventesima
)
NUMBER = (
    rf"(?:[IVXLCDM]+|\d+|[A-Z]|[Α-ΩϚ]{{1,3}}['ʹ΄’]|(?i:{NUMBER_WORDS})|(?i:{ORDINAL_WORDS}))"
    r"(?=$|[\s.:,;—–\-)])"
)

PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(rf"^(?:{RANK1})\s+{NUMBER}"), 1),
    (re.compile(rf"^(?:{RANK2})\s+{NUMBER}"), 2),
    (re.compile(rf"^(?:{STANDALONE})\.?$"), 2),
    (re.compile(rf"^(?:{ORDINALS_ROMANCE})\s+(?:parte|Parte|PARTE|partie|Partie|PARTIE)\b"), 1),
    # Ordinal word + keyword at the end of the line: "ENSIMMÄINEN LUKU", "Erstes Kapitel".
    (re.compile(r"^\S+\s+(?:Luku|LUKU|Kapitel|KAPITEL|Kapitel\.)\.?$"), 2),
    (re.compile(r"^\S+\s+(?:Osa|OSA|Buch|BUCH|Teil|TEIL)\.?$"), 1),
    # Chinese/Japanese: 第 + numeral + unit.
    # The unit must be followed by a space or the end of line: "第四回中…" is a
    # sentence ("in chapter four…"), not a heading.
    (re.compile(r"^第[\d一二三四五六七八九十百千零〇两]+[卷部](?=$|\s)"), 1),
    (re.compile(r"^第[\d一二三四五六七八九十百千零〇两]+[回章節节](?=$|\s)"), 2),
]
BARE_NUMERAL = re.compile(r"^(?:§\s*)?\(?(?:[IVXLC]{1,7}|\d{1,3})[º°]?\.?\)?$")
CJK_HEADING = re.compile(r"^\s*第[\d一二三四五六七八九十百千零〇两]+[回章節节卷部](?=$|\s)")
# "Chapitre I" alone (number only): the title may be on the next line.
BARE_HEADING = re.compile(
    rf"^(?:(?:{RANK1}|{RANK2})\s+\S+|\S+\s+(?:Luku|LUKU)|[IVXLC]{{1,7}}|\d{{1,3}}"
    rf"|第[\d一二三四五六七八九十百千零〇两]+[回章節节卷部])\.?$"
)


def is_standalone_line(line: str) -> bool:
    """Lines that must form their own block even without blank lines around
    them: separator lines and Chinese chapter headings (often glued to the
    previous paragraph in Gutenberg files)."""
    stripped = line.strip()
    return is_separator_line(line) or (bool(CJK_HEADING.match(stripped)) and len(stripped) <= 60)

TOC_HEADER = re.compile(
    r"^(?:CONTENTS|Contents|TABLE OF CONTENTS|Table of Contents|TABLE DES MATI[ÈE]RES|"
    r"Table des mati[èe]res|TABLE|INDEX|[ÍI]NDICE(?:\s.*)?|[ÍI]ndice|INHALT|Inhalt|"
    r"INHALTSVERZEICHNIS|Inhaltsverzeichnis|SISÄLLYS|Sisällys|SISÄLLYSLUETTELO|"
    r"ΠΕΡΙΕΧΟΜΕΝΑ|Περιεχόμενα|目錄|目录|目次)\.?$"
)
# Other lists that often follow a TOC: their entries are not headings, so they
# end the TOC (illustration lists, casts of characters).
LIST_HEADER = re.compile(
    r"^(?:(?:LIST OF )?ILLUSTRATIONS|(?:List of )?Illustrations|DRAMATIS PERS[OÆ]N[AÆ]E?|"
    r"Dramatis Pers[oæ]n[aæ]e?|PERSONS REPRESENTED|Persons Represented|PERSONNAGES|"
    r"Personnages|PERSONEN|Personen|PERSONAGGI|Personaggi|PERSONAJES|Personajes|"
    r"HENKILÖT|Henkilöt|ΠΡΟΣΩΠΑ|Πρόσωπα)\b"
)
_TOC_PAGE = re.compile(r"[\s.·…_]*\d+\s*$")
# Page references in TOC entries: "Capitolo I      »     5", "I     Pag.   VII",
# "Chapter I ........ 12". Everything from the first wide gap or leader is cut.
_TOC_TAIL = re.compile(r"(?:\s{3,}|\s*\.{3,}|\s+»|\s+Pag\.|\s+pag\.|\s+[Pp]age\b).*$")
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


_COLUMN_HEADER = re.compile(
    r"^(?:chapters?|chap|chapitres?|capitolo|capitoli|cap[ií]tulos?|kapitel|luku|"
    r"pages?|pag|seite|seiten|sivu|»|«|--|—|–|\.+|_+)\.?$",
    re.I,
)
_LEADING_NUMBER = re.compile(r"^(?:[IVXLC]{1,7}|\d{1,3})[.)]?\s+(?=\w)")


def clean_entry(line: str) -> str:
    """TOC line -> entry text, without page references or markup. Column
    headers ("CHAPTER", "Pages", "»") become empty and are dropped."""
    entry = _TOC_TAIL.sub("", line.strip()).strip()
    entry = _MARKUP_EDGES.sub("", entry).replace("_", "").strip()
    return "" if _COLUMN_HEADER.match(entry) or not re.search(r"\w", entry) else entry


def entry_variants(entry: str) -> list[list[str]]:
    """Token lists an entry may appear as in the body: as printed, and without
    its leading number ("1. Alt-Freibergs Romantik" -> "Alt-Freibergs Romantik")."""
    variants = [_tokens(entry)]
    stripped = _LEADING_NUMBER.sub("", entry)
    if stripped != entry:
        variants.append(_tokens(stripped))
    return [v for v in variants if v]


@dataclass
class Heading:
    block: int
    rank: int
    source: str  # "pattern", "toc", "pattern+toc"
    title_block: int | None = None
    # Decoration printed with the heading, e.g. "I" under "CHAPITRE PREMIER":
    # emitted as part of the heading, never as a section of its own.
    extra_blocks: list[int] = field(default_factory=list)
    # Text required after the heading; None = MIN_SECTION_TEXT. Poem titles
    # accept less: a quatrain is ~180 characters.
    min_text: int | None = None


@dataclass
class TocRegion:
    header: int  # block index of the "CONTENTS" line
    end: int  # first block after the table
    entries: list[str] = field(default_factory=list)


def _eligible(block: Block) -> bool:
    return (
        len(block.lines) <= MAX_HEADING_LINES
        and block.chars <= MAX_HEADING_CHARS
        and len(block.first) <= MAX_HEADING_LINE
    )


def _eligible_pattern(block: Block) -> bool:
    return (
        len(block.lines) <= MAX_PATTERN_LINES
        and block.chars <= MAX_PATTERN_CHARS
        and len(block.first) <= MAX_HEADING_LINE
    )


_DIALOGUE_START = re.compile(r"^[\"'“”«»‘’—–\-]")


def _eligible_toc_match(block: Block) -> bool:
    """TOC-only matches have no keyword to vouch for them: single-line blocks
    only, not dialogue, not a speaker label ('HAMLET.' followed by a speech)."""
    line = block.first
    return len(block.lines) == 1 and _eligible(block) and not _DIALOGUE_START.match(line) and not line.endswith(":")


_MARKUP_EDGES = re.compile(r"^[_=]+|[_=]+$")


def pattern_rank(line: str) -> int | None:
    line = _MARKUP_EDGES.sub("", line.strip())
    for pattern, rank in PATTERNS:
        if pattern.match(line):
            return rank
    if BARE_NUMERAL.match(line):
        return 3
    return None


def _tokens(text: str) -> list[str]:
    text = _TOC_PAGE.sub("", text)
    return _NON_WORD.sub(" ", text.casefold()).split()


def _token_prefix(a: list[str], b: list[str]) -> bool:
    """One token list is a prefix of the other ("chapitre i" vs "chapitre i
    monsieur myriel"), never a partial token ("chapitre i" vs "chapitre ii")."""
    if not a or not b:
        return False
    if a == b:
        return True  # exact match, even for "I" alone
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return long_[: len(short)] == short and len(" ".join(short)) >= 3


# --------------------------------------------------------------------------- #
# Tables of contents
# --------------------------------------------------------------------------- #

def find_toc_regions(blocks: list[Block]) -> list[TocRegion]:
    """A TOC starts at a 'CONTENTS'-like line. It ends where its first entry
    appears again (the body starts by repeating it), or at the first block that
    does not look like TOC entries (running prose)."""
    regions: list[TocRegion] = []
    i = 0
    while i < len(blocks):
        if not TOC_HEADER.match(clean_entry(blocks[i].first)):
            i += 1
            continue
        header = i
        entries = [clean_entry(l) for l in blocks[i].stripped[1:] if clean_entry(l)]
        added: list[int] = []  # entries contributed by each block after the header
        j = i + 1
        while j < len(blocks):
            block = blocks[j]
            if entries and _eligible(block) and _repeats(block, entries[0]):
                break  # first entry repeated: the body starts here
            if LIST_HEADER.match(block.first):
                break  # another list (illustrations, cast of characters)
            if not _looks_like_toc(block):
                # Running prose. If the block just before it is a heading (the
                # first chapter of the body), it does not belong to the table.
                prev = j - 1
                if prev > header and pattern_rank(blocks[prev].first) is not None and added:
                    del entries[len(entries) - added[-1]:]
                    j = prev
                break
            new = [clean_entry(l) for l in block.stripped if clean_entry(l)]
            entries.extend(new)
            added.append(len(new))
            j += 1
        regions.append(TocRegion(header, j, entries))
        i = j
    return regions


def _looks_like_toc(block: Block) -> bool:
    """Short block, or a block where most lines look like entries (heading
    keywords, page references, or short lines)."""
    if _eligible(block):
        return True
    lines = block.stripped
    entry_like = sum(
        1 for line in lines
        if pattern_rank(clean_entry(line)) is not None or _TOC_TAIL.search(line) or len(line) <= 60
    )
    return entry_like >= 0.5 * len(lines)


def _repeats(block: Block, entry: str) -> bool:
    body = _tokens(block.first)
    return any(_token_prefix(body, v) for v in entry_variants(entry))


def heading_like_entries(region: TocRegion) -> int:
    """Entries that look like headings (to compare with headings found). If
    none do (e.g. Faust's scene names), all entries count."""
    n = sum(1 for e in region.entries if pattern_rank(e) is not None)
    return n or len(region.entries)


def match_toc(blocks: list[Block], entries: list[str], start: int, limit: int) -> dict[int, int]:
    """Find each entry again in blocks[start:limit], in order. Returns
    {block: entry index}. Unfound entries are skipped without moving forward."""
    found: dict[int, int] = {}
    pointer = start
    for e, entry in enumerate(entries):
        variants = entry_variants(entry)
        if not variants:
            continue
        for b in range(pointer, limit):
            block = blocks[b]
            usable = _eligible_toc_match(block) or (
                pattern_rank(block.first) is not None and _eligible_pattern(block)
            )
            if usable and any(_token_prefix(_tokens(block.first), v) for v in variants):
                found[b] = e
                pointer = b + 1
                break
    return found


def toc_search_range(region: TocRegion, next_header: int, n_blocks: int) -> tuple[int, int]:
    """Where the entries of a table should be looked for: after it, or, for a
    table printed at the end of the book (common in French and Italian
    editions), before it."""
    if region.header > n_blocks // 2:
        return 0, region.header
    return region.end, next_header


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def detect_headings(
    blocks: list[Block],
    excluded: set[int],
    not_text: set[int] | None = None,
    poetry: bool = False,
    extra_entries: list[str] | None = None,
) -> tuple[list[Heading], list[TocRegion]]:
    """Return kept headings (in order) and the TOC regions.

    `excluded`: blocks that cannot be headings (annotations, speaker blocks).
    `not_text`: blocks that do not count as text after a heading (annotations).
    Speaker blocks are excluded from headings but *are* text."""
    not_text = excluded if not_text is None else not_text
    tocs = find_toc_regions(blocks)
    in_toc = {b for r in tocs for b in range(r.header, r.end)}

    candidates: dict[int, Heading] = {}
    bare_numerals = [
        i for i, b in enumerate(blocks)
        if i not in excluded and _eligible(b) and len(b.lines) == 1 and pattern_rank(b.first) == 3
    ]
    allow_bare = len(bare_numerals) >= MIN_BARE_NUMERALS
    for i, block in enumerate(blocks):
        if i in excluded or i in in_toc or not _eligible_pattern(block):
            continue
        rank = pattern_rank(block.first)
        if rank == 3 and not (allow_bare and len(block.lines) == 1):
            continue
        if rank is not None:
            candidates[i] = Heading(i, rank, "pattern")

    searches = []
    for n, region in enumerate(tocs):
        next_header = tocs[n + 1].header if n + 1 < len(tocs) else len(blocks)
        start, limit = toc_search_range(region, next_header, len(blocks))
        searches.append((region.entries, start, limit))
    if extra_entries:
        # e.g. the works named in a catalog title "Gladius Dei; Schwere Stunde"
        found = match_toc(blocks, extra_entries, 0, len(blocks))
        if len(found) >= 2:
            searches.append((extra_entries, 0, len(blocks)))
    for entries, start, limit in searches:
        for b in match_toc(blocks, entries, start, limit):
            if b in excluded or b in in_toc:
                continue
            if b in candidates:
                candidates[b].source = "pattern+toc"
            else:
                candidates[b] = Heading(b, pattern_rank(blocks[b].first) or 2, "toc")

    if poetry:
        _add_poem_titles(candidates, blocks, excluded | in_toc, not_text | in_toc)
    ordered = _drop_toc_echoes([candidates[i] for i in sorted(candidates)], blocks)
    ordered = _merge_numeral_decorations(ordered, blocks, not_text)
    kept = _keep_followed_by_text(ordered, blocks, not_text | in_toc)
    if not kept and not poetry:
        kept = _prose_upper_titles(blocks, excluded | in_toc, not_text | in_toc)
        ordered = kept
    _attach_titles(kept, blocks, excluded, {h.block for h in ordered})
    return kept, tocs


_KEYWORD_HEADING = [p for p, _ in PATTERNS[:2]]  # "CHAPTER …", "BOOK …"


def _merge_numeral_decorations(cands: list[Heading], blocks: list[Block], skip: set[int]) -> list[Heading]:
    """'CHAPITRE PREMIER' immediately followed by a bare 'I': the numeral is a
    decoration of the heading, not a section."""
    out: list[Heading] = []
    for c in cands:
        prev = out[-1] if out else None
        if (
            prev is not None and c.rank == 3
            and any(p.match(blocks[prev.block].first) for p in _KEYWORD_HEADING)
            and all(b in skip for b in range(prev.block + 1 + len(prev.extra_blocks), c.block))
        ):
            prev.extra_blocks.append(c.block)
            continue
        out.append(c)
    return out


MIN_PROSE_TITLE_TEXT = 1000
MIN_PROSE_TITLES = 3


def _prose_upper_titles(blocks: list[Block], excluded: set[int], skip: set[int]) -> list[Heading]:
    """Last resort for prose books without any recognised heading: chapters
    titled by a line in capitals ("THE CANNELL MYSTERY", "PRISONED"). Only if
    there are several, each followed by real text."""
    cands = [Heading(i, 2, "pattern") for i, b in enumerate(blocks)
             if i not in excluded and _is_upper_title(b)]
    kept = _keep_followed_by_text(cands, blocks, skip, min_text=MIN_PROSE_TITLE_TEXT)
    return kept if len(kept) >= MIN_PROSE_TITLES else []


_UPPER_TITLE = re.compile(r"[A-ZÀ-ÖØ-ÞΑ-Ω]{2}")


def _is_upper_title(block: Block) -> bool:
    line = block.first
    return (
        len(block.lines) == 1 and len(line) <= 60 and line == line.upper()
        and bool(_UPPER_TITLE.search(line)) and not is_separator(block)
    )


def _add_poem_titles(cands: dict[int, Heading], blocks: list[Block], excluded: set[int], not_text: set[int]) -> None:
    """Poetry collections: poems are titled by a line in capitals followed by
    verse ("L'ALBATROS"). A title directly followed by another title is a
    section of poems ("SPLEEN ET IDÉAL" then "BENEDICTION"). Only after the
    first regular heading, so the title page is not taken for poems."""
    if not cands:
        return
    start = min(cands)
    titles = [
        i for i in range(start + 1, len(blocks))
        if i not in excluded and i not in cands and _is_upper_title(blocks[i])
    ]
    title_set = set(titles)
    for i in titles:
        nxt = next((j for j in range(i + 1, len(blocks)) if j not in not_text), None)
        if nxt is None:
            continue
        if nxt in title_set:
            cands[i] = Heading(i, 1, "pattern")  # section of poems
        elif nxt in cands and cands[nxt].rank == 3:
            cands[i] = Heading(i, 2, "pattern")  # poem in parts: "UN FANTOME" then "I"
        else:
            cands[i] = Heading(i, 2, "pattern", min_text=MIN_POEM_TEXT)


def _drop_toc_echoes(cands: list[Heading], blocks: list[Block]) -> list[Heading]:
    """Tables of contents without a 'CONTENTS' line cannot be found as regions.
    Their entries still betray themselves: a heading located before the first
    chapter-level heading and repeated word for word later is a TOC echo."""
    standalone = re.compile(rf"^(?:{STANDALONE})\.?$")
    numbered = [c for c in cands if c.rank == 2 and not standalone.match(blocks[c.block].first)]
    if not numbered:
        return cands
    first_chapter = numbered[0].block
    keys = [" ".join(_tokens(blocks[c.block].first)) for c in cands]
    return [
        c for k, c in enumerate(cands)
        if c.block >= first_chapter or keys[k] not in keys[k + 1:]
    ]


def _keep_followed_by_text(
    cands: list[Heading], blocks: list[Block], skip: set[int], min_text: int = MIN_SECTION_TEXT
) -> list[Heading]:
    """Keep a candidate if enough text follows it before the next candidate, or
    if the next candidate is its kept child (lower level, i.e. higher rank)."""
    keep = [False] * len(cands)
    for k in range(len(cands) - 1, -1, -1):
        stop = cands[k + 1].block if k + 1 < len(cands) else len(blocks)
        own = set(cands[k].extra_blocks)
        text = sum(blocks[b].chars for b in range(cands[k].block + 1, stop) if b not in skip and b not in own)
        child = k + 1 < len(cands) and cands[k + 1].rank > cands[k].rank and keep[k + 1]
        needed = cands[k].min_text if cands[k].min_text is not None else min_text
        keep[k] = text >= needed or child
    return [c for c, ok in zip(cands, keep) if ok]


def _attach_titles(heads: list[Heading], blocks: list[Block], excluded: set[int], cands: set[int]) -> None:
    """'Chapitre I' followed by a one-line 'Monsieur Myriel': attach the title.
    Separator lines in between (Chinese files) are skipped."""
    for h in heads:
        if len(blocks[h.block].lines) != 1 or not BARE_HEADING.match(blocks[h.block].first):
            continue
        nxt = max([h.block, *h.extra_blocks]) + 1
        while nxt < len(blocks) and is_separator(blocks[nxt]):
            nxt += 1
        if nxt >= len(blocks) or (nxt in excluded and not is_separator(blocks[nxt])) or nxt in cands:
            continue
        line = blocks[nxt].first
        if (
            len(blocks[nxt].lines) == 1
            and len(line) <= 80
            and not re.match(r"^[\"'“”«»‘’—–\-(\[]", line)
            and not re.search(r"[.!?…:;,]$", line)
            and ("," not in line or line.isupper())  # "Tuttavia, il poetino" starts a sentence
        ):
            h.title_block = nxt


def assign_levels(heads: list[Heading]) -> dict[int, int]:
    """Map ranks actually used to levels 1..n (a book with only chapters gets
    chapters at level 1)."""
    ranks = sorted({h.rank for h in heads})
    return {rank: i + 1 for i, rank in enumerate(ranks)}
