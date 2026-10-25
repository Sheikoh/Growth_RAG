from growth_rag.cleaning.headings import is_standalone_line
from growth_rag.cleaning.text import (
    Block,
    decode,
    is_verse,
    join_lines,
    normalize,
    split_blocks,
    strip_markup,
    wrap_width,
)


def test_normalize_line_endings_bom_and_nfc():
    assert normalize("﻿é\r\nb\rc") == "é\nb\nc"


def test_decode_falls_back_for_non_utf8():
    text, fallback = decode("café".encode("cp1252"))
    assert text == "café" and fallback


def test_split_blocks_offsets_point_into_the_body():
    body = "First line\nsecond line\n\n\nNext block\n"
    blocks = split_blocks(body)
    assert [b.lines for b in blocks] == [["First line", "second line"], ["Next block"]]
    for b in blocks:
        assert body[b.start:b.end] == "\n".join(b.lines)


def test_separator_and_chinese_heading_lines_become_their_own_blocks():
    body = "雨村忙回頭看時\n第三回\n-----------\n　　卻說雨村\n回頭看時\n　　第二段"
    blocks = split_blocks(body, standalone=is_standalone_line)
    assert [b.first for b in blocks] == ["雨村忙回頭看時", "第三回", "-----------", "卻說雨村", "第二段"]


def test_chinese_sentence_starting_with_chapter_words_is_not_a_heading_line():
    assert not is_standalone_line("第四回中既將薛家母子在榮府內寄居等事略已表明")


def test_verse_vs_wrapped_prose():
    prose = Block([
        "It was the best of times, it was the worst of times, it was the age of",
        "wisdom, it was the age of foolishness, it was the epoch of belief, it",
        "was the epoch of incredulity.",
    ], 0, 0)
    verse = Block([
        "Souvent, pour s'amuser, les hommes d'équipage",
        "Prennent des albatros, vastes oiseaux des mers,",
        "Qui suivent, indolents compagnons de voyage,",
        "Le navire glissant sur les gouffres amers.",
    ], 0, 0)
    width = 72
    assert not is_verse(prose, width)
    assert is_verse(verse, width)


def test_wrap_width_has_a_floor_for_verse_books_but_not_for_chinese():
    verse_book = [Block(["Short verse line number one,", "and a second one."], 0, 0)]
    assert wrap_width(verse_book) == 60
    chinese = [Block(["此開卷第一回也．作者自云：因曾歷過一番夢幻之后"] * 3, 0, 0)]
    assert wrap_width(chinese) < 60


def test_join_lines_rules():
    assert join_lines(["ne fit-", "il jamais"], verse=False) == "ne fit-il jamais"
    assert join_lines(["un succès--", "succès"], verse=False) == "un succès-- succès"
    assert join_lines(["故將真事隱去，", "而借通靈之說"], verse=False) == "故將真事隱去，而借通靈之說"
    assert join_lines(["  Line one,", "  line two."], verse=True) == "Line one,\nline two."


def test_strip_markup_keeps_words():
    assert strip_markup("the _Revue des Deux-Mondes_ and =bold= but snake_case") == (
        "the Revue des Deux-Mondes and bold but snake_case"
    )
