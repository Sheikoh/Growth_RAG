"""Chunking rules on small books built through the real cleaning pipeline."""

from growth_rag.chunking.chunker import BookMeta, ChunkSettings, chunk_book
from growth_rag.chunking.sentences import sentence_spans
from growth_rag.chunking.tokens import ApproxTokenCounter, HFTokenCounter
from growth_rag.cleaning.pipeline import BookInput, clean_book

COUNTER = ApproxTokenCounter()  # ~4 characters per token
SMALL = ChunkSettings(target_tokens=100, max_tokens=150, min_tokens=30)


def paragraph(n_sentences: int, word: str = "word") -> str:
    """Hard-wrapped prose paragraph of n sentences (~80 characters each)."""
    sentence = " ".join([word] * 12) + " ends here."
    text = " ".join([sentence[0].upper() + sentence[1:]] * n_sentences)
    lines, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 > 70:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return "\n".join(lines + [line]) + "\n"


def build(body: str, settings=SMALL, play=False, language="en"):
    text, book = clean_book(BookInput(key="gutenberg:1", language=language, body=body, play_hint=play))
    chunks = chunk_book(text, book, BookMeta("gutenberg:1", "My Book", "Ann Author"), COUNTER, settings)
    return text, book, chunks


def test_prose_is_packed_within_sections_and_spans_are_exact():
    body = "".join(f"CHAPTER {n}\n\n" + "\n".join(paragraph(2) for _ in range(6)) + "\n" for n in ("I", "II"))
    text, book, chunks = build(body)
    assert all(c.text == text[c.start:c.end] for c in chunks)
    assert all(c.tokens <= SMALL.max_tokens for c in chunks)
    assert {c.section for c in chunks} == {0, 1}
    for c in chunks:  # never crosses a section
        s = book.sections[c.section]
        assert s.start <= c.start and c.end <= s.end
    assert "CHAPTER" not in "".join(c.text for c in chunks)  # headings are in the header, not the text
    assert chunks[0].key == "gutenberg:1/s0/c0" and chunks[0].header == "My Book — Ann Author\nCHAPTER I"
    assert [c.previous for c in chunks[1:]] == [c.key for c in chunks[:-1]]
    assert chunks[-1].next is None


def test_short_tail_is_merged_into_previous_chunk():
    body = "CHAPTER I\n\n" + "\n".join(paragraph(4) for _ in range(2)) + "\nShort end of chapter.\n"
    _, _, chunks = build(body)
    assert chunks[-1].tokens >= SMALL.min_tokens
    assert chunks[-1].text.endswith("Short end of chapter.")


def test_oversized_paragraph_is_split_at_sentences_in_balanced_parts():
    body = "CHAPTER I\n\n" + paragraph(12)  # ~240 tokens > max 150
    text, _, chunks = build(body)
    n = len(chunks)
    assert n >= 2 and [c.part for c in chunks] == [f"{k}/{n}" for k in range(1, n + 1)]
    assert all(c.text.endswith("ends here.") for c in chunks)  # cut at a sentence end
    sizes = [c.tokens for c in chunks]
    assert max(sizes) - min(sizes) <= 25  # balanced parts, no tiny tail


def test_scene_break_is_a_preferred_cut():
    body = "CHAPTER I\n\n" + paragraph(2) + "\n" + paragraph(2) + "\n* * *\n\n" + paragraph(2) + "\n" + paragraph(2)
    text, book, chunks = build(body)
    brk = next(a for a in book.annotations if a.type == "scene_break")
    assert any(c.end == brk.anchor for c in chunks)  # a chunk ends exactly at the break
    assert brk.key in chunks[0].annotations


STANZA = (
    "Souvent, pour s'amuser, les hommes d'équipage\n"
    "Prennent des albatros, vastes oiseaux des mers,\n"
    "Qui suivent, indolents compagnons de voyage,\n"
    "Le navire glissant sur les gouffres amers.\n"
)


def test_short_poem_is_one_chunk_even_below_minimum():
    body = f"PRÉFACE\n\n{paragraph(3)}\nL'ALBATROS\n\n{STANZA}\nLE CHAT\n\n{STANZA}\n{STANZA}\nLE CYGNE\n\n{STANZA}"
    settings = ChunkSettings(target_tokens=100, max_tokens=150, min_tokens=60)
    text, _, chunks = build(body, settings, language="fr")
    poems = [c for c in chunks if c.kind == "poem"]
    assert [c.section_path[-1] for c in poems] == ["L'ALBATROS", "LE CHAT", "LE CYGNE"]
    assert poems[0].tokens < settings.min_tokens  # kept whole anyway
    assert "d'équipage\nPrennent" in poems[0].text


def test_long_poem_is_split_at_stanzas():
    body = f"PRÉFACE\n\n{paragraph(3)}\nLE VOYAGE\n\n" + "\n".join([STANZA] * 8)
    text, book, chunks = build(body, language="fr")
    poem = [c for c in chunks if c.kind == "poem"]
    assert len(poem) > 1 and all(c.part for c in poem)
    stanza_starts = {p.start for p in book.paragraphs}
    assert all(c.start in stanza_starts for c in poem)  # every part starts at a stanza


def test_play_speeches_are_not_split_and_speakers_are_recorded():
    speech = "HAMLET.\nTo be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\n\n"
    reply = "OPHELIA.\nGood my lord,\nHow does your honour for this many a day?\n\n"
    scene = "ACT I\n\nSCENE I. A room.\n\nEnter Hamlet.\n\n" + (speech + reply) * 6
    text, book, chunks = build(scene, play=True)
    assert all(c.kind == "play" for c in chunks)
    speeches = [p for p in book.paragraphs if p.kind == "speech"]
    for c in chunks:  # every speech is entirely inside one chunk
        for p in speeches:
            assert p.end <= c.start or p.start >= c.end or (c.start <= p.start and p.end <= c.end)
    assert chunks[0].speaker_labels == ["HAMLET", "OPHELIA"]
    assert chunks[0].speakers[0] == "gutenberg:1/speaker/hamlet"


def test_front_matter_and_toc_are_not_chunked():
    body = (
        "MY BOOK\n\nby Someone\n\nCONTENTS\n\nCHAPTER I.\nCHAPTER II.\n\n"
        f"CHAPTER I.\n\n{paragraph(3)}\nCHAPTER II.\n\n{paragraph(3)}"
    )
    _, _, chunks = build(body)
    joined = "".join(c.text for c in chunks)
    assert "MY BOOK" not in joined and "CONTENTS" not in joined
    assert len(chunks) == 2


def test_footnotes_are_linked_to_their_chunk():
    body = "CHAPTER I\n\nA claim that needs a source[1] and more words.\n" + paragraph(2) + "\n[1] The source.\n"
    _, book, chunks = build(body)
    note = next(a for a in book.annotations if a.type == "footnote")
    owner = [c for c in chunks if note.key in c.annotations]
    assert len(owner) == 1 and owner[0].start <= note.anchor <= owner[0].end


def test_chinese_is_measured_and_split_without_spaces():
    sentence = "此開卷第一回也作者自云因曾歷過一番夢幻之后故將真事隱去。"
    body = "第一回　標題\n------\n　　" + sentence * 12  # ~330 CJK characters
    _, _, chunks = build(body, language="zh")
    assert len(chunks) >= 2 and all(c.text.endswith("。") for c in chunks)


def test_sentence_spans_respect_abbreviations_initials_and_cjk():
    text = "Mr. Smith met J. R. Tolkien. They talked! Then left."
    assert [text[a:b].strip() for a, b in sentence_spans(text)] == [
        "Mr. Smith met J. R. Tolkien.", "They talked!", "Then left."]
    zh = "他來了。她走了！好嗎？"
    assert [zh[a:b] for a, b in sentence_spans(zh)] == ["他來了。", "她走了！", "好嗎？"]


def test_hf_counter_uses_the_tokenizer_without_special_tokens():
    class FakeTokenizer:
        def __call__(self, texts, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [t.split() for t in texts]}

    counter = HFTokenCounter("fake/model", batch_size=2)
    counter._tokenizer = FakeTokenizer()
    assert counter.count(["a b", "c", "d e f"]) == [2, 1, 3]
    assert counter.name == "fake/model"
