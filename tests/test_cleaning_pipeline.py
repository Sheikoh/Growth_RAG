"""Annotations, speakers and offsets."""

from growth_rag.cleaning.pipeline import BookInput, clean_book

PROSE = (
    "This is a paragraph of running prose that is long enough to count as the\n"
    "real text of a chapter, so that the heading before it is kept by the\n"
    "detector. It goes on for a few lines to pass the minimum length rule.\n"
)


def run(body: str, key: str = "gutenberg:1", **kwargs):
    return clean_book(BookInput(key=key, language="en", body=body, **kwargs))


def test_offsets_are_consistent():
    body = f"Produced by Someone\n\nCHAPTER I.\n\n{PROSE}\n* * *\n\n{PROSE}\nCHAPTER II.\n\n{PROSE}"
    text, book = run(body)
    for p in book.paragraphs:
        section = book.sections[p.section]
        assert section.start <= p.start < p.end <= section.end
        assert text[p.start:p.end].strip()
    assert book.sections[-1].end == len(text)


def test_footnotes_are_anchored_and_matched_even_when_numbering_restarts():
    body = (
        "CHAPTER I.\n\nThe first claim[1] needs a source, and so does the second[2].\n"
        f"{PROSE}\n[1] Source one.\n\n[2] Source two,\non two lines.\n\n"
        f"CHAPTER II.\n\nAnother claim[1] here.\n{PROSE}\n[1] Source three.\n\n"
        "A reference without body[7].\n"
    )
    text, book = run(body)
    notes = [a for a in book.annotations if a.type == "footnote"]
    matched = [a for a in notes if a.status == "matched"]
    assert [a.text for a in matched] == ["Source one.", "Source two, on two lines.", "Source three."]
    assert text[matched[0].anchor - 11:matched[0].anchor] == "first claim"
    assert "[1]" not in text and "Source one" not in text  # removed from the running text
    assert matched[2].section == book.sections[-1].index  # restarted numbering, right chapter
    assert [a.status for a in notes if a.status != "matched"] == ["unmatched_reference"]
    assert book.metrics.footnote_references == 4 and book.metrics.footnotes_matched == 3


def test_illustrations_scene_breaks_and_credits_are_annotations():
    body = (
        "Produced by A Volunteer\n\nNote: the spelling was kept.\n\n"
        f"CHAPTER I.\n\n{PROSE}\n[Illustration: The old house]\n\n{PROSE}\n* * *\n\n{PROSE}"
    )
    text, book = run(body)
    by_type = {a.type: a for a in book.annotations}
    assert by_type["illustration"].text == "The old house"
    assert by_type["illustration"].anchor is not None
    assert by_type["scene_break"].paragraph is not None
    assert by_type["credit"].anchor is None  # book-level
    assert by_type["transcriber_note"].text == "Note: the spelling was kept."
    assert "Illustration" not in text and "Produced by" not in text


def test_speakers_are_book_scoped():
    scene = (
        "ACT I\n\nSCENE I. A room.\n\nEnter Hamlet.\n\n"
        + "HAMLET.\nTo be, or not to be, that is the question:\nWhether 'tis nobler.\n\n" * 5
        + "OPHELIA.\nGood my lord,\nHow does your honour?\n\n"
        + "[Exit.]\n\n"
    )
    _, one = run(scene, key="gutenberg:1524", play_hint=True)
    _, other = run(scene, key="gutenberg:9999", play_hint=True)
    speeches = [p for p in one.paragraphs if p.kind == "speech"]
    assert {p.speaker for p in speeches} == {"HAMLET", "OPHELIA"}
    assert one.cast[0].label == "HAMLET" and one.cast[0].speeches == 5
    # Same name in two books -> two different characters.
    assert one.cast[0].key == "gutenberg:1524/speaker/hamlet"
    assert other.cast[0].key == "gutenberg:9999/speaker/hamlet"
    kinds = [p.kind for p in one.paragraphs]
    assert kinds.count("stage_direction") == 2  # "Enter Hamlet." and "[Exit.]"


def test_speaker_label_that_looks_like_a_heading_stays_a_speaker():
    scene = (
        "ACT III\n\nSCENE II. A hall.\n\n"
        + "HAMLET.\nSpeak the speech, I pray you,\nas I pronounced it to you.\n\n" * 5
        + "PROLOGUE.\nFor us, and for our tragedy,\nHere stooping to your clemency.\n\n"
    )
    _, book = run(scene, play_hint=True)
    assert "PROLOGUE." not in [s.label for s in book.sections]
    assert "PROLOGUE" in [c.label for c in book.cast]


def test_greek_speaker_keys_do_not_collapse():
    from growth_rag.cleaning.speakers import speaker_key

    assert speaker_key("g:1", "ΑΝΤΙΓΟΝΗ") != speaker_key("g:1", "ΙΣΜΗΝΗ")
