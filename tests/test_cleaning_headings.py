"""Heading detection on small excerpts reproducing the layouts of the samples."""

from growth_rag.cleaning.pipeline import BookInput, clean_book

PROSE = (
    "This is a paragraph of running prose that is long enough to count as the\n"
    "real text of a chapter, so that the heading before it is kept by the\n"
    "detector. It goes on for a few lines to pass the minimum length rule.\n"
)


def sections(body: str, **kwargs):
    text, book = clean_book(BookInput(key="t:1", language=kwargs.pop("language", "en"), body=body, **kwargs))
    return text, book


def labels(book):
    return [s.label for s in book.sections]


def test_toc_entries_are_not_counted_twice():
    body = (
        "MY NOVEL\n\nCONTENTS.\n\nCHAPTER I.\nThe Start.\nCHAPTER II.\nThe End.\n\n"
        f"CHAPTER I.\n\n{PROSE}\nCHAPTER II.\n\n{PROSE}"
    )
    _, book = sections(body)
    assert labels(book) == ["Front matter", "CHAPTER I.", "CHAPTER II."]
    assert book.sections[0].role == "front_matter"
    toc = [p for p in book.paragraphs if p.kind == "toc"]
    assert toc  # the table is kept, marked as such


def test_hierarchy_with_title_on_next_line_and_restarting_numbers():
    body = (
        f"Livre premier--Un juste\n\nChapitre I\n\nMonsieur Myriel\n\n{PROSE}\n"
        f"Chapitre II\n\nUn autre titre\n\n{PROSE}\n"
        f"Livre deuxième--La chute\n\nChapitre I\n\nLe soir\n\n{PROSE}"
    )
    _, book = sections(body, language="fr")
    chapters = [s for s in book.sections if s.label == "Chapitre I"]
    assert len(chapters) == 2  # same printed label, different sections
    assert [s.index for s in chapters] != [chapters[0].index] * 2
    assert chapters[0].title == "Monsieur Myriel"
    assert chapters[1].path == ["Livre deuxième--La chute", "Chapitre I"]
    assert chapters[0].level == 2


def test_ordinal_words_and_long_wrapped_titles():
    body = (
        "Capítulo primero. Que trata de la condición y ejercicio del famoso\n"
        "hidalgo don Quijote de la Mancha, con otras cosas que no hacen al caso y\n"
        "que se dicen aquí sólo para que el título sea muy largo\n\n"
        f"{PROSE}\nCapítulo II. Que trata de la primera salida\n\n{PROSE}"
    )
    _, book = sections(body, language="es")
    assert len(book.sections) == 2
    assert book.sections[0].label.startswith("Capítulo primero.")


def test_keyword_followed_by_a_word_is_not_a_heading():
    body = f"Book in hand, he left.\n\n{PROSE}\nCHAPTER I.\n\n{PROSE}"
    _, book = sections(body)
    assert labels(book) == ["Front matter", "CHAPTER I."]


def test_toc_dictionary_finds_unnumbered_scene_names():
    body = (
        "Contents\n\n Zueignung\n Nacht\n Dom\n\n"
        f"Zueignung\n\n{PROSE}\nNacht\n\n{PROSE}\nDom\n\n{PROSE}"
    )
    _, book = sections(body, language="de")
    assert labels(book)[1:] == ["Zueignung", "Nacht", "Dom"]
    assert all(s.heading_source == "toc" for s in book.sections[1:])


def test_toc_without_header_is_recognised_by_its_echo():
    body = (
        "Segunda parte del ingenioso caballero don Quijote de la Mancha\n\n"
        + "Que trata de algo muy importante y largo\n\n" * 6
        + f"Capítulo primero. Inicio\n\n{PROSE}\n"
        f"Segunda parte del ingenioso caballero don Quijote de la Mancha\n\n"
        f"Capítulo primero. Otra vez\n\n{PROSE}"
    )
    _, book = sections(body, language="es")
    parts = [s for s in book.sections if s.label.startswith("Segunda parte")]
    assert len(parts) == 1 and parts[0].index > 1  # only the real one, after chapter 1


def test_chinese_chapters_glued_to_text():
    body = (
        "　　" + "這是第一回的正文。" * 30 + "\n第二回　標題\n------------\n"
        "　　" + "這是第二回的正文。" * 30
    )
    body = "第一回　開始\n------------\n" + body
    text, book = sections(body, language="zh")
    assert [s.label for s in book.sections] == ["第一回　開始", "第二回　標題"]
    assert "這是第二回的正文。這是" in text  # no spaces inserted between Chinese lines


def test_no_heading_falls_back_to_one_section():
    # A short story without chapters is normal: noted, not flagged.
    _, short = sections(PROSE * 3)
    assert [s.role for s in short.sections] == ["main_text"]
    assert short.status == "ok" and "no_headings_short_work" in short.flags
    # A long text without any heading needs review.
    _, long_ = sections(PROSE * 300)
    assert long_.status == "flagged" and "no_headings" in long_.flags


def test_poem_titles_in_a_verse_collection():
    stanza = (
        "Souvent, pour s'amuser, les hommes d'équipage\n"
        "Prennent des albatros, vastes oiseaux des mers,\n"
        "Qui suivent, indolents compagnons de voyage,\n"
        "Le navire glissant sur les gouffres amers.\n"
    )
    body = (
        f"PRÉFACE\n\n{PROSE}\nSPLEEN ET IDÉAL\n\nBENEDICTION\n\n{stanza}\n{stanza}\n"
        f"L'ALBATROS\n\n{stanza}\n{stanza}"
    )
    text, book = sections(body, language="fr")
    by_label = {s.label: s for s in book.sections}
    assert by_label["SPLEEN ET IDÉAL"].level < by_label["BENEDICTION"].level
    assert by_label["L'ALBATROS"].path == ["SPLEEN ET IDÉAL", "L'ALBATROS"]
    albatros = text[by_label["L'ALBATROS"].start:by_label["L'ALBATROS"].end]
    assert "d'équipage\nPrennent" in albatros  # verse line breaks kept


# --- Rules added after the first run on 111 books (37 flagged) ---------------

def test_toc_at_the_end_of_a_collection():
    stories = ["Due sposi in viaggio", "Nei boschi", "La vecchia casa"]
    body = "".join(f"{s.upper()}.\n\n{PROSE}\n" for s in stories)
    body += "INDICE.\n\n" + "\n\n".join(stories) + "\n"
    _, book = sections(body, language="it")
    assert [s.label for s in book.sections] == [f"{s.upper()}." for s in stories]


def test_numbered_toc_entries_match_unnumbered_headings():
    body = (
        "Inhaltsverzeichnis.\n\nSeite\n\n1. Alt-Freibergs Romantik        5--19\n\n"
        "2. Von festen Mauern     20--59\n\n"
        f"Alt-Freibergs Romantik.\n\n{PROSE}\nVon festen Mauern.\n\n{PROSE}"
    )
    _, book = sections(body, language="de")
    assert [s.label for s in book.sections][1:] == ["Alt-Freibergs Romantik.", "Von festen Mauern."]


def test_toc_written_differently_from_the_body_is_not_a_mismatch():
    entries = "\n".join(f"{n}. TITLE NUMBER {n}" for n in ("I", "II", "III", "IV", "V", "VI"))
    chapters = "".join(f"CHAPTER {n}\n\n{PROSE}\n" for n in ("I", "II", "III", "IV", "V", "VI"))
    _, book = sections(f"CONTENTS\n\nCHAPTER\n{entries}\n\n{chapters}")
    assert "toc_heading_mismatch" not in book.flags
    assert len([s for s in book.sections if s.label.startswith("CHAPTER")]) == 6


def test_more_number_forms():
    for heading in ("CHAPTER TWENTY-ONE.", "Chapter Eleven", "LETTRE XII.", "CHAPITRE TREIZIÈME",
                    "CAPITOLO UNDICESIMO"):
        _, book = sections(f"{heading}\n\n{PROSE}")
        assert book.sections[-1].label == heading, heading
    body = "".join(f"_({n}º)_\n\n{PROSE}\n" for n in range(1, 5))
    _, book = sections(body, language="it")
    assert [s.label for s in book.sections] == [f"_({n}º)_" for n in range(1, 5)]


def test_numeral_printed_under_a_heading_is_not_a_section():
    body = "".join(f"CHAPITRE {w}\n\n{n}\n\n{PROSE}\n" for w, n in (("PREMIER", "I"), ("DEUXIÈME", "II"), ("TROISIÈME", "III")))
    text, book = sections(body, language="fr")
    assert [s.label for s in book.sections] == ["CHAPITRE PREMIER", "CHAPITRE DEUXIÈME", "CHAPITRE TROISIÈME"]
    assert text.startswith("CHAPITRE PREMIER\nI")  # kept in the heading text


def test_toc_does_not_swallow_the_first_chapter():
    entries = "\n".join(f"CHAPITRE {n}" for n in ("I", "II", "III", "IV"))
    chapters = "".join(f"{n}\n\n{PROSE}\n" for n in ("I", "II", "III", "IV"))
    _, book = sections(f"TABLE\n\n{entries}\n\n{chapters}", language="fr")
    assert [s.label for s in book.sections][1:] == ["I", "II", "III", "IV"]


def test_capitalised_chapter_titles_in_prose_as_last_resort():
    body = "".join(f"{t}\n\n{PROSE * 6}\n" for t in ("THE CANNELL MYSTERY", "CANNELL'S STORY", "PRISONED"))
    _, book = sections(body)
    assert [s.label for s in book.sections] == ["THE CANNELL MYSTERY", "CANNELL'S STORY", "PRISONED"]


def test_works_named_in_the_catalog_title():
    body = f"Gladius Dei\n\n{PROSE}\nSchwere Stunde\n\n{PROSE}"
    _, book = sections(body, language="de", title="Gladius Dei; Schwere Stunde")
    assert [s.label for s in book.sections] == ["Gladius Dei", "Schwere Stunde"]
    # "; Or, …" is an alternative title, not a second work.
    _, other = sections(body, title="The Meadow-Brook Girls; Or, The Missing Pilot")
    assert "no_headings_short_work" in other.flags
