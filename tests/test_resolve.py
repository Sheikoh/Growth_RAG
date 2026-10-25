"""Book names -> registry keys."""

from growth_rag.index.resolve import normalize, resolve_books
from growth_rag.registry import Author, BookRecord


def book(n, title, author="Hugo, Victor", languages=("fr",), role=None):
    return BookRecord(key=f"gutenberg:{n}", source="gutenberg", source_id=str(n), title=title,
                      languages=list(languages), authors=[Author(name=author, role=role, raw=author)])


BOOKS = [
    book(17489, "Les misérables Tome I: Fantine"),
    book(17493, "Les misérables Tome II: Cosette"),
    book(1524, "Hamlet", "Shakespeare, William", ("en",)),
    book(2229, "Faust: Der Tragödie erster Teil", "Goethe, Johann Wolfgang von", ("de",)),
    book(779, "The Tragical History of Doctor Faustus", "Marlowe, Christopher", ("en",)),
    book(9999, "Hamlet, Prinz von Dänemark", "Shakespeare, William", ("de",)),
    book(5000, "Poems", "Smith, John", ("en",), role="Translator"),
]


def test_normalize():
    assert normalize("Les Misérables, Tome I") == "les miserables tome i"


def test_accents_case_and_all_volumes():
    res = resolve_books(BOOKS, title="les MISERABLES")
    assert res.keys == ["gutenberg:17489", "gutenberg:17493"]


def test_whole_words_only():
    assert resolve_books(BOOKS, title="Faust").keys == ["gutenberg:2229"]  # not "Faustus"


def test_translations_match_and_language_narrows():
    assert set(resolve_books(BOOKS, title="hamlet").keys) == {"gutenberg:1524", "gutenberg:9999"}
    assert resolve_books(BOOKS, title="hamlet", languages=["en"]).keys == ["gutenberg:1524"]


def test_author_matches_main_authors_only():
    assert set(resolve_books(BOOKS, author="Shakespeare").keys) == {"gutenberg:1524", "gutenberg:9999"}
    assert resolve_books(BOOKS, author="Smith").keys == []  # translator, not author
    assert resolve_books(BOOKS, title="fantine", author="hugo").keys == ["gutenberg:17489"]


def test_no_match_suggests_close_titles():
    res = resolve_books(BOOKS, title="hamlett")
    assert not res.books and any("Hamlet" in s for s in res.suggestions)
    res = resolve_books(BOOKS, title="miser")
    assert any("misérables" in s for s in res.suggestions)
