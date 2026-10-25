from growth_rag.sources.gutenberg.catalog import read_catalog
from growth_rag.sources.gutenberg.selection import Manifest, SelectionCriteria, build_manifest, select


def ids(catalog, **criteria):
    return select(catalog, SelectionCriteria(**criteria))[0]


def test_media_type_filter_excludes_non_text(catalog_file):
    assert 5000 not in ids(read_catalog(catalog_file))


def test_language_filter_matches_any_listed_language(catalog_file):
    catalog = read_catalog(catalog_file)
    assert ids(catalog, languages=["fr"]) == [1497, 17489, 99999]  # 1497 is "en; fr"


def test_bookshelf_filter_is_substring(catalog_file):
    catalog = read_catalog(catalog_file)
    # "Novels" also matches "Category: Historical Novels"...
    assert 17489 in ids(catalog, bookshelves=["Novels"])
    # ...the full category name does not.
    assert 17489 not in ids(catalog, bookshelves=["Category: Novels"])


def test_author_filter_or_within_and_between(catalog_file):
    catalog = read_catalog(catalog_file)
    assert ids(catalog, authors=["hugo", "Austen, Jane"]) == [1342, 17489]
    assert ids(catalog, authors=["hugo", "Austen, Jane"], languages=["fr"]) == [17489]
    assert ids(catalog, authors=["smith"]) == [2229]  # contributors count as authors


def test_exclude_subjects(catalog_file):
    assert 84 not in ids(read_catalog(catalog_file), exclude_subjects=["horror"])


def test_stratification_caps_each_language_and_is_deterministic(catalog_file):
    catalog = read_catalog(catalog_file)
    criteria = SelectionCriteria(languages=["en", "fr", "de"], per_language=1, seed=7)
    first, counts = select(catalog, criteria)
    second, _ = select(catalog, criteria)
    assert first == second
    assert {lang: c["selected"] for lang, c in counts.items()} == {"de": 1, "en": 1, "fr": 1}
    # 1497 ("en; fr") is counted once, in its first requested language.
    assert counts["en"]["candidates"] == 3 and counts["fr"]["candidates"] == 2


def test_manifest_round_trip(paths, catalog_file):
    manifest = build_manifest("pilot", read_catalog(catalog_file), catalog_file,
                              SelectionCriteria(languages=["de"]))
    path = paths.manifests / "pilot.json"
    manifest.save(path)
    loaded = Manifest.load(path)
    assert loaded.ids == [2229]
    assert loaded.catalog_snapshot == catalog_file.name
    assert loaded.criteria["languages"] == ["de"]
