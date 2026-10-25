"""Shared test fixtures: a tiny catalog with the edge cases found in the real one,
and a fake HTTP session so no test touches the network."""

from __future__ import annotations

from pathlib import Path

import pytest

from growth_rag.config import DataPaths

CATALOG_CSV = '''Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves
1342,Text,1998-06-01,Pride and Prejudice,en,"Austen, Jane, 1775-1817","Courtship -- Fiction; England -- Fiction",PR,"Best Books Ever Listings; Category: Novels"
84,Text,1993-10-01,"Frankenstein; Or, The Modern Prometheus",en,"Shelley, Mary Wollstonecraft, 1797-1851","Horror tales; Science fiction",PR,"Category: Novels; Gothic Fiction"
17489,Text,2006-01-09,"Les misérables
Tome I",fr,"Hugo, Victor, 1802-1885","Historical fiction",PQ,Category: Historical Novels
2229,Text,2000-07-01,Faust,de,"Goethe, Johann Wolfgang von, 1749-1832; Smith, John [Editor]",Drama,PT,Category: Plays/Films/Dramas
1497,Text,1998-05-01,The Republic,en; fr,"Plato, 428? BCE-348? BCE",Political science,JC,Category: Philosophy & Ethics
99999,Text,2020-01-01,Missing Book,fr,"Nobody, 1900-",Fiction,PQ,Category: Novels
5000,Sound,2004-01-01,Audio thing,en,"Reader, A",Audio,,Category: Novels
'''


@pytest.fixture
def paths(tmp_path: Path) -> DataPaths:
    return DataPaths(tmp_path / "data")


@pytest.fixture
def catalog_file(paths: DataPaths) -> Path:
    path = paths.catalog_dir("gutenberg") / "pg_catalog_2026-10-25.csv"
    path.parent.mkdir(parents=True)
    path.write_text(CATALOG_CSV, encoding="utf-8")
    return path


class FakeResponse:
    def __init__(self, status_code=200, content=b"", content_type="text/plain; charset=utf-8"):
        self.status_code = status_code
        self.content = content
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Maps URL -> FakeResponse, an exception instance, or a list of those
    (consumed in order, to simulate a failure followed by a success)."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        outcome = self.routes.get(url, FakeResponse(404, b"not found", "text/html"))
        if isinstance(outcome, list):
            outcome = outcome.pop(0) if len(outcome) > 1 else outcome[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def book_bytes(book_id: int) -> bytes:
    return (
        f"*** START OF THE PROJECT GUTENBERG EBOOK {book_id} ***\n"
        f"CHAPTER I\n\nText of {book_id}.\n"
        f"*** END OF THE PROJECT GUTENBERG EBOOK {book_id} ***\n"
    ).encode("utf-8")
