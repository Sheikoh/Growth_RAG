"""`rag eval`: search every question, score, save, compare (embedded Qdrant,
fake encoder: no server, no model)."""

from dataclasses import replace

import pytest

from fake_encoder import FakeEncoder
from fake_reranker import FakeReranker
from growth_rag.config import EmbeddingSettings, QdrantSettings, Settings
from growth_rag.embedding.run import embed_books
from growth_rag.evaluation.check import check_questions
from growth_rag.evaluation.dataset import dump_questions
from growth_rag.evaluation.report import compare_lines, failure_lines, overview_lines, slice_lines
from growth_rag.evaluation.run import EvalRun, build_run, run_eval
from growth_rag.index.qdrant import connect
from growth_rag.index.run import index_books
from test_evaluation import cleaned_and_chunked, ev, q


@pytest.fixture
def settings(tmp_path):
    return replace(Settings(), embedding=EmbeddingSettings(model="fake/bge", revision=None),
                   qdrant=QdrantSettings(path=str(tmp_path / "qdrant")))


@pytest.fixture
def registry(paths):
    return cleaned_and_chunked(paths)  # one book, cleaned and chunked


@pytest.fixture
def indexed(paths, registry, settings):
    embed_books(registry, paths, FakeEncoder(), settings.embedding)
    client = connect(settings.qdrant)
    index_books(registry, paths, client, settings.qdrant)
    yield registry, client
    client.close()


QUESTIONS = [
    q(id="island", query="Jackson Island nobody comes", relevant=[ev("Jackson's Island is good enough for me")]),
    q(id="filtered-out", query="Jackson Island", relevant=[ev("Jackson's Island")], filters={"title": "Hamlet"}),
    q(id="no-answer", query="Elizabeth Bennet"),
    q(id="miss", query="zebra quantum", relevant=[ev("the whale’s eye is so small")]),  # no shared word
    q(id="book", task="book", query="test book", relevant=[{"book": "gutenberg:1"}]),
    q(id="like-it", task="book", query="test book", relevant=[{"book": "gutenberg:1"}],
      seed_books=["gutenberg:1"]),
]


def run_questions(paths, indexed, settings, **kwargs):
    registry, client = indexed
    report = check_questions(QUESTIONS, registry, paths)
    return run_eval(QUESTIONS, report, registry, {"gutenberg:1"}, client, settings, FakeEncoder(), **kwargs)


def test_every_mode_is_searched_and_scored(paths, indexed, settings):
    modes = {m.mode: m for m in run_questions(paths, indexed, settings)}
    assert set(modes) == {"chunks:dense", "chunks:sparse", "chunks:hybrid", "books:metadata", "books:centroid"}
    sparse = modes["chunks:sparse"].questions
    assert sparse["island"].metrics["hit@1"] == 1.0 and sparse["island"].ranks == [1]
    assert sparse["filtered-out"].results == [] and sparse["filtered-out"].metrics["hit@10"] == 0
    assert sparse["no-answer"].metrics is None  # no answer in the corpus: searched, not scored
    books = modes["books:metadata"].questions
    assert books["book"].metrics["hit@1"] == 1.0
    assert "gutenberg:1" not in books["like-it"].results  # the seed book is never recommended


def test_skipped_questions_are_not_scored(paths, indexed, settings):
    modes = run_questions(paths, indexed, settings, chunk_modes=["sparse"], book_vectors=[], skip={"island"})
    assert [m.mode for m in modes] == ["chunks:sparse"]
    assert modes[0].questions["island"].metrics is None


def test_run_files_reports_and_comparison(paths, indexed, settings, tmp_path):
    registry, _ = indexed
    folder = tmp_path / "eval-x"
    folder.mkdir()
    dump_questions(folder / "questions.jsonl", QUESTIONS)
    a = build_run(folder, folder / "questions.jsonl", settings, run_questions(paths, indexed, settings), 20, {})
    a.save(tmp_path / "a.json")
    a = EvalRun.load(tmp_path / "a.json")
    b = build_run(folder, folder / "questions.jsonl",
                  replace(settings, chunking=replace(settings.chunking, target_tokens=300)),
                  run_questions(paths, indexed, settings, chunk_modes=["sparse"], depth=1), 1, {})

    overview = "\n".join(overview_lines(a))
    assert "Passages" in overview and "sparse" in overview and "Recommendations" in overview
    slices = "\n".join(slice_lines(a, QUESTIONS, registry))
    assert "same language" in slices
    assert "filtered-out" in "\n".join(failure_lines(a, QUESTIONS, "chunks:sparse"))
    comparison = "\n".join(compare_lines(a, b))
    assert "setting chunking.target_tokens: 350 -> 300" in comparison and "chunks:sparse" in comparison


def test_cli_eval_saves_a_run_then_shows_and_compares_it(paths, indexed, settings, tmp_path, monkeypatch, capsys):
    import growth_rag.cli as cli

    _, client = indexed
    client.close()  # the CLI opens its own client on the same embedded store
    folder = tmp_path / "eval-cli"
    folder.mkdir()
    dump_questions(folder / "questions.jsonl", QUESTIONS[:4])
    monkeypatch.setenv("GROWTH_RAG_QDRANT_PATH", settings.qdrant.path)
    monkeypatch.setattr(cli, "make_encoder", lambda em: FakeEncoder())

    base = ["--data-dir", str(paths.root), "eval", "--set", str(folder)]
    cli.main(base + ["--modes", "sparse", "--book-vectors"])
    out = capsys.readouterr()
    assert "Passages" in out.out and "hit@5 by slice" in out.out
    assert "filtered-out" in out.err  # its filter excludes the relevant book: a dataset error, not scored
    [run_file] = (paths.root / "eval_runs").glob("eval-cli-*.json")

    cli.main(base + ["--run", run_file.stem, "--failures", "chunks:sparse"])
    assert "miss: zebra quantum" in capsys.readouterr().out  # a saved run: no search, no model
    cli.main(base + ["--run", run_file.stem, "--diagnose", "chunks:sparse"])
    out = capsys.readouterr().out
    assert "Best answer rank" in out and "Misses of chunks:sparse" in out and "miss" in out
    cli.main(base + ["--compare", str(run_file), run_file.stem])
    assert "chunks:sparse" in capsys.readouterr().out


def test_rerank_modes_reorder_the_hybrid_candidates(paths, indexed, settings):
    reranker = FakeReranker()
    modes = {m.mode: m for m in run_questions(paths, indexed, settings, chunk_modes=["hybrid"], book_vectors=[],
                                              reranker=reranker, rerank_pools=[2, 5])}
    assert set(modes) == {"chunks:hybrid", "chunks:rerank@2", "chunks:rerank@5"}
    island = modes["chunks:rerank@5"].questions["island"]
    assert island.metrics["hit@1"] == 1.0 and len(island.results) <= 5
    assert all(len(r.results) <= 2 for r in modes["chunks:rerank@2"].questions.values())
    assert modes["chunks:rerank@2"].questions["filtered-out"].results == []  # no candidates, no rerank call
    assert max(reranker.calls) <= 5


def test_compare_shows_modes_added_by_the_new_run(paths, indexed, settings, tmp_path):
    folder = tmp_path / "eval-r"
    folder.mkdir()
    dump_questions(folder / "questions.jsonl", QUESTIONS)
    a = build_run(folder, folder / "questions.jsonl", settings,
                  run_questions(paths, indexed, settings, chunk_modes=["hybrid"], book_vectors=[]), 20, {})
    b = build_run(folder, folder / "questions.jsonl", settings,
                  run_questions(paths, indexed, settings, chunk_modes=["hybrid"], book_vectors=[],
                                reranker=FakeReranker(), rerank_pools=[5]), 20, {"rerank": {"pools": [5]}})
    text = "\n".join(compare_lines(a, b))
    assert "chunks:rerank@5  (only in B" in text and "setting rerank.pools: None -> [5]" in text


def test_cli_eval_and_search_with_rerank(paths, indexed, settings, tmp_path, monkeypatch, capsys):
    import growth_rag.cli as cli
    import growth_rag.rerank as rerank_module

    _, client = indexed
    client.close()
    folder = tmp_path / "eval-rr"
    folder.mkdir()
    dump_questions(folder / "questions.jsonl", QUESTIONS[:1])
    monkeypatch.setenv("GROWTH_RAG_QDRANT_PATH", settings.qdrant.path)
    monkeypatch.setattr(cli, "make_encoder", lambda em: FakeEncoder())
    monkeypatch.setattr(rerank_module, "make_reranker", lambda rr: FakeReranker())

    cli.main(["--data-dir", str(paths.root), "eval", "--set", str(folder), "--modes", "hybrid",
              "--book-vectors", "--rerank", "5"])
    out = capsys.readouterr().out
    assert "rerank@5" in out
    [run_file] = (paths.root / "eval_runs").glob("eval-rr-*.json")
    assert EvalRun.load(run_file).settings["rerank"]["pools"] == [5]

    cli.main(["--data-dir", str(paths.root), "search", "Jackson Island", "--rerank", "5", "--k", "2"])
    out = capsys.readouterr().out
    assert " 1. " in out and "was " in out and "Jackson" in out
