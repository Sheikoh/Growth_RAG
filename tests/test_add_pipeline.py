"""`gutenberg add`: select -> download -> clean -> chunk -> embed -> index."""

from dataclasses import replace

import pytest

from conftest import FakeResponse, FakeSession
from fake_encoder import FakeEncoder
from growth_rag.chunking.tokens import ApproxTokenCounter
from growth_rag.config import ChunkingSettings, EmbeddingSettings, QdrantSettings, Settings
from growth_rag.index.qdrant import IndexUnavailable, connect
from growth_rag.registry import Registry
from growth_rag.sources.gutenberg.catalog import read_catalog
from growth_rag.sources.gutenberg.pipeline import describe_selection, run_pipeline
from growth_rag.sources.gutenberg.selection import SelectionCriteria, build_manifest, select

MIRROR = "https://mirror.test/{id}/pg{id}.txt"


def novel(book_id: int) -> bytes:
    body = "".join(f"CHAPTER {n}\n\n" + "Some prose that is long enough to be a chapter. " * 12 + "\n\n"
                   for n in ("I", "II"))
    return (f"*** START OF THE PROJECT GUTENBERG EBOOK {book_id} ***\n\n{body}"
            f"*** END OF THE PROJECT GUTENBERG EBOOK {book_id} ***\n").encode("utf-8")


def settings(tmp_path):
    base = Settings()
    return replace(
        base,
        gutenberg=replace(base.gutenberg, mirror_url=MIRROR),
        chunking=ChunkingSettings(tokenizer="approx-4cpt"),
        embedding=EmbeddingSettings(model="fake/bge"),
        qdrant=QdrantSettings(path=str(tmp_path / "qdrant")),
    )


def test_title_and_id_criteria(catalog_file):
    catalog = read_catalog(catalog_file)
    assert select(catalog, SelectionCriteria(titles=["pride PREJUDICE"]))[0] == [1342]
    assert select(catalog, SelectionCriteria(titles=["miserables"]))[0] == [17489]  # accents ignored
    assert select(catalog, SelectionCriteria(titles=["faust", "frankenstein"]))[0] == [84, 2229]
    assert select(catalog, SelectionCriteria(ids=[84, 5000]))[0] == [84]  # 5000 is not Text


def test_describe_selection(catalog_file):
    catalog = read_catalog(catalog_file)
    manifest = build_manifest("m", catalog, catalog_file, SelectionCriteria(languages=["en"]))
    lines = describe_selection(manifest, catalog)
    assert lines[0].startswith("3 book(s): en=3")
    assert any("Pride and Prejudice" in line for line in lines)


def run(paths, catalog_file, tmp_path, ids, client_factory=None, until="index"):
    catalog = read_catalog(catalog_file)
    manifest = build_manifest("eval", catalog, catalog_file, SelectionCriteria(ids=ids))
    session = FakeSession({MIRROR.format(id=i): FakeResponse(200, novel(i)) for i in ids})
    cfg = settings(tmp_path)
    clients = []

    def default_client():
        clients.append(connect(cfg.qdrant))
        return clients[-1]

    registry = Registry.load(paths.registry)
    report = run_pipeline(
        registry, manifest, catalog, catalog_file.name, paths, cfg, until=until,
        session=session, counter=ApproxTokenCounter(), encoder_factory=FakeEncoder,
        client_factory=client_factory or default_client, delay_s=0, sleep=lambda s: None,
    )
    return report, registry, clients


def test_full_pipeline_indexes_the_selected_books(paths, catalog_file, tmp_path):
    report, registry, clients = run(paths, catalog_file, tmp_path, [1342, 84])
    assert report.steps_run == ["download", "clean", "chunk", "embed", "index"]
    assert report.status["index"] == {"ok": 2} and report.stopped_at is None
    total = sum(registry.get(k).chunking.chunks for k in ("gutenberg:1342", "gutenberg:84"))
    assert clients[0].count("chunks", exact=True).count == total
    clients[0].close()


def test_rerun_skips_what_is_done_and_adds_new_books(paths, catalog_file, tmp_path):
    _, _, clients = run(paths, catalog_file, tmp_path, [1342])
    clients[0].close()
    report, registry, clients = run(paths, catalog_file, tmp_path, [1342, 84])  # one more book
    assert report.status["index"] == {"ok": 2}
    assert registry.get("gutenberg:1342").index.status == "ok"
    clients[0].close()


def test_until_stops_after_a_step(paths, catalog_file, tmp_path):
    report, registry, _ = run(paths, catalog_file, tmp_path, [1342], until="chunk")
    assert report.steps_run == ["download", "clean", "chunk"]
    assert registry.get("gutenberg:1342").embedding is None


def test_qdrant_down_keeps_everything_up_to_the_vectors(paths, catalog_file, tmp_path):
    def down():
        raise IndexUnavailable("Cannot reach Qdrant")

    report, registry, _ = run(paths, catalog_file, tmp_path, [1342], client_factory=down)
    assert report.stopped_at == "index"
    assert registry.get("gutenberg:1342").embedding.status == "ok"
    assert report.status["index"] == {"not done": 1}


def test_cli_dry_run_downloads_nothing(paths, catalog_file, capsys):
    from growth_rag.sources.gutenberg.cli import main

    main(["--data-dir", str(paths.root), "add", "--title", "frankenstein", "--dry-run"])
    assert not paths.registry.exists() and not list(paths.manifests.glob("*.json"))


def test_cli_refuses_an_empty_selection(paths, catalog_file):
    from growth_rag.sources.gutenberg.cli import main

    with pytest.raises(SystemExit):
        main(["--data-dir", str(paths.root), "add"])
