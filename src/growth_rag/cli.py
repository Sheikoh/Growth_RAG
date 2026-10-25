"""`rag`: the source-independent steps, for every book in the registry.

    uv run rag chunk                       cleaned books -> chunks
    uv run rag embed                       chunks -> vectors on disk (bge-m3)
    uv run rag embed --backend flag        same with FlagEmbedding
    uv run rag embed-check --book 1524     compare the two backends on real chunks
    uv run rag index                       vectors -> Qdrant (docker compose up -d first)
    uv run rag index --recreate            rebuild both collections
    uv run rag search "a ship in a storm"  hybrid search (dense + sparse)
    uv run rag search "madness" --speaker OPHELIA --title Hamlet
    uv run rag search "the bishop" --title "les miserables"   # every volume
    uv run rag search "whaling" --author Melville --mode sparse
    uv run rag search "sea adventure" --books
    uv run rag eval-check                  check eval/eval-v2 against the chunked books
    uv run rag eval-check --show hamlet-ghost   the chunks that answer a question
    uv run rag eval                        score it: dense / sparse / hybrid, books (saved run)
    uv run rag eval --failures             ... and list what was not found
    uv run rag eval --run RUN --diagnose   where the answers are (rank buckets, misses)
    uv run rag eval --compare RUN_A RUN_B  what changed between two runs
    uv run rag model                       where the pinned model files are (download once)
    uv run rag model --rerank              same for the reranker
    uv run rag search "madness" --rerank   hybrid top 20 reordered by the cross-encoder
    uv run rag rerank-check                manual vs FlagEmbedding reranker on real chunks
    uv run rag eval --rerank 20 50         + modes chunks:rerank@20, chunks:rerank@50
    uv run rag status

Source-specific steps (catalog, select, download, clean) are in `gutenberg`.
Run from the stage folder (03_RAG) or pass --data-dir.
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from dataclasses import replace
from pathlib import Path

from growth_rag.chunking.run import TokenizerUnavailable
from growth_rag.chunking.store import iter_chunks
from growth_rag.cli_common import (
    add_chunk_command,
    add_common_args,
    add_selection_args,
    cmd_status,
    require_registry,
    run,
    selected_keys,
    start,
)
from growth_rag.config import DataPaths, Settings
from growth_rag.embedding.encoders import EncoderUnavailable, make_encoder
from growth_rag.registry import RegistryFormatError, make_key

log = logging.getLogger("growth_rag")


def _embedding_settings(args, settings: Settings):
    em = settings.embedding
    for field_name in ("backend", "batch_size", "device"):
        value = getattr(args, field_name, None)
        if value is not None:
            em = replace(em, **{field_name: value})
    return em


def _rerank_settings(args, settings: Settings):
    rr = settings.rerank
    if getattr(args, "rerank_backend", None):
        rr = replace(rr, backend=args.rerank_backend)
    return rr


def cmd_embed(args, settings: Settings, paths: DataPaths) -> None:
    from growth_rag.embedding.run import embed_books

    registry = require_registry(paths)
    if registry is None:
        return
    em = _embedding_settings(args, settings)
    summary = embed_books(registry, paths, make_encoder(em), em, selected_keys(args), force=args.force)
    if summary.interrupted:
        log.warning("Interrupted: registry saved after each book, re-run to resume.")
    log.info("Done: %s, %d chunks | vectors in %s", summary.by_status or "nothing to do",
             summary.chunks, paths.root / "vectors")


def cmd_embed_check(args, settings: Settings, paths: DataPaths) -> None:
    from growth_rag.embedding.check import compare_encoders

    registry = require_registry(paths)
    if registry is None:
        return
    key = make_key("gutenberg", args.book) if args.book else next(
        (r.key for r in registry if r.chunking and r.chunking.status == "ok"), None)
    record = registry.get(key) if key else None
    if record is None or not record.chunking or not record.chunking.path:
        log.error("No chunked book %s: run `rag chunk` first.", key or "")
        return
    chunks = list(iter_chunks(paths.resolve(record.chunking.path)))[: args.n]
    texts = [c.embedding_text() for c in chunks]
    em = _embedding_settings(args, settings)
    manual = make_encoder(replace(em, backend="manual"))
    flag = make_encoder(replace(em, backend="flag"))
    result = compare_encoders(manual, flag, texts)
    log.info("Compared on %d chunks of %s (%s)", result.texts, record.key, record.title)
    log.info("  dense cosine   min %.5f  mean %.5f   (expect > 0.999)",
             result.dense_cosine_min, result.dense_cosine_mean)
    log.info("  sparse tokens  Jaccard mean %.4f          (expect ~1.0)", result.sparse_jaccard_mean)
    log.info("  sparse score   max relative diff %.4f   (expect < 0.01)", result.sparse_score_rel_diff_max)


def _connect(settings: Settings):
    from growth_rag.index.qdrant import connect

    return connect(settings.qdrant)


def cmd_index(args, settings: Settings, paths: DataPaths) -> None:
    from growth_rag.index.run import index_books

    registry = require_registry(paths)
    if registry is None:
        return
    client = _connect(settings)
    summary = index_books(registry, paths, client, settings.qdrant, selected_keys(args),
                          force=args.force, recreate=args.recreate)
    if summary.interrupted:
        log.warning("Interrupted: registry saved, re-run the same command to resume.")
    log.info("Done: %s, %d points | collections %s / %s", summary.by_status or "nothing to do",
             summary.points, settings.qdrant.chunks_collection, settings.qdrant.books_collection)


def _book_filter(args, paths: DataPaths) -> tuple[list[str] | None, bool]:
    """Book keys from --book / --title / --author. Returns (keys, ok); ok is
    False when a title/author was given but matched nothing."""
    from growth_rag.index.resolve import resolve_books

    keys = [make_key("gutenberg", b) for b in args.book] if args.book else []
    if args.title or args.author:
        registry = require_registry(paths)
        if registry is None:
            return None, False
        res = resolve_books(registry, title=args.title, author=args.author, languages=args.language)
        if not res.books:
            what = " / ".join(x for x in (args.title, args.author) if x)
            hint = f" Closest titles: {'; '.join(res.suggestions)}" if res.suggestions else ""
            log.warning("No book matches %r.%s", what, hint)
            return None, False
        not_indexed = [b for b in res.books if not (b.index and b.index.status == "ok")]
        log.info("Filtering on %d book(s):", len(res.books))
        for b in res.books:
            note = "  (not indexed: run `rag index`)" if b in not_indexed else ""
            log.info("  %s  %s [%s]%s", b.key, b.title[:70], ", ".join(b.languages), note)
        keys += res.keys
    return (sorted(set(keys)) or None), True


def cmd_search(args, settings: Settings, paths: DataPaths) -> None:
    from growth_rag.index.search import build_filter, search_books, search_chunks

    book_keys, ok = _book_filter(args, paths)
    if not ok:
        return
    client = _connect(settings)
    encoder = make_encoder(_embedding_settings(args, settings))
    if args.books:
        hits = search_books(client, settings.qdrant, encoder, args.query, vector=args.vector,
                            k=args.k, languages=args.language, book_keys=book_keys)
        for rank, h in enumerate(hits, 1):
            p = h.payload
            print(f"{rank:>2}. {h.score:.3f}  {p['title'][:60]} — {p.get('authors_display', '')} "
                  f"[{', '.join(p.get('languages', []))}]  ({p['key']})")
        return
    speakers = [s.upper() for s in args.speaker] if args.speaker else None
    flt = build_filter(languages=args.language, book_keys=book_keys, kinds=args.kind, speaker_labels=speakers)
    pool = args.rerank if args.rerank else args.k
    hits = search_chunks(client, settings.qdrant, encoder, args.query, mode=args.mode, k=pool, query_filter=flt)
    if args.rerank:
        from growth_rag.rerank import make_reranker, rerank
        from growth_rag.rerank.base import sigmoid

        before = {h.payload["key"]: r for r, h in enumerate(hits, 1)}
        hits = rerank(make_reranker(_rerank_settings(args, settings)), args.query, hits)[:args.k]
        log.info("Reranked the %s top %d (score: reranker logit, sigmoid in brackets; was: rank before)",
                 args.mode, pool)
    for rank, h in enumerate(hits, 1):
        p = h.payload
        where = " / ".join(p.get("section_path", []))
        who = f"  speakers: {', '.join(p['speaker_labels'])}" if p.get("speaker_labels") else ""
        score = (f"{h.score:.2f} ({sigmoid(h.score):.2f}, was {before[p['key']]})" if args.rerank
                 else f"{h.score:.3f}")
        print(f"\n{rank:>2}. {score}  {p['title'][:50]} — {where[:60]}  [{p['kind']}, {p['language']}]{who}")
        print(textwrap.indent(textwrap.shorten(p["text"].replace("\n", " "), width=320), "    "))
        print(f"    ({p['key']}, characters {p['start']}–{p['end']})")


def cmd_eval_check(args, settings: Settings, paths: DataPaths) -> None:
    from growth_rag.evaluation.check import check_questions, coverage
    from growth_rag.evaluation.dataset import load_questions
    from growth_rag.sources.gutenberg.selection import Manifest

    registry = require_registry(paths)
    if registry is None:
        return
    folder = Path(args.set)
    questions = load_questions(folder / "questions.jsonl")
    manifest_path = folder / "manifest.json"
    manifest_ids = None
    if manifest_path.exists():
        manifest = Manifest.load(manifest_path)
        manifest_ids = {make_key(manifest.source, i) for i in manifest.ids}
        missing = sorted(manifest_ids - {r.key for r in registry})
        if missing:
            log.warning("%d manifest book(s) not in the registry: uv run gutenberg add --manifest %s",
                        len(missing), manifest_path)
    report = check_questions(questions, registry, paths, manifest_ids)

    if args.show:
        _show_targets(args.show, questions, report, registry, paths)
        return
    passages = [q for q in questions if q.task == "passage"]
    n_chunks = [len({c for t in report.targets[q.id] for c in t.chunks}) for q in passages if q.answerable]
    log.info("%s: %d questions (%d passage, %d book); %d relevant chunks in total, %.1f per answerable question",
             folder.name, len(questions), len(passages), len(questions) - len(passages),
             sum(n_chunks), sum(n_chunks) / max(len(n_chunks), 1))
    for name, counts in coverage(questions, registry).items():
        log.info("  %-15s %s", name, ", ".join(f"{k}={v}" for k, v in counts.most_common()))
    for level in ("error", "warning", "expected"):
        found = [p for p in report.problems if p.level == level]
        if found:
            title = {"error": "Errors (fix the dataset or the data)", "warning": "Warnings",
                     "expected": "Known gaps (declared in the questions)"}[level]
            log.log(logging.ERROR if level == "error" else logging.INFO, "%s: %d", title, len(found))
            for p in found:
                log.info("  %-24s %s", p.question, p.message)
    if not report.errors:
        log.info("OK: every evidence quote was found exactly once.")


def _show_targets(ids, questions, report, registry, paths) -> None:
    """Print the ground truth of some questions: the chunks that answer them."""
    from growth_rag.evaluation.check import BookStore

    store = BookStore(registry, paths)
    by_id = {q.id: q for q in questions}
    for qid in ids:
        q = by_id.get(qid)
        if q is None:
            log.warning("No question %r", qid)
            continue
        print(f"\n[{q.id}] ({q.task}, {q.language}) {q.query}")
        if q.filters:
            print(f"  filters: {q.filters.model_dump(exclude_none=True)}")
        if q.known_gap:
            print(f"  known gap: {q.known_gap}")
        if not q.relevant:
            print(f"  no answer in the corpus. {q.notes or ''}")
        for t in report.targets.get(q.id, []):
            record = registry.get(t.book)
            title = record.title[:60] if record else t.book
            if q.task == "book":
                print(f"  grade {t.grade}  {t.book}  {title}")
                continue
            loaded = store.get(t.book)
            chunks = {c.key: c for c in loaded[2]} if loaded else {}
            where = f"annotation {t.annotation}" if t.annotation else f"characters {t.span[0]}–{t.span[1]}"
            print(f"  grade {t.grade}  {title} — {where}")
            for key in t.chunks:
                c = chunks[key]
                print(f"    {key}  [{' / '.join(c.section_path)[:60]}]  {c.kind}, {c.tokens} tokens")
                print("      " + _excerpt(c, t.span))


def _excerpt(chunk, span: tuple[int, int] | None, width: int = 300) -> str:
    """The part of the chunk around the quote (the chunk start for footnotes).
    Sliced, not word-wrapped: also works for scripts without spaces."""
    centre = (max(span[0], chunk.start) - chunk.start) if span else 0
    begin = max(0, centre - width // 3)
    text = chunk.text[begin:begin + width].replace("\n", " ")
    return ("…" if begin else "") + text + ("…" if begin + width < len(chunk.text) else "")


def cmd_eval(args, settings: Settings, paths: DataPaths) -> None:
    from growth_rag.evaluation.check import check_questions
    from growth_rag.evaluation.dataset import load_questions
    from growth_rag.evaluation.report import compare_lines, failure_lines, overview_lines, slice_lines
    from growth_rag.evaluation.run import EvalRun, build_run, default_run_path, run_eval
    from growth_rag.sources.gutenberg.selection import Manifest

    runs_dir = paths.root / "eval_runs"
    if args.compare:
        a, b = (EvalRun.load(_run_path(p, runs_dir)) for p in args.compare)
        for line in compare_lines(a, b, k=args.k):
            print(line)
        return

    registry = require_registry(paths)
    if registry is None:
        return
    folder = Path(args.set)
    questions = load_questions(folder / "questions.jsonl")
    manifest_ids = None
    if (folder / "manifest.json").exists():
        manifest = Manifest.load(folder / "manifest.json")
        manifest_ids = {make_key(manifest.source, i) for i in manifest.ids}
    else:
        log.warning("No manifest.json in %s: searching every indexed book.", folder)

    report = None
    if args.run:  # re-read a saved run: no search, no model
        run = EvalRun.load(_run_path(args.run, runs_dir))
    else:
        report = check_questions(questions, registry, paths, manifest_ids)
        skip = {p.question for p in report.errors}
        if skip:
            log.warning("%d question(s) with dataset errors are not scored (see `rag eval-check`): %s",
                        len(skip), ", ".join(sorted(skip)))
        em = _embedding_settings(args, settings)
        settings = replace(settings, embedding=em)
        log.info("Encoding %d questions, then searching (%s; %s)...", len(questions),
                 ", ".join(args.modes), ", ".join(args.book_vectors))
        reranker, extra = None, {"restricted_to_manifest": manifest_ids is not None}
        if args.rerank:
            from growth_rag.rerank import make_reranker

            rr = _rerank_settings(args, settings)
            reranker = make_reranker(rr)
            extra["rerank"] = {"model": rr.model, "revision": rr.revision, "backend": rr.backend,
                               "candidates_from": "hybrid", "pools": args.rerank}
            log.info("Then reranking the hybrid top %s with %s.", " / ".join(map(str, args.rerank)), rr.model)
        modes = run_eval(questions, report, registry, manifest_ids, _connect(settings), settings,
                         make_encoder(em), chunk_modes=args.modes, book_vectors=args.book_vectors,
                         depth=args.depth, skip=skip, reranker=reranker, rerank_pools=args.rerank or ())
        run = build_run(folder, folder / "questions.jsonl", settings, modes, args.depth, extra)
        if not args.no_save:
            out = default_run_path(runs_dir, folder.name)
            run.save(out)
            log.info("Run saved: %s", out)

    print()
    for line in overview_lines(run):
        print(line)
    for task in ("chunks", "books"):
        lines = slice_lines(run, questions, registry, metric=args.metric, task=task)
        if lines:
            print()
            for line in lines:
                print(line)
    if args.failures:
        print()
        for line in failure_lines(run, questions, args.failures, k=10):
            print(line)
    if args.diagnose:
        from growth_rag.evaluation.diagnose import diagnose_lines, with_targets
        from growth_rag.evaluation.run import entries_for

        if any(not r.targets for m in run.modes for r in m.questions.values() if r.metrics is not None):
            # Run saved before targets were stored: take them from the current data.
            report = report or check_questions(questions, registry, paths, manifest_ids)
            with_targets(run, {q.id: [sorted(e.keys) for e in entries_for(q, report)] for q in questions})
            log.info("Targets taken from a check of the current data (valid if the chunking is unchanged).")
        print()
        for line in diagnose_lines(run, questions, mode=args.diagnose, k=10):
            print(line)


def _run_path(value: str, runs_dir: Path) -> Path:
    """A run file path, or a name in data/eval_runs (with or without .json)."""
    path = Path(value)
    if path.exists():
        return path
    candidate = runs_dir / (value if value.endswith(".json") else f"{value}.json")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No run file {value!r} (looked in {runs_dir})")


def cmd_model(args, settings: Settings, paths: DataPaths) -> None:
    """Locate (or download once) the pinned model files; later loads are local."""
    from growth_rag.utils.hf import folder_size

    if args.rerank:
        from growth_rag.rerank.files import PATTERNS, model_folder

        cfg, who = _rerank_settings(args, settings), "Both rerankers load"
    else:
        from growth_rag.embedding.encoders.files import PATTERNS, model_folder

        cfg, who = settings.embedding, "Both encoders and the tokenizer load"
    folder = model_folder(cfg)
    log.info("%s at revision %s", cfg.model, cfg.revision or "main")
    log.info("  folder: %s", folder)
    for name, size in folder_size(folder, PATTERNS).items():
        log.info("  %-26s %9.1f MB", name, size / 1e6)
    log.info("%s from this folder: no Hugging Face request once it is complete.", who)


def cmd_rerank_check(args, settings: Settings, paths: DataPaths) -> None:
    """Manual reranker vs FlagEmbedding on real chunks of one book."""
    from growth_rag.rerank.check import compare_rerankers
    from growth_rag.rerank.flag import FlagReranker
    from growth_rag.rerank.manual import ManualReranker

    registry = require_registry(paths)
    if registry is None:
        return
    record = registry.get(make_key("gutenberg", args.book))
    if record is None or not record.chunking or not record.chunking.path:
        log.error("No chunked book gutenberg:%s: run `rag chunk` first.", args.book)
        return
    chunks = list(iter_chunks(paths.resolve(record.chunking.path)))
    passages = [c.embedding_text() for c in chunks[: args.n]]
    # Questions: the first sentence-ish of chunks spread over the book (one is in the passages).
    step = max(1, len(chunks) // args.questions)
    cases = [(chunks[i].text[:160], passages + [chunks[i].embedding_text()])
             for i in range(0, step * args.questions, step)]
    rr = settings.rerank
    result = compare_rerankers(ManualReranker(rr), FlagReranker(rr), cases)
    log.info("Compared on %d questions x %d passages of %s (%s)", result.questions,
             result.pairs // max(result.questions, 1), record.key, record.title)
    log.info("  max |logit difference|   %.4f   (expect < 0.05 in fp16)", result.max_abs_diff)
    log.info("  rank correlation (min)   %.4f   (expect > 0.99)", result.spearman_min)
    log.info("  same best passage        %d/%d", result.same_top1, result.questions)


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, settings)
    sub = parser.add_subparsers(dest="command", required=True)

    add_chunk_command(sub, source=None)

    def embedding_options(p):
        p.add_argument("--backend", choices=["manual", "flag"], help="default from settings (manual)")
        p.add_argument("--batch-size", type=int, help=f"default {settings.embedding.batch_size}")
        p.add_argument("--device", help='"cuda", "cpu" or "auto" (default)')

    p = sub.add_parser("embed", help="embed chunks (dense + sparse) and book vectors, write them to disk")
    add_selection_args(p)
    embedding_options(p)
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("embed-check", help="compare the manual and FlagEmbedding encoders on real chunks")
    p.add_argument("--book", type=int, help="Gutenberg book number (default: first chunked book)")
    p.add_argument("--n", type=int, default=32, help="number of chunks to compare")
    embedding_options(p)
    p.set_defaults(func=cmd_embed_check)

    p = sub.add_parser("index", help="upload vectors to Qdrant (chunks + books collections)")
    add_selection_args(p)
    p.add_argument("--recreate", action="store_true", help="delete and rebuild both collections first")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="query the index (to check it; retrieval proper comes next)")
    p.add_argument("query")
    p.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--language", nargs="+", help="filter on chunk language, e.g. fr en")
    p.add_argument("--book", type=int, action="append", help="filter on a Gutenberg book number (repeatable)")
    p.add_argument("--title", help='filter on books whose title contains these words, e.g. "les miserables"')
    p.add_argument("--author", help="filter on books by this author, e.g. Melville")
    p.add_argument("--kind", nargs="+", choices=["prose", "poem", "play"])
    p.add_argument("--speaker", nargs="+", help="speaker labels, e.g. OPHELIA (plays)")
    p.add_argument("--rerank", nargs="?", type=int, const=settings.rerank.candidates, metavar="N",
                   help=f"rerank the top N with the cross-encoder, show the top k (N default {settings.rerank.candidates})")
    p.add_argument("--rerank-backend", choices=["manual", "flag"])
    p.add_argument("--books", action="store_true", help="search books instead of chunks")
    p.add_argument("--vector", choices=["metadata", "centroid"], default="metadata",
                   help="book vector to use with --books")
    embedding_options(p)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("eval-check", help="check an evaluation set against the cleaned and chunked books")
    p.add_argument("--set", default="eval/eval-v2", help="evaluation set folder (default: eval/eval-v2)")
    p.add_argument("--show", nargs="+", metavar="ID", help="print the chunks that answer these questions")
    p.set_defaults(func=cmd_eval_check)

    p = sub.add_parser("eval", help="score an evaluation set against the index (saved as a run)")
    p.add_argument("--set", default="eval/eval-v2", help="evaluation set folder (default: eval/eval-v2)")
    p.add_argument("--modes", nargs="+", choices=["dense", "sparse", "hybrid"], default=["dense", "sparse", "hybrid"])
    p.add_argument("--book-vectors", nargs="*", choices=["metadata", "centroid"], default=["metadata", "centroid"])
    p.add_argument("--depth", type=int, default=20, help="results kept per question (default 20)")
    p.add_argument("--metric", default="hit@5",
                   choices=["hit@1", "hit@3", "hit@5", "hit@10", "recall@10", "rr", "ndcg@10"],
                   help="metric of the by-slice tables (default hit@5)")
    p.add_argument("--failures", nargs="?", const="chunks:hybrid", metavar="MODE",
                   help="list questions not found in the top 10 (default mode chunks:hybrid)")
    p.add_argument("--diagnose", nargs="?", const="chunks:hybrid", metavar="MODE",
                   help="where the answers are: rank buckets, dense vs sparse, misses (default mode chunks:hybrid)")
    p.add_argument("--run", help="show a saved run instead of searching (file or name in data/eval_runs)")
    p.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="compare two saved runs")
    p.add_argument("--k", type=int, default=5, help="k of the hit@k changes listed by --compare")
    p.add_argument("--no-save", action="store_true", help="do not write the run file")
    p.add_argument("--rerank", nargs="+", type=int, metavar="N",
                   help="add modes chunks:rerank@N: the hybrid top N reordered by the cross-encoder (e.g. 20 50)")
    p.add_argument("--rerank-backend", choices=["manual", "flag"])
    embedding_options(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("model", help="locate (or download once) the pinned model files")
    p.add_argument("--rerank", action="store_true", help="the reranker's files instead of bge-m3's")
    p.set_defaults(func=cmd_model)

    p = sub.add_parser("rerank-check", help="compare the manual and FlagEmbedding rerankers on real chunks")
    p.add_argument("--book", type=int, default=1524, help="Gutenberg book number (default 1524, Hamlet)")
    p.add_argument("--n", type=int, default=16, help="passages per question")
    p.add_argument("--questions", type=int, default=4)
    p.set_defaults(func=cmd_rerank_check)

    p = sub.add_parser("status", help="summarise the registry")
    p.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> None:
    from growth_rag.index.qdrant import IndexUnavailable

    settings = Settings.from_env()
    args = build_parser(settings).parse_args(argv)
    paths = start(args, "rag")
    from growth_rag.evaluation.dataset import EvalFormatError
    from growth_rag.rerank import RerankerUnavailable
    from growth_rag.utils.hf import ModelFilesUnavailable

    run(args, settings, paths, (RegistryFormatError, FileNotFoundError, TokenizerUnavailable, EncoderUnavailable,
                                IndexUnavailable, EvalFormatError, ModelFilesUnavailable, RerankerUnavailable))


if __name__ == "__main__":
    main()
