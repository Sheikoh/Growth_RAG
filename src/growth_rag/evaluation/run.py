"""Score an evaluation set against the index: `rag eval`.

    questions --encode once (one batch)--> query vectors
              --for each mode: Qdrant search with the question's filters,
                restricted to the set's manifest books--> top `depth` keys
              --metrics.py against the targets of check.py--> scores

Modes: "chunks:dense", "chunks:sparse", "chunks:hybrid" for passage
questions; "chunks:rerank@N" (hybrid top N reordered by the cross-encoder,
`--rerank N`); "books:metadata", "books:centroid" for recommendations.

Every run is saved as JSON (settings, and per question the keys returned,
the ranks and the metrics), so that two runs can be compared after a change
(`rag eval --compare`), without querying again.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from growth_rag.config import Settings
from growth_rag.embedding.encoders.base import Encoder
from growth_rag.embedding.run import encode_sorted
from growth_rag.evaluation.check import CheckReport
from growth_rag.evaluation.dataset import Question
from growth_rag.evaluation.metrics import Entry, entry_ranks, score
from growth_rag.index.resolve import resolve_books
from growth_rag.index.search import build_filter, query_books, query_chunks
from growth_rag.registry import Registry
from growth_rag.utils.io import atomic_write_json, read_json, utc_now

CHUNK_MODES = ("dense", "sparse", "hybrid")
BOOK_VECTORS = ("metadata", "centroid")
DEPTH = 20  # results kept per question: enough for k = 10 and for a reranker later


class QuestionRun(BaseModel):
    results: list[str]  # keys returned, best first
    labels: list[str] = []  # "title — section" of the first results, to read failures
    ranks: list[int | None] = []  # per relevant entry: rank of the first result satisfying it
    targets: list[list[str]] = []  # per relevant entry: the keys that satisfy it (for --diagnose)
    grades: list[int] = []  # per relevant entry: 2 answer, 1 partial / related
    metrics: dict[str, float] | None = None  # None: not scored (no answer in the corpus, or dataset error)


class ModeRun(BaseModel):
    mode: str  # "chunks:hybrid", "books:metadata", ...
    questions: dict[str, QuestionRun]


class EvalRun(BaseModel):
    set_name: str
    created_at: datetime
    questions_sha256: str
    settings: dict
    depth: int
    modes: list[ModeRun]

    def save(self, path: Path) -> None:
        atomic_write_json(path, self.model_dump(mode="json"))

    @classmethod
    def load(cls, path: Path) -> EvalRun:
        return cls.model_validate(read_json(path))

    def mode(self, name: str) -> ModeRun | None:
        return next((m for m in self.modes if m.mode == name), None)


def run_eval(
    questions: list[Question],
    report: CheckReport,
    registry: Registry,
    manifest_keys: set[str] | None,
    client,
    settings: Settings,
    encoder: Encoder,
    *,
    chunk_modes=CHUNK_MODES,
    book_vectors=BOOK_VECTORS,
    depth: int = DEPTH,
    skip: set[str] = frozenset(),
    reranker=None,
    rerank_pools=(),
) -> list[ModeRun]:
    """`skip`: question ids not to score (dataset errors found by the check).
    `rerank_pools`: for each N, a mode "chunks:rerank@N" where the hybrid top N
    is reordered by `reranker` (results are then N long)."""
    vectors = encode_sorted(encoder, [q.query for q in questions], settings.embedding.batch_size)
    titles = {r.key: r.title for r in registry}
    runs: list[ModeRun] = []

    passages = [(i, q) for i, q in enumerate(questions) if q.task == "passage"]
    for mode in chunk_modes:
        out = {}
        for i, q in passages:
            flt, possible = question_filter(q, registry, manifest_keys)
            hits = query_chunks(client, settings.qdrant, vectors.dense[i], vectors.sparse[i], mode, depth,
                                flt) if possible else []
            out[q.id] = _scored(q, [h.payload["key"] for h in hits], _labels(hits, titles), report, skip)
        runs.append(ModeRun(mode=f"chunks:{mode}", questions=out))

    if reranker is not None:
        from growth_rag.rerank import rerank

        for pool in rerank_pools:
            out = {}
            for i, q in passages:
                flt, possible = question_filter(q, registry, manifest_keys)
                hits = query_chunks(client, settings.qdrant, vectors.dense[i], vectors.sparse[i], "hybrid", pool,
                                    flt) if possible else []
                hits = rerank(reranker, q.query, hits)
                out[q.id] = _scored(q, [h.payload["key"] for h in hits], _labels(hits, titles), report, skip)
            runs.append(ModeRun(mode=f"chunks:rerank@{pool}", questions=out))

    books = [(i, q) for i, q in enumerate(questions) if q.task == "book"]
    restrict = sorted(manifest_keys) if manifest_keys else None
    for vector in book_vectors:
        out = {}
        for i, q in books:
            hits = query_books(client, settings.qdrant, vectors.dense[i], vector, depth + len(q.seed_books),
                               languages=q.filters.language if q.filters else None, book_keys=restrict)
            keys = [h.payload["key"] for h in hits if h.payload["key"] not in q.seed_books][:depth]
            out[q.id] = _scored(q, keys, [titles.get(k, k)[:60] for k in keys[:3]], report, skip)
        runs.append(ModeRun(mode=f"books:{vector}", questions=out))
    return runs


def _labels(hits, titles: dict[str, str]) -> list[str]:
    """'book title — section path' of the first three results (to read failures)."""
    return [f"{titles.get(h.payload['book_key'], h.payload['book_key'])[:40]} — "
            f"{' / '.join(h.payload.get('section_path', []))[:50]}" for h in hits[:3]]


def question_filter(q: Question, registry: Registry, manifest_keys: set[str] | None):
    """The `rag search` filters of a question, restricted to the manifest's
    books. Returns (filter or None, possible): possible is False when the
    filters leave no book at all (then no search is made)."""
    f = q.filters
    books = set(manifest_keys) if manifest_keys else None
    if f and (f.title or f.author):
        found = set(resolve_books(registry, title=f.title, author=f.author, languages=f.language).keys)
        books = found if books is None else books & found
    if f and f.book:
        books = set(f.book) if books is None else books & set(f.book)
    if books is not None and not books:
        return None, False
    flt = build_filter(
        languages=f.language if f else None,
        book_keys=sorted(books) if books is not None else None,
        kinds=f.kind if f else None,
        speaker_labels=[s.upper() for s in f.speaker] if f and f.speaker else None,
    )
    return flt, True


def entries_for(q: Question, report: CheckReport) -> list[Entry]:
    targets = report.targets.get(q.id, [])
    if q.task == "book":
        return [Entry(frozenset({t.book}), t.grade) for t in targets]
    return [Entry(frozenset(t.chunks), t.grade) for t in targets]


def _scored(q: Question, results: list[str], labels: list[str], report: CheckReport, skip) -> QuestionRun:
    if not q.answerable or q.id in skip:
        return QuestionRun(results=results, labels=labels)
    entries = entries_for(q, report)
    return QuestionRun(results=results, labels=labels, ranks=entry_ranks(results, entries),
                       targets=[sorted(e.keys) for e in entries], grades=[e.grade for e in entries],
                       metrics=score(results, entries))


def build_run(set_folder: Path, questions_file: Path, settings: Settings, modes: list[ModeRun],
              depth: int, extra: dict) -> EvalRun:
    from growth_rag.chunking import CHUNKING_VERSION
    from growth_rag.cleaning import CLEANING_VERSION

    em = settings.embedding
    return EvalRun(
        set_name=set_folder.name,
        created_at=utc_now(),
        questions_sha256=hashlib.sha256(questions_file.read_bytes()).hexdigest(),
        settings={
            "embedding": {"model": em.model, "revision": em.revision, "backend": em.backend},
            "chunking": {"tokenizer": settings.chunking.tokenizer, "version": CHUNKING_VERSION,
                         "target_tokens": settings.chunking.target_tokens},
            "cleaning_version": CLEANING_VERSION,
            "qdrant": settings.qdrant.path or settings.qdrant.url,
            **extra,
        },
        depth=depth,
        modes=modes,
    )


def default_run_path(runs_dir: Path, set_name: str) -> Path:
    return runs_dir / f"{set_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
