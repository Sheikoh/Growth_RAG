"""Chunking step over the registry (any source): cleaned outputs -> chunks JSONL.

A book is (re)chunked when it has never been chunked, when the chunking rules
(CHUNKING_VERSION), the tokenizer or the size settings changed, when it was
re-cleaned since, when its output is missing, or with --force.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from growth_rag.chunking import CHUNKING_VERSION
from growth_rag.chunking.chunker import BookMeta, ChunkSettings, chunk_book
from growth_rag.chunking.store import save_chunks
from growth_rag.chunking.tokens import TokenCounter
from growth_rag.cleaning.store import load_cleaned
from growth_rag.config import ChunkingSettings, DataPaths
from growth_rag.registry import Author, BookRecord, ChunkingInfo, Registry
from growth_rag.utils.io import utc_now

log = logging.getLogger(__name__)


class TokenizerUnavailable(RuntimeError):
    def __init__(self, name: str, cause: Exception):
        super().__init__(
            f"Cannot load tokenizer {name!r} ({type(cause).__name__}: {cause}). It is downloaded "
            "from Hugging Face on first use: check the connection and that `transformers` is "
            "installed (uv sync), or use --tokenizer approx-4cpt for a quick trial."
        )


@dataclass
class ChunkSummary:
    chunked: int = 0
    skipped: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    chunks: int = 0
    interrupted: bool = False


def display_authors(authors: list[Author], limit: int = 2) -> str:
    """Main authors (no translators/illustrators) as 'First Last'."""
    names = []
    for a in authors:
        if a.role is not None:
            continue
        parts = [p.strip() for p in a.name.split(",")]
        names.append(f"{parts[1]} {parts[0]}" if len(parts) == 2 and parts[1] else a.name)
    return ", ".join(names[:limit])


def needs_chunking(record: BookRecord, paths: DataPaths, settings: ChunkingSettings, force: bool = False) -> bool:
    cleaning = record.cleaning
    if cleaning is None or cleaning.status == "error" or not cleaning.text_path:
        return False  # nothing to chunk yet
    info = record.chunking
    if force or info is None or info.status == "error":
        return True
    changed = (
        info.version != CHUNKING_VERSION
        or info.tokenizer != settings.tokenizer
        or (info.target_tokens, info.max_tokens, info.min_tokens)
        != (settings.target_tokens, settings.max_tokens, settings.min_tokens)
        # Re-cleaned with a different result. A re-clean producing the same
        # text (e.g. a rules change that did not affect this book) does not
        # require re-chunking.
        or info.cleaned_text_sha256 != cleaning.text_sha256
    )
    return changed or not (info.path and paths.resolve(info.path).exists())


def chunk_record(record: BookRecord, paths: DataPaths, counter: TokenCounter,
                 settings: ChunkingSettings) -> ChunkingInfo:
    """Chunk one cleaned book. Never raises: errors are returned as status 'error'."""
    cleaning = record.cleaning
    base = dict(version=CHUNKING_VERSION, tokenizer=counter.name,
                target_tokens=settings.target_tokens, max_tokens=settings.max_tokens,
                min_tokens=settings.min_tokens, cleaned_text_sha256=cleaning.text_sha256)
    try:
        text, book = load_cleaned(paths.resolve(cleaning.text_path), paths.resolve(cleaning.structure_path))
        chunks = chunk_book(
            text, book,
            BookMeta(key=record.key, title=record.title, authors=display_authors(record.authors)),
            counter,
            ChunkSettings(target_tokens=settings.target_tokens, max_tokens=settings.max_tokens,
                          min_tokens=settings.min_tokens),
        )
        stem = PurePosixPath(cleaning.text_path).stem
        path = paths.chunks_dir(record.source) / f"{stem}.jsonl"
        save_chunks(path, chunks)
        tokens = [c.tokens for c in chunks] or [0]
        return ChunkingInfo(
            status="ok", chunked_at=utc_now(), path=paths.relative(path), chunks=len(chunks),
            tokens_total=sum(tokens), tokens_median=int(statistics.median(tokens)),
            tokens_max=max(tokens), by_kind=dict(Counter(c.kind for c in chunks)), **base,
        )
    except Exception as exc:  # one bad book must not stop the run
        log.exception("Chunking failed for %s", record.key)
        return ChunkingInfo(status="error", chunked_at=utc_now(), error=f"{type(exc).__name__}: {exc}", **base)


def chunk_books(
    registry: Registry,
    paths: DataPaths,
    counter: TokenCounter,
    settings: ChunkingSettings,
    keys: Iterable[str] | None = None,
    *,
    source: str | None = None,
    force: bool = False,
    save_every: int = 10,
) -> ChunkSummary:
    wanted = set(keys) if keys is not None else None
    records = [
        r for r in registry
        if (source is None or r.source == source) and (wanted is None or r.key in wanted)
        and r.cleaning is not None and r.cleaning.status != "error"
    ]
    todo = [r for r in records if needs_chunking(r, paths, settings, force)]
    summary = ChunkSummary(skipped=len(records) - len(todo))
    if todo:
        # Load the tokenizer once, before the loop: if it cannot be loaded
        # (no internet, package missing), every book would fail the same way.
        try:
            counter.count(["warm-up"])
        except Exception as exc:
            raise TokenizerUnavailable(counter.name, exc) from exc
    log.info("%d cleaned books | %d up to date | %d to chunk (tokenizer: %s)",
             len(records), summary.skipped, len(todo), counter.name)
    try:
        for i, record in enumerate(todo, 1):
            info = chunk_record(record, paths, counter, settings)
            record.chunking = info
            summary.chunked += 1
            summary.chunks += info.chunks or 0
            summary.by_status[info.status] = summary.by_status.get(info.status, 0) + 1
            log.info("[%d/%d] %-16s %-5s chunks=%-5s median=%-4s %s %s", i, len(todo), record.key,
                     info.status, info.chunks, info.tokens_median, record.title[:40], info.error or "")
            if i % save_every == 0:
                registry.save()
    except KeyboardInterrupt:
        summary.interrupted = True
    finally:
        registry.save()
    return summary
