"""Pieces shared by the command lines (`gutenberg`, `rag`): data folder and
logging options, book selection, error handling, and the generic commands that
both expose (status, chunk)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from growth_rag.config import DataPaths, Settings
from growth_rag.registry import Registry, make_key
from growth_rag.utils.log import setup_logging

log = logging.getLogger("growth_rag")


def add_common_args(parser: argparse.ArgumentParser, settings: Settings) -> None:
    parser.add_argument("--data-dir", default=str(settings.data_dir),
                        help=f"root data folder (default: {settings.data_dir})")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def add_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", help="only the books of this manifest")
    p.add_argument("--book", type=int, action="append",
                   help="only this Gutenberg book number (repeatable)")
    p.add_argument("--key", action="append", help='only this registry key, e.g. "gutenberg:1524" (repeatable)')
    p.add_argument("--force", action="store_true", help="redo even if up to date")


def selected_keys(args, default_source: str = "gutenberg") -> list[str] | None:
    """None = every book; otherwise the union of --manifest, --book and --key."""
    keys: list[str] | None = None
    if getattr(args, "manifest", None):
        from growth_rag.sources.gutenberg.selection import Manifest

        manifest = Manifest.load(Path(args.manifest))
        keys = [make_key(manifest.source, i) for i in manifest.ids]
    if getattr(args, "book", None):
        keys = (keys or []) + [make_key(default_source, i) for i in args.book]
    if getattr(args, "key", None):
        keys = (keys or []) + list(args.key)
    return keys


def start(args, log_name: str, creates_data: bool = False) -> DataPaths:
    """Logging to console + data/logs/<log_name>.log. No log file where there is
    no data folder yet, so a wrong working directory leaves no stray folders."""
    paths = DataPaths(args.data_dir)
    log_file = paths.logs / f"{log_name}.log" if paths.root.exists() or creates_data else None
    setup_logging(args.log_level, log_file=log_file)
    return paths


def run(args, settings: Settings, paths: DataPaths, known_errors: tuple[type[Exception], ...]) -> None:
    try:
        args.func(args, settings, paths)
    except known_errors as exc:
        log.error("%s", exc)
        sys.exit(1)


def require_registry(paths: DataPaths) -> Registry | None:
    if not paths.registry.exists():
        log.warning("No registry at %s. Run the command from the stage folder (03_RAG) "
                    "or pass --data-dir.", paths.registry.resolve())
        return None
    return Registry.load(paths.registry)


def cmd_status(args, settings: Settings, paths: DataPaths) -> None:
    registry = require_registry(paths)
    if registry is not None:
        print(json.dumps(registry.summary(), indent=2, ensure_ascii=False))


def add_chunk_command(sub, source: str | None) -> None:
    p = sub.add_parser("chunk", help="cut cleaned books into chunks for embedding")
    add_selection_args(p)
    p.add_argument("--tokenizer", help='tokenizer name (default from settings; "approx-4cpt" = no download)')
    p.set_defaults(func=cmd_chunk, source=source)


def cmd_chunk(args, settings: Settings, paths: DataPaths) -> None:
    from growth_rag.chunking.run import chunk_books
    from growth_rag.chunking.tokens import make_counter

    registry = require_registry(paths)
    if registry is None:
        return
    chunking = settings.chunking
    if args.tokenizer:
        chunking = replace(chunking, tokenizer=args.tokenizer)
    summary = chunk_books(registry, paths, make_counter(chunking.tokenizer), chunking,
                          selected_keys(args), source=args.source, force=args.force)
    if summary.interrupted:
        log.warning("Interrupted: registry saved, re-run the same command to resume.")
    log.info("Done: %s, %d chunks", summary.by_status or "nothing to do", summary.chunks)
