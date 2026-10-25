"""Command line for the Gutenberg source. Argument handling only: every command
delegates to the functions of catalog / selection / download / migrate.

    uv run gutenberg add --title "moby dick" --languages en --dry-run   # everything in one go
    uv run gutenberg catalog
    uv run gutenberg select --name pilot --languages en fr de it fi \
        --bookshelf "Category: Novels" --per-language 20
    uv run gutenberg select --name hugo_fr --author "Hugo, Victor" --languages fr
    uv run gutenberg download --manifest data/manifests/pilot.json
    uv run gutenberg clean                                    # every downloaded book
    uv run gutenberg clean --manifest data/manifests/pilot.json
    uv run gutenberg clean --book 1524 --book 2229 --force
    uv run gutenberg chunk                                    # every cleaned book
    uv run gutenberg chunk --manifest data/manifests/pilot.json --tokenizer approx-4cpt
    uv run gutenberg status
    uv run gutenberg migrate        # once, for data written by gutenberg_acquire.py

The data folder defaults to ./data (relative to where you run the command), or
GROWTH_RAG_DATA_DIR, or --data-dir.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from growth_rag.chunking.run import TokenizerUnavailable
from growth_rag.cli_common import (
    add_chunk_command,
    add_common_args,
    add_selection_args,
    cmd_status,
    run,
    selected_keys,
    start,
)
from growth_rag.config import DataPaths, Settings
from growth_rag.registry import Registry, RegistryFormatError
from growth_rag.sources.gutenberg import SOURCE
from growth_rag.sources.gutenberg.catalog import fetch_catalog, latest_snapshot, read_catalog
from growth_rag.sources.gutenberg.clean import clean_books
from growth_rag.sources.gutenberg.download import download_manifest
from growth_rag.sources.gutenberg.migrate import migrate_v1
from growth_rag.sources.gutenberg.selection import Manifest, SelectionCriteria, build_manifest
from growth_rag.utils.http import HttpError, make_session

log = logging.getLogger("gutenberg")


def cmd_catalog(args, settings: Settings, paths: DataPaths) -> None:
    session = make_session(settings.http.user_agent)
    path = fetch_catalog(session, paths.catalog_dir(SOURCE), pg=settings.gutenberg,
                         http=settings.http, force=args.force)
    log.info("%d catalog entries", len(read_catalog(path)))


def add_criteria_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--title", nargs="+", help='titles containing these words, e.g. "moby dick" hamlet')
    p.add_argument("--id", type=int, nargs="+", help="explicit book numbers, e.g. 1342 84")
    p.add_argument("--languages", nargs="+", help="ISO codes, e.g. en fr de (default: all)")
    p.add_argument("--bookshelf", nargs="+", help='bookshelves containing any of these, e.g. "Category: Novels"')
    p.add_argument("--author", nargs="+", help="authors matching any of these, e.g. Hugo 'Austen, Jane'")
    p.add_argument("--subject", nargs="+", help="subjects containing any of these")
    p.add_argument("--exclude-subject", nargs="+", help="drop books whose subjects contain any of these")
    p.add_argument("--per-language", type=int, help="max books per language (default: no cap)")
    p.add_argument("--seed", type=int, default=42, help="random seed for reproducible sampling")


def criteria_from_args(args) -> SelectionCriteria:
    return SelectionCriteria(
        titles=args.title,
        ids=args.id,
        languages=args.languages,
        bookshelves=args.bookshelf,
        authors=args.author,
        subjects=args.subject,
        exclude_subjects=args.exclude_subject,
        per_language=args.per_language,
        seed=args.seed,
    )


def cmd_select(args, settings: Settings, paths: DataPaths) -> None:
    out = paths.manifests / f"{args.name}.json"
    if out.exists() and not args.force:
        sys.exit(f"Manifest {out} already exists (choose another --name or use --force).")
    snapshot = latest_snapshot(paths.catalog_dir(SOURCE))
    catalog = read_catalog(snapshot)
    manifest = build_manifest(args.name, catalog, snapshot, criteria_from_args(args))
    log.info("Catalog %s: %d entries", snapshot.name, len(catalog))
    for lang, c in manifest.counts.items():
        log.info("  %5s: %6d candidates -> %d selected", lang, c["candidates"], c["selected"])
    manifest.save(out)
    log.info("Wrote %s (%d books)", out, len(manifest.ids))


def cmd_download(args, settings: Settings, paths: DataPaths) -> None:
    manifest = Manifest.load(Path(args.manifest))
    catalog_dir = paths.catalog_dir(SOURCE)
    # Use the snapshot the manifest was built from, so metadata matches the selection.
    snapshot = catalog_dir / manifest.catalog_snapshot
    if not snapshot.exists():
        snapshot = latest_snapshot(catalog_dir)
        log.warning("%s missing, using %s", manifest.catalog_snapshot, snapshot.name)
    registry = Registry.load(paths.registry)
    summary = download_manifest(
        manifest,
        read_catalog(snapshot),
        snapshot.name,
        registry,
        paths,
        make_session(settings.http.user_agent),
        http=settings.http,
        mirror_url=args.mirror or settings.gutenberg.mirror_url,
        delay_s=settings.gutenberg.download_delay_s if args.delay is None else args.delay,
        retry_missing=args.retry_missing,
        verify_hash=args.verify,
    )
    if summary.interrupted:
        log.warning("Interrupted: registry saved, re-run the same command to resume.")
    log.info("Done: %s | registry: %s", summary.by_status or "nothing to do", paths.registry)


def cmd_clean(args, settings: Settings, paths: DataPaths) -> None:
    registry = Registry.load(paths.registry)
    summary = clean_books(registry, paths, selected_keys(args), force=args.force)
    if summary.interrupted:
        log.warning("Interrupted: registry saved, re-run the same command to resume.")
    log.info("Done: %s | outputs in %s", summary.by_status or "nothing to do", paths.clean_dir(SOURCE))


def cmd_add(args, settings: Settings, paths: DataPaths) -> None:
    """Select books and run every step for them (see sources/gutenberg/pipeline.py)."""
    from dataclasses import replace

    from growth_rag.chunking.tokens import make_counter
    from growth_rag.embedding.encoders import make_encoder
    from growth_rag.index.qdrant import connect
    from growth_rag.sources.gutenberg.pipeline import (
        CONFIRM_ABOVE,
        default_manifest_name,
        describe_selection,
        log_report,
        run_pipeline,
    )

    catalog_dir = paths.catalog_dir(SOURCE)
    if args.manifest:
        manifest = Manifest.load(Path(args.manifest))
        snapshot = catalog_dir / manifest.catalog_snapshot
        if not snapshot.exists():
            snapshot = latest_snapshot(catalog_dir)
    else:
        if not list(catalog_dir.glob("pg_catalog_*.csv")):
            log.info("No catalog snapshot yet: downloading it first.")
            fetch_catalog(make_session(settings.http.user_agent), catalog_dir,
                          pg=settings.gutenberg, http=settings.http)
        snapshot = latest_snapshot(catalog_dir)
        criteria = criteria_from_args(args)
        if not any([args.title, args.id, args.author, args.bookshelf, args.subject, args.languages]):
            sys.exit("Say which books to add: --title, --id, --author, --bookshelf, --subject "
                     "and/or --languages (or --manifest).")
        manifest = build_manifest(args.name or default_manifest_name(), read_catalog(snapshot),
                                  snapshot, criteria)
    catalog = read_catalog(snapshot)

    for line in describe_selection(manifest, catalog):
        log.info("%s", line)
    if not manifest.ids:
        log.warning("Nothing matches. Try fewer words in --title, or check --languages.")
        return
    if args.dry_run:
        log.info("Dry run: nothing downloaded. Re-run without --dry-run to add these books.")
        return
    if len(manifest.ids) > CONFIRM_ABOVE and not args.yes:
        sys.exit(f"{len(manifest.ids)} books selected (more than {CONFIRM_ABOVE}). Narrow the "
                 "selection (--languages, --per-language) or confirm with --yes.")

    if not args.manifest:
        out = paths.manifests / f"{manifest.name}.json"
        if out.exists():
            sys.exit(f"Manifest {out} already exists: choose another --name, or resume it with --manifest {out}")
        manifest.save(out)
        log.info("Selection saved: %s (re-run with --manifest %s to resume)", out, out)

    chunking = replace(settings.chunking, tokenizer=args.tokenizer) if args.tokenizer else settings.chunking
    embedding = settings.embedding
    if args.backend:
        embedding = replace(embedding, backend=args.backend)
    if args.batch_size:
        embedding = replace(embedding, batch_size=args.batch_size)
    settings = replace(settings, chunking=chunking, embedding=embedding)

    registry = Registry.load(paths.registry)
    report = run_pipeline(
        registry, manifest, catalog, snapshot.name, paths, settings,
        until=args.until,
        session=make_session(settings.http.user_agent),
        counter=make_counter(chunking.tokenizer),
        encoder_factory=lambda: make_encoder(embedding),
        client_factory=lambda: connect(settings.qdrant),
        delay_s=args.delay,
    )
    log_report(report)


def cmd_migrate(args, settings: Settings, paths: DataPaths) -> None:
    report = migrate_v1(paths)
    log.info("Migration: %s", report)


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gutenberg", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, settings)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("catalog", help="download a dated catalog snapshot")
    p.add_argument("--force", action="store_true", help="re-download today's snapshot")
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser("select", help="filter the catalog and write a manifest")
    p.add_argument("--name", required=True, help="manifest name (file name without .json)")
    add_criteria_args(p)
    p.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("add", help="select books and run every step: download, clean, chunk, embed, index")
    add_criteria_args(p)
    p.add_argument("--manifest", help="use (or resume) an existing manifest instead of a new selection")
    p.add_argument("--name", help="name of the manifest to create (default: add-<date>-<time>)")
    p.add_argument("--dry-run", action="store_true", help="show the selection, do nothing")
    p.add_argument("--yes", action="store_true", help="confirm a selection of more than 50 books")
    p.add_argument("--until", choices=["download", "clean", "chunk", "embed", "index"], default="index",
                   help="last step to run (default: index)")
    p.add_argument("--tokenizer", help='chunking tokenizer (default from settings; "approx-4cpt" = no download)')
    p.add_argument("--backend", choices=["manual", "flag"], help="embedding backend (default from settings)")
    p.add_argument("--batch-size", type=int, help="embedding batch size")
    p.add_argument("--delay", type=float, help="seconds between downloads (default from settings)")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("download", help="download the texts listed in a manifest")
    p.add_argument("--manifest", required=True, help="path to a manifest JSON")
    p.add_argument("--delay", type=float, help="seconds between books (default from settings)")
    p.add_argument("--mirror", help="mirror URL template with {id} (default from settings)")
    p.add_argument("--retry-missing", action="store_true", help="retry books previously 404")
    p.add_argument("--verify", action="store_true", help="check SHA-256 of existing files")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("clean", help="clean downloaded books (text + structure + annotations)")
    add_selection_args(p)
    p.set_defaults(func=cmd_clean)

    add_chunk_command(sub, source=SOURCE)  # same as `rag chunk`, limited to Gutenberg books

    p = sub.add_parser("status", help="summarise the registry")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("migrate", help="convert data written by the old single-file script")
    p.set_defaults(func=cmd_migrate)
    return parser


def main(argv: list[str] | None = None) -> None:
    settings = Settings.from_env()
    args = build_parser(settings).parse_args(argv)
    from growth_rag.embedding.encoders import EncoderUnavailable

    paths = start(args, "gutenberg", creates_data=args.command in ("catalog", "select", "migrate", "add"))
    run(args, settings, paths, (RegistryFormatError, FileNotFoundError, HttpError, TokenizerUnavailable,
                                EncoderUnavailable))


if __name__ == "__main__":
    main()
