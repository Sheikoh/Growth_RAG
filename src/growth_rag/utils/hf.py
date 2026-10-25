"""Hugging Face model files: from the local cache first, the Hub only if missing.

`from_pretrained("BAAI/bge-m3")` contacts the Hub on every call (about ten
metadata requests) even when everything is cached, and may pick files from
another branch. Here the files are resolved once, at a fixed revision, into a
local folder; loaders then get that folder, so they make no network call.

    folder = local_snapshot("BAAI/bge-m3", revision="5617a9f...", patterns=[...], required=[...])
    AutoModel.from_pretrained(folder)

Only the listed files are downloaded (bge-m3's repository also holds a 2.2 GB
ONNX copy of the weights and images that we never use).
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class ModelFilesUnavailable(RuntimeError):
    pass


def local_snapshot(
    repo: str,
    revision: str | None,
    patterns: list[str],
    required: list[str | tuple[str, ...]],
) -> Path:
    """Folder holding `repo` at `revision` with every `required` file (a tuple
    means: one of these). `repo` may also be a local folder, used as is.

    1. local cache only (no network): enough when the files are there;
    2. otherwise download the files matching `patterns` (only the missing ones).
    Set HF_HUB_OFFLINE=1 to forbid step 2.
    """
    if Path(repo).is_dir():
        folder = Path(repo)
        _check(folder, required, repo)
        return folder

    from huggingface_hub import snapshot_download

    try:
        folder = Path(snapshot_download(repo, revision=revision, allow_patterns=patterns, local_files_only=True))
        if not _missing(folder, required):
            return folder
    except FileNotFoundError:  # LocalEntryNotFoundError: not cached at this revision
        pass
    log.info("Downloading %s (%s) from Hugging Face: %s", repo, revision or "main", ", ".join(patterns))
    try:
        folder = Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))
    except Exception as exc:
        raise ModelFilesUnavailable(
            f"{repo} at revision {revision or 'main'} is not in the local cache and cannot be "
            f"downloaded ({type(exc).__name__}: {exc}). Check the connection, or unset HF_HUB_OFFLINE."
        ) from exc
    _check(folder, required, repo)
    return folder


def _missing(folder: Path, required: list[str | tuple[str, ...]]) -> list[str]:
    out = []
    for item in required:
        options = (item,) if isinstance(item, str) else item
        if not any((folder / name).exists() for name in options):
            out.append(" or ".join(options))
    return out


def _check(folder: Path, required, repo: str) -> None:
    missing = _missing(folder, required)
    if missing:
        raise ModelFilesUnavailable(f"{repo}: missing {', '.join(missing)} in {folder}")


def folder_size(folder: Path, names: list[str]) -> dict[str, int]:
    """Sizes in bytes of the given files of a snapshot (cache symlinks resolved)."""
    return {n: (folder / n).resolve().stat().st_size for n in names if (folder / n).exists()}
