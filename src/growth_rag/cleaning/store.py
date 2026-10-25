"""Reading and writing cleaned books: `<stem>.txt` + `<stem>.json`."""

from __future__ import annotations

from pathlib import Path

from growth_rag.cleaning.models import CleanedBook
from growth_rag.utils.io import atomic_write_bytes, atomic_write_json, read_json


def save_cleaned(directory: Path, stem: str, text: str, book: CleanedBook) -> tuple[Path, Path]:
    text_path = directory / f"{stem}.txt"
    structure_path = directory / f"{stem}.json"
    atomic_write_bytes(text_path, text.encode("utf-8"))
    atomic_write_json(structure_path, book.model_dump(mode="json"))
    return text_path, structure_path


def load_cleaned(text_path: Path, structure_path: Path) -> tuple[str, CleanedBook]:
    # newline="" keeps the text exactly as written, so offsets stay valid.
    with text_path.open(encoding="utf-8", newline="") as f:
        text = f.read()
    return text, CleanedBook.model_validate(read_json(structure_path))
