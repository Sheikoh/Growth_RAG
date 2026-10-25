"""The bge-reranker-v2-m3 files, resolved once into a local folder (cache first,
download only what is missing; see utils/hf.py). The repository's `assets/`
images are never downloaded."""

from __future__ import annotations

from pathlib import Path

from growth_rag.config import RerankSettings
from growth_rag.utils.hf import local_snapshot

PATTERNS = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "model.safetensors",
    "pytorch_model.bin",  # not at the pinned revision; taken if another revision only has it
]
REQUIRED: list[str | tuple[str, ...]] = [
    "config.json",
    "tokenizer_config.json",
    ("tokenizer.json", "sentencepiece.bpe.model"),
    ("model.safetensors", "pytorch_model.bin"),
]


def model_folder(settings: RerankSettings) -> Path:
    return local_snapshot(settings.model, settings.revision, PATTERNS, REQUIRED)
