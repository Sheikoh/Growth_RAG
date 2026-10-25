"""The bge-m3 files both encoders need, resolved once into a local folder.

At the pinned revision the repository holds the weights as `pytorch_model.bin`
only (no `.safetensors` on main); transformers loads it safely with
`torch.load(weights_only=True)` (torch >= 2.6). The ONNX copy (2.2 GB) and
the images are never downloaded.
"""

from __future__ import annotations

from pathlib import Path

from growth_rag.config import EmbeddingSettings
from growth_rag.utils.hf import local_snapshot

PATTERNS = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "model.safetensors",  # not on main at the pinned revision; taken if a later revision has it
    "pytorch_model.bin",
    "sparse_linear.pt",  # the sparse head (1024 -> 1)
    "colbert_linear.pt",  # unused here, but FlagEmbedding re-initialises *both* heads randomly if one is missing
]
REQUIRED: list[str | tuple[str, ...]] = [
    "config.json",
    "tokenizer_config.json",
    ("tokenizer.json", "sentencepiece.bpe.model"),
    ("model.safetensors", "pytorch_model.bin"),
    "sparse_linear.pt",
    "colbert_linear.pt",
]


def model_folder(settings: EmbeddingSettings) -> Path:
    """Local folder with the model files (cache first; downloads only what is missing)."""
    return local_snapshot(settings.model, settings.revision, PATTERNS, REQUIRED)
