"""Token counting. Chunk sizes are measured in the *embedding model's* tokens:
characters or words would be wrong for Chinese or Greek, and the model's limit
is in tokens.

Two implementations behind one small interface:
- `HFTokenCounter`: the real tokenizer (default "BAAI/bge-m3"), via transformers;
  downloads the tokenizer files once (no model weights).
- `ApproxTokenCounter`: no download, ~4 characters per token for alphabetic
  scripts and ~1 token per CJK character. For tests and quick trials only.

The counter's `name` is stored with every chunk: changing tokenizer triggers
re-chunking.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Protocol

_CJK = re.compile(r"[⺀-鿿豈-﫿぀-ヿ가-힯]")


class TokenCounter(Protocol):
    name: str

    def count(self, texts: Sequence[str]) -> list[int]: ...


class ApproxTokenCounter:
    name = "approx-4cpt"

    def count(self, texts: Sequence[str]) -> list[int]:
        out = []
        for text in texts:
            cjk = len(_CJK.findall(text))
            out.append(cjk + math.ceil((len(text) - cjk) / 4))
        return out


TOKENIZER_FILES = ["*.json", "*.model", "*.txt"]  # tokenizer.json, sentencepiece model, vocab files


class HFTokenCounter:
    """Lazy: the tokenizer is loaded on first use, so importing this module or
    building settings never triggers a download."""

    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 256, revision: str | None = None):
        self.name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self._tokenizer = None

    def _load(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # heavy import, only when needed

            from growth_rag.utils.hf import local_snapshot

            # Tokenizer files only, from the local cache when present (no Hub call).
            folder = local_snapshot(self.name, self.revision, TOKENIZER_FILES, ["tokenizer_config.json"])
            self._tokenizer = AutoTokenizer.from_pretrained(folder)
        return self._tokenizer

    def count(self, texts: Sequence[str]) -> list[int]:
        tokenizer = self._load()
        counts: list[int] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
            counts.extend(len(ids) for ids in encoded)
        return counts


def make_counter(name: str) -> TokenCounter:
    from growth_rag.config import BGE_M3, BGE_M3_REVISION

    if name == ApproxTokenCounter.name:
        return ApproxTokenCounter()
    return HFTokenCounter(name, revision=BGE_M3_REVISION if name == BGE_M3 else None)
