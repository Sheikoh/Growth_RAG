"""bge-reranker-v2-m3 through FlagEmbedding (the authors' implementation): the
reference `rag rerank-check` compares the manual version against. Optional
dependency: `uv sync --extra flag`. Imported on first use only."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from growth_rag.config import RerankSettings
from growth_rag.rerank.base import RerankerUnavailable


class FlagReranker:
    backend = "flag"

    def __init__(self, settings: RerankSettings):
        self.name = settings.model
        self.revision = settings.revision
        self.settings = settings
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import FlagReranker as _FlagReranker

            from growth_rag.rerank.files import model_folder

            folder = str(model_folder(self.settings))  # same local files as the manual version
            device = None if self.settings.device == "auto" else self.settings.device
            try:  # FlagEmbedding >= 1.3
                self._model = _FlagReranker(folder, use_fp16=self.settings.fp16, devices=device,
                                            max_length=self.settings.max_length)
            except TypeError:  # older releases
                self._model = _FlagReranker(folder, use_fp16=self.settings.fp16)
        except Exception as exc:
            raise RerankerUnavailable(self.backend, self.name, exc) from exc
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        model = self._load()
        pairs = [[query, p] for p in passages]
        try:
            out = model.compute_score(pairs, batch_size=self.settings.batch_size,
                                      max_length=self.settings.max_length, normalize=False)
        except TypeError:  # older releases: fewer keyword arguments
            out = model.compute_score(pairs)
        return np.atleast_1d(np.asarray(out, dtype=np.float32))
