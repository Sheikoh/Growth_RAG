"""bge-m3 through FlagEmbedding, the reference implementation by the model's
authors. Same vectors as `manual.py`, less code to own.

Optional dependency: `uv sync --extra flag` (pulls FlagEmbedding and its own
dependencies). Imported on first use only.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from growth_rag.config import EmbeddingSettings
from growth_rag.embedding.encoders.base import Encoded, EncoderUnavailable, l2_normalize, sparse_from_dict


class FlagBgeM3:
    backend = "flag"

    def __init__(self, settings: EmbeddingSettings):
        self.name = settings.model
        self.revision = settings.revision
        self.settings = settings
        self.dim = 1024
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import BGEM3FlagModel

            from growth_rag.embedding.encoders.files import model_folder

            # A local folder: same files as the manual encoder, and FlagEmbedding
            # does not download the whole repository (ONNX copy included).
            folder = str(model_folder(self.settings))
            device = None if self.settings.device == "auto" else self.settings.device
            try:  # FlagEmbedding >= 1.3
                self._model = BGEM3FlagModel(folder, use_fp16=self.settings.fp16, devices=device)
            except TypeError:  # older releases
                self._model = BGEM3FlagModel(folder, use_fp16=self.settings.fp16, device=device)
        except Exception as exc:
            raise EncoderUnavailable(self.backend, self.name, exc) from exc
        return self._model

    def encode(self, texts: Sequence[str]) -> Encoded:
        model = self._load()
        out = model.encode(
            list(texts),
            batch_size=len(texts),  # batching is done by the caller (sorted by length)
            max_length=self.settings.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = l2_normalize(np.asarray(out["dense_vecs"], dtype=np.float32))
        self.dim = dense.shape[1]
        return Encoded(dense=dense, sparse=[sparse_from_dict(w) for w in out["lexical_weights"]])
