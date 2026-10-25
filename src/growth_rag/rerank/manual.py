"""bge-reranker-v2-m3 with plain transformers: every step visible.

    (question, passage) --tokenizer, as ONE sequence--> <s> question </s></s> passage </s>
        --XLM-RoBERTa (reranker weights)--> hidden states
        --classification head on the first position--> one logit
    score = logit                      (higher = more relevant)

The model is a sequence classifier with a single output (num_labels = 1);
the head (dense + tanh + linear) is part of the checkpoint. Truncation keeps
the question whole and cuts the passage if the pair exceeds max_length
("only_second"), as FlagEmbedding does (it also caps the question at 3/4 of
max_length, irrelevant for short questions); with chunks of ~350 tokens no
truncation happens at 1024.
Heavy imports happen on first use only.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from growth_rag.config import RerankSettings
from growth_rag.rerank.base import RerankerUnavailable, length_order


class ManualReranker:
    backend = "manual"

    def __init__(self, settings: RerankSettings):
        self.name = settings.model
        self.revision = settings.revision
        self.settings = settings
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            from growth_rag.rerank.files import model_folder

            device = self.settings.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if (self.settings.fp16 and device.startswith("cuda")) else torch.float32
            self._torch, self.device = torch, device
            self.folder = model_folder(self.settings)  # local: no Hub call once cached
            self.tokenizer = AutoTokenizer.from_pretrained(self.folder)
            self.model = (AutoModelForSequenceClassification.from_pretrained(self.folder, dtype=dtype)
                          .to(device).eval())
        except Exception as exc:
            raise RerankerUnavailable(self.backend, self.name, exc) from exc
        self._loaded = True

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        self._load()
        torch = self._torch
        scores = np.zeros(len(passages), dtype=np.float32)
        order = length_order(passages)
        size = self.settings.batch_size
        for start in range(0, len(order), size):
            idx = order[start:start + size]
            batch = self.tokenizer(
                [query] * len(idx), [passages[i] for i in idx],
                padding=True, truncation="only_second", max_length=self.settings.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**batch).logits.view(-1).float().cpu().numpy()  # [pairs]
            scores[idx] = logits
        return scores
