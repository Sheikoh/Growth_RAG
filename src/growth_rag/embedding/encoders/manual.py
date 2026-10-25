"""bge-m3 with plain transformers: every step of the computation is visible.

    texts --tokenizer--> input_ids, attention_mask
          --XLM-RoBERTa (bge-m3 weights)--> last_hidden_state [n, tokens, 1024]
          --[:, 0] + L2 norm--> dense [n, 1024]
          --sparse_linear (1024 -> 1) + ReLU--> one weight per token
          --max per token id, no special tokens--> sparse {token_id: weight}

The sparse head is not part of the transformers checkpoint: it is a separate
file in the model repository (`sparse_linear.pt`, a 1024 -> 1 linear layer).
Files come from a local folder at the pinned revision (`files.py`): no Hub
call once cached. Heavy imports (torch, transformers) happen on first use only.
"""

from __future__ import annotations

from collections.abc import Sequence

from growth_rag.config import EmbeddingSettings
from growth_rag.embedding.encoders.base import (
    Encoded,
    EncoderUnavailable,
    cls_pooling,
    l2_normalize,
    lexical_weights,
)


class ManualBgeM3:
    backend = "manual"

    def __init__(self, settings: EmbeddingSettings):
        self.name = settings.model
        self.revision = settings.revision
        self.settings = settings
        self.dim = 1024
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            from growth_rag.embedding.encoders.files import model_folder

            device = self.settings.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if (self.settings.fp16 and device.startswith("cuda")) else torch.float32

            self._torch = torch
            self.device = device
            self.folder = model_folder(self.settings)  # local: from_pretrained makes no Hub call
            self.tokenizer = AutoTokenizer.from_pretrained(self.folder)
            # .safetensors if the folder has it, else pytorch_model.bin
            self.model = AutoModel.from_pretrained(self.folder, dtype=dtype).to(device).eval()
            self.dim = self.model.config.hidden_size

            # The sparse head: a separate small file in the model repository.
            self.sparse_head = torch.nn.Linear(self.dim, 1)
            self.sparse_head.load_state_dict(
                torch.load(self.folder / "sparse_linear.pt", map_location="cpu", weights_only=True))
            self.sparse_head = self.sparse_head.to(device=device, dtype=dtype).eval()

            special = {self.tokenizer.cls_token_id, self.tokenizer.eos_token_id,
                       self.tokenizer.pad_token_id, self.tokenizer.unk_token_id}
            self.ignore_ids = {i for i in special if i is not None}
        except Exception as exc:
            raise EncoderUnavailable(self.backend, self.name, exc) from exc
        self._loaded = True

    def encode(self, texts: Sequence[str]) -> Encoded:
        self._load()
        torch = self._torch
        batch = self.tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.settings.max_length, return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state  # [n, tokens, dim]
            dense = cls_pooling(hidden)  # [n, dim]
            token_weights = torch.relu(self.sparse_head(hidden)).squeeze(-1)  # [n, tokens]

        dense_np = l2_normalize(dense.float().cpu().numpy())
        weights_np = token_weights.float().cpu().numpy()
        ids_np = batch["input_ids"].cpu().numpy()
        mask = batch["attention_mask"].cpu().numpy().astype(bool)
        # Drop padding positions before building sparse vectors.
        sparse = lexical_weights(
            (ids[m] for ids, m in zip(ids_np, mask)),
            (w[m] for w, m in zip(weights_np, mask)),
            self.ignore_ids,
        )
        return Encoded(dense=dense_np, sparse=sparse)

