"""bge-m3 encoders. Both return an `Encoded` batch: L2-normalised dense vectors
and sparse (lexical) vectors as (token ids, weights)."""

from __future__ import annotations

from growth_rag.config import EmbeddingSettings
from growth_rag.embedding.encoders.base import Encoded, Encoder, EncoderUnavailable


def make_encoder(settings: EmbeddingSettings) -> Encoder:
    if settings.backend == "manual":
        from growth_rag.embedding.encoders.manual import ManualBgeM3

        return ManualBgeM3(settings)
    if settings.backend == "flag":
        from growth_rag.embedding.encoders.flag import FlagBgeM3

        return FlagBgeM3(settings)
    raise ValueError(f"Unknown embedding backend {settings.backend!r} (expected 'manual' or 'flag')")


__all__ = ["Encoded", "Encoder", "EncoderUnavailable", "make_encoder"]
