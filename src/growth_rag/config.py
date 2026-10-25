"""Central configuration.

Every value has a default and can be overridden with a GROWTH_RAG_* environment
variable (for instance from a .env file), so a local run and a cloud run differ
only by their settings, never by their code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# bge-m3 pinned to a commit of its Hugging Face repository (main, 2024-07-03):
# the same weights for every run, and the files are found in the local cache
# without asking the Hub which commit "main" points to today.
BGE_M3 = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
# Multilingual cross-encoder (same XLM-RoBERTa base as bge-m3), pinned likewise.
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"
BGE_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"GROWTH_RAG_{name}", default)


@dataclass(frozen=True)
class HttpSettings:
    # Identify the client: polite, and lets server operators contact you if needed.
    user_agent: str = "RAG-Growth-ingestion/0.2 (personal research project)"
    timeout_s: float = 60.0
    max_retries: int = 3


@dataclass(frozen=True)
class GutenbergSettings:
    catalog_url: str = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
    # Official high-speed mirror run by Project Gutenberg. Automated downloads go
    # here, not to www.gutenberg.org (robot access policy). {id} = book number.
    mirror_url: str = "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt"
    download_delay_s: float = 2.0


@dataclass(frozen=True)
class ChunkingSettings:
    # Embedding model's tokenizer (same pinned revision as the model);
    # "approx-4cpt" = no download (tests, trials).
    tokenizer: str = BGE_M3
    target_tokens: int = 350
    max_tokens: int = 512
    min_tokens: int = 80


@dataclass(frozen=True)
class EmbeddingSettings:
    backend: str = "manual"  # "manual" (transformers, readable) or "flag" (FlagEmbedding)
    model: str = BGE_M3  # a Hugging Face repo id, or a local folder with the same files
    revision: str | None = BGE_M3_REVISION  # commit hash (reproducible) or branch; None = main
    device: str = "auto"  # "auto" = cuda if available, else cpu
    fp16: bool = True  # on GPU only
    batch_size: int = 64
    max_length: int = 1024  # tokens; chunks are <= ~512 + header


@dataclass(frozen=True)
class RerankSettings:
    backend: str = "manual"  # "manual" (transformers, readable) or "flag" (FlagEmbedding)
    model: str = BGE_RERANKER  # a Hugging Face repo id, or a local folder with the same files
    revision: str | None = BGE_RERANKER_REVISION
    device: str = "auto"
    fp16: bool = True  # on GPU only
    batch_size: int = 16  # (question, passage) pairs per forward pass
    max_length: int = 1024  # tokens of question + passage together; chunks are <= ~512 + header
    candidates: int = 20  # hybrid results reranked by `rag search --rerank`; 20 beat 50 on eval-v2 (README 6.1)


@dataclass(frozen=True)
class QdrantSettings:
    url: str = "http://localhost:6333"  # Docker locally; a cloud URL later
    api_key: str | None = None  # needed for Qdrant Cloud
    path: str | None = None  # embedded local mode (no server), e.g. for tests
    chunks_collection: str = "chunks"
    books_collection: str = "books"


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    http: HttpSettings = field(default_factory=HttpSettings)
    gutenberg: GutenbergSettings = field(default_factory=GutenbergSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = field(default_factory=RerankSettings)
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)

    @classmethod
    def from_env(cls) -> Settings:
        http_defaults, pg_defaults, ck = HttpSettings(), GutenbergSettings(), ChunkingSettings()
        em, rr, qd = EmbeddingSettings(), RerankSettings(), QdrantSettings()
        return cls(
            data_dir=Path(_env("DATA_DIR", "data")),
            http=HttpSettings(
                user_agent=_env("USER_AGENT", http_defaults.user_agent),
                timeout_s=float(_env("HTTP_TIMEOUT_S", str(http_defaults.timeout_s))),
                max_retries=int(_env("HTTP_MAX_RETRIES", str(http_defaults.max_retries))),
            ),
            gutenberg=GutenbergSettings(
                catalog_url=_env("GUTENBERG_CATALOG_URL", pg_defaults.catalog_url),
                mirror_url=_env("GUTENBERG_MIRROR_URL", pg_defaults.mirror_url),
                download_delay_s=float(
                    _env("GUTENBERG_DELAY_S", str(pg_defaults.download_delay_s))
                ),
            ),
            chunking=ChunkingSettings(
                tokenizer=_env("TOKENIZER", ck.tokenizer),
                target_tokens=int(_env("CHUNK_TARGET_TOKENS", str(ck.target_tokens))),
                max_tokens=int(_env("CHUNK_MAX_TOKENS", str(ck.max_tokens))),
                min_tokens=int(_env("CHUNK_MIN_TOKENS", str(ck.min_tokens))),
            ),
            embedding=EmbeddingSettings(
                backend=_env("EMBED_BACKEND", em.backend),
                model=_env("EMBED_MODEL", em.model),
                revision=_env("EMBED_REVISION", em.revision or "") or None,
                device=_env("EMBED_DEVICE", em.device),
                fp16=_env("EMBED_FP16", str(em.fp16)).lower() in ("1", "true", "yes"),
                batch_size=int(_env("EMBED_BATCH_SIZE", str(em.batch_size))),
                max_length=int(_env("EMBED_MAX_LENGTH", str(em.max_length))),
            ),
            rerank=RerankSettings(
                backend=_env("RERANK_BACKEND", rr.backend),
                model=_env("RERANK_MODEL", rr.model),
                revision=_env("RERANK_REVISION", rr.revision or "") or None,
                device=_env("RERANK_DEVICE", rr.device),
                fp16=_env("RERANK_FP16", str(rr.fp16)).lower() in ("1", "true", "yes"),
                batch_size=int(_env("RERANK_BATCH_SIZE", str(rr.batch_size))),
                max_length=int(_env("RERANK_MAX_LENGTH", str(rr.max_length))),
                candidates=int(_env("RERANK_CANDIDATES", str(rr.candidates))),
            ),
            qdrant=QdrantSettings(
                url=_env("QDRANT_URL", qd.url),
                api_key=os.environ.get("GROWTH_RAG_QDRANT_API_KEY") or None,
                path=os.environ.get("GROWTH_RAG_QDRANT_PATH") or None,
                chunks_collection=_env("QDRANT_CHUNKS", qd.chunks_collection),
                books_collection=_env("QDRANT_BOOKS", qd.books_collection),
            ),
        )


class DataPaths:
    """Layout of the data folder. Everything derives from one root so the folder
    can be moved (or later mapped to a bucket) as a whole.

        <root>/
            registry.json              what we have (all sources)
            manifests/<name>.json      what we want (dataset definitions)
            catalog/<source>/...       catalog snapshots, never modified
            raw/<source>/...           files exactly as served
            clean/<source>/...         cleaned text + structure (never overwrites raw)
            chunks/<source>/...        chunks, one JSON Lines file per book
            vectors/<source>/...       embeddings, one .npz per book (re-index without re-embedding)
            logs/                      run logs
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.registry = self.root / "registry.json"
        self.manifests = self.root / "manifests"
        self.logs = self.root / "logs"

    def catalog_dir(self, source: str) -> Path:
        return self.root / "catalog" / source

    def raw_dir(self, source: str) -> Path:
        return self.root / "raw" / source

    def clean_dir(self, source: str) -> Path:
        return self.root / "clean" / source

    def chunks_dir(self, source: str) -> Path:
        return self.root / "chunks" / source

    def vectors_dir(self, source: str) -> Path:
        return self.root / "vectors" / source

    def relative(self, path: Path) -> str:
        """Path stored in records: relative to the root, POSIX separators."""
        return path.relative_to(self.root).as_posix()

    def resolve(self, relative: str) -> Path:
        return self.root / relative
