"""Schema of a chunk: the contract between chunking and Qdrant ingestion.

`text` is always exactly `cleaned_text[start:end]` (offsets in characters, as in
cleaning), so citations point back to the cleaned file. `header` is the context
prepended to the text *for embedding only* (book, author, section path).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

ChunkKind = Literal["prose", "poem", "play"]


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str  # "<book_key>/s<section>/c<n>", e.g. "gutenberg:1342/s3/c12"
    book_key: str
    index: int  # position in the book (reading order)
    section: int  # sequential section index from cleaning
    index_in_section: int
    section_path: list[str]
    section_title: str | None = None
    kind: ChunkKind
    language: str
    start: int
    end: int
    text: str
    header: str
    tokens: int  # of `text`, with `tokenizer`
    tokenizer: str
    paragraphs: list[int]  # first and last paragraph index covered
    part: str | None = None  # "2/3" when a single paragraph or poem was split
    speakers: list[str] = []  # book-scoped speaker keys, in order of appearance
    speaker_labels: list[str] = []
    annotations: list[str] = []  # keys of annotations anchored inside the chunk
    previous: str | None = None  # reading order, across sections
    next: str | None = None

    def embedding_text(self) -> str:
        """What the embedding model sees: context header + text."""
        return f"{self.header}\n\n{self.text}" if self.header else self.text
