"""Schema of an evaluation question (one JSON object per line of questions.jsonl).

    {"id": "hamlet-ophelia-flowers",
     "task": "passage",
     "query": "Which flowers does Ophelia hand out, and what do they mean?",
     "language": "en",
     "tags": ["play", "speaker"],
     "filters": {"title": "Hamlet", "speaker": ["OPHELIA"]},
     "relevant": [{"book": "gutenberg:1524",
                   "evidence": "There’s rosemary, that’s for remembrance", "grade": 2}],
     "answer": "Rosemary for remembrance, pansies for thoughts, ..."}

- task "passage": find the chunks that answer; every relevant entry has an
  `evidence` quote (in the cleaned text, or in an annotation with "in":
  "annotation" for footnotes). An empty `relevant` list means the corpus
  does not answer the question (the generator should say so).
- task "book": recommend books; relevant entries name books, no evidence.
- grade: 2 = answers / fits, 1 = partially answers / related (for nDCG).
- filters: the same as `rag search` (title, author, speaker, language,
  kind, book). A question without filters tests unfiltered retrieval.
- seed_books: for "books like X" questions, the books to leave out of the
  results (recommending X itself is not an answer).
- known_gap: why the current pipeline is expected to fail this question.
  Keeps known failures visible without counting them as dataset errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class EvalFormatError(ValueError):
    pass


class Relevant(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    book: str  # registry key, e.g. "gutenberg:1524"
    evidence: str | None = None  # verbatim quote (whitespace, case and quote marks are normalised)
    in_: Literal["text", "annotation"] = Field("text", alias="in")
    grade: Literal[1, 2] = 2


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    author: str | None = None
    speaker: list[str] | None = None
    language: list[str] | None = None
    kind: list[Literal["prose", "poem", "play"]] | None = None
    book: list[str] | None = None  # registry keys


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    task: Literal["passage", "book"]
    query: str
    language: str  # language of the query (ISO 639-1)
    tags: list[str] = []
    filters: Filters | None = None
    relevant: list[Relevant] = []
    seed_books: list[str] = []
    answer: str | None = None  # reference answer, for generation evaluation later
    known_gap: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _evidence_matches_task(self) -> Question:
        for r in self.relevant:
            if self.task == "passage" and not r.evidence:
                raise ValueError(f"passage question: relevant {r.book} needs an evidence quote")
            if self.task == "book" and r.evidence:
                raise ValueError(f"book question: relevant {r.book} takes no evidence")
        return self

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)

    @property
    def books(self) -> list[str]:
        return list(dict.fromkeys(r.book for r in self.relevant))


def load_questions(path: Path) -> list[Question]:
    """Read questions.jsonl; errors name the line. Blank lines are allowed."""
    questions: list[Question] = []
    seen: dict[str, int] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            q = Question.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise EvalFormatError(f"{path.name}, line {n}: {exc}") from exc
        if q.id in seen:
            raise EvalFormatError(f"{path.name}, line {n}: id {q.id!r} already used on line {seen[q.id]}")
        seen[q.id] = n
        questions.append(q)
    return questions


def dump_questions(path: Path, questions: list[Question]) -> None:
    """Write one question per line, defaults left out (keeps the file readable)."""
    lines = [json.dumps(q.model_dump(mode="json", by_alias=True, exclude_defaults=True), ensure_ascii=False)
             for q in questions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
