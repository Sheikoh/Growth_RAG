"""Evaluation sets: questions with the passages (or books) that answer them.

An evaluation set is a folder (03_RAG/eval/<name>/) with:
- `manifest.json`: the books it is built on (a normal manifest, so
  `uv run gutenberg add --manifest eval/<name>/manifest.json` loads them);
- `questions.jsonl`: one question per line (schema in `dataset.py`).

Ground truth is a verbatim quote, not chunk keys: chunk keys change whenever
chunking changes, quotes don't. `rag eval-check` locates every quote in the
cleaned books and lists the chunks that contain it (`check.py`).
"""
