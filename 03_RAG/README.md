# 03_RAG — Retrieval-Augmented Generation over a book corpus

Stage 3 of Growth_RAG: give the local Mistral model access to a corpus of
full-text books through a vector database, without LangChain or similar
frameworks. Every component (acquisition, cleaning, chunking, embedding,
indexing, retrieval, integration with the model) is built by hand, to master
the details of each step.

**Use cases (initial):** generating text grounded in retrieved passages, and
recommending books. The scope may grow (graph RAG in the next stage).

**Corpus:** Project Gutenberg (public domain in the US) and, later, legitimately
owned books. Many languages, not only English and French.

---

## Pipeline and status

| # | Step | Status | Code |
|---|------|--------|------|
| 1 | Acquisition: catalog → manifest → raw texts + registry | ✅ done | `growth_rag.sources.gutenberg` |
| 2 | Cleaning: encoding, boilerplate, structure, paragraphs, annotations, speakers | ✅ v1 done | `growth_rag.cleaning` + `sources.gutenberg.clean` |
| 3 | Chunking: prose / poems / plays, graph-ready keys, annotations and speakers linked | ✅ v1 done | `growth_rag.chunking` |
| 4 | Embedding: dense + sparse (bge-m3, two encoders), book vectors, on disk | ✅ v1 done | `growth_rag.embedding` |
| 5 | Indexing in Qdrant (Docker now, cloud later) + check search | ✅ v1 done | `growth_rag.index` |
| 6 | Retrieval: hybrid search, reranking (bge-reranker-v2-m3); neighbours, query understanding next | ✅ reranker v1; next upgrades listed in 6 | `growth_rag.index`, `growth_rag.rerank` |
| 7 | Integration with the LLM: pipeline RAG, then retriever as an agent tool | planned | — |
| 8 | Evaluation: question set with verified ground truth, retrieval scoring (`rag eval`), then RAGAS | ✅ set eval-v1, check, scoring v1; RAGAS later | `growth_rag.evaluation`, `eval/` |

---

## Repository layout

Reusable code lives in the `growth_rag` package (`src/`), stage folders hold
notebooks, data and stage-specific notes. Tests are at the repository root.

```
Growth_RAG/
├── pyproject.toml               # dependencies, CLI entry points (gutenberg, rag)
├── src/growth_rag/
│   ├── cli.py                   # `rag`: chunk, embed, embed-check, index, search, eval-check, eval, model, rerank-check, status
│   ├── cli_common.py            # options, selection, errors shared by both CLIs
│   ├── config.py                # Settings (GROWTH_RAG_* env vars) + DataPaths
│   ├── registry.py              # BookRecord schema (Pydantic) + Registry
│   ├── cleaning/                # generic cleaning, reused by every source
│   │   ├── models.py            # CleanedBook schema: sections, paragraphs, annotations, cast
│   │   ├── text.py              # normalisation, blocks, verse test, line joining, markup
│   │   ├── headings.py          # patterns, TOC dictionary, TOC echoes, poem titles
│   │   ├── annotations.py       # footnotes, illustrations, credits, transcriber's notes
│   │   ├── speakers.py          # plays: speaker labels, book-scoped keys
│   │   ├── pipeline.py          # body text -> (cleaned text, CleanedBook)
│   │   └── store.py             # <stem>.txt + <stem>.json
│   ├── chunking/                # generic chunking, reads only cleaning outputs
│   │   ├── models.py            # Chunk schema (the contract with Qdrant ingestion)
│   │   ├── tokens.py            # token counters: bge-m3 tokenizer / approximate
│   │   ├── sentences.py         # rule-based sentence spans (oversized paragraphs)
│   │   ├── chunker.py           # pieces, modes (prose/poem/play), packing, linking
│   │   ├── store.py             # JSON Lines
│   │   └── run.py               # over the registry: re-chunk rules, ChunkingInfo
│   ├── embedding/               # chunks -> vectors on disk
│   │   ├── encoders/base.py     # bge-m3 maths: CLS pooling, sparse weights (no torch)
│   │   ├── encoders/manual.py   # transformers + sparse_linear.pt: every step visible
│   │   ├── encoders/flag.py     # FlagEmbedding (reference), optional extra
│   │   ├── encoders/files.py    # the bge-m3 files both encoders read (pinned, local)
│   │   ├── books.py             # book-level vectors: metadata text, centroid
│   │   ├── store.py             # one .npz per book (dense fp16 + sparse CSR)
│   │   ├── check.py             # compare two encoders (rag embed-check)
│   │   └── run.py               # over the registry: batching by length, re-embed rules
│   ├── index/                   # Qdrant
│   │   ├── qdrant.py            # connect, collections, UUIDv5 ids, delete + upsert
│   │   ├── run.py               # vectors -> points, re-index rules
│   │   ├── resolve.py           # titles / authors -> book keys (registry)
│   │   └── search.py            # dense / sparse / hybrid (RRF), filters, books
│   ├── rerank/                  # second stage: cross-encoder reranking (see 6.1)
│   │   ├── base.py              # Reranker protocol, rerank(), passage text
│   │   ├── manual.py            # transformers: every step visible
│   │   ├── flag.py              # FlagEmbedding (reference), optional extra
│   │   ├── files.py             # pinned model files, local
│   │   └── check.py             # manual vs FlagEmbedding (rag rerank-check)
│   ├── evaluation/              # evaluation sets (see 8)
│   │   ├── dataset.py           # Question schema, load / dump questions.jsonl
│   │   ├── check.py             # locate evidence quotes -> relevant chunks (rag eval-check)
│   │   ├── metrics.py           # hit@k, recall@k, MRR, nDCG (pure functions)
│   │   ├── run.py               # search every question in every mode, run files (rag eval)
│   │   ├── diagnose.py          # rank buckets, dense vs sparse, misses (rag eval --diagnose)
│   │   └── report.py            # overview, by-slice, failures, comparison of two runs
│   ├── utils/                   # generic, no knowledge of books or RAG
│   │   ├── io.py                # atomic writes, JSON, hashing, timestamps
│   │   ├── http.py              # session + retries on transient errors only
│   │   ├── log.py               # console + file logging
│   │   ├── hf.py                # model files: local cache first, Hub only if missing
│   │   └── text_norm.py         # case / accent-insensitive matching of names
│   └── sources/
│       └── gutenberg/           # everything Gutenberg-specific
│           ├── catalog.py       # snapshots + CSV parsing (pure functions)
│           ├── selection.py     # filters, stratification, Manifest model
│           ├── download.py      # mirror download, idempotent and resumable
│           ├── boilerplate.py   # licence markers (START/END)
│           ├── clean.py         # registry -> raw -> cleaning pipeline -> outputs
│           ├── pipeline.py      # `gutenberg add`: select -> ... -> index in one run
│           ├── migrate.py       # one-off: old single-file layout -> current
│           └── cli.py           # argparse only, delegates to the modules above
├── tests/                       # pytest, no network (fake HTTP session);
│                                # test_cleaning_samples.py runs on the 11 samples
│                                # if present (GROWTH_RAG_SAMPLES_DIR or data/raw)
└── 03_RAG/
    ├── README.md                # this file
    ├── docker-compose.yml       # Qdrant (data in the Docker named volume growth_rag_qdrant)
    ├── eval/eval-v1/            # evaluation set, versioned: manifest.json + questions.jsonl (frozen)
    ├── eval/eval-v2/            # current set: v1 + dataset fixes (8.7)
    └── data/                    # git-ignored, see "Data layout"
```

Dependencies point one way: `cli` → source modules → `cleaning` / `registry` /
`config` → `utils`. `utils` never imports anything from the project, and
`cleaning` knows nothing about Gutenberg (owned books will reuse it with their
own adapter).

---

## Usage

Run from `03_RAG/` (the data folder defaults to `./data`; override with
`--data-dir` or `GROWTH_RAG_DATA_DIR`). After changing `pyproject.toml`, run
`uv sync` once so the `gutenberg` and `rag` commands are installed
(`uv sync --extra flag` adds FlagEmbedding for the reference encoder).

```powershell
uv run gutenberg catalog                       # dated snapshot of pg_catalog.csv
uv run gutenberg select --name pilot --languages en fr de it fi `
    --bookshelf "Category: Novels" --per-language 20
uv run gutenberg select --name hugo_fr --author "Hugo, Victor" --languages fr
uv run gutenberg download --manifest data/manifests/pilot.json
uv run gutenberg clean                         # every downloaded book not yet up to date
uv run gutenberg clean --manifest data/manifests/cleaning_samples.json
uv run gutenberg clean --book 1524 --force
uv run gutenberg chunk                         # every cleaned book not yet up to date
uv run gutenberg chunk --tokenizer approx-4cpt # quick trial, no tokenizer download
uv run gutenberg status
uv run gutenberg migrate                       # once, for data from gutenberg_acquire.py

docker compose up -d                           # Qdrant, from 03_RAG
uv run rag embed                               # chunks -> data/vectors (bge-m3, manual encoder)
uv run rag embed-check --book 1524             # manual vs FlagEmbedding on 32 real chunks
uv run rag index                               # vectors -> Qdrant (--recreate to rebuild)
uv run rag search "a ship lost in a storm"     # hybrid by default; --mode dense|sparse
uv run rag search "madness" --speaker OPHELIA --title Hamlet
uv run rag search "the bishop" --title "les miserables"   # all volumes found
uv run rag search "whaling" --author Melville --mode sparse
uv run rag search "sea adventure novel" --books --language en
uv run gutenberg add --manifest eval/eval-v1/manifest.json   # the evaluation books
uv run rag eval-check                          # locate every ground-truth quote (eval-v2)
uv run rag eval-check --show hamlet-ghost lm-fantine-sells   # the chunks that answer
uv run rag eval                                # score: dense / sparse / hybrid, books (saved run)
uv run rag eval --failures                     # ... and what was not found (hybrid)
uv run rag eval --compare eval-v1-20261025-2010 eval-v1-20261026-0930   # before / after
uv run rag model                               # pinned bge-m3 files: where they are (download once)
uv run rag model --rerank                      # same for the reranker
uv run rag rerank-check                        # manual vs FlagEmbedding reranker
uv run rag search "madness" --title Hamlet --rerank   # hybrid top 20 reordered by the cross-encoder
uv run rag eval --depth 100 --rerank 20 50 --diagnose  # baseline + reranker modes, where the answers are
uv run rag status
uv run pytest                                  # from the repository root
```

Filters are case-insensitive substring matches; values inside one option are
OR-ed, different options are AND-ed. `download` skips books already present,
skips known 404s unless `--retry-missing`, and can be interrupted (Ctrl-C) and
resumed with the same command. `clean` re-cleans a book only when it was never
cleaned, when `CLEANING_VERSION` changed, when its raw file changed (checksum),
when its outputs are missing, or with `--force`. Logs go to the console and
`data/logs/gutenberg.log`.

---

## Adding books (one command)

`uv run gutenberg add` runs every step for a selection: **select → download →
clean → chunk → embed → index** (code: `src/growth_rag/sources/gutenberg/pipeline.py`).
Each step is the same function as its own command, and idempotent: books
already done are skipped, so an interrupted run resumes when re-run.

```powershell
uv run gutenberg add --title "moby dick" --languages en --dry-run   # 1. look
uv run gutenberg add --id 2701                                      # 2. add exactly what you want
uv run gutenberg add --title hamlet --author Shakespeare --languages en fr de
uv run gutenberg add --author "Verne, Jules" --languages fr --per-language 5 --name verne-fr
uv run gutenberg add --manifest data/manifests/verne-fr.json        # resume / re-run
uv run gutenberg add --title faust --until chunk                    # stop after a step
```

- **Always start with `--dry-run`.** Title matching is on whole words, so
  "hamlet" also finds critical studies, other authors' works and *The memorials
  of the hamlet of Knightsbridge*; several editions of the same book are common
  (3 × *Moby Dick*). Narrow with `--author` / `--languages`, or use `--id`.
- Selection options are those of `gutenberg select` plus `--title` (accent- and
  case-insensitive, any of several titles) and `--id`. More than 50 books need
  `--yes`.
- Each run saves its selection as a manifest (`--name`, default
  `add-<date>-<time>`): what was added is recorded and reproducible — an
  evaluation set is a manifest plus its questions.
- If no catalog snapshot exists yet, it is downloaded first.
- If Qdrant is not running, everything up to the vectors on disk is done and the
  run says to start Qdrant and run `rag index`.
- Options passed to the steps: `--tokenizer`, `--backend`, `--batch-size`,
  `--delay`; `--until download|clean|chunk|embed|index`.
- The summary lists the status of every stage for the selected books and the
  books flagged by cleaning (usable, worth a look).

## Data layout

```
data/
├── registry.json              # what we have: one BookRecord per book, all sources
├── manifests/<name>.json      # what we want: criteria + catalog snapshot + IDs
├── catalog/gutenberg/         # catalog snapshots, never modified
├── raw/gutenberg/pg<id>.txt   # byte-exact files as served by the mirror
├── clean/gutenberg/pg<id>.txt # cleaned text (offsets refer to this file)
├── clean/gutenberg/pg<id>.json# structure: sections, paragraphs, annotations, cast, metrics
├── chunks/gutenberg/pg<id>.jsonl # one chunk per line
├── vectors/gutenberg/pg<id>.npz  # dense (fp16) + sparse vectors + book vectors
├── eval_runs/<set>-<date>.json   # one per `rag eval`: settings, results, ranks, metrics
└── logs/
```

- **Manifest vs registry.** The manifest is a dataset definition (you write or
  generate it, several can coexist, version them). The registry is the state
  of the local corpus (written by the pipeline, deduplicated across manifests,
  records which manifests requested each book).
- **Registry keys** are `<source>:<source_id>` (`gutenberg:1342`), so other
  sources can share the registry. The record schema is `BookRecord` in
  `registry.py`; it will become the payload of the Qdrant `books` collection
  and the properties of Book nodes in the graph stage.
- **Raw files are never modified.** Cleaning reads `raw/` and writes elsewhere,
  so cleaning rules can change without re-downloading.
- **Paths in records are relative** to the data root, so the folder can move
  (and later map to a bucket).

---

## Decisions taken

| Decision | Reason |
|---|---|
| No LangChain / LlamaIndex | Learning goal: implement and understand every step. |
| Pipeline RAG first, then retriever as a tool | Measure "is retrieval good?" separately from "does the model use the tool well?". Same retriever serves both. |
| Catalog CSV + official mirror (`gutenberg.pglaf.org`), not a RapidAPI wrapper | Wrappers only return metadata + links, add a key and rate limits. Gutenberg's robot policy points automated access to mirrors. |
| Selection by manifest, stratified by language with a seed | Controls corpus composition (Gutenberg is ~80% English), reproducible datasets for evaluation. |
| Explicit registry schema (Pydantic) | It is the contract between steps; validation catches errors early; maps directly to Qdrant payloads. |
| Embedding model separate from the LLM | Retrieval only needs query and documents in the same space; the LLM only sees text. Decoder token embeddings are not trained for similarity. |
| **bge-m3** as embedding model *(proposed, to confirm)* | 8192-token input, 100+ languages, dense **and** sparse vectors from one model (hybrid search in Qdrant without another model). multilingual-e5 is limited to 512 tokens. |
| Qdrant in Docker, configured by settings | Same client code locally and in the cloud: only URL / API key change. |
| Book-level vectors from the start | Recommendations need book-level similarity; aggregating chunk hits is biased toward long books. |
| Cleaning output = `.txt` + `.json`, offsets in **characters** (Python string positions) | Readable text; one offset unit for chunking, Qdrant and the graph (bytes differ from characters in Greek, Chinese…). |
| Generic `cleaning` package + thin source adapter | Owned books (EPUB, PDF) will only need their own adapter. |
| Books where detection fails are **kept** (one `main_text` section, status `flagged`) | A detection failure costs chapter information, not the whole book; flagged books can be listed, reviewed or filtered. |
| Registry: optional `cleaning` field, schema version unchanged | Policy: additive optional fields do not bump `SCHEMA_VERSION`; breaking changes do. |
| Speaker keys include the book key | `HAMLET` in two books gives two characters; merging is a graph-stage decision. |
| Chunk sizes in **bge-m3 tokens**: target 350, max 512, min 80 | Characters/words are wrong for Chinese or Greek; a few hundred tokens keeps retrieval precise (bge-m3 accepts 8192). Values to tune with evaluation. |
| Chunking rule chosen **per paragraph/section**, not per book | Books mix forms (prose preface before poems, prose speeches in plays). |
| No overlap; `previous`/`next` links instead | Overlap duplicates text and blurs citations; neighbours can be added to the context at retrieval time. Both options can be compared with RAGAS. |
| Context header (book, author, section path) **embedded, not stored in the text** | "He opened the door" alone means little; the stored text stays an exact span for citations. |
| Front matter, apparatus and TOC not chunked in v1 | Kept in the cleaned files; can be indexed later. |
| Two bge-m3 encoders: `manual` (default) and `flag` | `manual` shows each step (CLS pooling, sparse head); `flag` is the reference; `rag embed-check` verifies they agree. |
| Vectors written to disk before upload | Re-indexing, rebuilding collections or moving to a cloud Qdrant never requires re-embedding. |
| Chunks sorted by length before batching (batch 64) | Less padding: the largest speed gain; VRAM measured at ~1.4 GB peak for 32 × 512 tokens. |
| Delete a book's points before uploading it again | A book re-chunked into fewer chunks leaves no stale points. |
| Hybrid search = RRF over dense + sparse | Rank-based fusion: dense and sparse scores are on different scales. |

### Qdrant design (implemented in steps 4–5)

- Collection **`chunks`**: named vectors `dense` (bge-m3, 1024, cosine) and
  `sparse` (bge-m3 lexical weights); payload: the whole chunk (key, book key,
  section index and path, kind, language, offsets, text, header, speakers,
  annotations, previous/next) plus title, authors, source and embedding model;
  payload indexes on `book_key`, `language`, `kind`, `speakers`,
  `speaker_labels`, `section`.
- Collection **`books`**: one point per book, named vectors declared at
  creation even if filled later — `metadata` (title + authors + subjects +
  bookshelves, available now), `centroid` (mean of the book's chunk vectors,
  after step 4), `summary` (LLM-generated description, later). Points may lack
  some named vectors, so the collection does not need to be rebuilt.
- Point IDs: UUIDv5 derived from the deterministic key (`gutenberg:1342`,
  `gutenberg:1342/s3/c12`), so re-ingestion upserts instead of duplicating, and
  the graph stage can reference the same IDs.
- **Annotations** (footnotes, illustrations, … see 2.8): chunk payloads carry
  the keys of their annotations. Footnotes with substantial text can also be
  indexed as their own points (`kind: "note"`, `parent_chunk_key`), so they are
  retrievable and filterable; at answer time, a retrieved chunk's footnotes can
  be added to the context.
- **Graph stage** (planned links): `Chunk –HAS_NOTE→ Footnote`,
  `Chunk –NEAR→ Illustration`, `Section/Book –HAS_NOTE→ TranscriberNote`,
  `Book –PRODUCED_BY→ Person` (from credit lines).
- Store the embedding model name and version in the payload, to detect stale
  vectors after a model change.

---

## Points skimmed over and improvements left, per step

### 1. Acquisition (done)

Skimmed over / known limitations:
- **Translations and volumes are separate catalog entries.** Verne has 162
  entries in 11 languages; long novels are split into tomes (37 Hugo entries in
  French). Downloading is unaffected, but recommendations must not suggest a
  translation of a book already read → needs a **Work vs Edition** notion
  (graph stage), or at least deduplication on normalised author + title.
- **Two bookshelf taxonomies** coexist: 72 broad `Category: …` shelves and 340
  older specific ones. Substring matching means `Novels` also matches
  `Category: Historical Novels`.
- **Author parsing is best effort.** ~40 irregular entries remain (corporate
  authors, titles of nobility, conferences); `raw` is always kept. Contributors
  (translators, editors, illustrators) are listed as authors: `--author
  Baudelaire` also returns the Poe stories he translated. Filter on `role` if needed.
- **Copyright outside the US.** Gutenberg texts are public domain in the US;
  in the EU the rule is life + 70 years. `death_year` is in the registry, so a
  check is possible but not implemented (matters only if the corpus is
  redistributed or deployed publicly).
- Stratification is by language only.

Improvements left:
- Download the **HTML** edition as an alternative: chapter headings are marked
  up (`<h2>`, etc.), which can make structure detection far more reliable than
  heuristics on plain text.
- Mirror fallback (other mirrors from the official list) when one fails.
- Compare catalog snapshots to detect new or changed books.
- Registry as JSON is fine for thousands of books; move to SQLite (or the graph
  DB) beyond that.
- Store license / attribution text per book if the corpus is ever published.

### 2. Cleaning (v1 done)

Design based on the measurement of 11 deliberately varied samples
(`data/manifests/cleaning_samples.json`: en, fr, de, es, it, fi, el, zh; novels,
a multi-volume work, plays, poetry, a low-ID file).

#### 2.1 File level — simpler than expected

| Check | Result on all 11 samples |
|---|---|
| Encoding | Valid UTF-8, no BOM, already NFC |
| Line endings | **CRLF everywhere** → normalise to LF first |
| START/END markers | Present, modern wording only |
| Licence footer | Exactly 18,389 characters in every file |
| Header before START | 730–980 characters |

The mirror serves regenerated files in one uniform format, so marker-based
stripping is reliable. Keep the fallback anyway (books without markers are
logged and flagged, not guessed), and keep NFC normalisation as a cheap safety
step.

#### 2.2 Front matter — classify, do not delete

Between START and the first real chapter, the samples contain: credit lines
("Produced by …"), transcriber's notes (Italian, Greek), title pages (sometimes
centred with leading spaces), tables of contents (8 of 11 books), and **real
text by other authors** (Frichet's preface to *Fleurs du Mal*, Scherillo's essay
before *I promessi sposi*, Cervantes' legal preliminaries).

→ Detect structure first; everything before the first body heading becomes a
`front_matter` section. It is stored and labelled; chunking decides whether to
index it. Transcriber's notes are also metadata (the Italian one documents
`_italics_` and `=bold=`).

#### 2.3 Headings — the main difficulty

- **Every table of contents repeats the headings**: *Huckleberry Finn* has 84
  heading-like lines for 42 chapters, *Moby Dick* 270 for 135. A TOC entry is
  followed by another heading or a one-line summary; a body heading by prose.
  → A heading counts only if a substantial block of text follows it.
- **Patterns vary a lot:**

  | Book | Body heading | Note |
  |---|---|---|
  | Huckleberry Finn | `CHAPTER I.` | title only in the TOC |
  | Moby Dick | `CHAPTER 1. Loomings.` | number and title on one line |
  | Les Misérables | `Livre premier--Un juste`, then `Chapitre I` + **title on next line** | TOC puts both on one line |
  | Don Quijote | `Capítulo primero.`, then `Capítulo II.` | ordinal word, then roman numerals |
  | I promessi sposi | `CAPITOLO PRIMO.`, then `CAPITOLO II.` | same switch |
  | Seitsemän veljestä | `ENSIMMÄINEN LUKU`, `TOINEN LUKU` … | ordinal words only, no TOC |
  | Λουκής Λάρας | `ΚΕΦΑΛΑΙΟΝ Α'.` | Greek numerals |
  | 紅樓夢 | `第一回　title` + `-----` line | Chinese numerals, variant `第一一九回` |
  | Hamlet | `ACT I` / `SCENE I.` | TOC uses `Scene I.` |
  | Faust | `Zueignung`, `Nacht`, `Dom` … | **no numbering**, some names repeated |
  | Fleurs du Mal | `I`, `II` … then a title line | numbering restarts per section |

- **Strategy:** when a TOC exists, use it as a **heading dictionary** and find
  each entry again later as a standalone line (language-independent; the only
  way to recover *Faust*'s structure). Fall back to per-language pattern tables,
  including ordinal words, for books without a TOC (Finnish, Greek).
- Patterns must be **case-sensitive and line-anchored**: a case-insensitive
  `capítulo` matched a sentence in Cervantes' prologue.

#### 2.4 Hierarchy — numbering restarts

Livre → Chapitre; Parte → Capítulo (numbering restarts in part II, and part I is
itself divided into four "partes"); ACT → SCENE; section → poem.

→ Section identity is a **sequential index** within the book (`s0`, `s1`, …),
never the printed number ("Chapitre I" occurs eight times in Tome I). Printed
labels and the hierarchy path (`Livre 2 / Chapitre 1`) are metadata. For the
graph: generic `Section` nodes with a `level`, not a fixed Book → Chapter model.

#### 2.5 Paragraphs and line wrapping

- Prose is hard-wrapped at ~70 characters (median line length 65–72),
  paragraphs separated by blank lines → join the lines of a block with a space.
- **Verse must not be joined**: *Faust*, *Fleurs du Mal* and much of *Hamlet*
  have median line lengths of 32–45 and meaningful line breaks. Detect per
  block (*Hamlet* mixes verse and prose). Plays also have speaker lines and
  stage directions.
- **Chinese breaks the blank-line rule**: lines wrap at ~35 characters
  mid-sentence and paragraphs are marked by full-width indentation (`　　`).
  Measured with blank lines: 425 "paragraphs", the largest 48,651 characters.
  → Split on the indentation and **join lines without a space**.
- Paragraph sizes: medians 50–290 characters, but single paragraphs reach 9,180
  (Les Misérables) and 6,813 (Quijote) → chunking must both merge and split.

#### 2.6 Inline markup and conventions

- `_italics_` (935 in *Huckleberry Finn*), `=bold=` (Italian): remove the
  markers, keep the text; optionally record the spans for later rendering.
- `--` used as a dash (French, Finnish, Italian) vs real em dashes (*Moby
  Dick*): normalise if it matters for tokenisation.
- Footnote references `[1]` with bodies elsewhere, `[Illustration]` markers,
  separator lines (`-----`, `* * *`): **extracted as annotations** (2.8), not
  deleted.

#### 2.7 Several works or authors in one file

The Italian volume holds Scherillo's essay **and** the novel; *Don Quijote*
holds both parts; *Fleurs du Mal* opens with another author's preface.
→ Each section has a `role`: `front_matter`, `main_text`, `apparatus` (notes,
texts by other authors). Needed for recommendations and correct citations.

#### 2.8 Annotations — kept and linked, not deleted

Footnotes, illustrations, transcriber's notes, credits and structural breaks
are removed from the running text but **kept as annotation records** pointing to
where they occurred, so they can later be linked to chunks and become graph
nodes.

What the samples show for footnotes (*I promessi sposi*): references are inline
(`gloria[1]».`); bodies are separate blocks starting with `[1]`, placed right
after the paragraph, sometimes wrapped over several lines; 45 references for 44
bodies; numbering restarts.

Annotation record (one per element, stored with the cleaned book):

| Field | Meaning |
|---|---|
| `key` | deterministic, sequential per book: `gutenberg:45334/a12` |
| `type` | `footnote`, `illustration`, `transcriber_note`, `credit`, `scene_break`, … |
| `label` | printed marker (`[1]`, `[Illustration: …]`) |
| `text` | footnote body or caption (may be empty) |
| `anchor` | character offset **in the cleaned text** where the reference occurred |
| `section` | sequential section index |
| `raw_span` | start/end offsets in the raw file (provenance, debugging) |
| `status` | `matched`, `unmatched_reference`, `unmatched_body` |

Rules:
- **Record anchors while building the cleaned text** (offset = current output
  length when the marker is met), not on the raw text: every later
  transformation (line joining, markup removal) would shift raw offsets.
- **Footnote matching**: because numbering restarts, link a body `[n]` to the
  nearest preceding unmatched reference `[n]` in the same section. Unmatched
  references or bodies are kept and flagged, never dropped.
- Book-level elements (credits, transcriber's notes) anchor to the book or to
  the section they appear in, not to a text position.
- Scene breaks (`* * *`) are both annotations and **preferred chunk
  boundaries**.
- Linking to chunks happens at **chunking time** (step 3), since chunk
  boundaries don't exist yet during cleaning. Because annotations keep offsets,
  re-chunking only recomputes the mapping.

#### 2.8b Speakers in plays — included in v1

Needed to answer queries such as "everything Ophelia says". Both plays in the
samples use the same convention: speaker name in capitals with a period, alone
on its line, followed by the speech; stage directions on their own lines
(*Hamlet*: 1,122 labels — HAMLET 358, HORATIO 107, KING 102 …; *Faust*: 801 —
MEPHISTOPHELES 219, FAUST 200 …).

- Each block of a play gets a `kind` (`speech`, `stage_direction`) and a
  `speaker`; each book gets a cast list (labels with counts).
- Chunking avoids splitting a speech where possible and records the speakers of
  each chunk in the payload → Qdrant filter on `speaker`. Graph:
  `Chunk –SPOKEN_BY→ Character`.

Points of attention:
- **False positives**: chapter headings have the same shape (`CAPITOLO II.`,
  `ENSIMMÄINEN LUKU`). Speaker detection runs *after* heading detection, and
  only in books that behave like plays (bookshelf "Plays/Films/Dramas" or a high
  density of such labels).
- **One character, several labels**: *Faust* has MARGARETE (67) and GRETCHEN
  (14) for the same person; `FIRST CLOWN` vs `CLOWN`. Labels are stored as
  printed; merging aliases into one character is a graph-stage task.
- **One name, several characters (homonyms across books)**: speaker keys always
  include the book key (`gutenberg:1524/speaker/hamlet`), so the Hamlet of
  Shakespeare and a Hamlet in another play are never confused. The reverse
  problem — the *same* character across editions or translations of one work
  (Hamlet in English and in a French translation) — needs the Work vs Edition
  notion of the graph stage; it is not resolved by cleaning.
- **Other formats** are not supported in v1: abbreviated inline names
  (`Ham. To be …`), speakers inside the speech line, verse dramas with other
  conventions.
- **Dialogue in novels** (`“…,” said Huck`) is a different, much harder
  problem: quote detection plus attribution to a character (pronouns, implicit
  turns). It needs NLP (coreference) or an LLM pass, and belongs to the graph
  stage. Until then, character queries on novels rely on plain retrieval of the
  name.

#### 2.9 Output of the cleaning step

Per book, in a separate folder (`data/clean/<source>/`), never overwriting raw:
- normalised main text;
- sections: index, level, label, hierarchy path, role, start/end offsets;
- paragraphs (offsets), with a `verse` flag;
- annotations (2.8);
- validation metrics: headings found vs TOC entries, section length
  distribution, share of front matter, footnote match rate.

The registry records the cleaning status and the **cleaning version**, so books
can be re-cleaned when the rules change. Books whose metrics disagree are
flagged for review instead of silently accepted.

Improvements left: the HTML edition as an alternative input (headings and
footnotes are marked up); other speaker formats in plays and dialogue
attribution in novels (2.8b); small caps; recording italic spans; a review tool
for flagged books.

#### 2.10 v1 implementation: results on the samples and known limits

Acceptance results (`tests/test_cleaning_samples.py`, all passing):

| Book | Expected structure | v1 result |
|---|---|---|
| Huckleberry Finn | 42 chapters + "Chapter the Last" | 43 ✓ |
| Hamlet | 5 acts, 20 scenes; speakers | 5 + 20 ✓; 1,121 speeches, HAMLET 358, "PROLOGUE" kept as a character |
| Don Quijote | 2 parts (part I in 4 *partes*), 126 chapters + prologue | ✓ (headerless TOC recognised by its echo) |
| Faust | 28 scenes (no numbering) | 28 ✓ via TOC dictionary; verse 72 % |
| Moby Dick | 135 chapters + Epilogue + Etymology + Extracts | 138 ✓ |
| Les Fleurs du Mal | sections → titled poems → numbered parts | 13 sections, 109 poems, 23 parts; verse 93 % |
| Seitsemän veljestä | 14 chapters (ordinal words) | 14 ✓ |
| Les Misérables I | 8 *livres*, 70 *chapitres*, titles on next line | ✓, titles attached |
| 紅樓夢 | 120 chapters, headings glued to text | 120 ✓, no spaces inserted |
| Λουκής Λάρας | Greek-numeral chapters | 10 ✓ |
| I promessi sposi | essay (10) + introduction + 38 chapters; footnotes | ✓; 44 of 45 footnotes matched, 46 illustrations |

Rules added while testing (each one fixed a sample, and is covered by a test):
- **Verse test**: a line break is *voluntary* when the next word would have
  fitted on the line; a block is verse when most breaks are voluntary. Width =
  99th percentile of line lengths, with a 60-character floor for Western texts
  (mostly-verse books would otherwise give a too-small width), none for Chinese.
- **Keyword + number**: a keyword (Chapter, Livre, Capítulo…) counts only when
  followed by a numeral or an ordinal word, so "Book in hand, he…" is not a
  heading; long wrapped titles are accepted (up to 4 lines / 320 characters).
- **TOC regions** end where the first entry is repeated, at running prose, or
  at another list (illustrations, *Dramatis Personæ*). Page references
  (`»  5`, `Pag. VII`, dot leaders) are removed from entries. TOC-only matches
  must be single lines that are not dialogue.
- **TOC echoes**: a heading before the first numbered chapter that is repeated
  word for word later is a TOC entry (tables without a "Contents" line).
- **Chinese**: `第…回` lines must be followed by a space or the end of line
  (`第四回中…` is a sentence) and are split from glued text.
- **Plays**: speaker blocks cannot be headings but count as text after a
  heading; the bookshelf hint still requires ≥ 5 labels.
- **Poetry collections** (≥ 15 % verse blocks, not a play): capitalised
  single-line titles become poem headings after the first regular heading; a
  title directly followed by another title is a section of poems.

Known limits of v1 (candidates for v2):
- **Roles** are rule-based: `front_matter` before the first heading,
  `apparatus` only for sections titled Notes / Index / Appendix / Contents….
  Scherillo's essay in the Italian volume is detected as its own sections but
  marked `main_text`: telling "a different work by another author" from the
  main work needs metadata or an LLM pass.
- **Stage directions** without brackets in the middle of a scene (common in
  German) are attributed to the current speaker; speaker formats other than
  `NAME.` on its own line are not recognised (2.8b).
- **Transcriber's notes** spread over several blocks: only the first block is
  captured; the rest stays in the text. Notes at the end of the book are
  recognised only if their first line matches the known starters.
- **Front-matter TOC paragraphs** are kept as `toc` paragraphs (to be skipped by
  chunking); a TOC without header is only detected through its echoes, so its
  lines stay as ordinary front-matter text.
- **Poem titles** are only detected when written in capitals; a collection with
  title-case titles falls back to numbered parts or a single section.
- Heading keyword tables cover en, fr, de, es, it, fi, el, zh (+ some pt/la by
  overlap). Other languages rely on the TOC dictionary or fall back to one
  flagged section — run `gutenberg status` to see `cleaning_issues`.
- Footnotes in the `[Footnote 1: …]` and `[1]` formats are supported; notes
  referenced with superscripts or `{1}` are not.

#### 2.11 First run on 111 books (v1.0 → v1.1)

v1.0 flagged 37 of 111 books. Diagnosis on the raw files:

| Flag (v1.0) | Books | Cause | Fix in v1.1 |
|---|---|---|---|
| `toc_heading_mismatch` | 21 | Mostly a **false alarm**: TOC printed as a column headed "CHAPTER" with entries `I. THE DRAMA…`, body `CHAPTER I`. Chapters were found by patterns, but the check only counted TOC matches | Check compares the TOC with headings found by *either* method; column headers (`CHAPTER`, `Pages`, `Seite`, `»`) are dropped from entries |
| | | French/Italian collections: **TOC printed at the end of the book** | Tables in the second half of the book are searched *before* themselves |
| | | Numbered entries `1. Alt-Freibergs Romantik`, body `Alt-Freibergs Romantik.` | Entries are also matched without their leading number |
| | | A chapter `I` swallowed by the TOC just before the prose | The heading just before running prose is given back to the body |
| `no_headings` | 14 | Short stories without chapters (23–46k characters) | Not severe under 60k characters: `no_headings_short_work` |
| | | Capitalised unnumbered chapter titles in pulp fiction (`THE CANNELL MYSTERY`) | Last resort for prose: ≥ 3 capitalised titles each followed by ≥ 1,000 characters |
| | | Sections `_(6º)_` | Bare numerals accept `º`/`°`, parentheses, markup |
| `single_dominant_section` | 5 | `CHAPTER ELEVEN`… (number words beyond ten), `LETTRE I.` | English number words with compounds (`TWENTY-ONE`), French `-ième`, Italian `-esimo`; letter keywords (Letter, Lettre, Lettera, Carta, Brief, Kirje) |
| | | `CHAPITRE PREMIER` with a bare `I` printed below it, counted twice | The numeral is merged into the heading |
| (all) | | Collections named in the catalog title (`Gladius Dei; Schwere Stunde`) | Title parts are searched as headings when ≥ 2 are found (`; Or, …` alternative titles excluded) |

Result on the same 37 books: **30 ok, 7 flagged**, all justified:
- *Gladius Dei; Schwere Stunde*: the titles appear only in the Gutenberg header
  and an editor's note, not as headings in the text.
- *La gran rivale*: all 8 stories found; the end-of-book TOC runs into the
  publisher's catalogue (`ROMANZI ITALIANI …`), which inflates the entry count.
- 5 long texts with no recognisable heading: dated letters (*Natalie*),
  title-case chapter titles with a period (`Das grüne Holz.`), a 16th-century
  Latin/French text, German sketches with title-case titles.

Still open (v1.2 candidates): stop a TOC at a publisher's catalogue; title-case
chapter titles in prose (risky: indistinguishable from short lines of dialogue
without more context); dated letters as sections (`den 17. December.`).

`CLEANING_VERSION` is now 1.1.0: the next `gutenberg clean` re-cleans every
book automatically.

#### 2.12 Found by the evaluation (to fix later)

Evaluation reads chunks closely, and showed two structure errors in
*Les Fleurs du Mal* (6099):
- **Poems with the same title merged.** The four poems titled "Spleen" ended up
  in one section, so the poem chunker never saw the boundaries between them:
  chunk `s62/c2` holds the end of "Je suis comme le roi d'un pays pluvieux" and
  only the first stanza of "Quand le ciel bas et lourd" (the rest is in `c3`).
  Its vector is dominated by the other poem; even the exact line does not
  retrieve it. Repeated titles must still split poems.
- **A poem taken as a parent section.** Many poems carry the path
  "LE MASQUE / …": the poem "Le Masque" (with its subtitle) was read as a
  section heading. Its title is then embedded in the header of every following
  poem: noise in those vectors.

Both change the chunks, so the book is re-chunked and re-embedded, and the
evaluation targets move: measure with `rag eval --compare`, after the
retrieval upgrades (one change at a time).

### 3. Chunking (v1 done)

#### 3.1 Rules

Within each section (sections are never crossed), paragraphs of kind `heading`
and `toc` are skipped (headings go to the header), then:

| Mode (per section) | Detected by | Packing |
|---|---|---|
| **prose** | default | paragraphs packed up to 350 tokens; cut preferred at scene breaks (`* * *`); a tail under 80 tokens merged into the previous chunk |
| **poem** | ≥ 80 % verse paragraphs, not a play | the whole poem is **one chunk if ≤ 512 tokens, even below 80**; longer poems packed by stanza, never cut inside a stanza; parts labelled `k/n` |
| **play** | any `speech` / `stage_direction` paragraph | speeches packed within the scene; a speech continuing over several stanzas stays one piece; stage directions stay where they are |

Oversized paragraphs (alone > 512 tokens) are cut at sentence ends (prose) or
lines (verse) into **balanced** parts (700 tokens → 2 × 350, never 512 + 188),
labelled `k/n`. A single sentence above the maximum is cut at words (characters
for scripts without spaces).

Each chunk carries: `key` (`gutenberg:1342/s3/c12`), exact `start`/`end` and
`text` (= `cleaned_text[start:end]`), `header` (book — author, section path and
title) used only for embedding (`embedding_text()`), `tokens` and `tokenizer`,
`section_path`, `kind`, `language`, `speakers` (book-scoped keys) and
`speaker_labels`, `annotations` (keys of annotations anchored inside),
`previous`/`next` (reading order, across sections), `part`.

Re-chunking happens when the rules (`CHUNKING_VERSION`), the tokenizer or the
size settings change, when the **cleaned text changed** (checksum, not date: a
re-clean producing the same text costs nothing), when the output is missing, or
with `--force`. A tokenizer that cannot be loaded stops the run before the
first book, with a hint.

Settings: `GROWTH_RAG_TOKENIZER` (default `BAAI/bge-m3`, downloaded on first use,
tokenizer files only), `GROWTH_RAG_CHUNK_TARGET_TOKENS` / `_MAX_TOKENS` /
`_MIN_TOKENS`. `approx-4cpt` (~4 characters per token, 1 per CJK character)
needs no download and is used by the tests.

#### 3.2 Results on the samples (approximate counter)

9,145 chunks for the 11 samples; no chunk above 512; medians 230–315 tokens.
Fleurs du Mal: 135 poem chunks, > 75 % of poems in one piece. Hamlet: speeches
never split except the one longer than a chunk (*O, what a rogue and peasant
slave am I*, 611 tokens, cut at verse lines). Every annotation anchored in an
indexed section is linked to exactly one chunk. Acceptance tests:
`tests/test_cleaning_samples.py` (chunking part), unit tests:
`tests/test_chunking.py`, `tests/test_chunking_run.py`.

Cleaning was bumped to **1.2.0** along the way: a poem title followed by a
single quatrain (~180 characters) was dropped by the 200-character rule for
headings; poem titles now need 60 characters of text.

#### 3.3 Points of attention and improvements left

- **Sizes are measured with the tokenizer you run with.** The sample figures
  above use the approximate counter; with bge-m3, chunk counts change slightly.
  Run with the real tokenizer before indexing (the registry records which one).
- **Sentence splitting is rule-based**: abbreviations list is short and
  language-agnostic; a neural segmenter (e.g. wtpsplit) can replace it if
  evaluation shows bad cuts. It is only used for oversized paragraphs.
- **Dialogue in novels** is packed like any prose (no speaker metadata, see 2.8b).
- **Prose speeches in plays** that run over 512 tokens are cut at sentences, as
  any prose.
- **Front matter / apparatus / TOC** are not chunked; prefaces by the author
  (e.g. Cervantes' prologues, kept as `main_text`) are.
- **Sections that are only a heading** (a section title directly followed by a
  sub-section) produce no chunk: their label lives in the children's paths.
- To evaluate with RAGAS: target size (250 / 350 / 500), overlap vs neighbour
  expansion, header on/off, poem-as-one-chunk vs stanza chunks.

### 4. Embedding (v1 done)

#### 4.1 What bge-m3 computes (both encoders)

One transformer pass (XLM-RoBERTa large, 568M parameters) gives the last hidden
states `[batch, tokens, 1024]`. From them:
- **dense** = hidden state of the first token (`<s>`, the CLS position),
  L2-normalised → cosine similarity is a dot product;
- **sparse** = `ReLU(sparse_linear(hidden))`, one weight per token; per text,
  the **maximum weight of each token id** is kept, special tokens (`<s>`,
  `</s>`, padding, unknown) are dropped. The result is a vector over the
  tokenizer's ~250k-token vocabulary with a few dozen non-zero entries: a
  *learned* BM25 that favours exact terms (names, rare words). The linear head
  is not in the transformers checkpoint but in a separate file of the model
  repository (`sparse_linear.pt`).
- The third bge-m3 output (multi-vector "ColBERT") is not used.

Nothing is trained here: the weights are BAAI's, used as published (what is
pretrained and what is our code, and how the sparse weights were learned:
section 8.6).

`encoders/manual.py` does exactly this with transformers; `encoders/flag.py`
calls FlagEmbedding. The model-independent maths (`cls_pooling`,
`l2_normalize`, `lexical_weights`) lives in `encoders/base.py` and is tested
without torch. `rag embed-check` encodes the same chunks with both and reports
the dense cosine (expect > 0.999), sparse token overlap (expect ~1.0) and the
sparse score difference (expect < 1 %); small deviations come from fp16
rounding.

#### 4.2 Run

- Text embedded = `chunk.embedding_text()` (header + text). Queries are embedded
  as they are (bge-m3 needs no instruction prefix).
- Chunks are sorted by length, batched (64 by default), and put back in order.
- Per book, `data/vectors/<source>/<stem>.npz`: dense vectors in **float16**
  (half the size; negligible effect on cosine), sparse vectors in CSR layout,
  plus the book's `metadata` vector (title, author, subjects, bookshelves,
  language) and `centroid` (normalised mean of its chunk vectors).
- Re-embedding when the chunks file changes (checksum), the model or
  `EMBEDDING_VERSION` changes, the output is missing, or `--force`. The backend
  is recorded but does not trigger re-embedding (same vectors expected).
- The model is loaded once before the first book; if it cannot be loaded, the
  run stops with a hint instead of failing book after book.
- **Model files** (`encoders/files.py`, `utils/hf.py`): bge-m3 is pinned to a
  commit of its repository (`BGE_M3_REVISION` in `config.py`, main of
  2024-07-03). The needed files are resolved once into a local folder (Hugging
  Face cache first, download only what is missing), and both encoders and the
  chunking tokenizer load from that folder: **no Hub request once cached**,
  and the same weights every run. `rag model` shows the folder and sizes (and
  downloads on a new machine, e.g. a cloud VM). The revision is recorded in
  each book's `EmbeddingInfo`; changing it re-embeds (books embedded before it
  was recorded are not redone for that).
- Settings: `GROWTH_RAG_EMBED_BACKEND`, `_EMBED_MODEL` (repo id or local
  folder), `_EMBED_REVISION` (commit, branch, or empty for main), `_EMBED_DEVICE`,
  `_EMBED_FP16`, `_EMBED_BATCH_SIZE`, `_EMBED_MAX_LENGTH`. `HF_HUB_OFFLINE=1`
  forbids any download (an incomplete cache then gives a clear error).

#### 4.3 Points of attention

- **VRAM** (RTX 5070, 12 GB, measured): weights 1.08 GB in fp16, peak 1.45 GB
  for 32 × 512 tokens, ~1.65 GB reserved. Check the baseline with `nvidia-smi`
  (another process — a notebook kernel holding a model — may already use GB).
  Indexing does not need Mistral loaded; at query time bge-m3 can run on CPU.
- **Windows**: when VRAM is full the driver may spill into shared system memory
  instead of failing (everything becomes very slow). NVIDIA Control Panel →
  "CUDA – Sysmem Fallback Policy" → "Prefer No Sysmem Fallback" gives a clear
  error instead.
- **Weights file**: bge-m3's main branch has `pytorch_model.bin` but no
  `.safetensors`. v1 asked for safetensors only, so transformers took them from
  an unmerged conversion pull request (`refs/pr/130`), and FlagEmbedding
  downloaded the whole main branch (with a 2.2 GB ONNX copy). Both encoders now
  read the same main-branch files; `embed-check` measured identical vectors
  (cosine 1.00000) from the two copies. Loading `.bin` is safe with torch ≥ 2.6
  (`weights_only=True`). Cleanup of the Hugging Face cache, optional: the
  `refs/pr/130` revision and the `onnx/` files are no longer used (`hf cache
  scan`, or `hf cache ls` in recent versions, shows what takes space).
- **FlagEmbedding and transformers 5**: works (FlagEmbedding installed with
  transformers 5.x). `embed-check` on 32 Hamlet chunks: dense cosine 1.00000,
  sparse Jaccard 1.0000, sparse score difference 0.0000.
- Improvements left: ColBERT vectors (late interaction, Qdrant multivectors);
  quality on low-resource languages; `summary` book vectors from an LLM.

### 5. Indexing (Qdrant, v1 done)

- `docker compose up -d` from `03_RAG` (dashboard on
  http://localhost:6333/dashboard). Pin `QDRANT_VERSION` (in `03_RAG/.env`) to
  your qdrant-client version.
- **Persistence**: storage is the Docker named volume `growth_rag_qdrant`, which
  survives `stop` and `down`; only `docker compose down -v` or `docker volume rm`
  delete it. Not a bind-mounted Windows folder: on Docker Desktop/WSL, Qdrant
  storage on a host folder is known to lose or corrupt data after restarts.
- **The source of truth is `data/vectors`**, not Qdrant: if the volume is lost,
  `uv run rag index --recreate` rebuilds both collections from disk without
  re-embedding. `--recreate` (or `--force`) is needed because the registry still
  records the books as indexed.
- Cloud later: set `GROWTH_RAG_QDRANT_URL` and `GROWTH_RAG_QDRANT_API_KEY`;
  `rag index` uploads the same vectors (the target is recorded per book, so a
  new target triggers a full upload). `GROWTH_RAG_QDRANT_PATH` runs Qdrant
  embedded in the process (tests).
- Re-upload when the vectors file changes (checksum), the target or collection
  names change, or `--force`; `--recreate` deletes and rebuilds both
  collections (needed if the vector size changes: the command says so).
- Upload per book: delete its chunk points, upsert in batches of 256, upsert
  its book point (`metadata` + `centroid`; `summary` stays empty).
- `rag search`: dense, sparse or hybrid (RRF over 4 × k candidates of each),
  filters on language, book, kind and speaker label; `--books` searches the
  books collection with the `metadata` or `centroid` vector. It is a check of
  the index, not yet the retrieval step.
- Improvements left: quantisation (scalar/binary) when the corpus grows;
  snapshots/backups; collection aliases for zero-downtime re-indexing. The
  chunk text is stored twice (Qdrant payload and chunks file): fine at this
  scale, revisit if storage matters.

### 6. Retrieval (reranking v1 done)

**What exists (baseline, `rag search`)**: hybrid dense + sparse with RRF,
filters on language, book, kind, speaker; books named by `--title` / `--author`
(resolved in the registry: accent- and case-insensitive, whole words, every
matching volume or edition; suggestions when nothing matches); book search on
the `metadata` or `centroid` vector.

**Why it is only a baseline**: it returns the k nearest chunks and stops. It
does not check that they actually answer the question, does not give the model
enough surrounding text, and does not understand the question beyond its
embedding. Upgrades, in order of expected value:

1. **Evaluation set first.** Done: `eval/eval-v1`, 89 questions across
   languages and kinds (prose, poems, plays, footnotes), ground truth checked
   by `rag eval-check`, scored by `rag eval` (hit@k, recall@k, MRR, nDCG per
   mode and slice; section 8). RAGAS context precision / recall later. Every
   change below is measured with `rag eval --compare`.
2. **Reranking.** Done (v1, 6.1): the hybrid top N is rescored, pair by pair,
   by a multilingual cross-encoder (bge-reranker-v2-m3). Bi-encoders compare
   two separate vectors; a cross-encoder reads query and passage together.
3. **Context expansion.** Add `previous` / `next` chunks (same section) around
   each hit, merge adjacent hits, attach linked footnotes; for poems, the whole
   poem. Replaces overlap (README 3.1).
4. **Diversity and deduplication.** Several hits from the same passage, or from
   two editions/translations of the same work, waste the context budget: MMR,
   a cap per book, Work-level dedup when available.
5. **Query understanding** (with Mistral): extract filters from natural language
   (title, author, speaker, language) and resolve them with the same resolver;
   rewrite or decompose questions (multi-query); optionally HyDE.
6. **Fusion tuning.** RRF vs weighted fusion, candidate counts per branch;
   sparse matters for names and rare words, dense for paraphrases.
7. **Cross-lingual policy.** A French question can retrieve German chunks; decide
   per use case whether to filter by language or let the model translate.
8. **Two-stage for recommendations.** Books first (metadata/centroid), then
   chunks within the chosen books; exclude other editions of a book already read.
9. **Context assembly** for the LLM (step 7): token budget, ordering, citation
   ids (`key` + offsets), headers.

#### 6.1 Reranking (v1 done): `rerank/`

```powershell
uv run rag model --rerank                      # locate / download once the pinned reranker files (2.3 GB)
uv run rag rerank-check                        # manual implementation vs FlagEmbedding, on Hamlet chunks
uv run rag search "madness" --title Hamlet --rerank          # hybrid top 20, reordered, top 5 shown
uv run rag search "the bishop's candlesticks" --rerank 20 --k 3
uv run rag eval --depth 100 --rerank 20 50 --diagnose        # + modes chunks:rerank@20 / @50
```

**Why** (diagnosis of eval-v1, 8.6): hybrid search has an answer in its top 100
for 60 of 69 questions, but first for only 23. The answers are collected; the
order is wrong. Reordering is what a reranker does.

**What a cross-encoder is, compared with our search.** bge-m3 is a
*bi-encoder*: question and chunk are encoded separately, each into its own
vector, and only the vectors are compared. That is what makes search fast
(chunk vectors are computed once, at indexing), but the model never sees the
question and the passage together. A *cross-encoder* reads them as one input:
```
<s> question </s></s> passage </s>  →  transformer  →  one number (a logit)
```
Every word of the question can attend to every word of the passage: "who
rescues Ishmael" can be matched with "the Rachel… picked me up", not just with
passages that are about Ishmael in general. Much more accurate, much slower:
one full model pass per (question, passage) pair, nothing can be precomputed.
Hence two stages: the fast search collects N candidates (hybrid: the best pool,
8.6), the reranker reorders only those N.

**The model**: bge-reranker-v2-m3 (BAAI), a cross-encoder fine-tuned from
bge-m3's base (XLM-RoBERTa large, multilingual, 568M parameters, ~1.1 GB in
fp16, so it fits next to bge-m3 on the GPU). Used as published: nothing is
trained here. Pinned to a commit (`BGE_RERANKER_REVISION`), loaded from a local
folder like bge-m3 (`rerank/files.py`, `utils/hf.py`). Its main branch has
`model.safetensors`.

**Our code** (`rerank/manual.py`): tokenizer on the (question, passage) pair,
`AutoModelForSequenceClassification` (the classification head on the first
position is part of the checkpoint, one output), the logit is the score.
`rerank/flag.py` calls FlagEmbedding's `FlagReranker`, the reference;
`rag rerank-check` compares the two on real chunks (expect logits within 0.05
in fp16, rank correlation > 0.99, same best passage).

**What the reranker reads**: the same text the chunk was embedded with,
header + text (`passage_text`), so title, author and section path help it.
Question + chunk stay well under `max_length` (1024 tokens); nothing is
truncated in practice.

**Scores**: logits, comparable within one question only (they rank, they are
not probabilities). `rag search --rerank` also shows sigmoid(logit) in
brackets, easier to read but not calibrated, and the candidate's rank before
reranking ("was 7").

**Candidate pool N**: the reranker can only promote what the search returned
(the ceiling in `--diagnose`). A bigger N reaches more answers but costs more
(N pairs per question) and gives the reranker more chances to be fooled by a
near-miss. `rag eval --rerank 20 50` measures both; on eval-v2, 20 wins (see
results), so the default of `rag search --rerank` is 20
(`GROWTH_RAG_RERANK_CANDIDATES`).

**Cost**: N model passes per question, in batches of 16 pairs (sorted by
length). For 69 questions × (20 + 50) pairs, about a minute on the RTX 5070.

**Settings**: `GROWTH_RAG_RERANK_BACKEND` (manual / flag), `_RERANK_MODEL`,
`_RERANK_REVISION`, `_RERANK_DEVICE`, `_RERANK_FP16`, `_RERANK_BATCH_SIZE`,
`_RERANK_MAX_LENGTH`, `_RERANK_CANDIDATES`.

**Results** (eval-v2, run `eval-v2-20261025-235752`, 69 scored questions;
`rag rerank-check`: manual and FlagEmbedding identical, max logit difference
0.0000, rank correlation 1.0000, same best passage 4/4):

| Mode | hit@1 | hit@5 | hit@10 | recall@10 | MRR | nDCG@10 |
|------|-------|-------|--------|-----------|-----|---------|
| dense | 0.42 | 0.58 | 0.68 | 0.66 | 0.50 | 0.52 |
| sparse | 0.17 | 0.33 | 0.39 | 0.38 | 0.24 | 0.26 |
| hybrid | 0.35 | 0.61 | 0.67 | 0.65 | 0.46 | 0.49 |
| **rerank@20** | **0.59** | **0.77** | **0.78** | **0.77** | **0.65** | **0.67** |
| rerank@50 | 0.55 | 0.72 | 0.78 | 0.77 | 0.63 | 0.65 |

- **The answer comes first for 41 questions instead of 24** (hybrid → rerank@20),
  and is in the top 5 for 53 instead of 42.
- **rerank@20 almost reaches its ceiling.** 56 questions have an answer in the
  hybrid top 20; the reranker puts it in the top 5 for 53 of them. What it
  still misses is mostly what it never sees: the 13 questions whose answer is
  not among the 20 candidates.
- **20 candidates beat 50** on every metric but hit@10 (equal): more candidates
  reach a few more answers, but give the reranker more near-misses to prefer.
  The gap is small (3 questions at hit@1), but consistent, and 20 costs 2.5
  times fewer model passes: 20 is the default.
- **By slice** (small n): the gain is large in the same language (hit@5 0.58 →
  0.83), smaller across languages (0.64 → 0.70). It fixes both near-quotes
  (`sv-names`, `fdm-spleen-lid`), the Finnish and Greek questions, and most play
  and paraphrase misses. Unchanged: footnotes and the other known gaps, Italian
  queries (2/5), dialect, speaker questions (already 5/6).
- **What is left is candidate recall.** Of the 13 answers outside the hybrid top
  20, 4 are known gaps (footnotes ×2, front matter, Gretchen/Margarete) and 9
  are not found by the search at all; 6 of those 9 are cross-lingual
  (hamlet-play-trap es→en, fdm-carrion it→fr, ps-farewell fr→it,
  hlm-flower-burial en→zh, sv-impivaara en→fi, dq-library fr→es). Next levers:
  question translation / multi-query (upgrades 5 and 7), then the data fixes of
  the known gaps.

**Points of attention / next**
- Evaluated as extra modes (`chunks:rerank@N`) next to the plain ones, on
  eval-v2: compare `hybrid` with `rerank@20` and `rerank@50` in the same run.
- Not yet used by anything but `rag search` and `rag eval`; the retriever tool
  for the LLM (step 7) will call search + rerank.
- If the reranker helps hit@1 but not hit@10, the remaining misses are beyond
  its reach (answers not in the candidates): neighbour expansion (upgrade 3)
  and query understanding (upgrade 5) come next.

### 7. Integration with the LLM

- Context assembly: format chunks with source IDs for citations, enforce a
  token budget (start with k = 3–5).
- Pipeline RAG first; then register the retriever as a tool in the existing
  agent framework (tool description must say *when* to search).
- Log retrieved contexts at every call: RAGAS needs them, especially in
  agentic mode where retrieval happens inside the loop.

### 8. Evaluation (current set eval-v2; check, scoring and diagnosis v1 done)

#### 8.1 The set: `eval/eval-v1/`

```powershell
uv run gutenberg add --manifest eval/eval-v1/manifest.json   # books already done are skipped
uv run rag eval-check                                        # every quote found exactly once?
uv run rag eval-check --show md-why-sea ps-fn-verri          # read the ground truth
```

- **48 books** (`manifest.json`): the 11 cleaning samples and the 37 books
  flagged in the first run. 15 of them carry questions; the other 33 are
  distractors, which retrieval has to rank below the right passage.
- **89 questions** (`questions.jsonl`): 71 passage questions (69 answerable,
  2 with no answer in the corpus) and 18 book recommendations.

| Dimension | What is covered |
|-----------|-----------------|
| Kinds | prose (51), verse plays (13: Hamlet, Faust), poems (5: Les Fleurs du Mal), a poem inside prose (紅樓夢) |
| Book languages | en, fr, de, es, it, fi, el (katharevousa), zh |
| Query languages | en 52, fr 14, de 7, it 6, es 5, fi 2, zh 2, el 1; **39 cross-lingual** (query language ≠ book language) |
| Structure | speakers (6), footnotes (3), front matter, letters (Lettres persanes), two works in one file (Thomas Mann), a book without headings, a quote crossing a chunk boundary |
| Query style | paraphrase (23), keyword (15), near-quote (2), dialect (Huckleberry Finn), spelling variants (18th-century French, katharevousa vs modern Greek, traditional vs simplified Chinese) |
| Difficulty | answers spread over 2–3 passages (aggregation), a multi-hop question with arithmetic (Yorick), unanswerable questions including a hard negative (Javert's death is in tome V, not in the corpus) |
| Filters | 10 questions use the `rag search` filters (title, author, speaker) |
| Recommendations | one obvious book, many relevant books (8 juvenile novels), cross-lingual ("novela picaresca" → a German *Schelmenroman*), "books like X" with X excluded |

Counts by tag and language are printed by `rag eval-check`.

#### 8.2 Format, and why

One JSON object per line (schema and example in `evaluation/dataset.py`):
easy to diff in git, and a broken line is reported with its line number.

- **Ground truth is a verbatim quote, not chunk keys.** Chunk keys depend on
  the tokenizer, `CHUNKING_VERSION` and the target size. Your bge-m3 chunks
  already differ from approximate-counter chunks. `rag eval-check` locates the
  quote in the cleaned text and lists the chunks that overlap it, so changing
  the chunking means re-running the check, not relabelling.
- **Matching** ignores case, whitespace (also needed for Chinese, which has no
  spaces), curly vs straight quote marks, and Unicode compatibility forms
  (NFKC). The quote must occur **exactly once**, which forces precise ground
  truth: an ambiguous quote is reported as an error.
- A quote crossing a chunk boundary makes **both chunks relevant**
  (`hlm-couplet`). A question counts as found at k when *any* relevant chunk
  is in the top k.
- **Footnotes** (`"in": "annotation"`): the quote is located in an annotation;
  the relevant chunk is the one that carries that annotation's key.
- **Grades**: 2 answers the question, 1 partially answers it or is related.
  This feeds nDCG; recall can use grade 2 only.
- **`known_gap`**: expected failures stay in the set, documented: footnote text
  not embedded (3), front matter not chunked (1), GRETCHEN / MARGARETE speaker
  labels not merged (1). When a fix lands, the score moves, and the check
  shows these as "known gaps", not errors.
- **`answer`**: reference answers for the generation evaluation (step 7:
  correctness, faithfulness). I wrote them from the passages; re-read them.
- **`seed_books`**: for "books like X", X is excluded from the results.

#### 8.3 How it was built, and how to extend it

Every quote was copied from the cleaned text of these exact Gutenberg files and
checked with `rag eval-check`. If a quote is not found on your side, the book
was probably updated on Gutenberg since (another revision of the file): fix
the quote, not the code.

To add a question: find the passage (`rag search`, or `--show` on a neighbour
question), copy a short distinctive quote, add a line, run `rag eval-check`.
Write the question *before* re-reading the passage when you can: questions
written from the text reuse its words, which flatters sparse retrieval
(the `keyword` and `quote` tags mark these on purpose).

Treat a set as **frozen** once you report numbers with it; changes go into
`eval-v2` (copy the folder), otherwise scores from different dates can't be
compared.

To scale beyond hand-written questions: generate questions per chunk with an
LLM (cheap, but biased towards the chunk's own vocabulary and towards
single-chunk facts), filter them, spot-check them, and keep the hand-written
set as the reference.

#### 8.4 Limits of this sample set

- Small slices: 1–6 questions per language or element. Per-slice scores are
  indicative only; always report the count next to a score.
- One author wrote the questions and answers, from the passages (see 8.3).
- Recommendation relevance was judged only within these 48 books. In a larger
  registry, unjudged relevant books appear, and other editions or translations
  of the same work too (Hamlet exists in 6 languages). For comparable numbers,
  score with the search restricted to the manifest's books.
- In *Gladius Dei; Schwere Stunde* the section labels are wrong (cleaning flag
  `single_dominant_section`): its questions rely on the title filter only.

**Dataset errors found by the first runs** (fix in `eval-v2`, keep `eval-v1`
frozen for comparisons already made):
- `fdm-spleen-lid` is ambiguous: Baudelaire's poem "Le Couvercle" is also about
  the sky as a lid ("Le Ciel ! couvercle noir de la grande marmite"), and dense
  ranks it first. Add it as an answer (grade 1 or 2) or reword the question.
- `lm-petit-gervais` quotes the wrong moment: the theft ("Jean Valjean posa le
  pied dessus") is about 3,000 characters before the quoted passage (where he
  notices the coin). The chunks returned at ranks 1–2 are most likely the
  right ones.
- To review with `rag eval-check --show`: `ps-opening` (the opening describes
  the lake over several chunks; the answer may not be in only the first one),
  `lp-persian` (answer not in the top 100 while a neighbouring chunk is
  returned), `hamlet-ophelia-death` (the churchyard scene discusses her
  drowning: a partial answer).

#### 8.5 Scoring: `rag eval` (v1 done)

```powershell
uv run rag eval                          # all modes, saved in data/eval_runs/
uv run rag eval --failures               # + questions not found in the top 10 (hybrid)
uv run rag eval --failures chunks:dense  # same for another mode
uv run rag eval --run eval-v1-20261025-230519 --diagnose   # where the answers are
uv run rag eval --metric rr              # slice tables with MRR instead of hit@5
uv run rag eval --run eval-v1-20261025-2010            # show a saved run again (no model)
uv run rag eval --compare eval-v1-20261025-2010 eval-v1-20261026-0930
```

**What it does** (`evaluation/run.py`):
1. runs `eval-check` first; questions with dataset errors are not scored;
2. encodes all the questions **once** (one batch), loading the model once;
3. for each mode, searches each question with its filters, **restricted to
   the set's manifest books** (comparable numbers whatever else is indexed),
   and keeps the top 20 (`--depth`):
   - passage questions on the `chunks` collection: `dense`, `sparse`, `hybrid`;
   - book questions on the `books` collection: `metadata`, `centroid`
     (seed books of "books like X" removed);
4. scores each question against its targets and saves a run file.

**Metrics** (`evaluation/metrics.py`; an *entry* is one evidence quote, or
one book; its rank is the first result that satisfies it):

| Metric | Question it answers | Notes |
|--------|--------------------|-------|
| hit@k (k = 1, 3, 5, 10) | Is an answer in the top k? | The main RAG number: the generator sees the top k |
| recall@10 | How much of the answer is in the top 10? | Differs from hit@k for aggregation / multi-hop questions |
| MRR | How high is the first answer? | Mean of 1 / rank (0 if not found) |
| nDCG@10 | Are the best answers first? | Uses the grades; gain credited at the first result covering an entry |

Questions with no answer in the corpus are searched (their results are in
the run file) but not scored: they test generation, not retrieval.

**Reading the output**
- Overview: one row per mode. Expect sparse to be strong on names and rare
  words, dense on paraphrases and cross-lingual questions, hybrid at least as
  good as the better of the two on average.
- By slice: the same metric by same / cross-lingual, query language, known
  gaps and tags. n is printed on every row: with n = 2, one question moves the
  score by 0.5.
- `--failures`: what came back instead of the answer (book and section of the
  top 3). Read these before changing anything: a wrong book, the right book
  but the wrong chapter, or the right passage just below the cut need different
  fixes (filters / query understanding, reranking, context expansion).
- `--diagnose [MODE]` (`evaluation/diagnose.py`), to choose an upgrade before
  building it. Run it on a deep run (`rag eval --depth 100`), since ranks beyond
  the depth are unknown:
  - **best answer rank per mode**, in buckets (1, 2–5, 6–10, 11–20, 21–depth,
    beyond): answers present but not first can be fixed by reordering;
    answers beyond the depth cannot;
  - **reordering ceiling**: what hit@1 and hit@5 would be if a perfect reranker
    reordered the top `depth` (the most any reranker can gain);
  - **dense vs sparse vs hybrid**: questions only one branch finds, their union
    (what a perfect fusion could reach), questions dense finds and hybrid loses;
  - **misses** of one mode (default hybrid): best rank, and how close the
    returned chunks are, read from the keys: same section and how many chunks
    away, same book, other book.
  Run files store each question's target keys; for older runs they are taken
  from a check of the current data (valid while the chunking is unchanged).
- `--compare A B`: settings that differ, metric changes per mode, and the
  questions that gained or lost hit@5 (`--k`). Use it after every change
  (chunk size, fusion, reranker...). A run is only comparable to another on
  the same question file (a warning says when the files differ).

**Run files** (`data/eval_runs/<set>-<date>.json`): embedding model, revision
and backend, chunking tokenizer / version / target size, cleaning version,
Qdrant target, depth; then for each mode and question the keys returned, the
labels of the first three, the entry ranks and the metrics. Kept small and
plain JSON, so they can be diffed or loaded in a notebook.

**Limits and next steps**
- Baseline numbers: to measure on the real index (the development sandbox has
  no GPU or model; the pipeline was run end to end there on the 48 books with
  a stand-in encoder, which only checks the mechanics).
- Each question is searched once per mode; reranking (section 6, upgrade 2)
  will add a mode that rescores the hybrid top 20 — the depth is already 20 for
  that reason.
- RAGAS (context precision / recall, faithfulness) comes with the LLM step
  (7) and reuses the same targets and the `answer` fields.

#### 8.6 Understanding the numbers: from embeddings to metrics

This section explains, in words first, what every line of `rag eval` computes
and why. The equations are there to support the text, not replace it.

##### What is pretrained, and what is ours

**We did not train or fine-tune anything.** bge-m3 is used as published by
its authors (BAAI): we download its weights and only run it (inference,
`model.eval()` and `torch.inference_mode()`); no weight changes.

| Part | Where it comes from |
|------|--------------------|
| The transformer (XLM-RoBERTa large, further pre-trained by BAAI with RetroMAE, inputs up to 8192 tokens) | `pytorch_model.bin`, trained by BAAI |
| The dense vector | nothing extra: it *is* the transformer's output at the first position (CLS) |
| The sparse head: one 1024 → 1 linear layer (1024 weights W and a bias b) | `sparse_linear.pt`, trained by BAAI |
| Tokenizer | the model repository's tokenizer files |
| Calling the model, taking the CLS state and normalising it, applying the sparse head, keeping the max weight per token, dropping special tokens | **our code** (`encoders/manual.py`, `encoders/base.py`), checked against BAAI's FlagEmbedding by `rag embed-check` |
| Storing, indexing, searching, fusing, evaluating | **our code** (`index/`, `evaluation/`) |

**Where the sparse weights are learned from.** BAAI trained the whole model,
transformer and both heads together, on retrieval: for a question, the score
of the passage that answers it must come out higher than the scores of
passages that don't (a contrastive loss, InfoNCE). They applied that rule to
the dense score, the sparse score and the multi-vector score alike. On top,
"self-knowledge distillation": the sum of the three scores acts as a teacher
that each score individually learns to imitate. Data: about 1.2 billion
unlabelled text pairs (Wikipedia, mC4, CC-News, translation pairs from NLLB
and CCMatrix, …), then labelled question–passage datasets (MS MARCO, NQ,
TriviaQA, HotpotQA, Mr. TyDi, MIRACL, DuReader, …) and GPT-3.5-generated
questions on long documents. So "learned" means: W and b were adjusted until
products of token weights ranked the right passages first, over that data.
Source: Chen et al., *M3-Embedding* (2024), arXiv:2402.03216.

##### Dense vs sparse: two different vectors, not one vector with its zeros removed

bge-m3 reads the text once, then two separate heads turn what it has read into
two different representations.

**The dense vector is a summary of meaning.**
- It has 1024 numbers, and almost none of them are zero.
- No single number means anything by itself ("dimension 417" isn't "about the
  sea"). What carries the meaning is the direction the whole vector points in.
- Training taught the model to make texts with the same meaning point the same
  way: a question and the passage that answers it, or a sentence and its
  translation.
- It's computed from one special position, the first token (CLS), whose final
  state acts as a summary of the whole text:
  ```
  q = h_CLS / ‖h_CLS‖        (divided by its length, so every vector has length 1)
  ```

**The sparse vector is a weighted list of the words actually present.**
- It has one dimension per token of the model's vocabulary, about 250,000 of
  them, and each dimension does mean something: "the token ▁sea", "the token
  ▁Valjean".
- A text uses only a few dozen of those tokens, so almost everything is zero.
  That's why it's called sparse, and why it's stored as a short list of
  (token, weight) pairs rather than 250,000 numbers. Sparse *storage* is a
  general idea (any vector with many zeros can be stored that way); the sparse
  *vector* of bge-m3 is a specific thing: weights over the vocabulary.
- The weight is how important the model judges that token to be in this text.
  "Why does Ishmael go to sea?" gives roughly ▁Is·hma·el → high, ▁sea → high,
  ▁go → medium, ▁does and ▁to → close to 0.
- The model computes this by looking at each token in its context and
  outputting one number, never negative. If a token appears several times, the
  highest value is kept:
  ```
  w(t) = max over the positions of t of ReLU(W · h_t + b)
  ```
  The idea is close to BM25 keyword weighting, except that the weights are
  learned (see above) instead of counted (term frequency and rarity).

**So the two can't be converted into each other.** Dropping the dense vector's
zeros wouldn't give the sparse one: the dense vector has no zeros, and its 1024
dimensions aren't words. They live in two different spaces.

##### The two scores

**Dense: how closely two texts point in the same direction.**
```
s(q,c) = q · c = Σ over the 1024 dimensions of q_i × c_i = cos(angle between q and c)
```
- It's a similarity, not a distance: higher means closer in meaning. About 1
  means the same meaning, around 0 unrelated. Search keeps the largest values.
- Because both vectors have length 1, the plain product (q · c) equals the
  cosine, which is cheap to compute.
- Why it's used: two texts can match without sharing a single word, as with a
  paraphrase or a translation. That's why dense handles the cross-lingual
  questions of eval-v1 (hit@5 0.58).

**Sparse: how many important words the two texts share.**
```
s(q,c) = Σ over the tokens present in BOTH q and c of w_q(t) × w_c(t)
```
- Only shared tokens count, and a shared token counts a lot only if it's
  important in both texts. Sharing "Valjean" weighs heavily; sharing "the"
  weighs almost nothing.
- Why it's used: exact names and rare words. Dense may blur them ("a French
  bishop" vs "Monseigneur Bienvenu"); sparse doesn't.
- Its limit follows directly from the equation: no shared token means a score
  of 0. An English question shares almost no tokens with French text
  (cross-lingual sparse hit@5: 0.18). Chinese characters overlap even between
  traditional and simplified writing (sparse found both Chinese questions).

##### Hybrid: candidate pools and rank fusion

**The candidate pool ("4 × 20").** We want 20 final results (`--depth`), so
each branch first returns its own top 80. If each returned only 20, fusion
could only reorder chunks both branches already ranked highly: a chunk that
sparse ranks 5th but dense ranks 30th would be lost before fusion saw it. The
factor 4 is a rule of thumb (big enough to catch such cases, small enough to
stay fast), set in `index/search.py`; it is one of the "fusion tuning" settings
(section 6, upgrade 6).

**The fusion: each list awards points by position.** First place earns 1/2,
second 1/3, third 1/4, and so on. A chunk's final score is the sum of the
points it earned in the two lists; a list where it doesn't appear gives 0.
```
s(c) = 1/(2 + r_dense(c)) + 1/(2 + r_sparse(c))      r = position in that list, counting from 0
```
- **Why ranks rather than scores:** a cosine (between 0 and 1) and a sparse sum
  (unbounded, often 0.05–0.4) aren't on the same scale, so adding them makes no
  sense. Positions are comparable. This is Reciprocal Rank Fusion (RRF),
  computed inside Qdrant.
- **Why the constant 2:** it's Qdrant's value, and it sets how steep the points
  scale is. With 2, first place is worth 0.50 and tenth 0.083: the top
  positions dominate. With the 60 common in papers, first and tenth are worth
  almost the same (1/60 vs 1/69). The scores of a `rag search` confirm it:
  1.000 = 1/(2+0) + 1/(2+0) (first in both lists), 0.476 = 1/(2+1) + 1/(2+5).
- **Consequence:** a chunk that both lists rank 2nd (1/3 + 1/3 = 0.67) beats a
  chunk that only dense ranks 1st (0.50). Fusion rewards agreement. When the
  sparse list is poor, that "agreement" can push dense's correct first answer
  down: in eval-v1, hybrid hit@1 (0.33) is below dense hit@1 (0.39).
- **Scores are not similarities:** a hybrid score says where a chunk was
  ranked, not how relevant it is. Never put a threshold on it.

Book searches (`books` collection) are plain cosine on one book vector:
`metadata` (the embedded catalog text: title, authors, subjects, bookshelves)
or `centroid` (the normalised mean of the book's chunk vectors). Books listed in
`seed_books` ("books like X") are removed from the results.

##### Entries, K_e, rank_e and g_e: what the metrics are built on

**An entry is one piece of ground truth.** For a passage question, it's one
evidence quote; for a recommendation question, one book. Most questions have
one entry; "What does Fantine sell?" has three (her hair, her teeth sold, the
earlier offer).

**K_e is the set of chunks that contain that quote.** `eval-check` finds the
quote in the book and notes which chunks overlap it. Usually that's one chunk;
it's two when the quote straddles a chunk boundary. Either chunk counts as
finding the entry.

**rank_e is how far down the result list you go before meeting one of those chunks.**
```
rank_e = min { i : results[i] ∈ K_e }        (positions counted from 1)
```
- Results [c7, c12, c3] with K_e = {c12}: rank 2.
- K_e = {c12, c13} with c13 first in the list: rank 1.
- If none of its chunks is in the 20 results, the rank is None: "not found".

**g_e is a grade written by hand in the question file.** It's never computed.
2: this passage answers the question; 1: it answers partly, or is related.
Example from `lm-galley-years`: "J'ai passé dix-neuf ans au bagne" has grade 2,
while "pendant dix-neuf ans un matelas de bois" has grade 1 (it implies the
answer without stating it). Only nDCG uses the grade; hit, recall and MRR treat
every entry the same.

##### The metrics, and why each one exists (`evaluation/metrics.py`)

**Answers vs related passages**: hit@k, recall@k and RR only count the
question's best-graded entries (grade 2, its real answers, when it has any);
grade-1 entries only add to nDCG (rule introduced with eval-v2, 8.7).

**hit@k: would the LLM see at least one answer if we gave it the top k chunks?**
```
hit@k = 1 if some entry has rank_e ≤ k, else 0
```
The number that matters most for a RAG: the generator only reads what you hand
it, typically 3 to 5 chunks.

**recall@k: how much of the answer is in the top k?**
```
recall@k = (entries found within the top k) / (all entries)
```
For single-entry questions it equals hit@k. It differs for answers spread over
several passages: finding the hair but not the teeth gives hit = 1 but
recall = 1/2.

**RR (averaged into MRR): how far down is the first answer?**
```
RR = 1 / (rank of the first entry found)        (0 if none)
```
Rank 1 gives 1, rank 2 gives 0.5, rank 3 gives 0.33, rank 10 gives 0.1. It
rewards putting an answer first, and sees the difference between rank 1 and
rank 4, which hit@5 doesn't.

**nDCG@10: is the whole ranking good, counting both which passages are better
answers and where they appear?** Each entry earns points when its first chunk
appears: a full answer (grade 2) earns 3, a partial one (grade 1) earns 1, so a
real answer counts three times a related passage:
```
gain = 2^g_e − 1
```
Points shrink with depth: full value at position 1, divided by 1.58 at
position 2, by 2 at position 3, by 3.46 at position 10. Lower positions count
less, but the decline is gradual:
```
DCG = Σ over positions i of (gains first earned at i) / log2(i + 1)
```
The total is then divided by the best score possible for that question, so 1
means a perfect ordering, whatever the number of entries:
```
nDCG = DCG / best possible DCG
```
Worked example (illustrative ranks), `lm-fantine-sells`, entries hair (2),
teeth sold (2), teeth offer (1); hair found at rank 2, offer at rank 4, sale
missing: hit@1 = 0, hit@5 = 1, recall@10 = 2/3, RR = 1/2;
DCG = 3/log2(3) + 1/log2(5) = 1.893 + 0.431 = 2.324;
best = 3/1 + 3/log2(3) + 1/log2(4) = 3 + 1.893 + 0.5 = 5.393; nDCG@10 = 0.43.

**Table cells: the average of a metric over the questions in that row.**
```
cell = (1/n) × Σ over the n questions of their score
```
Every question weighs the same. For 0/1 metrics a cell is simply a share:
hit@5 = 0.58 means 40 of 69 questions had an answer in the top 5.

##### What `rag eval` runs, step by step (`cmd_eval` in `cli.py`)

1. `check_questions` (`evaluation/check.py`): each relevant entry becomes a
   target, the chunk keys overlapping its quote (K_e) and its grade (the same
   step as `rag eval-check`). Questions with dataset errors are not scored.
2. `encode_sorted` (`embedding/run.py`): all questions encoded in one batch,
   the model loaded once from the local folder (no Hub request).
3. `run_eval` (`evaluation/run.py`): every question searched in every mode,
   restricted to the manifest's books plus the question's own filters
   (`question_filter`), top 20 kept.
4. `score` per question (`metrics.py`), then `mean_metrics` per mode and per
   slice, `failure_lines` (`report.py`).
5. `build_run` + `save`: everything written to `data/eval_runs/<set>-<date>.json`.

**Slice tables** (`slices`): same language / cross-lingual (the query language
is or isn't among the relevant books' languages), query language, known gap,
tags. A slice score moves in steps of 1/n: with n = 1 or 2, a row is an
anecdote, not a measurement.

**Failure list** (`failure_lines`): questions with hit@10 = 0 in one mode, and
the labels ("book title — section path", truncated) of the first three results.
Two identical labels are two chunks of the same section. The ground truth is
chunk-level, so "right chapter, wrong chunk" is still a miss.

##### Reference: eval-v1 baseline, 2026-10-25

bge-m3 at the pinned revision, chunks of ~350 bge-m3 tokens, cleaning 1.2.0,
48 books, run `eval-v1-20261025-204525`.

| Mode | n | hit@1 | hit@5 | hit@10 | recall@10 | MRR | nDCG@10 |
|------|---|-------|-------|--------|-----------|-----|---------|
| chunks: dense | 69 | 0.39 | 0.54 | 0.64 | 0.62 | 0.46 | 0.50 |
| chunks: sparse | 69 | 0.17 | 0.33 | 0.39 | 0.37 | 0.24 | 0.26 |
| chunks: hybrid | 69 | 0.33 | 0.58 | 0.62 | 0.61 | 0.44 | 0.47 |
| books: metadata | 18 | 0.89 | 1.00 | 1.00 | 0.95 | 0.93 | 0.88 |
| books: centroid | 18 | 0.67 | 0.83 | 0.94 | 0.84 | 0.73 | 0.67 |

Checks that hold: dense hit@1 = 27/69, hybrid hit@10 = 43/69 (69 − 43 = the 26
questions of the hybrid failure list), cross-lingual sparse hit@5 = 6/33,
known gap 0.20 = 1/5 (`ps-fn-lebrun` is the one found: the four others are in
the failure list).

The 26 hybrid failures (not in the top 10), grouped by what came back:

| What came back | Questions |
|---|---|
| Right section at rank 1–3, wrong chunk | hamlet-ghost, huck-hell, md-doubloon, lm-petit-gervais, ps-opening, ps-bravi, sv-names, mann-gladius (right book; its section labels are wrong) |
| Right book, wrong place | md-whiteness, md-rescue, huck-island, hamlet-ophelia-death, hamlet-play-trap, lm-confession, dq-library, ps-farewell, sv-cantor, lp-persian, fdm-spleen-lid |
| Wrong book | fdm-carrion (Italian query → Italian books), hlm-flower-burial (English query → *Red Money*; 紅樓夢 at rank 2), sv-impivaara |
| Declared known gaps | huck-notice, faust-spinning-wheel, ps-fn-ragguaglio, ps-fn-verri |

What the baseline says (diagnosis, before any upgrade):
- Hybrid helps from hit@5 on but hurts hit@1: rank fusion rewards agreement,
  and the sparse branch is weak on this set (see "Hybrid" above).
- Sparse depends on shared tokens: weak across languages, strong on Chinese.
  Dense is about as good across languages as within one.
- Language attraction: some cross-lingual questions are pulled towards books
  written in the query's language.
- Metadata beats centroid for books, partly because the recommendation queries
  were written like catalog subjects (a bias of the set), partly because a mean
  over thousands of chunks washes out what makes a book distinctive.
- Many misses are near misses: 8 of 26 have the right section in the top 3.

Diagnosis of the same setup at depth 100 (`eval-v1-20261025-230519`,
`--diagnose`):

| Mode | n | 1 | 2–5 | 6–10 | 11–20 | 21–100 | >100 |
|------|---|---|-----|------|-------|--------|------|
| chunks: dense | 69 | 27 | 10 | 7 | 6 | 12 | 7 |
| chunks: sparse | 69 | 12 | 11 | 4 | 6 | 11 | 25 |
| chunks: hybrid | 69 | 23 | 16 | 4 | 10 | 7 | 9 |

| k | dense | sparse | hybrid | union | dense only | sparse only |
|---|-------|--------|--------|-------|------------|-------------|
| 1 | 27 | 12 | 23 | 30 | 18 | 3 |
| 5 | 37 | 23 | 39 | 42 | 19 | 5 |
| 10 | 44 | 27 | 43 | 48 | 21 | 4 |
| 20 | 50 | 33 | 53 | 54 | 21 | 4 |

What it says:
- **The answers are mostly there, badly ordered.** Hybrid has an answer in its
  top 100 for 60 of 69 questions, but first for only 23. A perfect reordering
  of its top 100 would give hit@1 and hit@5 of 60/69; of its top 20, 53/69.
  This is the case for a **reranker**.
- **Hybrid is the better candidate pool, dense the better order.** At 20,
  hybrid finds 53 (dense 50, union 54: fusion already collects almost every
  answer either branch finds), but at 1 it is behind dense (23 vs 27): fusion
  reorders the top badly. Five questions are found by dense and lost by hybrid
  (md-doubloon, md-whiteness, hamlet-ghost, lm-confession, sv-names); sparse
  alone rescues four (Chinese ×2, a name, dialect). So: hybrid to collect the
  candidates, a reranker to order them.
- **Beyond the reach of reranking**: the 9 hybrid answers not in the top 100.
  They are known gaps (3), dataset errors (`lm-petit-gervais`, see 8.4), and
  questions far from the text's wording; these need data, chunking or question
  fixes, not reordering.
- Of the 26 hybrid misses at 10, 12 have a returned chunk in the answer's
  section: neighbour expansion may put the answer in the context of several of
  them (re-run `--diagnose` on a new run for exact chunk distances: this run
  predates stored targets, and its sections were matched against another
  chunking).

#### 8.7 eval-v2 (current set)

`eval/eval-v2/` = eval-v1 with the dataset errors found by the first runs
fixed (8.4). Same 89 questions, same ids and order, same 48 books; every
changed question says what changed in its `notes` ("v2: …"). eval-v1 stays
frozen, so the comparisons made with it remain valid; `rag eval` and
`rag eval-check` now default to eval-v2 (`--set eval/eval-v1` for the old one).

| Question | Change |
|----------|--------|
| `lm-petit-gervais` | The theft itself ("Jean Valjean posa le pied dessus.") is the answer; v1's quote (he notices the coin later) becomes grade 1 |
| `fdm-spleen-lid` | "Le Couvercle" (the sky as a lid) added as a partial answer (grade 1) |
| `ps-opening` | The lake and Lecco are described over several chunks: the Lecco passage added (grade 2); the introduction, which quotes the opening line, added as grade 1 |
| `lp-persian` | Letter XXX opens with the Parisians' reaction: added as an answer (grade 2) |
| `hamlet-ophelia-death` | The gravediggers discussing her drowning added (grade 1) |
| `dq-library` | The burning itself (chapter VII) added as an answer (grade 2) |

**Metric rule changed at the same time** (`metrics.py`): hit@k, recall@k and
MRR count only a question's *best-graded* entries (its real answers: grade 2
when it has any); grade-1 entries (partial answers, related passages or books)
only add to nDCG. Without this, adding a related passage to the ground truth
would raise the hit rate. It also applies to eval-v1 runs made from now on, so
an eval-v1 run made today can differ slightly from the 2026-10-25 reference:
compare runs made with the same code.

**New baseline**: re-run on eval-v2 before judging any upgrade, in one run
with the reranker modes:
```powershell
uv run rag eval --depth 100 --rerank 20 50 --diagnose
```
Done on 2026-10-25 (`eval-v2-20261025-235752`); results in 6.1. Compared with
eval-v1, hybrid hit@10 rose from 0.62 to 0.67 with the same index: that change
comes from the dataset fixes (and the grading rule), not from the system.

### Cross-cutting

- Convert remaining notebooks' reusable parts into package modules as they
  stabilise; keep notebooks for exploration.
- Settings could move to `pydantic-settings` with `.env` support when
  configuration grows (Qdrant, model paths, cloud credentials).
- Delete `03_RAG/gutenberg_acquire.py` once the data is migrated (git keeps it).

---

## Troubleshooting

**About ten `HTTP Request: ... huggingface.co` lines on every command, or the model "downloading" again**

Not a download: `from_pretrained("BAAI/bge-m3")` asks the Hub which files and
commit to use on every call, even with everything cached (the log showed
`Download complete: 0.00B`). Fixed: the files are resolved once at a pinned
revision into a local folder and loaded from there (README 4.2, `rag model`).
The request lines themselves came from the `httpx` logger, now set to WARNING.

**`SSL: CERTIFICATE_VERIFY_FAILED ... self-signed certificate in certificate chain`**

Not necessarily interception. Here the cause was an outdated `certifi`
(2022.12.7) that did not contain the *SSL.com TLS RSA Root CA 2022* used by
gutenberg.org. It was stuck because the PyTorch index (`download.pytorch.org`)
also hosts old copies of `certifi` and `requests`, and uv's default
`first-index` strategy took them from there.

Fix: declare the PyTorch index with `explicit = true` and assign only
`torch`/`torchvision`/`torchaudio` to it in `[tool.uv.sources]`, then
`uv lock --upgrade` and `uv sync`.

Diagnosis steps that helped: browser certificate issuer, `openssl s_client
-connect www.gutenberg.org:443 -showcerts`, `uv pip show certifi`,
`uv tree --invert --package certifi`, and the `source` of the package in `uv.lock`.

**conda `(base)` active alongside uv.** Conda sets `SSL_CERT_FILE` (and other
variables) that leak into the uv environment. `requests` ignores it, but
`httpx` (used by `huggingface_hub`, `qdrant-client`) reads it. Prefer
`conda config --set auto_activate_base false` when working in uv projects.

Never "fix" certificate errors with `verify=False`.
