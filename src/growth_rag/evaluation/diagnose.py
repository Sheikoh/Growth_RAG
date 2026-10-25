"""Where are the answers? `rag eval --diagnose` (pure functions returning lines).

Three views of a saved run, to choose an upgrade before building it:

1. Rank buckets, per mode: for each scored question, the rank of its best
   answer (1, 2-5, 6-10, 11-20, 21-depth, beyond depth). Answers present but
   not first are what a reranker can reorder (its ceiling); answers beyond the
   depth are out of its reach (representation, chunking or dataset problems).
2. Branch complementarity (dense vs sparse vs hybrid at k): questions only one
   branch finds, the union of both (what a perfect fusion could reach), and
   questions dense finds but hybrid loses (fusion hurting).
3. Misses of one mode at k: best rank, and how close the returned chunks are
   to the answer, read from the keys (`<book>/s<section>/c<index>`):
   same section (and how many chunks away), same book, other book.
"""

from __future__ import annotations

import re

from growth_rag.evaluation.dataset import Question
from growth_rag.evaluation.run import EvalRun, QuestionRun

_KEY = re.compile(r"^(?P<book>.+?)/s(?P<section>\d+)/c(?P<chunk>\d+)$")


def best_rank(r: QuestionRun) -> int | None:
    """Rank of the best answer, among the best-graded entries (as hit@k)."""
    ranks = r.ranks
    if r.grades and len(r.grades) == len(r.ranks):
        top = max(r.grades)
        ranks = [x for x, g in zip(r.ranks, r.grades) if g == top]
    found = [x for x in ranks if x is not None]
    return min(found) if found else None


def buckets(depth: int) -> list[tuple[str, int, int]]:
    """(label, lowest rank, highest rank), cut at the run's depth."""
    out = []
    for low, high in ((1, 1), (2, 5), (6, 10), (11, 20), (21, depth)):
        if low > depth:
            break
        high = min(high, depth)
        out.append((str(low) if low == high else f"{low}-{high}", low, high))
    return out


def bucket_lines(run: EvalRun) -> list[str]:
    cols = buckets(run.depth) + [(f">{run.depth}", 0, 0)]
    lines = ["Best answer rank, per mode (questions)",
             f"  {'mode':<16}{'n':>4}" + "".join(f"{label:>9}" for label, _, _ in cols)]
    for m in run.modes:
        ranks = [best_rank(r) for r in m.questions.values() if r.metrics is not None]
        counts = [sum(1 for x in ranks if x is not None and lo <= x <= hi) for _, lo, hi in cols[:-1]]
        counts.append(sum(1 for x in ranks if x is None))
        lines.append(f"  {m.mode:<16}{len(ranks):>4}" + "".join(f"{c:>9}" for c in counts))
    return lines


def ceiling_lines(run: EvalRun, mode: str) -> list[str]:
    m = run.mode(mode)
    if m is None:
        return []
    ranks = [best_rank(r) for r in m.questions.values() if r.metrics is not None]
    n = len(ranks)
    present = sum(1 for x in ranks if x is not None)
    lines = [f"Reordering ceiling for {mode} (a perfect reranker of its top {run.depth}):"]
    for k in (1, 5):
        now = sum(1 for x in ranks if x is not None and x <= k)
        lines.append(f"  hit@{k}: {now}/{n} now, at most {present}/{n} "
                     f"({present - now} answer(s) present below rank {k})")
    return lines


def complementarity_lines(run: EvalRun, ks=(1, 5, 10, 20)) -> list[str]:
    modes = {m.mode: m for m in run.modes}
    names = ("chunks:dense", "chunks:sparse", "chunks:hybrid")
    if not all(n in modes for n in names):
        return []
    dense, sparse, hybrid = (modes[n] for n in names)
    ids = [qid for qid, r in dense.questions.items() if r.metrics is not None]

    def found(mode, qid, k):
        rank = best_rank(mode.questions[qid])
        return rank is not None and rank <= k

    lines = ["Dense vs sparse vs hybrid (questions with an answer in the top k)",
             f"  {'k':>4}{'dense':>8}{'sparse':>8}{'hybrid':>8}{'union':>8}{'dense only':>12}{'sparse only':>13}"]
    for k in (k for k in ks if k <= run.depth):
        d = {q for q in ids if found(dense, q, k)}
        s = {q for q in ids if found(sparse, q, k)}
        h = {q for q in ids if found(hybrid, q, k)}
        lines.append(f"  {k:>4}{len(d):>8}{len(s):>8}{len(h):>8}{len(d | s):>8}{len(d - s):>12}{len(s - d):>13}")
    k = min(10, run.depth)
    lost = [q for q in ids if found(dense, q, k) and not found(hybrid, q, k)]
    gained = [q for q in ids if found(hybrid, q, k) and not found(dense, q, k)]
    lines.append(f"  found by dense, lost by hybrid (top {k}): {', '.join(lost) or '-'}")
    lines.append(f"  found by hybrid, not by dense (top {k}): {', '.join(gained) or '-'}")
    return lines


def closeness(results: list[str], targets: list[list[str]], k: int) -> str:
    """How close the top-k results come to the answer, from the keys alone."""
    wanted = [m for keys in targets for key in keys if (m := _KEY.match(key))]
    if not wanted:
        return "not indexed"
    got = [m for key in results[:k] if (m := _KEY.match(key))]
    same_section = [abs(int(g["chunk"]) - int(w["chunk"])) for g in got for w in wanted
                    if g["book"] == w["book"] and g["section"] == w["section"]]
    if same_section:
        return f"same section, {min(same_section)} chunk(s) away"
    if any(g["book"] == w["book"] for g in got for w in wanted):
        return "same book, other section"
    return "other book"


def miss_lines(run: EvalRun, questions: list[Question], mode: str, k: int = 10) -> list[str]:
    m = run.mode(mode)
    if m is None:
        return [f"No mode {mode!r} in this run"]
    by_id = {q.id: q for q in questions}
    misses = [(qid, r) for qid, r in m.questions.items() if r.metrics is not None
              and not ((b := best_rank(r)) is not None and b <= k)]
    lines = [f"Misses of {mode} at top {k}: {len(misses)} — best rank of the answer, closest result"]
    for qid, r in misses:
        rank = best_rank(r)
        gap = "  [known gap]" if (q := by_id.get(qid)) is not None and q.known_gap else ""
        if mode.startswith("books:"):
            where = ", ".join(r.results[:3])  # book keys: no sections to compare
        else:
            where = closeness(r.results, r.targets, k) if r.targets else "targets not in the run file"
        lines.append(f"  {qid:<26}{'>' + str(run.depth) if rank is None else rank:>6}   {where}{gap}")
    return lines


def diagnose_lines(run: EvalRun, questions: list[Question], mode: str = "chunks:hybrid", k: int = 10) -> list[str]:
    lines = bucket_lines(run) + [""]
    if mode.startswith("chunks:"):
        lines += ceiling_lines(run, mode) + [""]
        comp = complementarity_lines(run)
        lines += comp + ([""] if comp else [])
    return lines + miss_lines(run, questions, mode, k)


def with_targets(run: EvalRun, targets: dict[str, list[list[str]]]) -> EvalRun:
    """Fill missing targets (runs saved before they were stored) from a check
    of the current data. Valid as long as the chunking did not change since."""
    for m in run.modes:
        for qid, r in m.questions.items():
            if not r.targets and qid in targets:
                r.targets = targets[qid]
    return run
