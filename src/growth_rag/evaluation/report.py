"""Tables from evaluation runs (pure functions returning lines of text).

- overview:  one row per mode, the mean of every metric;
- slices:    one metric (default hit@5) by slice (cross-lingual, query
             language, tag) and by mode: where retrieval fails;
- failures:  questions with no answer in the top k, with what came back;
- compare:   two runs side by side, and the questions that changed.

A mean over a handful of questions moves by 0.2 when one question flips:
every row shows its n.
"""

from __future__ import annotations

from collections import defaultdict

from growth_rag.evaluation.dataset import Question
from growth_rag.evaluation.run import EvalRun, ModeRun
from growth_rag.registry import Registry

METRICS = ("hit@1", "hit@5", "hit@10", "recall@10", "rr", "ndcg@10")
LABEL = {"rr": "MRR"}


def mean_metrics(mode: ModeRun, ids: list[str] | None = None) -> tuple[int, dict[str, float]]:
    scored = [r.metrics for qid, r in mode.questions.items()
              if r.metrics is not None and (ids is None or qid in ids)]
    if not scored:
        return 0, {}
    return len(scored), {m: sum(s[m] for s in scored) / len(scored) for m in METRICS}


def overview_lines(run: EvalRun) -> list[str]:
    header = f"  {'mode':<16}{'n':>4}" + "".join(f"{LABEL.get(m, m):>10}" for m in METRICS)
    lines: list[str] = []
    for task, title in (("chunks", "Passages (chunks collection)"), ("books", "Recommendations (books collection)")):
        modes = [m for m in run.modes if m.mode.startswith(task + ":")]
        if not modes:
            continue
        lines += [title, header]
        for m in modes:
            n, means = mean_metrics(m)
            lines.append(f"  {m.mode.split(':')[1]:<16}{n:>4}" + "".join(f"{means.get(k, 0):>10.2f}" for k in METRICS))
    return lines


def slices(questions: list[Question], registry: Registry) -> dict[str, list[str]]:
    """Slice name -> question ids. Order: same/cross-lingual, query language,
    known gaps, then tags by size."""
    lingual: dict[str, list[str]] = defaultdict(list)
    language: dict[str, list[str]] = defaultdict(list)
    tags: dict[str, list[str]] = defaultdict(list)
    gaps: list[str] = []
    for q in questions:
        langs = {lang for b in q.books if (r := registry.get(b)) for lang in r.languages}
        if langs:
            lingual["cross-lingual" if q.language not in langs else "same language"].append(q.id)
        language[f"query in {q.language}"].append(q.id)
        if q.known_gap:
            gaps.append(q.id)
        for t in q.tags:
            tags[f"tag {t}"].append(q.id)

    def by_size(d: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
        return sorted(d.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    ordered = dict(by_size(lingual) + by_size(language))
    if gaps:
        ordered["known gap"] = gaps
    ordered.update(by_size(tags))
    return ordered


def slice_lines(run: EvalRun, questions: list[Question], registry: Registry,
                metric: str = "hit@5", task: str = "chunks") -> list[str]:
    modes = [m for m in run.modes if m.mode.startswith(task + ":")]
    if not modes:
        return []
    names = [m.mode.split(":")[1] for m in modes]
    lines = [f"{LABEL.get(metric, metric)} by slice ({task})",
             f"  {'slice':<24}{'n':>4}" + "".join(f"{n:>10}" for n in names)]
    for name, ids in slices([q for q in questions if (q.task == "passage") == (task == "chunks")], registry).items():
        row = [mean_metrics(m, ids) for m in modes]
        n = row[0][0]
        if n == 0:
            continue
        lines.append(f"  {name:<24}{n:>4}" + "".join(f"{means[metric]:>10.2f}" for _, means in row))
    return lines


def failure_lines(run: EvalRun, questions: list[Question], mode: str, k: int = 10) -> list[str]:
    m = run.mode(mode)
    if m is None:
        return [f"No mode {mode!r} in this run (modes: {', '.join(x.mode for x in run.modes)})"]
    by_id = {q.id: q for q in questions}
    failed = [(qid, r) for qid, r in m.questions.items() if r.metrics is not None and r.metrics[f"hit@{k}"] == 0]
    lines = [f"Not found in the top {k} ({mode}): {len(failed)} question(s)"]
    for qid, r in failed:
        q = by_id.get(qid)
        gap = "  [known gap]" if q is not None and q.known_gap else ""
        lines.append(f"  {qid}{gap}: {q.query if q else ''}")
        lines += [f"      {i}. {label}" for i, label in enumerate(r.labels, 1)]
    return lines


def compare_lines(a: EvalRun, b: EvalRun, k: int = 5) -> list[str]:
    lines = [f"A: {a.set_name} {a.created_at:%Y-%m-%d %H:%M} UTC   B: {b.set_name} {b.created_at:%Y-%m-%d %H:%M} UTC"]
    if a.questions_sha256 != b.questions_sha256:
        lines.append("  Warning: the question files differ between the two runs.")
    for key in sorted(set(_flat(a.settings)) | set(_flat(b.settings))):
        va, vb = _flat(a.settings).get(key), _flat(b.settings).get(key)
        if va != vb:
            lines.append(f"  setting {key}: {va} -> {vb}")
    for ma in a.modes:
        mb = b.mode(ma.mode)
        if mb is None:
            continue
        (na, xa), (nb, xb) = mean_metrics(ma), mean_metrics(mb)
        lines.append(f"{ma.mode}  (n {na} -> {nb})")
        lines.append("  " + "  ".join(f"{LABEL.get(m, m)} {xa.get(m, 0):.2f}->{xb.get(m, 0):.2f}" for m in METRICS))
        for qid, ra in ma.questions.items():
            rb = mb.questions.get(qid)
            if ra.metrics is None or rb is None or rb.metrics is None:
                continue
            before, after = ra.metrics[f"hit@{k}"], rb.metrics[f"hit@{k}"]
            if before != after:
                lines.append(f"  {'gained' if after > before else 'lost  '} hit@{k}: {qid}")
    for run, other, label in ((b, a, "only in B"), (a, b, "only in A")):
        for m in run.modes:
            if other.mode(m.mode) is None:
                n, x = mean_metrics(m)
                lines.append(f"{m.mode}  ({label}, n {n})")
                lines.append("  " + "  ".join(f"{LABEL.get(k_, k_)} {x.get(k_, 0):.2f}" for k_ in METRICS))
    return lines


def _flat(d: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(_flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out
