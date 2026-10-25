"""`rag eval --diagnose`: rank buckets, reranking ceiling, branch complementarity, misses."""

from growth_rag.evaluation.dataset import Question
from growth_rag.evaluation.diagnose import (
    best_rank,
    bucket_lines,
    buckets,
    ceiling_lines,
    closeness,
    complementarity_lines,
    diagnose_lines,
    with_targets,
)
from growth_rag.evaluation.run import EvalRun, ModeRun, QuestionRun
from growth_rag.utils.io import utc_now

T = [["b:1/s3/c4"]]  # one entry, satisfied by chunk 4 of section 3 of book b:1


def qr(rank, results=(), targets=T):
    return QuestionRun(results=list(results), ranks=[rank], targets=targets,
                       metrics={"hit@10": float(rank is not None and rank <= 10)})


def make_run(modes: dict[str, dict[str, QuestionRun]], depth=100) -> EvalRun:
    return EvalRun(set_name="t", created_at=utc_now(), questions_sha256="x", settings={}, depth=depth,
                   modes=[ModeRun(mode=name, questions=qs) for name, qs in modes.items()])


def Q(qid, gap=None):
    return Question(id=qid, task="passage", query="?", language="en", known_gap=gap)


def test_best_rank_and_buckets_follow_the_depth():
    assert best_rank(QuestionRun(results=[], ranks=[None, 7, 3])) == 3
    assert best_rank(QuestionRun(results=[], ranks=[None])) is None
    assert [b[0] for b in buckets(100)] == ["1", "2-5", "6-10", "11-20", "21-100"]
    assert [b[0] for b in buckets(8)] == ["1", "2-5", "6-8"]


def test_bucket_table_counts_each_question_once():
    run = make_run({"chunks:hybrid": {"a": qr(1), "b": qr(3), "c": qr(15), "d": qr(None),
                                      "unscored": QuestionRun(results=[])}})
    row = bucket_lines(run)[2].split()
    assert row == ["chunks:hybrid", "4", "1", "1", "0", "1", "0", "1"]


def test_ceiling_is_what_a_perfect_reordering_could_reach():
    run = make_run({"chunks:hybrid": {"a": qr(1), "b": qr(3), "c": qr(40), "d": qr(None)}})
    text = "\n".join(ceiling_lines(run, "chunks:hybrid"))
    assert "hit@1: 1/4 now, at most 3/4" in text and "hit@5: 2/4 now, at most 3/4" in text


def test_complementarity_shows_union_and_fusion_losses():
    run = make_run({
        "chunks:dense": {"a": qr(1), "b": qr(None), "c": qr(2)},
        "chunks:sparse": {"a": qr(None), "b": qr(1), "c": qr(None)},
        "chunks:hybrid": {"a": qr(1), "b": qr(4), "c": qr(30)},
    })
    lines = complementarity_lines(run)
    assert lines[2].split() == ["1", "1", "1", "1", "2", "1", "1"]  # k=1: dense a; sparse b; hybrid a
    assert lines[3].split() == ["5", "2", "1", "2", "3", "2", "1"]  # k=5: dense a,c; hybrid a,b
    assert lines[-2].endswith("c")  # dense found c, hybrid lost it
    assert lines[-1].endswith("b")


def test_closeness_reads_positions_from_the_keys():
    assert closeness(["b:1/s3/c6", "b:2/s1/c0"], T, 10) == "same section, 2 chunk(s) away"
    assert closeness(["b:1/s9/c0"], T, 10) == "same book, other section"
    assert closeness(["b:2/s3/c4"], T, 10) == "other book"
    assert closeness(["b:1/s3/c4"], [[]], 10) == "not indexed"  # e.g. front matter


def test_diagnose_lists_misses_with_known_gaps(tmp_path):
    run = make_run({"chunks:hybrid": {"hit": qr(2), "miss": qr(14, ["b:1/s3/c5"]),
                                      "gap": qr(None, ["b:1/s8/c0"], targets=[[]])}})
    text = "\n".join(diagnose_lines(run, [Q("hit"), Q("miss"), Q("gap", gap="not chunked")]))
    assert "Misses of chunks:hybrid at top 10: 2" in text
    assert "same section, 1 chunk(s) away" in text and "not indexed  [known gap]" in text
    assert "hit " not in text.split("Misses")[1]


def test_old_runs_get_their_targets_from_the_current_data():
    run = make_run({"chunks:hybrid": {"a": QuestionRun(results=["b:1/s3/c5"], ranks=[None], metrics={})}})
    with_targets(run, {"a": T})
    assert run.modes[0].questions["a"].targets == T


def test_best_rank_ignores_related_entries():
    r = QuestionRun(results=[], ranks=[1, 9], grades=[1, 2])
    assert best_rank(r) == 9
