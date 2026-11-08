from src.evaluation.judge_check import agreement, kappa, needs_review, sample


def rec(i, kind, abstained, correctness, answer="The system uses X."):
    return {"id": i, "kind": kind, "abstained": abstained, "correctness": correctness, "answer": answer}


def test_kappa_perfect_and_chance():
    assert kappa([("y", "y"), ("n", "n"), ("y", "y"), ("n", "n")]) == 1.0
    assert kappa([("y", "n"), ("n", "y")]) == -1.0
    assert kappa([]) is None


def test_sample_is_stratified_and_deterministic():
    records = [rec(f"a{i}", "answerable", False, "correct") for i in range(10)]
    records += [rec(f"u{i}", "near_miss", True, "not_applicable", "The passages do not state it.") for i in range(10)]
    s1, s2 = sample(records, 4, 3), sample(records, 4, 3)
    assert [r["id"] for r in s1] == [r["id"] for r in s2]
    assert sum(r["kind"] == "answerable" for r in s1) == 4 and len(s1) == 7


def test_agreement_counts_only_filled_rows():
    records = [rec("a", "answerable", False, "correct"), rec("b", "answerable", False, "partial"),
               rec("c", "near_miss", True, "not_applicable")]
    rows = [
        {"id": "a", "human_abstained": "n", "human_correctness": "correct"},
        {"id": "b", "human_abstained": "n", "human_correctness": "incorrect"},
        {"id": "c", "human_abstained": "y", "human_correctness": ""},
    ]
    out = agreement(rows, records)
    assert out["abstained"]["n"] == 3 and out["abstained"]["agreement"] == 1.0
    assert out["correctness"]["n"] == 2 and out["correctness"]["agreement"] == 0.5


def test_needs_review_flags_disagreements_and_misses():
    refusal = "The passages do not state the accuracy."
    assert needs_review(rec("a", "near_miss", False, "not_applicable", refusal))
    assert not needs_review(rec("b", "near_miss", True, "not_applicable", refusal))
    assert needs_review(rec("c", "answerable", False, "incorrect"))
    assert not needs_review(rec("d", "answerable", False, "correct"))


def test_sample_always_includes_flagged_records():
    records = [rec(f"a{i}", "answerable", False, "correct") for i in range(10)]
    records.append(rec("bad", "answerable", True, "incorrect", "The passages do not state it."))
    assert "bad" in [r["id"] for r in sample(records, 2, 0)]
