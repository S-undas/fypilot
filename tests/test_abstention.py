from src.abstention.claims import split_claims
from src.abstention.gates import apply_nli_gate, apply_retrieval_gate, auc, best_threshold, sweep, threshold_keeping
from src.abstention.nli import support_score, support_scores, windows


def test_split_claims_strips_citations_markdown_and_refusals():
    answer = (
        "The system uses **YOLO nano** on a Raspberry Pi 4 [2][3].\n"
        "- It streams video over a local network connection.\n"
        "The passages do not state the measured latency.\n"
        "Yes."
    )
    assert split_claims(answer) == [
        "The system uses YOLO nano on a Raspberry Pi 4.",
        "It streams video over a local network connection.",
    ]


def test_split_claims_pure_refusal_is_empty():
    assert split_claims("The provided passages do not specify any particular future improvements.") == []
    assert split_claims("No numeric accuracy result is given for this test case.") == []
    assert split_claims("The win rate for that topology is not reported in the document.") == []


class FakeScorer:
    def __init__(self, table):
        self.table = table

    def entailment(self, pairs):
        return [self.table.get((p, c), 0.0) for p, c in pairs]


def test_support_score_best_premise_per_claim_weakest_claim_overall():
    answer = "The model runs on a small board. It was trained for ten days."
    claims = split_claims(answer)
    scorer = FakeScorer({("p1", claims[0]): 0.9, ("p2", claims[0]): 0.1, ("p1", claims[1]): 0.2, ("p2", claims[1]): 0.4})
    assert support_score(scorer, answer, ["p1", "p2"]) == 0.4
    assert support_score(scorer, "The duration is not stated in the passages.", ["p1"]) is None
    assert support_score(scorer, answer, []) is None


def test_auc():
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.5], [0.5]) == 0.5
    assert auc([], [0.1]) is None


def test_best_threshold_separates_and_prefers_fewer_refusals():
    keep, drop = [0.8, 0.9, 0.7], [0.2, 0.3, 0.1]
    tau = best_threshold(keep, drop)
    assert 0.3 < tau <= 0.7
    assert best_threshold([], drop) is None
    assert sweep(keep, drop, [0.5]) == [{"tau": 0.5, "answerable_kept": 1.0, "unanswerable_refused": 1.0}]


def rec(i, kind, abstained, **extra):
    return {"id": i, "kind": kind, "abstained": abstained, "correctness": "correct", "supported_by_cited": "yes",
            "supported_by_retrieved": "yes", "n_valid_citations": 1, **extra}


def test_retrieval_gate_refuses_low_scores_and_keeps_existing_refusals():
    records = [rec("a", "answerable", False, s=0.9), rec("b", "answerable", False, s=0.2),
               rec("c", "near_miss", True, s=0.1), rec("d", "off_topic", False, s=0.3)]
    out = apply_retrieval_gate(records, "s", 0.5)
    assert out.changed == ["b", "d"]
    by = {r["id"]: r for r in out.records}
    assert by["b"]["abstained"] is True and by["b"]["correctness"] == "incorrect"
    assert by["d"]["correctness"] == "not_applicable" and by["d"]["gated_by"] == "retrieval"
    assert by["a"]["abstained"] is False and "gated_by" not in by["c"]


def test_nli_gate_only_touches_answered_records_with_a_score():
    records = [rec("a", "answerable", False), rec("b", "near_miss", False), rec("c", "near_miss", True)]
    out = apply_nli_gate(records, {"a": 0.95, "b": 0.1, "c": 0.0}, 0.5)
    assert out.changed == ["b"]
    assert {r["id"]: r["abstained"] for r in out.records} == {"a": False, "b": True, "c": True}


def test_list_lead_ins_are_not_claims():
    answer = "The planned evaluation metrics are:\n- Mean Average Precision to gauge detection accuracy.\n- Detection latency of the whole pipeline."
    claims = split_claims(answer)
    assert len(claims) == 2 and all(not c.endswith(":") for c in claims)


def test_support_scores_min_and_fraction():
    answer = "The model runs on a small board. It was trained for ten days. It is lightweight and fast."
    claims = split_claims(answer)
    scorer = FakeScorer({("p", claims[0]): 0.9, ("p", claims[1]): 0.1, ("p", claims[2]): 0.7})
    out = support_scores(scorer, answer, ["p"])
    assert out["min"] == 0.1 and out["claims"] == 3 and abs(out["frac"] - 2 / 3) < 1e-9
    assert support_scores(scorer, "The duration is not stated in the passages.", ["p"]) is None


def test_threshold_keeping_allows_few_refusals():
    scores = [i / 20 for i in range(20)]
    tau = threshold_keeping(scores, 0.9)
    assert sum(s >= tau for s in scores) == 18
    assert threshold_keeping([0.2, 0.4], 0.94) < 0.2
    assert threshold_keeping([], 0.94) is None


def test_windows_cover_long_text_with_overlap():
    text = " ".join(f"w{i}" for i in range(200))
    ws = windows(text, size=50, stride=25)
    assert all(len(w.split()) <= 50 for w in ws) and len(ws) > 4
    assert ws[0].split()[0] == "w0" and ws[-1].split()[-1] == "w199"
    assert set(ws[0].split()) & set(ws[1].split())
    assert windows("short text here") == ["short text here"]
