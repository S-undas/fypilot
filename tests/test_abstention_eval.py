import json

import src.config as cfg
from src.evaluation.abstention_eval import (
    arms_table, build_arms, choose_signal, curve_rows, is_bad, nli_pools, nli_premises, pick, score_nli, signal_analysis,
)


def rec(i, kind, abstained=False, correctness="correct", split="dev", max_dense=0.5, top_score=0.0, **extra):
    return {"id": i, "kind": kind, "abstained": abstained, "correctness": correctness, "split": split,
            "max_dense": max_dense, "top_score": top_score, "supported_by_cited": "yes", "supported_by_retrieved": "yes",
            "n_valid_citations": 1, "gold_in_top5": True, "retrieved": ["c1"], "cited_chunks": [], "answer": "The system uses X for Y.", **extra}


def test_is_bad():
    assert not is_bad(rec("a", "answerable"))
    assert is_bad(rec("b", "answerable", correctness="incorrect"))
    assert is_bad(rec("c", "near_miss", correctness="not_applicable"))


def test_signal_analysis_and_choice():
    records = [rec("a1", "answerable", max_dense=0.8, top_score=5.0), rec("a2", "answerable", max_dense=0.7, top_score=4.0),
               rec("u1", "off_topic", max_dense=0.4, top_score=-6.0), rec("u2", "near_miss", max_dense=0.75, top_score=-3.0)]
    out = signal_analysis(records)
    assert out["top_score"]["auc"] == 1.0 and out["max_dense"]["auc"] < 1.0
    assert out["top_score"]["auc_by_kind"]["off_topic"] == 1.0
    assert choose_signal(out) == "top_score"


def test_nli_premises_prefers_cited_chunks():
    texts = {"c1": "one", "c2": "two", "c3": "three"}
    assert nli_premises({"cited_chunks": ["c2"], "retrieved": ["c1", "c2"]}, texts) == ["two"]
    assert nli_premises({"cited_chunks": [], "retrieved": ["c1", "c3", "zz"]}, texts) == ["one", "three"]


class FakeScorer:
    def entailment(self, pairs):
        return [0.9 if "supported" in c else 0.1 for _, c in pairs]


def test_score_nli_skips_refusals_and_caches(monkeypatch, tmp_path):
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(cfg.settings, "eval_dir", tmp_path)
    records = [rec("a", "answerable", answer="The supported claim is stated here."),
               rec("b", "near_miss", abstained=True, answer="The passages do not state it.")]
    scores = score_nli("r", records, FakeScorer(), {"c1": "evidence"})
    assert scores["a"]["frac"] == 1.0 and scores["a"]["min"] == 0.9 and scores["b"] is None
    assert json.loads((tmp_path / "results" / "nli3_r.json").read_text(encoding="utf-8")) == scores
    assert score_nli("r", records, None, {}) == scores
    assert pick(scores) == {"a": 1.0, "b": None} and pick(scores, "min")["a"] == 0.9


def test_nli_pools_split_good_and_bad():
    runs = {"g": [rec("a", "answerable"), rec("b", "near_miss", correctness="not_applicable"), rec("c", "answerable", split="test")]}
    good, bad = nli_pools(runs, {"g": {"a": 0.9, "b": 0.2, "c": 0.8}}, "dev")
    assert good == [0.9] and bad == [0.2]


def test_build_arms_and_table():
    plain = [rec("a", "answerable", max_dense=0.8), rec("u", "off_topic", correctness="not_applicable", max_dense=0.2),
             rec("n", "near_miss", correctness="not_applicable", max_dense=0.8)]
    taus = {"balanced": 0.5, "no_harm": 0.3}
    arms = build_arms(plain, plain, "max_dense", taus, {"a": 0.9, "u": None, "n": 0.1}, 0.5)
    assert list(arms) == ["plain", "prompt refusal", "+ threshold (balanced)", "+ threshold (no-harm)",
                          "prompt refusal + threshold (no-harm)", "+ threshold + NLI (no-harm)"]
    assert arms["+ threshold (balanced)"]["changed"] == ["u"] and arms["+ threshold (no-harm)"]["changed"] == ["u"]
    assert arms["+ threshold + NLI (no-harm)"]["changed"] == ["n"]
    table = arms_table(arms, "dev")
    assert "| unanswerable questions answered anyway | 2 | 2 | 1 | 1 | 1 | 0 |" in table
    assert list(build_arms(plain, plain, "max_dense", {}, {}, None)) == ["plain", "prompt refusal"]


def test_curve_rows():
    rows = curve_rows("g", "dev", [0.8, 0.9], [0.1, 0.2], n=5)
    assert len(rows) == 5 and rows[0]["answerable_kept"] == 1.0 and rows[-1]["unanswerable_refused"] == 1.0
    assert curve_rows("g", "dev", [], []) == []
