from src.evaluation.metrics import summarize_by_split
import json

import src.config as cfg
from src.evaluation.report import fmt, judge_agreement_section, metrics_table, retrieval_table


def rec(kind, abstained, correctness, split):
    return {"kind": kind, "abstained": abstained, "correctness": correctness, "supported_by_cited": "yes",
            "supported_by_retrieved": "yes", "n_valid_citations": 1, "gold_in_top5": True, "split": split}


def test_fmt():
    assert fmt(None) == "-" and fmt(0.5) == "0.500" and fmt(7) == "7"


def test_table_only_includes_runs_that_have_the_split():
    both = summarize_by_split([rec("answerable", False, "correct", "dev"), rec("near_miss", True, "not_applicable", "test")])
    dev_only = summarize_by_split([rec("answerable", False, "partial", "dev")])
    table = metrics_table({"a": both, "b": dev_only}, "dev")
    assert "| metric | a | b |" in table and "answered correctly | 1.000 | 0.000" in table
    test_table = metrics_table({"a": both, "b": dev_only}, "test")
    assert "| metric | a |" in test_table and " b " not in test_table.splitlines()[0]
    assert metrics_table({"b": dev_only}, "test") == ""


def test_retrieval_table(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("arm,split,n,mrr\ndense,dev,19,0.67\nhybrid,dev,19,0.76\n", encoding="utf-8")
    out = retrieval_table(p)
    assert "| dense | 19 | 0.67 |" in out and "**dev**" in out


def test_judge_agreement_section(monkeypatch, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(cfg.settings, "eval_dir", tmp_path)
    assert judge_agreement_section("r") == ""
    record = {"id": "a", "kind": "answerable", "abstained": False, "correctness": "correct"}
    (results / "judged_r.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    header = "id,kind_hidden,question,reference_answer,system_answer,human_abstained,human_correctness,note"
    (results / "judge_check_r.csv").write_text(header + "\na,,q,ref,ans,n,correct,\n", encoding="utf-8")
    out = judge_agreement_section("r")
    assert "| refused or answered | 1 | 1.000 |" in out and "| correct, partial or incorrect | 1 | 1.000 |" in out
