from src.evaluation.final import comparison_table, counts, error_causes, error_table, gate_table, panel_misses, panel_table, parse_tables


def rec(kind, abstained, correctness="correct", gold=True):
    return {"kind": kind, "abstained": abstained, "correctness": correctness, "gold_in_top5": gold}


RECORDS = [
    rec("answerable", False), rec("answerable", True, "incorrect", gold=False), rec("answerable", False, "incorrect", gold=False),
    rec("answerable", False, "incorrect", gold=True), rec("near_miss", False, "n/a"), rec("misleading", True, "n/a"), rec("off_topic", False, "n/a"),
]


def test_counts_and_comparison_table():
    c = counts(RECORDS)
    assert c == {"answerable": 4, "correct": 1, "wrongly_refused": 1, "unanswerable": 3, "refused": 1, "answered_anyway": 2}
    table = comparison_table({"A": RECORDS})
    assert "| answerable questions answered correctly | 1 of 4 |" in table and "| unanswerable questions answered anyway | 2 of 3 |" in table


def test_error_causes_split_retrieval_from_generation():
    e = error_causes(RECORDS)
    assert e["retrieval_miss_refused"] == 1 and e["retrieval_miss_answered"] == 1 and e["generation_error"] == 1
    assert e["hallucinated_by_kind"] == {"near_miss": 1, "off_topic": 1}
    assert "| run A | 1 | 1 | 1 | 1 / 0 / 1 |" in error_table({"run A": RECORDS})


SUMMARY = """# Title

## Qwen test split

| metric | plain | refusal |
|---|---|---|
| answered correctly | 1.000 | 0.900 |
| wrongly refused | 0.000 | 0.100 |
| other row | 1 | 2 |
"""


def test_parse_and_render_gate_table():
    tables = parse_tables(SUMMARY)
    assert tables["Qwen test split"]["columns"] == ["plain", "refusal"] and tables["Qwen test split"]["wrongly refused"] == ["0.000", "0.100"]
    out = gate_table(tables, "Qwen test split")
    assert "| answered correctly | 1.000 | 0.900 |" in out and "other row" not in out
    assert gate_table(tables, "missing heading") == ""


def test_panel_misses_classify_question_types():
    rows = [{"qid": "Q1", "human_covered": "n", "question": "Why did you choose YOLO?"},
            {"qid": "Q2", "human_covered": "n", "question": "How did you validate the results?"},
            {"qid": "Q3", "human_covered": "n", "question": "What remains open?"},
            {"qid": "Q4", "human_covered": "n", "question": "Why this?"},
            {"qid": "Q5", "human_covered": "y", "question": "Why that?"},
            {"qid": "Q6", "human_covered": "", "question": "Unlabeled"}]
    key = {"Q1": "a", "Q2": "b", "Q3": "c", "Q4": "d", "Q5": "e", "Q6": "f"}
    status = {"a": "COVERED", "b": "COVERED", "c": "COVERED", "d": "NOT COVERED", "e": "COVERED", "f": "COVERED"}
    records = {k: {"source": "doc", "status": v} for k, v in status.items()}
    res = panel_misses(rows, key, records)
    assert res["missed_by_type"] == {"why or justify": 1, "validate or test": 1, "other": 1}
    assert res["grid"][("NOT COVERED", "n")] == 1 and res["grid"][("COVERED", "y")] == 1 and ("COVERED", "") not in res["grid"]
    assert "Gaps the system missed (answered, but the person said not covered): 3." in panel_table(res)
    records["a"]["source"] = "generic"
    assert sum(panel_misses(rows, key, records)["missed_by_type"].values()) == 2
