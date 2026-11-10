from src.consistency.report import contradictions_section, human_section, missing_section, pct, per_doc, planted_section, promises_section


def test_pct():
    assert pct(1, 4) == "25% (1 of 4)" and pct(0, 0) == "-"


def test_per_doc_counts_each_document_and_stage():
    rows = [{"doc": "F25_044", "verdict": "present"}, {"doc": "F25_044", "verdict": "missing"}, {"doc": "F25_181", "verdict": "present"}]
    table = per_doc(rows, ["present", "partial", "missing"])
    assert "| F25_044 | rd | 2 | 2 | 1 | 0 | 1 |" in table and "| F25_181 | dev | 1 | 1 | 1 | 0 | 0 |" in table


def test_missing_section_compares_heading_rule_and_model():
    perturb = [{"variant": "remove", "heading": "missing", "verdict": "missing"}, {"variant": "empty", "heading": "present", "verdict": "partial"}]
    text = missing_section([{"doc": "F25_044", "verdict": "present"}], perturb)
    assert "| chapter deleted | 1 | 100% (1 of 1) | 100% (1 of 1) | 100% (1 of 1) |" in text
    assert "| heading kept, content removed | 1 | 0% (0 of 1) | 100% (1 of 1) | 0% (0 of 1) |" in text


def test_other_sections():
    assert "| F25_044 | 2 | 1 | 1 |" in contradictions_section([{"doc": "F25_044", "first": "conflict", "verdict": "confirmed"}, {"doc": "F25_044", "first": "consistent", "verdict": "consistent"}])
    assert "Pairs judged by: g/m (1)." in contradictions_section([{"doc": "F25_044", "first": "consistent", "verdict": "consistent", "served_by": "g/m"}])
    assert "| F25_182 | rd | 2 | 1 | 0 | 1 | 0 |" in promises_section([{"doc": "F25_182", "verdict": "not_shown"}])
    assert human_section(None) == "" and planted_section(None) == ""
    res = {"promise": {"planted": 10, "extracted": 9, "flagged": 8}, "contradiction": {"planted": 10, "found": 6}}
    assert "| contradiction (number changed in one place) | 10 | 60% (6 of 10) |" in planted_section(res)
