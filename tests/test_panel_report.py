from src.panel.categories import CATEGORIES
from src.panel.report import group_table, human_section, source_table
from src.panel.run_panel import COVERED, NOT_COVERED


def rec(i, doc, category, status, source="doc", specificity=0.6, **extra):
    return {"id": i, "doc": doc, "category": category, "status": status, "source": source, "specificity": specificity, **extra}


def test_report_tables():
    doc_rows = [rec("a", "D1", "motivation", COVERED, j_answers_question="yes", j_supported="yes"), rec("b", "D1", "ethics", NOT_COVERED)]
    gen_rows = [rec("g", "D1", "ethics", NOT_COVERED, source="generic", specificity=0.4)]
    table = source_table({"doc": doc_rows, "generic": gen_rows})
    assert "marked NOT COVERED | 50% (1 of 2) | 100% (1 of 1)" in table
    assert "| ethics | 1 | 100% (1 of 1) |" in group_table(doc_rows, "category", list(CATEGORIES))
    assert human_section(None) == ""
