from src.panel.judge import judge, parse
from src.panel.run_panel import COVERED, NOT_COVERED, panel_status, render_sheet
from src.qa.answer import build_messages


def test_panel_status():
    assert panel_status("NOT COVERED") == NOT_COVERED and panel_status("  Not covered.") == NOT_COVERED
    assert panel_status("The passages do not state the measured latency.") == NOT_COVERED
    assert panel_status("The system runs on a Raspberry Pi 4 board [1].") == COVERED


def test_panel_prompt_asks_for_not_covered(index):
    hits = index.search("budget", k=1, mode="bm25")
    system = build_messages("q", hits, prompt="panel")[0]["content"]
    assert "NOT COVERED" in system and "[2]" in system


def rec(i, doc, category, status):
    return {"id": i, "doc": doc, "category": category, "question": f"Question {i}?", "status": status, "answer": "Answer text [1].",
            "sources": ["[1] D, 1 Intro, p. 3"]}


def test_render_sheet_lists_weak_spots_and_hides_answers_for_not_covered():
    rows = [rec("a", "D1", "motivation", COVERED), rec("b", "D1", "ethics", NOT_COVERED), rec("c", "D2", "ethics", NOT_COVERED)]
    md = render_sheet("D1", rows)
    assert "1 are not covered" in md and "## Weak spots" in md and "(ethics) Question b?" in md
    assert "**Draft answer:** Answer text [1]." in md and md.count("**Draft answer:**") == 1
    assert "Question c?" not in md


def test_judge_parse():
    assert parse('{"answers_question": "yes", "supported": "partial"}') == {"answers_question": "yes", "supported": "partial"}
    assert parse('{"answers_question": "maybe", "supported": "yes"}') is None

    class Bad:
        def complete(self, messages):
            return "cannot judge"

    assert judge(Bad(), "q", "a", [])["answers_question"] == "parse_error"
