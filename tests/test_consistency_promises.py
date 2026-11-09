import json

from src.consistency.promises import extract, judge_promise, parse_promises, parse_verdict, render_sheet
from src.consistency.structure import Structure, chunks_in
from src.ingestion.chunks import Chunk

TEXT = "The system will detect objects in real time. We aim for 95% accuracy on the test set. Students enjoy robots."


def test_parse_promises_keeps_only_quotes_found_in_the_text():
    raw = json.dumps([
        {"promise": "Detect objects in real time", "quote": "will detect objects in real time"},
        {"promise": "Reach 95% accuracy", "quote": "aim for 95% accuracy on the test set"},
        {"promise": "Invented", "quote": "the battery lasts ten hours on one charge"},
        {"promise": 5, "quote": "x"},
    ])
    out = parse_promises(raw, TEXT)
    assert [p["promise"] for p in out] == ["Detect objects in real time", "Reach 95% accuracy"]
    assert parse_promises("no list", TEXT) == [] and parse_promises("[oops", TEXT) == []


def test_parse_verdict():
    assert parse_verdict('{"verdict": "not_shown", "evidence": [2, "x"], "reason": "only planned"}') == {
        "verdict": "not_shown", "evidence": [2], "reason": "only planned"}
    assert parse_verdict('{"verdict": "maybe"}') is None


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, messages):
        return self.replies.pop(0)


def chunk(i, section, text):
    return Chunk(chunk_id=f"D:{i}", doc_id="D", doc_type="reports", section=section, page_start=str(i + 1), page_end=str(i + 1), text=text)


def test_extract_drops_near_duplicate_promises():
    one = json.dumps([{"promise": "The system will detect objects in real time", "quote": "will detect objects in real time"}])
    dup = json.dumps([{"promise": "The system will detect objects in real time.", "quote": "detect objects in real time"}])
    chunks = [chunk(0, "1 Introduction", TEXT), chunk(1, "2.1 Goals", TEXT)]
    out = extract(ScriptedLLM([one, dup]), chunks)
    assert len(out) == 1 and out[0]["section"] == "1 Introduction" and out[0]["page"] == "1"


def test_chunks_in_selects_chapters_by_title():
    chunks = [chunk(0, "1 Introduction", "a"), chunk(1, "4 Results", "b"), chunk(2, "4.1 Setup", "c")]
    s = Structure(("Introduction", "Results"), {"Introduction": 1, "Results": 4}, {"D:0": 1, "D:1": 4, "D:2": 4})
    assert [c.chunk_id for c in chunks_in(s, chunks, "result")] == ["D:1", "D:2"]


def test_judge_promise_cites_evidence(index):
    p = {"promise": "budget estimate for hardware", "quote": "q", "section": "1 Intro", "page": "1"}
    keep = {c.chunk_id for c in index.chunks if c.doc_id == "B"}
    llm = ScriptedLLM([json.dumps({"verdict": "shown", "evidence": [1], "reason": "costs listed"})])
    out = judge_promise(p, "B", keep, index, None, llm)
    assert out["verdict"] == "shown" and out["evidence"] == ["1 Costs, p. 3"]
    bad = judge_promise(p, "B", keep, index, None, ScriptedLLM(["cannot say"]))
    assert bad["verdict"] == "parse_error"


def test_render_sheet_lists_unshown_first():
    rows = [
        {"doc": "D", "verdict": "shown", "promise": "A works", "section": "1 Intro", "page": "2", "quote": "a", "reason": "r1", "evidence": ["7 Results, p. 9"]},
        {"doc": "D", "verdict": "not_shown", "promise": "B is fast", "section": "2 Vision", "page": "", "quote": "b", "reason": "r2", "evidence": []},
    ]
    md = render_sheet("D", rows)
    assert "2 promises found. 1 are not shown or contradicted." in md
    assert md.index("B is fast") < md.index("A works") and "Evidence: 7 Results, p. 9" in md
