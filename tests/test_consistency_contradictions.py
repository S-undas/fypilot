import json

from src.consistency.contradictions import candidate_pairs, judge_pair, parse_facts, parse_verdict, render_sheet

TEXT = "Detections below 0.4 confidence are removed. The model is YOLOv8n trained for 50 epochs."


def test_parse_facts_keeps_only_quotes_found_in_the_text():
    raw = json.dumps([
        {"fact": "Detections below 0.4 confidence are removed", "quote": "Detections below 0.4 confidence are removed."},
        {"fact": "Made up", "quote": "the battery lasts ten hours"},
        {"fact": 3, "quote": "x"},
    ])
    out = parse_facts(raw, TEXT)
    assert len(out) == 1 and out[0]["fact"].startswith("Detections below 0.4")
    assert parse_facts("nothing", TEXT) == []


def test_parse_verdict_respects_allowed_values():
    assert parse_verdict('{"verdict": "conflict", "reason": "numbers differ"}', {"conflict", "consistent"}) == {"verdict": "conflict", "reason": "numbers differ"}
    assert parse_verdict('{"verdict": "confirmed"}', {"conflict"}) is None


def fact(cid, section, text):
    return {"chunk_id": cid, "section": section, "page": "1", "fact": text, "quote": text}


def test_candidate_pairs_link_same_topic_across_chunks(embedder):
    facts = [
        fact("D:0", "5.2 Data", "Detections with confidence below 0.4 are removed"),
        fact("D:1", "8.2 Outliers", "Detections with confidence below 25 percent are removed"),
        fact("D:2", "7.3 Hardware", "The camera is streamed over wifi to the board"),
        fact("D:0", "5.2 Data", "Detections with confidence below 0.4 are dropped from results"),
    ]
    pairs = candidate_pairs(facts, embedder, threshold=0.5)
    linked = {(i, j) for i, j, _ in pairs}
    assert (0, 1) in linked and (1, 3) in linked
    assert (0, 3) not in linked and not any(2 in p for p in linked)
    assert candidate_pairs(facts[:1], embedder) == []


class ScriptedLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def complete(self, messages, max_tokens=None):
        self.calls += 1
        return self.replies.pop(0)


def test_judge_pair_confirms_only_flagged_conflicts():
    a, b = fact("D:0", "5.2", "cutoff 0.4"), fact("D:1", "8.2", "cutoff 25%")
    llm = ScriptedLLM([json.dumps({"verdict": "conflict", "reason": "numbers differ"}), json.dumps({"verdict": "confirmed", "reason": "still differ"})])
    assert judge_pair(a, b, "t1", "t2", llm)["verdict"] == "confirmed" and llm.calls == 2
    llm = ScriptedLLM([json.dumps({"verdict": "conflict", "reason": "x"}), json.dumps({"verdict": "not_confirmed", "reason": "after tuning"})])
    out = judge_pair(a, b, "t1", "t2", llm)
    assert out["first"] == "conflict" and out["verdict"] == "not_confirmed"
    llm = ScriptedLLM([json.dumps({"verdict": "consistent", "reason": "same"})])
    assert judge_pair(a, b, "t1", "t2", llm)["verdict"] == "consistent" and llm.calls == 1
    assert judge_pair(a, b, "t1", "t2", ScriptedLLM(["??"]))["verdict"] == "parse_error"


def test_render_sheet_lists_confirmed_conflicts_only():
    a, b = fact("D:0", "5.2 Data", "cutoff 0.4"), fact("D:1", "8.2 Outliers", "cutoff 25%")
    rows = [{"doc": "D", "a": a, "b": b, "verdict": "confirmed", "reason": "numbers differ"},
            {"doc": "D", "a": a, "b": b, "verdict": "not_confirmed", "reason": "explained"}]
    md = render_sheet("D", rows)
    assert "2 statement pairs compared. 1 conflicts confirmed." in md and md.count("numbers differ") == 1 and "explained" not in md


def test_signature_needs_numbers_or_names_and_skips_equal_pairs(embedder):
    from src.consistency.contradictions import signature

    assert signature({"quote": "YOLOv8n reaches 0.63 precision, 25% recall"}) == {"YOLOv8n", "0.63", "25%"}
    assert signature({"quote": "the camera streams video"}) == frozenset()
    facts = [fact("D:0", "5", "ResNet model is used for crop disease"), fact("D:1", "8", "EfficientNet model is used for crop disease"),
             fact("D:2", "9", "ResNet model is used for crop disease today")]
    linked = {(i, j) for i, j, _ in candidate_pairs(facts, embedder, threshold=0.3)}
    assert (0, 1) in linked and (1, 2) in linked and (0, 2) not in linked


def test_cap_never_drops_number_pairs_that_share_wording(embedder):
    facts = [fact("D:0", "5.2", "Detections with confidence below 0.4 are removed"), fact("D:1", "8.2", "The project removed predictions below 25% confidence"),
             fact("D:2", "7", "The ResNet backbone feeds the Dense head"), fact("D:3", "9", "The EfficientNet backbone feeds the Dense head")]
    capped = {(i, j) for i, j, _ in candidate_pairs(facts, embedder, threshold=0.2, limit=1)}
    assert (0, 1) in capped and len(capped) >= 2
    assert {(i, j) for i, j, _ in candidate_pairs(facts, embedder, threshold=0.2, limit=100)} >= capped
