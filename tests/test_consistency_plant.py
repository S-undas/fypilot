import random

from src.consistency.plant import change_number, covers, edits_from, pick_contradictions, score, shared_numbers
from src.consistency.structure import apply_edits
from src.ingestion.chunks import Chunk


def test_change_number_changes_only_the_first_match():
    assert change_number("Detections below 0.4 confidence, 0.4 again", "0.4") == "Detections below 0.6 confidence, 0.4 again"
    assert change_number("Accuracy of 80% overall", "80%") == "Accuracy of 60% overall"
    assert change_number("Accuracy of 20% overall", "20%") == "Accuracy of 40% overall"
    assert change_number("Uses 50 images", "50") == "Uses 100 images"


def test_shared_numbers_skip_years_and_tiny_values():
    assert shared_numbers("in 2024 we used 3 cameras and 0.4 threshold", "threshold 0.4 and 3 cameras in 2024") == ["0.4"]
    assert shared_numbers("a 40 mm lens", "a 12 mm lens") == []


def fact(cid, section, quote):
    return {"chunk_id": cid, "section": section, "page": "1", "fact": quote, "quote": quote}


def test_pick_contradictions_uses_agreeing_pairs_with_verbatim_quotes(embedder):
    text = {"D:0": "Detections below 0.4 confidence are removed.", "D:1": "A 0.4 confidence threshold removes weak detections.",
            "D:2": "The camera streams video."}
    facts = {"D": [fact("D:0", "5.2", "Detections below 0.4 confidence are removed."), fact("D:1", "8.2", "A 0.4 confidence threshold removes weak detections."),
                   fact("D:2", "7.3", "The camera streams video.")]}
    picked = pick_contradictions(facts, embedder, text, 5, random.Random(1), min_sim=0.3)
    assert len(picked) == 1 and picked[0]["chunk_id"] == "D:1" and "0.6" in picked[0]["replace"] and picked[0]["find"] in text["D:1"]
    assert pick_contradictions(facts, embedder, {"D:1": "different text"}, 5, random.Random(1), min_sim=0.3) == []


def test_edits_apply_to_copies_and_score_matches_by_overlap():
    chunk = Chunk(chunk_id="D:1", doc_id="D", doc_type="reports", section="8.2", page_start="9", page_end="9", text="A 0.4 threshold removes weak detections.")
    plants = [{"kind": "contradiction", "doc": "D", "chunk_id": "D:1", "section": "8.2", "find": "0.4 threshold", "replace": "0.6 threshold"},
              {"kind": "promise", "doc": "D", "chunk_id": "D:1", "section": "8.2", "inserted": "The system will answer within 200 ms."}]
    edits = edits_from(plants, {"D:1": chunk.text})
    assert edits["D:1"] == "A 0.6 threshold removes weak detections. The system will answer within 200 ms."
    assert apply_edits([chunk], edits)[0].text == edits["D:1"] and chunk.text.startswith("A 0.4")
    prom = [{"doc": "D", "section": "8.2", "quote": "will answer within 200 ms", "verdict": "not_shown"}]
    conf = [{"doc": "D", "verdict": "confirmed", "a": {"chunk_id": "D:1", "quote": "A 0.6 threshold removes weak detections."},
             "b": {"chunk_id": "D:0", "quote": "threshold 0.4"}}]
    res = score(plants, prom, conf)
    assert res["promise"] == {"planted": 1, "extracted": 1, "flagged": 1} and res["contradiction"] == {"planted": 1, "found": 1}
    assert score(plants, [], [])["promise"]["extracted"] == 0 and covers("will answer within 200 ms", "will answer within 200 ms")
