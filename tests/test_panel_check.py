from src.panel.categories import CATEGORIES
from src.panel.check import sample, score
from src.panel.run_panel import COVERED, NOT_COVERED


def rec(i, doc, category, status, source="doc"):
    return {"id": i, "doc": doc, "category": category, "question": f"Question {i}?", "status": status, "source": source}


def test_sample_is_deterministic_and_stratified():
    docs = ["D1", "D2"]
    cats = list(CATEGORIES)
    doc_rows = [rec(f"{d}-{c}", d, c, COVERED) for d in docs for c in cats]
    gen_rows = [rec(f"g-{d}-{c}", d, c, COVERED, source="generic") for d in docs for c in cats]
    s1, s2 = sample(doc_rows, gen_rows, n_doc=6, n_generic=4), sample(doc_rows, gen_rows, n_doc=6, n_generic=4)
    assert [r["id"] for r in s1] == [r["id"] for r in s2]
    assert sum(r["source"] == "doc" for r in s1) == 6 and sum(r["source"] == "generic" for r in s1) == 4
    assert len({(r["doc"], r["category"]) for r in s1 if r["source"] == "doc"}) == 6


def test_score_precision_recall_and_relevance():
    records = {"a": rec("a", "D", "ethics", NOT_COVERED), "b": rec("b", "D", "ethics", NOT_COVERED),
               "c": rec("c", "D", "ethics", COVERED), "d": rec("d", "D", "ethics", COVERED, source="generic")}
    key = {"Q1": "a", "Q2": "b", "Q3": "c", "Q4": "d"}
    rows = [{"qid": "Q1", "human_covered": "n", "human_relevance": "3"}, {"qid": "Q2", "human_covered": "y", "human_relevance": "2"},
            {"qid": "Q3", "human_covered": "n", "human_relevance": "1"}, {"qid": "Q4", "human_covered": "p", "human_relevance": ""}]
    out = score(rows, key, records)["doc"]
    assert out["labeled"] == 3 and out["system_not_covered"] == 2 and out["human_not_covered"] == 2
    assert out["not_covered_precision"] == 0.5 and out["not_covered_recall"] == 0.5
    assert out["wrongly_not_covered"] == 1 and out["wrongly_covered"] == 1 and out["mean_relevance"] == 2.0
    assert score(rows, key, records)["generic"]["human_partial"] == 1
    assert out["not_covered_precision_lenient"] == 0.5 and out["not_covered_recall_lenient"] == 0.5
    lenient = score([{"qid": "Q1", "human_covered": "p", "human_relevance": "3"}], {"Q1": "a"}, {"a": records["a"]})["doc"]
    assert lenient["not_covered_precision"] == 0.0 and lenient["not_covered_precision_lenient"] == 1.0
