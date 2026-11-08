from src.evaluation.dataset import EvalItem
from src.evaluation.retrieval_eval import aggregate, evaluate_arm, gold_chunk_ids, per_item_ranks


def item(qid, doc, question, section, quote, split="dev", kind="answerable"):
    return EvalItem(id=qid, doc=doc, kind=kind, question=question, gold_answer="x" if kind == "answerable" else "",
                    gold_section=section, evidence_quote=quote, split=split)


def test_gold_chunk_is_the_chunk_containing_the_quote(index):
    it = item("q1", "A", "which detector", "1 Methods", "yolo object detection")
    assert gold_chunk_ids(it, index.chunks) == {"A:0"}


def test_ranks_and_aggregate(index):
    items = [
        item("q1", "A", "yolo object detection camera frames", "1 Methods", "yolo object detection", "dev"),
        item("q2", "B", "budget cost estimate hardware", "1 Costs", "budget and cost estimate", "test"),
        item("u1", "A", "unrelated", "", "", "dev", kind="off_topic"),
    ]
    search = lambda q, k, docs: index.search(q, k, "dense", docs)
    rows = per_item_ranks(items, index.chunks, search)
    assert [r["id"] for r in rows] == ["q1", "q2"]
    assert all(r["chunk_rank"] == 1 and r["section_rank"] == 1 for r in rows)
    agg = aggregate(rows)
    assert agg["recall@1"] == 1.0 and agg["mrr"] == 1.0 and agg["n"] == 2


def test_evaluate_arm_splits_and_misses(index):
    items = [
        item("q1", "A", "reinforcement learning agents", "1 Methods", "yolo object detection", "dev"),
        item("q2", "B", "budget cost estimate hardware", "1 Costs", "budget and cost estimate", "test"),
    ]
    out = evaluate_arm(items, index.chunks, lambda q, k, docs: index.search(q, 1, "dense", docs))
    assert out["dev"]["recall@1"] == 0.0 and out["dev"]["mrr"] == 0.0
    assert out["test"]["recall@1"] == 1.0
    assert out["all"]["n"] == 2
