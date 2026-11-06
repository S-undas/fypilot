import copy

import pytest

from src.retrieval.index import Index


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_relevant_chunk_ranks_first(index, mode):
    hits = index.search("yolo object detection camera", k=2, mode=mode)
    assert hits[0].chunk.chunk_id == "A:0"
    assert hits[0].rank == 1 and len(hits) == 2


def test_doc_filter(index):
    hits = index.search("object detection", k=4, mode="hybrid", docs=["B"])
    assert {h.chunk.doc_id for h in hits} == {"B"}
    assert index.search("object detection", docs=["missing"]) == []


def test_hit_has_both_scores(index):
    h = index.search("budget cost", k=1, mode="bm25")[0]
    assert h.chunk.chunk_id == "B:2"
    assert h.bm25 > 0 and -1 <= h.dense <= 1


def test_unknown_mode(index):
    with pytest.raises(ValueError):
        index.search("x", mode="nope")


def test_save_load_roundtrip(index, embedder, tmp_path):
    index.save(tmp_path)
    loaded = Index.load(tmp_path, embedder)
    a = index.search("reinforcement learning", k=3, mode="dense")
    b = loaded.search("reinforcement learning", k=3, mode="dense")
    assert [h.chunk.chunk_id for h in a] == [h.chunk.chunk_id for h in b]


def test_load_rejects_other_model(index, embedder, tmp_path):
    index.save(tmp_path)
    other = copy.copy(embedder)
    other.model_name = "different"
    with pytest.raises(RuntimeError):
        Index.load(tmp_path, other)
