from src.retrieval.rerank import Reranker, retrieve


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        return [1.0 if "budget" in text else 0.0 for _, text in pairs]


def test_rerank_reorders_by_cross_encoder_score(index):
    hits = index.search("object detection", k=4, mode="dense")
    rr = Reranker("fake")
    rr._model = FakeCrossEncoder()
    out = rr.rerank("object detection", hits)
    assert out[0].chunk.chunk_id == "B:2"
    assert [h.rank for h in out] == [1, 2, 3, 4]
    assert len(out) == len(hits)


def test_rerank_empty(index):
    rr = Reranker("fake")
    rr._model = FakeCrossEncoder()
    assert rr.rerank("q", []) == []


def test_retrieve_without_and_with_reranker(index):
    plain = retrieve(index, "object detection", 2, "dense", ["A", "B"])
    assert len(plain) == 2
    rr = Reranker("fake")
    rr._model = FakeCrossEncoder()
    out = retrieve(index, "object detection", 2, "dense", ["A", "B"], rr, pool=4)
    assert len(out) == 2 and out[0].chunk.chunk_id == "B:2"
