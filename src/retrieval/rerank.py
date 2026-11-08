from __future__ import annotations

from dataclasses import replace

from src.config import settings
from src.retrieval.index import Hit, index_text


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.reranker_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(query, index_text(h.chunk)) for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda p: -float(p[1]))
        return [replace(h, score=float(s), rank=i) for i, (h, s) in enumerate(ranked, 1)]


def retrieve(index, question: str, k: int, mode: str | None, docs: list[str] | None, reranker: Reranker | None = None,
             pool: int | None = None) -> list[Hit]:
    if reranker is None:
        return index.search(question, k, mode, docs)
    return reranker.rerank(question, index.search(question, pool or settings.rerank_pool, mode, docs))[:k]
