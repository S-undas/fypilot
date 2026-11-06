from __future__ import annotations

import os

import numpy as np

from src.config import settings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([QUERY_PREFIX + query])
