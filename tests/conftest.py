import re
import zlib

import numpy as np
import pytest

from src.ingestion.chunks import Chunk
from src.retrieval.index import Index


class FakeEmbedder:
    model_name = "fake"

    def _vec(self, text):
        v = np.zeros(64, dtype="float32")
        for t in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(t.encode()) % 64] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, q):
        return self._vec(q)[None, :]


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def index(embedder):
    rows = [
        ("A", "1 Methods", "yolo object detection on camera frames"),
        ("A", "2 Agents", "reinforcement learning agents for red and blue teams"),
        ("B", "1 Costs", "budget and cost estimate for hardware"),
        ("B", "2 Benchmark", "object detection benchmark results on the dataset"),
    ]
    chunks = [
        Chunk(chunk_id=f"{d}:{i}", doc_id=d, doc_type="reports", section=s, page_start=str(i + 1), page_end=str(i + 1), text=t)
        for i, (d, s, t) in enumerate(rows)
    ]
    return Index.build(chunks, embedder)
