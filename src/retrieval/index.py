from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import settings
from src.ingestion.chunks import Chunk
from src.retrieval.embedder import Embedder


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense: float
    bm25: float
    rank: int


def index_text(c: Chunk) -> str:
    return f"{c.section}\n{c.text}"


def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def digest(chunks: list[Chunk], model: str) -> str:
    h = hashlib.sha1(model.encode())
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.text.encode())
    return h.hexdigest()


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.faiss = faiss.IndexFlatIP(vectors.shape[1])
        self.faiss.add(vectors)
        self.bm25 = BM25Okapi([tokenize(index_text(c)) for c in chunks])
        self.docs = np.array([c.doc_id for c in chunks])

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "Index":
        return cls(chunks, embedder.encode_docs([index_text(c) for c in chunks]), embedder)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss, str(path / "dense.faiss"))
        (path / "chunks.jsonl").write_text(
            "\n".join(c.model_dump_json() for c in self.chunks) + "\n", encoding="utf-8"
        )
        meta = {"model": self.embedder.model_name, "digest": digest(self.chunks, self.embedder.model_name)}
        (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> "Index":
        lines = (path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        chunks = [Chunk.model_validate_json(line) for line in lines]
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta["digest"] != digest(chunks, embedder.model_name):
            raise RuntimeError("Index does not match the embedding model or chunks. Rebuild it.")
        dense = faiss.read_index(str(path / "dense.faiss"))
        return cls(chunks, dense.reconstruct_n(0, dense.ntotal), embedder)

    def search(self, query: str, k: int | None = None, mode: str | None = None, docs: list[str] | None = None) -> list[Hit]:
        k, mode = k or settings.top_k, mode or settings.retrieval_mode
        n = len(self.chunks)
        sims, ids = self.faiss.search(self.embedder.encode_query(query), n)
        dense = np.empty(n, dtype="float32")
        dense[ids[0]] = sims[0]
        bm25 = np.asarray(self.bm25.get_scores(tokenize(query)), dtype="float32")

        allowed = np.arange(n) if not docs else np.flatnonzero(np.isin(self.docs, docs))
        if len(allowed) == 0:
            return []

        def ranks(scores: np.ndarray) -> dict[int, int]:
            order = allowed[np.argsort(-scores[allowed], kind="stable")]
            return {int(i): r + 1 for r, i in enumerate(order)}

        r_dense, r_bm25 = ranks(dense), ranks(bm25)
        if mode == "dense":
            final = {i: float(dense[i]) for i in r_dense}
        elif mode == "bm25":
            final = {i: float(bm25[i]) for i in r_bm25}
        elif mode == "hybrid":
            final = {i: 1 / (settings.rrf_k + r_dense[i]) + 1 / (settings.rrf_k + r_bm25[i]) for i in r_dense}
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

        top = sorted(final, key=lambda i: -final[i])[:k]
        return [Hit(self.chunks[i], final[i], float(dense[i]), float(bm25[i]), r + 1) for r, i in enumerate(top)]


def load_all_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(settings.processed_dir.glob("*.jsonl")):
        chunks += [Chunk.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return chunks


def main() -> None:
    chunks = load_all_chunks()
    index = Index.build(chunks, Embedder())
    index.save(settings.index_dir)
    print(f"indexed {len(chunks)} chunks from {len(set(c.doc_id for c in chunks))} documents -> {settings.index_dir}")


if __name__ == "__main__":
    main()
