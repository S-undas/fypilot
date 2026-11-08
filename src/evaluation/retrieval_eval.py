from __future__ import annotations

import csv
import sys
from collections import defaultdict

from src.config import settings
from src.evaluation.dataset import EvalItem, load_items, norm
from src.ingestion.chunks import Chunk

KS = (1, 3, 5, 10)


def gold_chunk_ids(item: EvalItem, chunks: list[Chunk]) -> set[str]:
    quote = norm(item.evidence_quote)
    return {
        c.chunk_id
        for c in chunks
        if c.doc_id == item.doc and norm(c.section) == norm(item.gold_section) and quote in norm(c.text)
    }


def first_rank(hits, match) -> int | None:
    for h in hits:
        if match(h.chunk):
            return h.rank
    return None


def per_item_ranks(items: list[EvalItem], chunks: list[Chunk], search) -> list[dict]:
    rows = []
    for it in items:
        if not it.answerable:
            continue
        gold = gold_chunk_ids(it, chunks)
        hits = search(it.question, max(KS), [it.doc])
        rows.append(
            {
                "id": it.id,
                "split": it.split,
                "chunk_rank": first_rank(hits, lambda c: c.chunk_id in gold),
                "section_rank": first_rank(hits, lambda c: norm(c.section) == norm(it.gold_section)),
            }
        )
    return rows


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    out = {"n": n}
    for k in KS:
        out[f"recall@{k}"] = sum(r["chunk_rank"] is not None and r["chunk_rank"] <= k for r in rows) / n
    out["section_hit@5"] = sum(r["section_rank"] is not None and r["section_rank"] <= 5 for r in rows) / n
    out["mrr"] = sum(1 / r["chunk_rank"] for r in rows if r["chunk_rank"]) / n
    return out


def evaluate_arm(items: list[EvalItem], chunks: list[Chunk], search) -> dict[str, dict]:
    rows = per_item_ranks(items, chunks, search)
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
        by_split["all"].append(r)
    return {split: aggregate(rs) for split, rs in by_split.items()}


def main() -> None:
    from src.retrieval.embedder import Embedder
    from src.retrieval.index import Index

    sys.stdout.reconfigure(encoding="utf-8")
    items = load_items(settings.eval_dir / "questions.csv")
    index = Index.load(settings.index_dir, Embedder())
    arms: dict[str, callable] = {
        mode: (lambda q, k, docs, m=mode: index.search(q, k, m, docs)) for mode in ("dense", "bm25", "hybrid")
    }
    if settings.reranker_model:
        from src.retrieval.rerank import Reranker

        rr = Reranker()
        arms["hybrid+rerank"] = lambda q, k, docs: rr.rerank(q, index.search(q, 20, "hybrid", docs))[:k]

    results = {name: evaluate_arm(items, index.chunks, s) for name, s in arms.items()}
    cols = ["n", *[f"recall@{k}" for k in KS], "section_hit@5", "mrr"]
    out = settings.eval_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "retrieval_ablation.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "split", *cols])
        for arm, by_split in results.items():
            for split in ("dev", "test", "all"):
                m = by_split.get(split, {})
                w.writerow([arm, split, *[round(m.get(c, 0), 3) for c in cols]])
    for split in ("dev", "test", "all"):
        print(f"\n[{split}] " + "  ".join(f"{c:>13}" for c in cols))
        for arm, by_split in results.items():
            m = by_split.get(split, {})
            print(f"{arm:14}" + "  ".join(f"{m.get(c, 0):13.3f}" if c != "n" else f"{int(m.get(c, 0)):13d}" for c in cols))


if __name__ == "__main__":
    main()
