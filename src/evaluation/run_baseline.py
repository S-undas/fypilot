from __future__ import annotations

import argparse
import json
import sys
import time

from src.config import settings
from src.evaluation.dataset import load_items, norm
from src.evaluation.judge import judge
from src.evaluation.metrics import summarize_by_split
from src.evaluation.retrieval_eval import gold_chunk_ids
from src.qa.answer import answer
from src.qa.llm import LLM


def parse_chain(spec: str) -> list[tuple[str, str]]:
    return [tuple(part.split(":", 1)) for part in spec.split(",")]


def run(name: str, chain, splits: list[str], mode: str, k: int, limit: int | None, prompt: str = "grounded", rerank: bool = False) -> list[dict]:
    from src.retrieval.embedder import Embedder
    from src.retrieval.index import Index

    items = [i for i in load_items(settings.eval_dir / "questions.csv") if i.split in splits and i.verified in ("y", "fixed")]
    if limit:
        items = items[:limit]
    index = Index.load(settings.index_dir, Embedder())
    index.search("warm up", 1)
    reranker = None
    if rerank:
        from src.retrieval.rerank import Reranker

        reranker = Reranker()
    gen, grader = LLM(chain), LLM([settings.judge_model])
    out_dir = settings.eval_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for n, it in enumerate(items, 1):
        t0 = time.time()
        ans = answer(it.question, index, gen, [it.doc], k, mode, prompt=prompt, reranker=reranker)
        latency = time.time() - t0
        gold = gold_chunk_ids(it, index.chunks) if it.answerable else set()
        cited = [(c.n, ans.hits[c.n - 1].chunk.text) for c in ans.citations]
        retrieved = [(i, h.chunk.text) for i, h in enumerate(ans.hits, 1)]
        verdict = judge(grader, it.question, it.gold_answer, ans.text, cited, retrieved)
        records.append(
            {
                "id": it.id, "doc": it.doc, "kind": it.kind, "split": it.split, "question": it.question,
                "answer": ans.text, "served_by": "/".join(gen.last_ref or ()), "latency_s": round(latency, 2),
                "n_valid_citations": len(ans.citations), "invalid_refs": ans.invalid_refs,
                "retrieved": [h.chunk.chunk_id for h in ans.hits], "top_dense": round(ans.hits[0].dense, 4) if ans.hits else None,
                "gold_in_top5": any(h.chunk.chunk_id in gold for h in ans.hits), **verdict,
            }
        )
        print(f"[{n}/{len(items)}] {it.id} {latency:.1f}s abstained={verdict['abstained']} correctness={verdict['correctness']}", flush=True)
    with (out_dir / f"judged_{name}.jsonl").open("w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    return records


def print_summary(name: str, records: list[dict]) -> None:
    by_split = summarize_by_split(records)
    keys = list(by_split["all"])
    print(f"\n=== {name} ===")
    print(f"{'metric':26}" + "".join(f"{s:>10}" for s in ("dev", "test", "all") if s in by_split))
    for key in keys:
        cells = []
        for s in ("dev", "test", "all"):
            if s in by_split:
                v = by_split[s].get(key)
                cells.append(f"{'-':>10}" if v is None else (f"{v:10.3f}" if isinstance(v, float) else f"{v:10d}"))
        print(f"{key:26}" + "".join(cells))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--chain", default=",".join(f"{p}:{m}" for p, m in settings.llm_chain))
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--mode", default=settings.retrieval_mode)
    ap.add_argument("-k", type=int, default=settings.top_k)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--prompt", default="grounded", choices=["grounded", "bare"])
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()
    records = run(args.name, parse_chain(args.chain), args.splits.split(","), args.mode, args.k, args.limit, args.prompt, args.rerank)
    print_summary(args.name, records)


if __name__ == "__main__":
    main()
