from __future__ import annotations

import argparse
import sys

from src.config import settings
from src.qa.answer import answer
from src.qa.llm import LLM
from src.retrieval.embedder import Embedder
from src.retrieval.index import Index
from src.retrieval.rerank import Reranker


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--doc", action="append")
    ap.add_argument("-k", type=int, default=settings.top_k)
    ap.add_argument("--mode", default=settings.retrieval_mode, choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--allow-refusal", action="store_true")
    args = ap.parse_args()

    index = Index.load(settings.index_dir, Embedder())
    reranker = Reranker() if settings.reranker_model else None
    ans = answer(args.question, index, LLM(), args.doc, args.k, args.mode, args.allow_refusal, reranker=reranker)
    print(ans.text, "\n")
    for c in ans.citations:
        print(c)
    if ans.invalid_refs:
        print("invalid references:", ans.invalid_refs)


if __name__ == "__main__":
    main()
