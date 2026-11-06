from __future__ import annotations

import argparse
import sys

from src.config import settings
from src.retrieval.embedder import Embedder
from src.retrieval.index import Index


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--doc", action="append", help="restrict to a doc id, repeatable")
    ap.add_argument("-k", type=int, default=settings.top_k)
    ap.add_argument("--mode", default=settings.retrieval_mode, choices=["dense", "bm25", "hybrid"])
    args = ap.parse_args()

    index = Index.load(settings.index_dir, Embedder())
    for h in index.search(args.query, args.k, args.mode, args.doc):
        c = h.chunk
        pages = c.page_start if c.page_start == c.page_end else f"{c.page_start}-{c.page_end}"
        print(f"#{h.rank} {c.doc_id} | {c.section} | p.{pages} | dense={h.dense:.3f} bm25={h.bm25:.1f}")
        print("   " + c.text[:220].replace("\n", " "))


if __name__ == "__main__":
    main()
