from __future__ import annotations

import argparse
import csv
import json
import re
import sys

from src.abstention.claims import REFUSAL, split_claims
from src.config import settings
from src.panel.categories import CATEGORIES, GENERIC_QUESTIONS
from src.panel.generate import generate_questions
from src.panel.judge import judge
from src.qa.answer import answer
from src.qa.llm import LLM

DOCS = ["F25_044", "F25_051", "F25_181", "F25_182"]
NOT_COVERED, COVERED = "NOT COVERED", "COVERED"
CSV_COLUMNS = ["id", "doc", "source", "category", "question", "status", "answer", "sources", "specificity", "top_score"]


def panel_status(text: str) -> str:
    t = text.strip()
    if re.match(r"^\W*not covered", t, re.I):
        return NOT_COVERED
    if REFUSAL.search(t) and not split_claims(t):
        return NOT_COVERED
    return COVERED


def render_sheet(doc: str, records: list[dict]) -> str:
    rows = [r for r in records if r["doc"] == doc]
    weak = [r for r in rows if r["status"] == NOT_COVERED]
    lines = [f"# Defense prep sheet: {doc}", "", f"{len(rows)} likely panel questions. {len(weak)} are not covered by the documents.", ""]
    if weak:
        lines += ["## Weak spots (not covered)", ""] + [f"- ({r['category']}) {r['question']}" for r in weak] + [""]
    for category in CATEGORIES:
        group = [r for r in rows if r["category"] == category]
        if not group:
            continue
        lines += [f"## {category.capitalize()}", ""]
        for r in group:
            lines += [f"**Q:** {r['question']}", "", f"**Status:** {r['status']}", ""]
            if r["status"] == COVERED:
                lines += [f"**Draft answer:** {r['answer']}", "", "**Sources:** " + ("; ".join(r["sources"]) or "none"), ""]
    return "\n".join(lines)


def questions_for(mode: str, index, llm, reranker, doc: str) -> list[tuple[str, str]]:
    if mode == "generic":
        return [(c, q) for c, qs in GENERIC_QUESTIONS.items() for q in qs]
    return generate_questions(index, llm, reranker, doc)


def run(name: str, docs: list[str], mode: str) -> list[dict]:
    from src.retrieval.embedder import Embedder
    from src.retrieval.index import Index
    from src.retrieval.rerank import Reranker

    index = Index.load(settings.index_dir, Embedder())
    index.search("warm up", 1)
    reranker = Reranker()
    gen, grader = LLM(settings.llm_chain), LLM([settings.judge_model])
    out_dir = settings.eval_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for doc in docs:
        for n, (category, question) in enumerate(questions_for(mode, index, gen, reranker, doc), 1):
            ans = answer(question, index, gen, [doc], settings.top_k, settings.retrieval_mode, prompt="panel", reranker=reranker)
            status = panel_status(ans.text)
            rec = {
                "id": f"{doc}-{mode}-{n:02d}", "doc": doc, "source": mode, "category": category, "question": question,
                "status": status, "answer": ans.text, "sources": [str(c) for c in ans.citations],
                "specificity": round(index.search(question, 1, "dense", [doc])[0].dense, 4),
                "top_score": round(ans.hits[0].score, 4) if ans.hits else None,
                "served_by": "/".join(gen.last_ref or ()),
            }
            if status == COVERED:
                cited = [(c.n, ans.hits[c.n - 1].chunk.text) for c in ans.citations] or [(i, h.chunk.text) for i, h in enumerate(ans.hits[:3], 1)]
                try:
                    rec.update({f"j_{k}": v for k, v in judge(grader, question, ans.text, cited).items()})
                except Exception as exc:
                    print(f"judge failed for {rec['id']}: {type(exc).__name__}", flush=True)
                    rec.update(j_answers_question="judge_error", j_supported="judge_error")
            records.append(rec)
            print(f"{rec['id']} {category:11} {status}", flush=True)
        (out_dir / f"panel_sheet_{doc}_{mode}.md").write_text(render_sheet(doc, records), encoding="utf-8")
    (out_dir / f"panel_{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    with (out_dir / f"panel_{name}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({**r, "sources": "; ".join(r["sources"])})
    return records


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["doc", "generic"], required=True)
    ap.add_argument("--docs", default=",".join(DOCS))
    ap.add_argument("--name")
    a = ap.parse_args()
    records = run(a.name or a.mode, a.docs.split(","), a.mode)
    nc = sum(r["status"] == NOT_COVERED for r in records)
    print(f"\n{len(records)} questions, {nc} not covered ({nc / max(len(records), 1):.0%})")


if __name__ == "__main__":
    main()
