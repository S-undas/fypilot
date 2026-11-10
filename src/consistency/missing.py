from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace

from src.config import settings
from src.consistency.requirements import DOC_STAGES, DOC_TYPES, Requirement, applicable
from src.consistency.structure import FRONT, Structure, Variant, apply_variant, load_structure, targets
from src.qa.answer import pages_of
from src.retrieval.index import Hit

VERDICTS = {"present", "partial", "missing"}
SYSTEM = "You check whether a final year project report contains a part required by the course handbook. Reply with one JSON object and nothing else."


def build_messages(req: Requirement, titles: list[str], hits: list[Hit], subsections: list[str]) -> list[dict]:
    blocks = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        page = f", p. {pages_of(h)}" if pages_of(h) else ""
        blocks.append(f"[{i}] ({c.section}{page})\n{c.text[:1200]}")
    user = (
        f"Report contents list: {'; '.join(titles)}\n"
        + (f"Sections inside the matching chapter: {'; '.join(subsections)}\n" if subsections else "")
        + "\n"
        f"Required part: {req.name}, that is {req.what}.\n\n"
        "Passages retrieved from the report:\n\n" + "\n\n".join(blocks) + "\n\n"
        'Return a JSON object with keys "verdict", "evidence" and "reason".\n'
        '"verdict": "present" if the passages show real content of this part, "partial" if only some of it is there or it is thin, '
        '"missing" if the report has no such content. A title in the contents list with no matching content in the passages does not count as present. '
        "Judge by the section each passage comes from, not by passages that only mention the topic.\n"
        '"evidence": the numbers of the passages that show it (empty list if missing).\n'
        '"reason": at most 20 words.'
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if d.get("verdict") not in VERDICTS:
        return None
    ev = [n for n in d.get("evidence") or [] if isinstance(n, int)]
    return {"verdict": d["verdict"], "evidence": ev, "reason": str(d.get("reason", ""))[:200]}


def heading_verdict(req: Requirement, titles: list[str]) -> str:
    return "present" if any(req.matches(t) for t in titles) else "missing"


def anchors(req: Requirement, titles: list[str], structure: Structure, kept: list) -> tuple[list[Hit], list[str]]:
    """Opening passages and section titles of the chapter or front part whose title matches the requirement."""
    hits: list[Hit] = []
    subs: list[str] = []
    for t in (t for t in titles if req.matches(t)):
        if t in structure.chapters:
            part = [c for c in kept if structure.owner.get(c.chunk_id) == structure.chapters[t]]
        elif t in FRONT:
            part = [c for c in kept if c.section == t]
        else:
            continue
        subs += [s for s in dict.fromkeys(c.section for c in part) if s not in subs]
        hits += [Hit(c, 0.0, 0.0, 0.0, 0) for c in part[:2]]
    return hits, subs


def retrieve_in(index, query: str, doc: str, keep: set[str], reranker, first: list[Hit]) -> list[Hit]:
    n = sum(1 for c in index.chunks if c.doc_id == doc)
    hits = [h for h in index.search(query, n, settings.retrieval_mode, [doc]) if h.chunk.chunk_id in keep][: settings.rerank_pool]
    if reranker is not None:
        hits = reranker.rerank(query, hits)
    seen = {h.chunk.chunk_id for h in first}
    merged = (first + [h for h in hits if h.chunk.chunk_id not in seen])[: settings.top_k]
    return [replace(h, rank=i) for i, h in enumerate(merged, 1)]


def check(req: Requirement, doc: str, titles: list[str], structure: Structure, kept: list, index, reranker, llm) -> dict:
    base = {"requirement": req.key, "heading": heading_verdict(req, titles)}
    if req.pdf_only:
        return {**base, "verdict": base["heading"], "evidence": [], "reason": "looked for a References or Bibliography heading in the PDF", "method": "pdf"}
    first, subs = anchors(req, titles, structure, kept)
    hits = retrieve_in(index, req.query, doc, {c.chunk_id for c in kept}, reranker, first)
    parsed = parse(llm.complete(build_messages(req, titles, hits, subs))) if hits else {"verdict": "missing", "evidence": [], "reason": "no passages"}
    parsed = parsed or {"verdict": "parse_error", "evidence": [], "reason": ""}
    sources = [f"{hits[n - 1].chunk.section}, p. {pages_of(hits[n - 1])}" for n in parsed["evidence"] if 1 <= n <= len(hits)]
    return {**base, **parsed, "evidence": sources, "method": "retrieval+llm"}


def load_context():
    from src.qa.llm import LLM
    from src.retrieval.embedder import Embedder
    from src.retrieval.index import Index
    from src.retrieval.rerank import Reranker

    index = Index.load(settings.index_dir, Embedder())
    index.search("warm up", 1)
    return index, Reranker(), LLM(settings.consistency_chain)


def doc_chunks(index, doc: str):
    return [c for c in index.chunks if c.doc_id == doc]


def run_check(docs: list[str]) -> list[dict]:
    index, reranker, llm = load_context()
    rows = []
    for doc in docs:
        chunks = doc_chunks(index, doc)
        structure = load_structure(doc, chunks)
        for req in applicable(DOC_TYPES[doc], DOC_STAGES[doc]):
            rec = {"doc": doc, "type": DOC_TYPES[doc], "variant": "intact", "target": None,
                   **check(req, doc, list(structure.titles), structure, chunks, index, reranker, llm)}
            rows.append(rec)
            print(f"{doc} {req.key:18} heading={rec['heading']:8} llm={rec['verdict']}", flush=True)
    return rows


def run_perturb(docs: list[str]) -> list[dict]:
    index, reranker, llm = load_context()
    rows = []
    for doc in docs:
        chunks = doc_chunks(index, doc)
        structure: Structure = load_structure(doc, chunks)
        reqs = {r.key: r for r in applicable(DOC_TYPES[doc], DOC_STAGES[doc])}
        for key, title in targets(structure, chunks, list(reqs.values())).items():
            for kind in ("remove", "empty"):
                titles, kept = apply_variant(structure, chunks, Variant(kind, title))
                rec = {"doc": doc, "type": DOC_TYPES[doc], "variant": kind, "target": title,
                       **check(reqs[key], doc, titles, structure, kept, index, reranker, llm)}
                rows.append(rec)
                print(f"{doc} {key:18} {kind:6} heading={rec['heading']:8} llm={rec['verdict']}", flush=True)
    return rows


def render_sheet(doc: str, rows: list[dict]) -> str:
    mine = [r for r in rows if r["doc"] == doc]
    gaps = [r for r in mine if r["verdict"] != "present"]
    lines = [f"# Required parts check: {doc} ({DOC_TYPES[doc]})", "", f"{len(mine)} required parts checked. {len(gaps)} are missing or partial.", ""]
    names = {r.key: r.name for r in applicable(DOC_TYPES[doc], DOC_STAGES[doc])}
    for r in mine:
        lines.append(f"- **{names[r['requirement']]}**: {r['verdict']}" + (f" ({r['reason']})" if r["reason"] else "")
                     + (f" Evidence: {'; '.join(r['evidence'])}" if r["evidence"] else ""))
    return "\n".join(lines) + "\n"


def write(name: str, rows: list[dict], docs: list[str], sheets: bool) -> None:
    out = settings.eval_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"missing_{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    if sheets:
        for doc in docs:
            (out / f"missing_sheet_{doc}.md").write_text(render_sheet(doc, rows), encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check", "perturb"], required=True)
    ap.add_argument("--docs", default=",".join(DOC_TYPES))
    a = ap.parse_args()
    docs = a.docs.split(",")
    rows = run_check(docs) if a.mode == "check" else run_perturb(docs)
    write(a.mode, rows, docs, a.mode == "check")
    print(f"\n{len(rows)} checks written")


if __name__ == "__main__":
    main()
