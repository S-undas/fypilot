from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

from src.config import settings
from src.consistency.missing import doc_chunks, load_context, retrieve_in
from src.consistency.requirements import DOC_TYPES
from src.consistency.structure import apply_edits, chunks_in, load_structure
from src.panel.generate import similar
from src.qa.answer import pages_of
from src.retrieval.index import Hit

SOURCE = r"introduction|vision|approach|methodology"
EVIDENCE = r"implementation|test case|experiment|results|user manual|conclusion|work completed"
VERDICTS = {"shown", "not_shown", "contradicted"}
MAX_PER_CHUNK = 3
EXTRACT_SYSTEM = "You read one part of a final year project report and list the promises it makes. Reply with a JSON list and nothing else."
JUDGE_SYSTEM = "You check whether a final year project report backs up a promise it made. Reply with one JSON object and nothing else."


def extract_messages(section: str, text: str) -> list[dict]:
    user = (
        f"Section: {section}\n\n{text}\n\n"
        "List the promises or claims in this text that the rest of the report should back up with evidence: "
        "things the project says it will build, achieve, measure or evaluate, and target numbers such as accuracy or speed. "
        "Skip background, definitions, motivation and general statements.\n"
        f'Return a JSON list of at most {MAX_PER_CHUNK} objects with keys "promise" (one sentence, at most 25 words) and '
        '"quote" (the exact words from the text, at most 30 words). Return [] if there are none.'
    )
    return [{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": user}]


def judge_messages(promise: str, section: str, hits: list[Hit]) -> list[dict]:
    blocks = []
    for i, h in enumerate(hits, 1):
        page = f", p. {pages_of(h)}" if pages_of(h) else ""
        blocks.append(f"[{i}] ({h.chunk.section}{page})\n{h.chunk.text[:1200]}")
    user = (
        f"Promise made in section {section}: {promise}\n\n"
        "Passages from the implementation, testing, results and conclusion parts of the same report:\n\n" + "\n\n".join(blocks) + "\n\n"
        'Return a JSON object with keys "verdict", "evidence" and "reason".\n'
        '"verdict": "shown" if the passages give concrete evidence that the promise was carried out: an implemented feature described, '
        "or test or experiment results reported, with numbers if the promise has numbers. "
        '"not_shown" if the passages only repeat the plan, call it future work, do not mention it, or give no results for a promised measurement. '
        '"contradicted" if the passages say something that conflicts with the promise, such as a different number or method, or that it was not done.\n'
        '"evidence": the numbers of the passages that settle it (empty list if none).\n'
        '"reason": at most 25 words.'
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", s.lower())


def parse_promises(text: str, source: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    have = set(words(source))
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not isinstance(it.get("promise"), str) or not isinstance(it.get("quote"), str):
            continue
        q = words(it["quote"])
        if q and sum(w in have for w in q) / len(q) >= 0.9:
            out.append({"promise": it["promise"].strip(), "quote": it["quote"].strip()})
    return out[:MAX_PER_CHUNK]


def parse_verdict(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if d.get("verdict") not in VERDICTS:
        return None
    return {"verdict": d["verdict"], "evidence": [n for n in d.get("evidence") or [] if isinstance(n, int)], "reason": str(d.get("reason", ""))[:200]}


def extract(llm, chunks) -> list[dict]:
    found: list[dict] = []
    for c in chunks:
        for p in parse_promises(llm.complete(extract_messages(c.section, c.text), max_tokens=600), c.text):
            if not any(similar(p["promise"], f["promise"]) for f in found):
                found.append({**p, "section": c.section, "page": c.page_start or ""})
    return found


def judge_promise(p: dict, doc: str, evidence_ids: set[str], index, reranker, llm) -> dict:
    hits = retrieve_in(index, p["promise"], doc, evidence_ids, reranker, [])
    parsed = parse_verdict(llm.complete(judge_messages(p["promise"], p["section"], hits), max_tokens=200)) if hits else None
    parsed = parsed or {"verdict": "parse_error", "evidence": [], "reason": ""}
    sources = [f"{hits[n - 1].chunk.section}, p. {pages_of(hits[n - 1])}" for n in parsed["evidence"] if 1 <= n <= len(hits)]
    return {**p, **parsed, "evidence": sources}


def run(docs: list[str], edits: dict[str, str] | None = None) -> list[dict]:
    index, reranker, llm = load_context()
    rows = []
    for doc in docs:
        chunks = doc_chunks(index, doc)
        structure = load_structure(doc, chunks)
        promises = extract(llm, chunks_in(structure, apply_edits(chunks, edits), SOURCE))
        evidence_ids = {c.chunk_id for c in chunks_in(structure, chunks, EVIDENCE)}
        print(f"{doc}: {len(promises)} promises", flush=True)
        for p in promises:
            rows.append({"doc": doc, "type": DOC_TYPES[doc], **judge_promise(p, doc, evidence_ids, index, reranker, llm)})
            print(f"  {rows[-1]['verdict']:12} {p['promise'][:90]}", flush=True)
    return rows


def render_sheet(doc: str, rows: list[dict]) -> str:
    mine = [r for r in rows if r["doc"] == doc]
    flagged = [r for r in mine if r["verdict"] != "shown"]
    lines = [f"# Promises check: {doc}", "", f"{len(mine)} promises found. {len(flagged)} are not shown or contradicted.", ""]
    for r in sorted(mine, key=lambda r: r["verdict"] == "shown"):
        page = f", p. {r['page']}" if r["page"] else ""
        lines += [f"- **{r['verdict']}**: {r['promise']} ({r['section']}{page})", f"  - Quote: \"{r['quote']}\"", f"  - {r['reason']}"]
        if r["evidence"]:
            lines.append("  - Evidence: " + "; ".join(r["evidence"]))
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=",".join(DOC_TYPES))
    a = ap.parse_args()
    docs = a.docs.split(",")
    rows = run(docs)
    out = settings.eval_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "promises_check.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    for doc in docs:
        (out / f"promises_sheet_{doc}.md").write_text(render_sheet(doc, rows), encoding="utf-8")
    print("\n", dict(Counter(r["verdict"] for r in rows)))


if __name__ == "__main__":
    main()
