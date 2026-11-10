from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

import numpy as np

from src.config import settings
from src.consistency.missing import doc_chunks, load_context
from src.consistency.promises import words
from src.consistency.requirements import DOC_TYPES
from src.consistency.structure import apply_edits
from src.panel.generate import STOP, similar

MAX_FACTS_PER_CHUNK = 6
PAIR_THRESHOLD = 0.6
PAIRS_PER_FACT = 4
MAX_PAIRS_PER_DOC = 400
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?(?![\w])")
NAME = re.compile(r"\b(?:[A-Z][A-Za-z]*\d[\w.-]*|[A-Z]{2,}\w*|[A-Z][a-z]+[A-Z]\w*)")
FIRST = {"conflict", "consistent", "unrelated"}
FACT_SYSTEM = "You read one part of a final year project report and list its specific, checkable facts. Reply with a JSON list and nothing else."
PAIR_SYSTEM = "You compare two statements from the same final year project report. Reply with one JSON object and nothing else."


def fact_messages(section: str, text: str) -> list[dict]:
    user = (
        f"Section: {section}\n\n{text}\n\n"
        "List the specific facts in this text that another part of the report could contradict: numbers and what they measure "
        "(thresholds, counts, sizes, percentages, durations, versions), and the names of models, algorithms, datasets, tools, "
        "hardware and methods the project uses or plans to use. Skip general statements and opinions.\n"
        f'Return a JSON list of at most {MAX_FACTS_PER_CHUNK} objects with keys "fact" (one self-contained sentence naming the item it is about), '
        '"quote" (the exact words from the text, at most 30 words). Return [] if there are none.'
    )
    return [{"role": "system", "content": FACT_SYSTEM}, {"role": "user", "content": user}]


def pair_messages(a: dict, b: dict) -> list[dict]:
    user = (
        f"Statement A (section {a['section']}): {a['quote']}\n"
        f"Statement B (section {b['section']}): {b['quote']}\n\n"
        'Return a JSON object with keys "verdict" and "reason".\n'
        '"verdict": "conflict" if the two statements give different values, names or methods for the same item, so both cannot be true of the finished project. '
        '"consistent" if they agree or add detail to each other. "unrelated" if they are about different things. '
        "A change that the text itself explains, such as before and after tuning, a plan versus the final choice, or different models being compared, is not a conflict.\n"
        '"reason": at most 25 words.'
    )
    return [{"role": "system", "content": PAIR_SYSTEM}, {"role": "user", "content": user}]


def confirm_messages(a: dict, b: dict, text_a: str, text_b: str) -> list[dict]:
    user = (
        f"Statement A (section {a['section']}): {a['quote']}\nStatement B (section {b['section']}): {b['quote']}\n\n"
        f"Full passage of A:\n{text_a[:1500]}\n\nFull passage of B:\n{text_b[:1500]}\n\n"
        "A first check found that A and B conflict. Read the full passages. "
        'Return a JSON object with keys "verdict" and "reason". '
        '"verdict": "confirmed" if they still conflict about the same item, "not_confirmed" if the passages explain the difference, '
        "the statements are about different items, or they are compatible. "
        '"reason": at most 25 words.'
    )
    return [{"role": "system", "content": PAIR_SYSTEM}, {"role": "user", "content": user}]


def load_json(text: str, pattern: str):
    m = re.search(pattern, text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def parse_facts(text: str, source: str) -> list[dict]:
    items = load_json(text, r"\[.*\]")
    have = set(words(source))
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not isinstance(it.get("fact"), str) or not isinstance(it.get("quote"), str):
            continue
        q = words(it["quote"])
        if q and sum(w in have for w in q) / len(q) >= 0.9:
            out.append({"fact": it["fact"].strip(), "quote": it["quote"].strip()})
    return out[:MAX_FACTS_PER_CHUNK]


def parse_verdict(text: str, allowed: set[str]) -> dict | None:
    d = load_json(text, r"\{.*\}")
    if not isinstance(d, dict) or d.get("verdict") not in allowed:
        return None
    return {"verdict": d["verdict"], "reason": str(d.get("reason", ""))[:200]}


def extract_facts(llm, chunks) -> list[dict]:
    facts: list[dict] = []
    for c in chunks:
        for f in parse_facts(llm.complete(fact_messages(c.section, c.text), max_tokens=900), c.text):
            facts.append({**f, "chunk_id": c.chunk_id, "section": c.section, "page": c.page_start or ""})
    return facts


def signature(fact: dict) -> frozenset[str]:
    """Numbers and names (YOLOv8n, ResNet, GPU) a statement commits to; a conflict needs different ones on both sides."""
    q = fact["quote"]
    return frozenset(NUMBER.findall(q)) | frozenset(NAME.findall(q))


def content_words(quote: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]{4,}", quote.lower()) if w not in STOP}


def numeric_overlap(a: dict, b: dict) -> bool:
    """Both statements carry numbers and share at least two content words: the likeliest kind of number conflict."""
    return bool(NUMBER.search(a["quote"]) and NUMBER.search(b["quote"])) and len(content_words(a["quote"]) & content_words(b["quote"])) >= 2


def candidate_pairs(facts: list[dict], embedder, threshold: float = PAIR_THRESHOLD, per_fact: int = PAIRS_PER_FACT,
                    limit: int = MAX_PAIRS_PER_DOC) -> list[tuple[int, int, float]]:
    if len(facts) < 2:
        return []
    vecs = embedder.encode_docs([f["fact"] for f in facts])
    sims = vecs @ vecs.T
    seen: dict[tuple[int, int], float] = {}
    for i in range(len(facts)):
        taken = 0
        for j in np.argsort(-sims[i]):
            j = int(j)
            if sims[i, j] < threshold or taken >= per_fact:
                break
            if j == i or facts[i]["chunk_id"] == facts[j]["chunk_id"] or similar(facts[i]["quote"], facts[j]["quote"], 0.8):
                continue
            if not signature(facts[i]) or not signature(facts[j]) or signature(facts[i]) == signature(facts[j]):
                continue
            seen[(min(i, j), max(i, j))] = float(sims[i, j])
            taken += 1
    ranked = sorted(((i, j, s) for (i, j), s in seen.items()), key=lambda p: -p[2])
    return ranked[:limit] + [(i, j, s) for i, j, s in ranked[limit:] if numeric_overlap(facts[i], facts[j])]


def judge_pair(a: dict, b: dict, text_a: str, text_b: str, llm) -> dict:
    first = parse_verdict(llm.complete(pair_messages(a, b), max_tokens=200), FIRST) or {"verdict": "parse_error", "reason": ""}
    out = {"first": first["verdict"], "first_reason": first["reason"], "verdict": first["verdict"], "reason": first["reason"]}
    if first["verdict"] == "conflict":
        second = parse_verdict(llm.complete(confirm_messages(a, b, text_a, text_b), max_tokens=200), {"confirmed", "not_confirmed"}) or {"verdict": "parse_error", "reason": ""}
        out.update(verdict=second["verdict"], reason=second["reason"])
    return out


def contradictions_for(doc: str, doc_type: str, chunks: list, embedder, llm, progress=None) -> list[dict]:
    text = {c.chunk_id: c.text for c in chunks}
    facts = extract_facts(llm, chunks)
    pairs = candidate_pairs(facts, embedder)
    rows = []
    for n, (i, j, sim) in enumerate(pairs, 1):
        a, b = facts[i], facts[j]
        res = judge_pair(a, b, text[a["chunk_id"]], text[b["chunk_id"]], llm)
        rows.append({"doc": doc, "type": doc_type, "similarity": round(sim, 3), "a": a, "b": b, **res, "served_by": "/".join(llm.last_ref or ())})
        if progress:
            progress(n, len(pairs), f"{a['section']} vs {b['section']}")
    return rows


def run(docs: list[str], edits: dict[str, str] | None = None) -> list[dict]:
    index, _, llm = load_context()
    rows = []
    for doc in docs:
        got = contradictions_for(doc, DOC_TYPES[doc], apply_edits(doc_chunks(index, doc), edits), index.embedder, llm)
        rows += got
        print(f"{doc}: {len(got)} statement pairs compared", flush=True)
        for r in got:
            if r["verdict"] == "confirmed":
                print(f"  CONFLICT {r['a']['section']} vs {r['b']['section']}: {r['reason']}", flush=True)
    return rows


def render_sheet(doc: str, rows: list[dict]) -> str:
    mine = [r for r in rows if r["doc"] == doc]
    found = [r for r in mine if r["verdict"] == "confirmed"]
    lines = [f"# Contradictions check: {doc}", "", f"{len(mine)} statement pairs compared. {len(found)} conflicts confirmed.", ""]
    for r in found:
        a, b = r["a"], r["b"]
        lines += [f"- **{a['section']}, p. {a['page']}** vs **{b['section']}, p. {b['page']}**: {r['reason']}",
                  f"  - A: \"{a['quote']}\"", f"  - B: \"{b['quote']}\""]
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
    (out / "contradictions_check.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    for doc in docs:
        (out / f"contradictions_sheet_{doc}.md").write_text(render_sheet(doc, rows), encoding="utf-8")
    print("\n", dict(Counter(r["first"] for r in rows)), "confirmed:", sum(r["verdict"] == "confirmed" for r in rows))


if __name__ == "__main__":
    main()
