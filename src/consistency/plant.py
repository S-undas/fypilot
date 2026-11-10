from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys

from src.config import settings
from src.consistency import contradictions, promises
from src.consistency.missing import doc_chunks, load_context
from src.consistency.promises import words
from src.consistency.review import spread
from src.consistency.structure import chunks_in, load_structure
from src.panel.generate import similar

OUT = settings.eval_dir / "results"
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?(?![\w])")
PLANT_SYSTEM = "You edit final year project reports for a test. Reply with one sentence and nothing else."
COLUMNS = ["pid", "kind", "doc", "section", "page", "change", "keep", "note"]


def usable_number(tok: str) -> bool:
    v = float(tok.rstrip("%"))
    return not (1900 <= v <= 2100) and ("%" in tok or "." in tok or v >= 10)


def change_number(sentence: str, tok: str) -> str:
    v = float(tok.rstrip("%"))
    if tok.endswith("%"):
        new = f"{int(v * 2) if v <= 50 else int(v - 20)}%"
    elif "." in tok:
        new = f"{v * 1.5:.{len(tok.split('.')[1])}f}"
    else:
        new = str(int(v * 2))
    return sentence.replace(tok, new, 1)


def shared_numbers(a: str, b: str) -> list[str]:
    return [t for t in NUM.findall(a) if t in NUM.findall(b) and usable_number(t)]


def pick_contradictions(facts_by_doc: dict[str, list[dict]], embedder, text: dict[str, str], n: int, rng: random.Random,
                        min_sim: float = 0.75) -> list[dict]:
    """Two places that state the same number about the same thing; the number is changed in one place, so a conflict exists by construction."""
    pool = []
    for doc, facts in facts_by_doc.items():
        if len(facts) < 2:
            continue
        vecs = embedder.encode_docs([f["fact"] for f in facts])
        sims = vecs @ vecs.T
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                a, b = facts[i], facts[j]
                nums = shared_numbers(a["quote"], b["quote"])
                if (sims[i, j] >= min_sim and nums and a["chunk_id"] != b["chunk_id"] and b["quote"] in text.get(b["chunk_id"], "")
                        and not similar(a["quote"], b["quote"], 0.8)):
                    pool.append({"doc": doc, "chunk_id": b["chunk_id"], "section": b["section"], "page": b["page"], "find": b["quote"],
                                 "replace": change_number(b["quote"], nums[0]), "counterpart": f"{a['section']}, p. {a['page']}: {a['quote']}"})
    out, used = [], set()
    for p in spread(pool, len(pool), rng):
        if p["chunk_id"] in used or len(out) >= n:
            continue
        used.add(p["chunk_id"])
        out.append({"kind": "contradiction", **p})
    return out


def plant_messages(section: str, text: str) -> list[dict]:
    user = (
        f"Section: {section}\n\n{text[:1200]}\n\n"
        "Write one sentence that could be added to this section of the report. It must promise a new, specific, measurable capability or result "
        "(with a target number) that this text does not mention and that is plausible for this project. Write it in the report's own style, in the future tense."
    )
    return [{"role": "system", "content": PLANT_SYSTEM}, {"role": "user", "content": user}]


def pick_promises(index, llm, n: int, rng: random.Random, taken: set[str]) -> list[dict]:
    pool = []
    for doc in sorted(promises_docs()):
        chunks = doc_chunks(index, doc)
        s = load_structure(doc, chunks)
        pool += [{"doc": doc, "chunk": c} for c in chunks_in(s, chunks, promises.SOURCE) if c.chunk_id not in taken and len(c.text) > 300]
    out = []
    for item in spread([{"doc": p["doc"], "chunk": p["chunk"]} for p in pool], len(pool), rng):
        if len(out) >= n:
            break
        c = item["chunk"]
        sentence = llm.complete(plant_messages(c.section, c.text)).strip().split("\n")[0].strip()
        if 5 <= len(sentence.split()) <= 45:
            out.append({"kind": "promise", "doc": c.doc_id, "chunk_id": c.chunk_id, "section": c.section, "page": c.page_start or "", "inserted": sentence})
    return out


def promises_docs() -> list[str]:
    from src.consistency.requirements import DOC_TYPES

    return list(DOC_TYPES)


def edits_from(plants: list[dict], text: dict[str, str]) -> dict[str, str]:
    edits: dict[str, str] = {}
    for p in plants:
        cur = edits.get(p["chunk_id"], text[p["chunk_id"]])
        edits[p["chunk_id"]] = cur.replace(p["find"], p["replace"], 1) if p["kind"] == "contradiction" else cur.rstrip() + " " + p["inserted"]
    return edits


def covers(inserted: str, quote: str) -> bool:
    q = words(quote)
    return bool(q) and sum(w in set(words(inserted)) for w in q) / len(q) >= 0.6


def score(plants: list[dict], promise_rows: list[dict], conflict_rows: list[dict]) -> dict:
    out: dict = {"promise": {"planted": 0, "extracted": 0, "flagged": 0}, "contradiction": {"planted": 0, "found": 0}}
    for p in plants:
        if p["kind"] == "promise":
            mine = [r for r in promise_rows if r["doc"] == p["doc"] and r["section"] == p["section"] and covers(p["inserted"], r["quote"])]
            out["promise"]["planted"] += 1
            out["promise"]["extracted"] += bool(mine)
            out["promise"]["flagged"] += any(r["verdict"] in ("not_shown", "contradicted") for r in mine)
        else:
            new = set(NUM.findall(p["replace"])) - set(NUM.findall(p["find"]))
            hit = [r for r in conflict_rows if r["doc"] == p["doc"] and r["verdict"] == "confirmed"
                   and any(s["chunk_id"] == p["chunk_id"] and new & set(NUM.findall(s["quote"])) for s in (r["a"], r["b"]))]
            out["contradiction"]["planted"] += 1
            out["contradiction"]["found"] += bool(hit)
    return out


def make() -> None:
    index, _, llm = load_context()
    text = {c.chunk_id: c.text for c in index.chunks}
    rng = random.Random(7)
    facts = {doc: contradictions.extract_facts(llm, doc_chunks(index, doc)) for doc in promises_docs()}
    conflicts = pick_contradictions(facts, index.embedder, text, 14, rng)
    taken = {p["chunk_id"] for p in conflicts}
    plants = conflicts + pick_promises(index, llm, 14, rng, taken)
    for n, p in enumerate(plants, 1):
        p["pid"] = f"X{n:02d}"
    (OUT / "consistency_plants.json").write_text(json.dumps(plants, ensure_ascii=False, indent=1), encoding="utf-8")
    with (OUT / "consistency_plants.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for p in plants:
            change = (f"Changed \"{p['find']}\" to \"{p['replace']}\". Agrees elsewhere with: {p['counterpart']}" if p["kind"] == "contradiction"
                      else f"Added at the end of the section: \"{p['inserted']}\"")
            w.writerow([p["pid"], p["kind"], p["doc"], p["section"], p["page"], change, "", ""])
    print(f"wrote {len(plants)} planted changes ({len(conflicts)} contradictions) to {OUT / 'consistency_plants.csv'}")


def kept_plants() -> list[dict]:
    plants = json.loads((OUT / "consistency_plants.json").read_text(encoding="utf-8"))
    with (OUT / "consistency_plants.csv").open(encoding="utf-8-sig", newline="") as f:
        drop = {r["pid"] for r in csv.DictReader(f) if r["keep"].strip().lower() == "n"}
    return [p for p in plants if p["pid"] not in drop]


def evaluate(part: str = "both") -> None:
    plants = kept_plants()
    index, _, _ = load_context()
    text = {c.chunk_id: c.text for c in index.chunks}
    edits = edits_from(plants, text)
    docs = sorted({p["doc"] for p in plants})
    p_rows = promises.run(docs, edits) if part in ("both", "promises") else []
    c_rows = contradictions.run(docs, edits) if part in ("both", "contradictions") else []
    for name, rows in (("promises", p_rows), ("contradictions", c_rows)):
        if rows:
            lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
            (OUT / f"consistency_planted_{name}.jsonl").write_text(lines, encoding="utf-8")
    res = score(plants, p_rows, c_rows)
    (OUT / f"consistency_planted_score_{part}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["make", "evaluate"])
    ap.add_argument("--part", choices=["both", "promises", "contradictions"], default="both")
    a = ap.parse_args()
    make() if a.action == "make" else evaluate(a.part)


if __name__ == "__main__":
    main()
