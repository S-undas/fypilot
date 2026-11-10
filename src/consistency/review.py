from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict

from src.config import settings

COLUMNS = ["qid", "kind", "doc", "where", "text", "human_label", "note"]
OUT = settings.eval_dir / "results"


def load(name: str) -> list[dict]:
    path = OUT / f"{name}_check.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def spread(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Pick n rows, taking from each document in turn."""
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_doc[r["doc"]].append(r)
    for v in by_doc.values():
        rng.shuffle(v)
    out: list[dict] = []
    while len(out) < n and any(by_doc.values()):
        for d in sorted(by_doc):
            if by_doc[d] and len(out) < n:
                out.append(by_doc[d].pop())
    return out


def sample(promises: list[dict], conflicts: list[dict], n_flagged: int = 10, n_shown: int = 5, n_conflict: int = 12,
           n_dropped: int = 4, n_consistent: int = 3, seed: int = 21) -> list[dict]:
    rng = random.Random(seed)
    items = [{"kind": "promise", "group": "flagged" if r["verdict"] != "shown" else "shown", **r} for r in promises if r["verdict"] != "parse_error"]
    cons = [{"kind": "conflict", "group": g, **r} for r in conflicts
            for g in [("confirmed" if r["verdict"] == "confirmed" else "dropped" if r["first"] == "conflict" else "consistent" if r["first"] == "consistent" else "")] if g]
    picked: list[dict] = []
    for kind, group, n in (("promise", "flagged", n_flagged), ("promise", "shown", n_shown), ("conflict", "confirmed", n_conflict),
                           ("conflict", "dropped", n_dropped), ("conflict", "consistent", n_consistent)):
        pool = [r for r in (items if kind == "promise" else cons) if r["group"] == group]
        picked += spread(pool, n, rng)
    rng.shuffle(picked)
    return picked


def describe(r: dict) -> tuple[str, str]:
    if r["kind"] == "promise":
        page = f", p. {r['page']}" if r["page"] else ""
        return f"{r['section']}{page}", f"Promise: {r['promise']} | Quote: \"{r['quote']}\""
    a, b = r["a"], r["b"]
    return f"A: {a['section']}, p. {a['page']} | B: {b['section']}, p. {b['page']}", f"A: \"{a['quote']}\" | B: \"{b['quote']}\""


def make_sheet() -> None:
    picked = sample(load("promises"), load("contradictions"))
    key = {}
    with (OUT / "consistency_review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for n, r in enumerate(picked, 1):
            qid = f"R{n:02d}"
            where, text = describe(r)
            key[qid] = {"kind": r["kind"], "group": r["group"], "doc": r["doc"], "verdict": r["verdict"]}
            w.writerow([qid, r["kind"], r["doc"], where, text, "", ""])
    (OUT / "consistency_review_key.json").write_text(json.dumps(key), encoding="utf-8")
    print(f"wrote {len(picked)} items to {OUT / 'consistency_review.csv'}")


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def score(rows: list[dict], key: dict[str, dict]) -> dict:
    lab = {r["qid"]: r["human_label"].strip().lower() for r in rows if r["human_label"].strip().lower() in ("y", "n", "p") and r["qid"] in key}
    out: dict = {}
    for kind, group_names in (("promise", ("flagged", "shown")), ("conflict", ("confirmed", "dropped", "consistent"))):
        part = {g: [lab[q] for q, k in key.items() if q in lab and k["kind"] == kind and k["group"] == g] for g in group_names}
        out[kind] = {g: {"labeled": len(v), "y": v.count("y"), "n": v.count("n"), "p": v.count("p")} for g, v in part.items()}
    p, c = out["promise"], out["conflict"]
    p["flag_precision"] = rate(p["flagged"]["n"], p["flagged"]["labeled"])
    p["flag_precision_lenient"] = rate(p["flagged"]["n"] + p["flagged"]["p"], p["flagged"]["labeled"])
    p["shown_but_unsupported"] = rate(p["shown"]["n"], p["shown"]["labeled"])
    c["conflict_precision"] = rate(c["confirmed"]["y"], c["confirmed"]["labeled"])
    c["conflict_precision_lenient"] = rate(c["confirmed"]["y"] + c["confirmed"]["p"], c["confirmed"]["labeled"])
    c["real_among_dropped"] = rate(c["dropped"]["y"], c["dropped"]["labeled"])
    c["real_among_consistent"] = rate(c["consistent"]["y"], c["consistent"]["labeled"])
    return out


def load_labeled() -> dict | None:
    path, key_path = OUT / "consistency_review.csv", OUT / "consistency_review_key.json"
    if not (path.exists() and key_path.exists()):
        return None
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not any(r["human_label"].strip() for r in rows):
        return None
    return score(rows, json.loads(key_path.read_text(encoding="utf-8")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["make", "score"])
    a = ap.parse_args()
    if a.action == "make":
        make_sheet()
    else:
        print(json.dumps(load_labeled(), indent=2))


if __name__ == "__main__":
    main()
