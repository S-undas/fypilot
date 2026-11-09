from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter

from src.config import settings
from src.panel.run_panel import NOT_COVERED

COLUMNS = ["qid", "doc", "category", "question", "human_covered", "human_relevance", "note"]


def load(name: str) -> list[dict]:
    path = settings.eval_dir / "results" / f"panel_{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample(doc_records: list[dict], generic_records: list[dict], n_doc: int = 24, n_generic: int = 12, seed: int = 13) -> list[dict]:
    rng = random.Random(seed)
    docs = sorted({r["doc"] for r in doc_records})
    picked = []
    for d in docs:
        rows = [r for r in doc_records if r["doc"] == d]
        cats = sorted({r["category"] for r in rows})
        rng.shuffle(cats)
        for c in cats[: n_doc // len(docs)]:
            picked.append(rng.choice([r for r in rows if r["category"] == c]))
    pool = sorted(generic_records, key=lambda r: r["id"])
    rng.shuffle(pool)
    per_doc, taken = max(1, n_generic // len(docs)), Counter()
    for r in pool:
        if taken[r["doc"]] < per_doc and len([p for p in picked if p["source"] == "generic"]) < n_generic:
            picked.append(r)
            taken[r["doc"]] += 1
    rng.shuffle(picked)
    return picked


def make_sheet() -> None:
    picked = sample(load("doc"), load("generic"))
    out = settings.eval_dir / "results"
    key = {}
    with (out / "panel_check.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for n, r in enumerate(picked, 1):
            qid = f"Q{n:02d}"
            key[qid] = r["id"]
            w.writerow([qid, r["doc"], r["category"], r["question"], "", "", ""])
    (out / "panel_check_key.json").write_text(json.dumps(key), encoding="utf-8")
    print(f"wrote {len(picked)} questions to {out / 'panel_check.csv'}")


def score(rows: list[dict], key: dict[str, str], records: dict[str, dict]) -> dict:
    out: dict = {}
    for label, wanted in (("doc", {"doc"}), ("generic", {"generic"}), ("all", {"doc", "generic"})):
        labeled = [(records[key[r["qid"]]], r) for r in rows if r["human_covered"].strip().lower() in ("y", "n", "p") and r["qid"] in key]
        labeled = [(rec, r) for rec, r in labeled if rec["source"] in wanted]
        sys_nc = [(rec, r) for rec, r in labeled if rec["status"] == NOT_COVERED]
        hum_nc = [(rec, r) for rec, r in labeled if r["human_covered"].strip().lower() == "n"]
        tp = sum(1 for _, r in sys_nc if r["human_covered"].strip().lower() == "n")
        tp_len = sum(1 for _, r in sys_nc if r["human_covered"].strip().lower() in ("n", "p"))
        hum_len = sum(1 for _, r in labeled if r["human_covered"].strip().lower() in ("n", "p"))
        rel = [int(r["human_relevance"]) for _, r in labeled if r["human_relevance"].strip().isdigit()]
        out[label] = {
            "labeled": len(labeled),
            "system_not_covered": len(sys_nc),
            "human_not_covered": len(hum_nc),
            "not_covered_precision": tp / len(sys_nc) if sys_nc else None,
            "not_covered_recall": tp / len(hum_nc) if hum_nc else None,
            "not_covered_precision_lenient": tp_len / len(sys_nc) if sys_nc else None,
            "not_covered_recall_lenient": tp_len / hum_len if hum_len else None,
            "wrongly_not_covered": len(sys_nc) - tp,
            "wrongly_covered": sum(1 for rec, r in labeled if rec["status"] != NOT_COVERED and r["human_covered"].strip().lower() == "n"),
            "human_partial": sum(1 for _, r in labeled if r["human_covered"].strip().lower() == "p"),
            "mean_relevance": sum(rel) / len(rel) if rel else None,
        }
    return out


def load_labeled() -> dict | None:
    path = settings.eval_dir / "results" / "panel_check.csv"
    key_path = settings.eval_dir / "results" / "panel_check_key.json"
    if not (path.exists() and key_path.exists()):
        return None
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not any(r["human_covered"].strip() for r in rows):
        return None
    records = {r["id"]: r for name in ("doc", "generic") for r in load(name)}
    return score(rows, json.loads(key_path.read_text(encoding="utf-8")), records)


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
