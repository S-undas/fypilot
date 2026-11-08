from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter

from src.config import settings

COLUMNS = ["id", "kind_hidden", "question", "reference_answer", "system_answer", "human_abstained", "human_correctness", "note"]


def load_records(name: str) -> list[dict]:
    path = settings.eval_dir / "results" / f"judged_{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


REFUSAL = re.compile(
    r"(do(es)? not|don't|doesn't|cannot|can't|no (information|mention|specific))[^.]{0,60}"
    r"(state|report|specify|provide|mention|contain|include|find|give|record)",
    re.I,
)


def needs_review(r: dict) -> bool:
    """Judge and a simple text rule disagree on abstention, or an answerable question was not judged correct."""
    rule = bool(REFUSAL.search(r["answer"][:300]))
    return rule != bool(r["abstained"]) or (r["kind"] == "answerable" and r["correctness"] != "correct")


def sample(records: list[dict], n_answerable: int, n_unanswerable: int, seed: int = 13) -> list[dict]:
    rng = random.Random(seed)
    must = [r for r in records if needs_review(r)]
    rest = [r for r in records if not needs_review(r)]
    ans = sorted((r for r in rest if r["kind"] == "answerable"), key=lambda r: r["id"])
    unans = sorted((r for r in rest if r["kind"] != "answerable"), key=lambda r: r["id"])
    rng.shuffle(ans)
    rng.shuffle(unans)
    picked = must + ans[:n_answerable] + unans[:n_unanswerable]
    rng.shuffle(picked)
    return picked


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agreement(rows: list[dict], records: list[dict]) -> dict:
    by_id = {r["id"]: r for r in records}
    abst, corr = [], []
    for row in rows:
        r = by_id[row["id"]]
        h_abs = row["human_abstained"].strip().lower()
        if h_abs in ("y", "n"):
            abst.append((h_abs, "y" if r["abstained"] else "n"))
        h_cor = row["human_correctness"].strip().lower()
        if h_cor in ("correct", "partial", "incorrect") and r["correctness"] in ("correct", "partial", "incorrect"):
            corr.append((h_cor, r["correctness"]))
    return {
        "abstained": {"n": len(abst), "agreement": sum(a == b for a, b in abst) / len(abst) if abst else None, "kappa": kappa(abst)},
        "correctness": {"n": len(corr), "agreement": sum(a == b for a, b in corr) / len(corr) if corr else None, "kappa": kappa(corr)},
    }


def make_sheet(name: str, n_answerable: int, n_unanswerable: int) -> None:
    records = load_records(name)
    gold = {}
    with (settings.eval_dir / "questions.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            gold[row["id"]] = row["gold_answer"]
    out = settings.eval_dir / "results" / f"judge_check_{name}.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in sample(records, n_answerable, n_unanswerable):
            w.writerow([r["id"], "", r["question"], gold[r["id"]] or "NONE (not in the documents)", r["answer"], "", "", ""])
    print("wrote", out)


def score_sheet(name: str, path: str) -> None:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print(json.dumps(agreement(rows, load_records(name)), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["make", "score"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--sheet")
    ap.add_argument("--answerable", type=int, default=9)
    ap.add_argument("--unanswerable", type=int, default=8)
    a = ap.parse_args()
    make_sheet(a.name, a.answerable, a.unanswerable) if a.action == "make" else score_sheet(a.name, a.sheet)


if __name__ == "__main__":
    main()
