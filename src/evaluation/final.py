from __future__ import annotations

import csv
import json
import re
from collections import Counter

from src.config import settings

RESULTS = settings.eval_dir / "results"
RUNS = [
    ("Qwen, bare prompt", "qwen_bare"),
    ("Qwen, grounded prompt", "qwen_main"),
    ("Qwen, grounded + reranker (plain RAG)", "qwen_rerank"),
    ("Qwen, + refusal instruction", "qwen_refusal"),
    ("gpt-oss, plain RAG", "gptoss_plain"),
    ("gpt-oss, + refusal instruction", "gptoss_refusal"),
]
GATE_ROWS = ["answered correctly", "wrongly refused", "refused (all unanswerable)", "unanswerable questions answered anyway"]
WHY = re.compile(r"\b(why|justify|justif\w*)\b", re.I)
VALIDATE = re.compile(r"\b(validate|validated|validation|verify|verified|evaluate|evaluated|test|tested)\b", re.I)


def load_run(name: str) -> list[dict]:
    path = RESULTS / f"judged_{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def counts(records: list[dict]) -> dict:
    ans = [r for r in records if r["kind"] == "answerable"]
    unans = [r for r in records if r["kind"] != "answerable"]
    return {
        "answerable": len(ans),
        "correct": sum(r["correctness"] == "correct" for r in ans),
        "wrongly_refused": sum(r["abstained"] is True for r in ans),
        "unanswerable": len(unans),
        "refused": sum(r["abstained"] is True for r in unans),
        "answered_anyway": sum(r["abstained"] is False for r in unans),
    }


def comparison_table(runs: dict[str, list[dict]]) -> str:
    names = list(runs)
    c = {n: counts(runs[n]) for n in names}
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label in (("correct", "answerable questions answered correctly"), ("wrongly_refused", "answerable questions wrongly refused"),
                       ("refused", "unanswerable questions refused"), ("answered_anyway", "unanswerable questions answered anyway")):
        lines.append(f"| {label} | " + " | ".join(f"{c[n][key]} of {c[n]['answerable' if key in ('correct', 'wrongly_refused') else 'unanswerable']}" for n in names) + " |")
    return "\n".join(lines)


def error_causes(records: list[dict]) -> dict:
    errors = [r for r in records if r["kind"] == "answerable" and r["correctness"] != "correct"]
    unans = [r for r in records if r["kind"] != "answerable" and r["abstained"] is False]
    return {
        "retrieval_miss_refused": sum((not r["gold_in_top5"]) and r["abstained"] for r in errors),
        "retrieval_miss_answered": sum((not r["gold_in_top5"]) and not r["abstained"] for r in errors),
        "generation_error": sum(bool(r["gold_in_top5"]) for r in errors),
        "hallucinated_by_kind": dict(Counter(r["kind"] for r in unans)),
    }


def error_table(runs: dict[str, list[dict]]) -> str:
    lines = ["| run | right passage missed, then refused | right passage missed, then wrong answer | wrong answer with the right passage | unanswerable answered anyway (near-miss / misleading / off-topic) |",
             "|---|---|---|---|---|"]
    for name, records in runs.items():
        e = error_causes(records)
        k = e["hallucinated_by_kind"]
        lines.append(f"| {name} | {e['retrieval_miss_refused']} | {e['retrieval_miss_answered']} | {e['generation_error']} | "
                     f"{k.get('near_miss', 0)} / {k.get('misleading', 0)} / {k.get('off_topic', 0)} |")
    return "\n".join(lines)


def parse_tables(md: str) -> dict[str, dict[str, list[str]]]:
    """Heading -> {metric: values by column}, plus a 'columns' entry, for the pipe tables of a generated summary."""
    out: dict[str, dict[str, list[str]]] = {}
    heading = None
    for line in md.splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
        elif heading and line.startswith("|") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            table = out.setdefault(heading, {})
            if cells[0] == "metric":
                table["columns"] = cells[1:]
            elif "columns" in table:
                table[cells[0]] = cells[1:]
    return out


def gate_table(tables: dict[str, dict[str, list[str]]], heading: str) -> str:
    t = tables.get(heading)
    if not t:
        return ""
    cols = t["columns"]
    lines = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += [f"| {m} | " + " | ".join(t[m]) + " |" for m in GATE_ROWS if m in t]
    return "\n".join(lines)


def panel_misses(rows: list[dict], key: dict[str, str], records: dict[str, dict]) -> dict:
    """Among document-specific questions a person labeled, how the system's status compares, and what kind of question it missed."""
    grid: Counter = Counter()
    missed: Counter = Counter()
    for r in rows:
        label = r["human_covered"].strip().lower()
        rec = records.get(key.get(r["qid"], ""))
        if rec is None or rec["source"] != "doc" or label not in ("y", "n", "p"):
            continue
        grid[(rec["status"], label)] += 1
        if rec["status"] == "COVERED" and label == "n":
            q = r["question"]
            missed["why or justify" if WHY.search(q) else "validate or test" if VALIDATE.search(q) else "other"] += 1
    return {"grid": dict(grid), "missed_by_type": dict(missed)}


def panel_table(res: dict) -> str:
    g = res["grid"]
    lines = ["| the person said | system said COVERED | system said NOT COVERED |", "|---|---|---|"]
    for label, name in (("y", "covered"), ("p", "partly covered"), ("n", "not covered")):
        lines.append(f"| {name} | {g.get(('COVERED', label), 0)} | {g.get(('NOT COVERED', label), 0)} |")
    m = res["missed_by_type"]
    total = sum(m.values())
    detail = ", ".join(f"{k}: {v}" for k, v in sorted(m.items(), key=lambda kv: -kv[1]))
    return "\n".join(lines) + f"\n\nGaps the system missed (answered, but the person said not covered): {total}. By question type, {detail}."


def load_panel() -> dict | None:
    path, key_path = RESULTS / "panel_check.csv", RESULTS / "panel_check_key.json"
    if not (path.exists() and key_path.exists()):
        return None
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    records = {}
    for name in ("doc", "generic"):
        for line in (RESULTS / f"panel_{name}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                records[r["id"]] = r
    return panel_misses(rows, json.loads(key_path.read_text(encoding="utf-8")), records)


def build() -> str:
    runs = {name: load_run(run) for name, run in RUNS}
    parts = ["# Final results: plain RAG against RAG with abstention, and where it fails", "",
             "Produced by `python -m src.evaluation.final` from the saved runs of Phases 3 to 5. No model is called.", "",
             "## Plain RAG against RAG with abstention (all 57 questions: 38 answerable, 19 unanswerable)", "", comparison_table(runs), ""]
    phase4 = RESULTS / "phase4_summary.md"
    if phase4.exists():
        tables = parse_tables(phase4.read_text(encoding="utf-8"))
        for heading, label in (("Qwen 3.8 27B: test split", "Qwen, test split (19 answerable, 9 unanswerable)"),
                               ("gpt-oss-120b: test split", "gpt-oss, test split (19 answerable, 9 unanswerable)")):
            table = gate_table(tables, heading)
            if table:
                parts += [f"## Refusal methods on the test split: {label}", "", table, ""]
    parts += ["## Where the question answering fails", "", error_table(runs), ""]
    panel = load_panel()
    if panel:
        parts += ["## Where the panel questions fail (document-specific questions a person labeled)", "", panel_table(panel), ""]
    return "\n".join(parts)


def main() -> None:
    text = build()
    out = RESULTS / "final_summary.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
