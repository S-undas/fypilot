from __future__ import annotations

import argparse
import csv
import json

from src.config import settings
from src.evaluation.judge_check import agreement, load_records
from src.evaluation.metrics import summarize_by_split

ROWS = [
    ("n_answerable", "answerable questions"),
    ("correct_rate", "answered correctly"),
    ("false_refusal_rate", "wrongly refused"),
    ("gold_in_top5", "right passage in top 5"),
    ("faithful_rate", "answer supported by retrieved passages"),
    ("citation_support_rate", "citations support the answer"),
    ("n_unanswerable", "unanswerable questions"),
    ("abstain_rate", "refused (all unanswerable)"),
    ("abstain_rate_near_miss", "refused (near-miss)"),
    ("abstain_rate_misleading", "refused (misleading)"),
    ("abstain_rate_off_topic", "refused (off-topic)"),
    ("abstention_precision", "refusals that were justified"),
]


RETRIEVAL_NOTE = (
    "How to read: recall@k is the share of questions whose right passage is in the top k results. "
    "mrr is the average of 1 divided by the rank of the right passage (1 is best). "
    "section_hit@5 means a passage from the right section is in the top 5."
)
ANSWER_NOTE = (
    "How to read: answered correctly and wrongly refused are shares of answerable questions. "
    "The refused rows are shares of unanswerable questions the system declined to answer. "
    "Labels come from an LLM judge, whose agreement with a person is reported at the end."
)


def fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def load_run(name: str) -> list[dict]:
    path = settings.eval_dir / "results" / f"judged_{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics_table(runs: dict[str, dict[str, dict]], split: str) -> str:
    names = [n for n, by_split in runs.items() if split in by_split]
    if not names:
        return ""
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label in ROWS:
        lines.append(f"| {label} | " + " | ".join(fmt(runs[n][split].get(key)) for n in names) + " |")
    return "\n".join(lines)


def retrieval_table(path) -> str:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    cols = [c for c in rows[0] if c not in ("arm", "split")]
    out = []
    for split in ("dev", "test", "all"):
        sel = [r for r in rows if r["split"] == split]
        if not sel:
            continue
        out += [f"**{split}**", "", "| search | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        out += [f"| {r['arm']} | " + " | ".join(r[c] for c in cols) + " |" for r in sel]
        out.append("")
    return "\n".join(out)


def judge_agreement_section(name: str) -> str:
    path = settings.eval_dir / "results" / f"judge_check_{name}.csv"
    if not path.exists():
        return ""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not any(r["human_abstained"].strip() for r in rows):
        return ""
    result = agreement(rows, load_records(name))
    lines = [f"## Judge agreement with human labels ({name})", "", "| label | labeled by a person | agreement | kappa |", "|---|---|---|---|"]
    for key, label in (("abstained", "refused or answered"), ("correctness", "correct, partial or incorrect")):
        r = result[key]
        lines.append(f"| {label} | {r['n']} | {fmt(r['agreement'])} | {fmt(r['kappa'])} |")
    return "\n".join(lines) + "\n"


def build(names: list[str]) -> str:
    runs = {n: summarize_by_split(load_run(n)) for n in names}
    parts = ["# Phase 3 results", "", "Produced by `python -m src.evaluation.report`. Judge labels come from an LLM judge, and its agreement with a person is reported at the end.", ""]
    ablation = settings.eval_dir / "results" / "retrieval_ablation.csv"
    if ablation.exists():
        parts += ["## Retrieval comparison (answerable questions, search inside the right document)", "", RETRIEVAL_NOTE, "", retrieval_table(ablation)]
    for split in ("dev", "test", "all"):
        table = metrics_table(runs, split)
        if table:
            parts += [f"## Answers and refusals: {split}", "", ANSWER_NOTE, "", table, ""]
    for name in names:
        section = judge_agreement_section(name)
        if section:
            parts.append(section)
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    text = build(args.runs)
    out = settings.eval_dir / "results" / "phase3_summary.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
