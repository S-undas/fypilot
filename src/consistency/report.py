from __future__ import annotations

import json
from collections import Counter

from src.config import settings
from src.consistency.requirements import DOC_STAGES, DOC_TYPES
from src.consistency.review import load_labeled

OUT = settings.eval_dir / "results"


def load(name: str) -> list[dict] | None:
    path = OUT / f"{name}.jsonl"
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(num: int, den: int) -> str:
    return f"{num / den:.0%} ({num} of {den})" if den else "-"


def per_doc(rows: list[dict], verdicts: list[str], key: str = "verdict") -> str:
    lines = ["| document | type | stage | checked | " + " | ".join(verdicts) + " |", "|---|---|---|---|" + "---|" * len(verdicts)]
    for doc in DOC_TYPES:
        mine = [r for r in rows if r["doc"] == doc]
        c = Counter(r[key] for r in mine)
        lines.append(f"| {doc} | {DOC_TYPES[doc]} | {DOC_STAGES[doc]} | {len(mine)} | " + " | ".join(str(c[v]) for v in verdicts) + " |")
    return "\n".join(lines)


def missing_section(check: list[dict], perturb: list[dict] | None) -> str:
    parts = ["## Missing required parts", "", per_doc(check, ["present", "partial", "missing"]), ""]
    if perturb:
        lines = ["| test | cases | plain heading rule flags it | retrieval + model flags it | model says missing |", "|---|---|---|---|---|"]
        for kind, label in (("remove", "chapter deleted"), ("empty", "heading kept, content removed")):
            rows = [r for r in perturb if r["variant"] == kind]
            lines.append(f"| {label} | {len(rows)} | {pct(sum(r['heading'] != 'present' for r in rows), len(rows))} | "
                         f"{pct(sum(r['verdict'] != 'present' for r in rows), len(rows))} | {pct(sum(r['verdict'] == 'missing' for r in rows), len(rows))} |")
        parts += ["Removal test: each required chapter was deleted from a copy of the report, or emptied while its heading stayed.", "", *lines, ""]
    return "\n".join(parts)


def promises_section(rows: list[dict]) -> str:
    return "\n".join(["## Promises without evidence", "", per_doc(rows, ["shown", "not_shown", "contradicted"]), ""])


def contradictions_section(rows: list[dict]) -> str:
    docs = {}
    for doc in DOC_TYPES:
        mine = [r for r in rows if r["doc"] == doc]
        docs[doc] = (len(mine), sum(r["first"] == "conflict" for r in mine), sum(r["verdict"] == "confirmed" for r in mine))
    lines = ["| document | pairs compared | flagged by first check | confirmed by second check |", "|---|---|---|---|"]
    lines += [f"| {d} | {a} | {b} | {c} |" for d, (a, b, c) in docs.items()]
    served = Counter(r.get("served_by") or "unknown" for r in rows)
    models = "Pairs judged by: " + ", ".join(f"{m} ({n})" for m, n in served.most_common()) + "."
    return "\n".join(["## Contradictions inside a report", "", *lines, "", models, ""])


def human_section(res: dict | None) -> str:
    if not res:
        return ""
    p, c = res["promise"], res["conflict"]
    fmt = lambda v: "-" if v is None else f"{v:.0%}"
    lines = ["## Human check", "", "A person labeled a sample of the flags without seeing the system's verdict.", "",
             "| measure | value |", "|---|---|",
             f"| promise flags that were right ({p['flagged']['labeled']} labeled) | {fmt(p['flag_precision'])} (lenient {fmt(p['flag_precision_lenient'])}) |",
             f"| promises marked shown that were not backed up ({p['shown']['labeled']} labeled) | {fmt(p['shown_but_unsupported'])} |",
             f"| conflicts confirmed that were real ({c['confirmed']['labeled']} labeled) | {fmt(c['conflict_precision'])} (lenient {fmt(c['conflict_precision_lenient'])}) |",
             f"| conflicts dropped by the second check that were real ({c['dropped']['labeled']} labeled) | {fmt(c['real_among_dropped'])} |",
             f"| pairs judged consistent that were real conflicts ({c['consistent']['labeled']} labeled) | {fmt(c['real_among_consistent'])} |", ""]
    return "\n".join(lines)


def planted_section(res: dict | None) -> str:
    if not res:
        return ""
    p, c = res["promise"], res["contradiction"]
    lines = ["## Planted mistakes", "", "Mistakes were inserted into copies of the extracted text, then the checkers were run on the copies.", "",
             "| kind | planted | found |", "|---|---|---|",
             f"| unproven promise (extracted by the system) | {p['planted']} | {pct(p['extracted'], p['planted'])} |",
             f"| unproven promise (extracted and flagged) | {p['planted']} | {pct(p['flagged'], p['planted'])} |",
             f"| contradiction (number changed in one place) | {c['planted']} | {pct(c['found'], c['planted'])} |", ""]
    return "\n".join(lines)


def build() -> str:
    check, perturb = load("missing_check"), load("missing_perturb")
    promises, conflicts = load("promises_check"), load("contradictions_check")
    planted = OUT / "consistency_planted_score_both.json"
    parts = ["# Phase 6 results: gap and consistency checks", "",
             "Produced by `python -m src.consistency.report`. Each report is checked against the handbook's required parts, "
             "for promises it never backs up, and for statements that contradict each other.", ""]
    for text in (missing_section(check, perturb) if check else "", promises_section(promises) if promises else "",
                 contradictions_section(conflicts) if conflicts else "", human_section(load_labeled()),
                 planted_section(json.loads(planted.read_text(encoding="utf-8")) if planted.exists() else None)):
        if text:
            parts.append(text)
    return "\n".join(parts)


def main() -> None:
    text = build()
    out = OUT / "phase6_summary.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
