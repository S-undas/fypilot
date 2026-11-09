from __future__ import annotations

from collections import Counter, defaultdict

from src.config import settings
from src.evaluation.report import fmt
from src.panel.categories import CATEGORIES
from src.panel.check import load, load_labeled
from src.panel.run_panel import COVERED, NOT_COVERED


def pct(num: int, den: int) -> str:
    return f"{num / den:.0%} ({num} of {den})" if den else "-"


def source_table(by_source: dict[str, list[dict]]) -> str:
    lines = ["| | document-specific questions | generic questions |", "|---|---|---|"]
    stats = {}
    for name, rows in by_source.items():
        covered = [r for r in rows if r["status"] == COVERED]
        judged = [r for r in covered if r.get("j_answers_question") in ("yes", "partial", "no")]
        stats[name] = {
            "n": str(len(rows)),
            "nc": pct(sum(r["status"] == NOT_COVERED for r in rows), len(rows)),
            "spec": fmt(sum(r["specificity"] for r in rows) / len(rows)) if rows else "-",
            "ans": pct(sum(r["j_answers_question"] == "yes" for r in judged), len(judged)),
            "sup": pct(sum(r["j_supported"] == "yes" for r in judged), len(judged)),
        }
    for key, label in (("n", "questions"), ("nc", "marked NOT COVERED"), ("spec", "mean similarity to the document (specificity)"),
                       ("ans", "covered answers that address the question (judge)"), ("sup", "covered answers supported by their sources (judge)")):
        lines.append(f"| {label} | " + " | ".join(stats[s][key] for s in ("doc", "generic")) + " |")
    return "\n".join(lines)


def group_table(rows: list[dict], key: str, order: list[str]) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    lines = [f"| {key} | questions | not covered |", "|---|---|---|"]
    for g in order:
        if g in groups:
            lines.append(f"| {g} | {len(groups[g])} | {pct(sum(r['status'] == NOT_COVERED for r in groups[g]), len(groups[g]))} |")
    return "\n".join(lines)


def human_section(result: dict | None) -> str:
    if result is None:
        return ""
    lines = ["## Human check of coverage status", "",
             "A person looked up each sampled question in the documents without seeing the system's status.", "",
             "| | document-specific | generic | all |", "|---|---|---|---|"]
    for key, label in (("labeled", "questions labeled"), ("system_not_covered", "marked NOT COVERED by the system"),
                       ("human_not_covered", "not covered according to the person"), ("not_covered_precision", "NOT COVERED precision"),
                       ("not_covered_recall", "NOT COVERED recall"),
                       ("not_covered_precision_lenient", "precision if partly covered counts as not covered"),
                       ("not_covered_recall_lenient", "recall if partly covered counts as not covered"),
                       ("wrongly_not_covered", "wrongly marked NOT COVERED"),
                       ("wrongly_covered", "wrongly marked covered"), ("human_partial", "partly covered (counted as covered)"),
                       ("mean_relevance", "mean relevance rating (1 to 3)")):
        lines.append(f"| {label} | " + " | ".join(fmt(result[s][key]) for s in ("doc", "generic", "all")) + " |")
    return "\n".join(lines) + "\n"


def build() -> str:
    doc_rows, gen_rows = load("doc"), load("generic")
    parts = ["# Phase 5 results: panel questions", "",
             "Produced by `python -m src.panel.report`. Questions are generated per document and category, answered from the document, "
             "and marked NOT COVERED when the passages do not answer them.", "",
             "## Document-specific vs generic questions", "", source_table({"doc": doc_rows, "generic": gen_rows}), "",
             "## Document-specific questions by category", "", group_table(doc_rows, "category", list(CATEGORIES)), "",
             "## Document-specific questions by document", "", group_table(doc_rows, "doc", sorted({r["doc"] for r in doc_rows})), ""]
    labeled = human_section(load_labeled())
    if labeled:
        parts.append(labeled)
    return "\n".join(parts)


def main() -> None:
    text = build()
    out = settings.eval_dir / "results" / "phase5_summary.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
