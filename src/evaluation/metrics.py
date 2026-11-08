from __future__ import annotations

from collections import defaultdict


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def summarize(records: list[dict]) -> dict:
    """Aggregate judged records. A record needs: kind, abstained, correctness,
    supported_by_cited, supported_by_retrieved, n_valid_citations, gold_in_top5."""
    ans = [r for r in records if r["kind"] == "answerable"]
    unans = [r for r in records if r["kind"] != "answerable"]
    answered = [r for r in ans if r["abstained"] is False]
    with_cites = [r for r in answered if r["n_valid_citations"] > 0]
    out = {
        "n_answerable": len(ans),
        "correct_rate": rate(sum(r["correctness"] == "correct" for r in ans), len(ans)),
        "partial_rate": rate(sum(r["correctness"] == "partial" for r in ans), len(ans)),
        "false_refusal_rate": rate(sum(r["abstained"] is True for r in ans), len(ans)),
        "gold_in_top5": rate(sum(bool(r["gold_in_top5"]) for r in ans), len(ans)),
        "faithful_rate": rate(sum(r["supported_by_retrieved"] == "yes" for r in answered), len(answered)),
        "citation_present_rate": rate(len(with_cites), len(answered)),
        "citation_support_rate": rate(sum(r["supported_by_cited"] == "yes" for r in with_cites), len(with_cites)),
        "n_unanswerable": len(unans),
        "abstain_rate": rate(sum(r["abstained"] is True for r in unans), len(unans)),
    }
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for r in unans:
        by_kind[r["kind"]].append(r)
    for kind, rs in sorted(by_kind.items()):
        out[f"abstain_rate_{kind}"] = rate(sum(r["abstained"] is True for r in rs), len(rs))
    abstained_all = [r for r in records if r["abstained"] is True]
    out["abstention_precision"] = rate(sum(r["kind"] != "answerable" for r in abstained_all), len(abstained_all))
    out["parse_errors"] = sum(r["correctness"] == "parse_error" for r in records)
    return out


def summarize_by_split(records: list[dict]) -> dict[str, dict]:
    splits: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        splits[r["split"]].append(r)
        splits["all"].append(r)
    return {s: summarize(rs) for s, rs in splits.items()}
