from __future__ import annotations

import csv
import json
import sys

from src.abstention.gates import apply_nli_gate, apply_retrieval_gate, auc, best_threshold, sweep, threshold_keeping
from src.abstention.nli import support_scores
from src.config import settings
from src.evaluation.metrics import summarize_by_split
from src.evaluation.report import fmt, load_run

SIGNALS = ("max_dense", "top_score")
SIGNAL_LABEL = {"max_dense": "best dense similarity in the top 5", "top_score": "reranker score of the top passage"}
GENERATORS = {"qwen": "Qwen 3.8 27B", "gptoss": "gpt-oss-120b"}
KINDS = ("near_miss", "misleading", "off_topic")
MIN_KEEP = 0.94


def results_dir():
    return settings.eval_dir / "results"


def is_bad(r: dict) -> bool:
    """An answer we would want to block: it answers an unanswerable question or is not judged correct."""
    return r["kind"] != "answerable" or r["correctness"] != "correct"


def signal_analysis(records: list[dict], split: str = "dev") -> dict:
    rows = [r for r in records if r["split"] == split]
    out = {}
    for s in SIGNALS:
        keep = [r[s] for r in rows if r["kind"] == "answerable" and r.get(s) is not None]
        drop = [r[s] for r in rows if r["kind"] != "answerable" and r.get(s) is not None]
        out[s] = {
            "auc": auc(keep, drop),
            "auc_by_kind": {k: auc(keep, [r[s] for r in rows if r["kind"] == k and r.get(s) is not None]) for k in KINDS},
            "tau": best_threshold(keep, drop),
            "tau_no_harm": threshold_keeping(keep, MIN_KEEP),
        }
    return out


def choose_signal(dev_analysis: dict) -> str:
    return max(SIGNALS, key=lambda s: (dev_analysis[s]["auc"] or 0, s == "max_dense"))


def nli_premises(record: dict, texts: dict[str, str]) -> list[str]:
    ids = record.get("cited_chunks") or record["retrieved"]
    return [texts[i] for i in ids if i in texts]


def score_nli(name: str, records: list[dict], scorer, texts: dict[str, str], tag: str = "nli3") -> dict[str, dict | None]:
    path = results_dir() / f"{tag}_{name}.json"
    scores: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for r in records:
        if r["id"] in scores:
            continue
        scores[r["id"]] = support_scores(scorer, r["answer"], nli_premises(r, texts)) if r["abstained"] is False else None
    path.write_text(json.dumps(scores), encoding="utf-8")
    return scores


def pick(scores: dict[str, dict | None], key: str = "frac") -> dict[str, float | None]:
    return {i: (v[key] if v else None) for i, v in scores.items()}


def nli_pools(runs: dict[str, list[dict]], scores: dict[str, dict], split: str) -> tuple[list[float], list[float]]:
    good, bad = [], []
    for name, records in runs.items():
        for r in records:
            s = scores[name].get(r["id"])
            if r["split"] != split or r["abstained"] is not False or s is None:
                continue
            (bad if is_bad(r) else good).append(s)
    return good, bad


def build_arms(plain: list[dict], refusal: list[dict], signal: str, taus: dict[str, float | None],
               nli: dict[str, float | None], tau_nli: float | None) -> dict[str, dict]:
    arms = {"plain": {"records": plain, "changed": []}, "prompt refusal": {"records": refusal, "changed": []}}
    if taus.get("balanced") is not None:
        g = apply_retrieval_gate(plain, signal, taus["balanced"])
        arms["+ threshold (balanced)"] = {"records": g.records, "changed": g.changed}
    base = plain
    if taus.get("no_harm") is not None:
        g = apply_retrieval_gate(plain, signal, taus["no_harm"])
        arms["+ threshold (no-harm)"] = {"records": g.records, "changed": g.changed}
        base = g.records
        both = apply_retrieval_gate(refusal, signal, taus["no_harm"])
        arms["prompt refusal + threshold (no-harm)"] = {"records": both.records, "changed": both.changed}
    if tau_nli is not None:
        g = apply_nli_gate(base, nli, tau_nli)
        arms["+ threshold + NLI (no-harm)"] = {"records": g.records, "changed": g.changed}
    return arms


ROWS = [
    ("n_answerable", "answerable questions"),
    ("correct_rate", "answered correctly"),
    ("false_refusal_rate", "wrongly refused"),
    ("n_unanswerable", "unanswerable questions"),
    ("abstain_rate", "refused (all unanswerable)"),
    ("abstain_rate_near_miss", "refused (near-miss)"),
    ("abstain_rate_misleading", "refused (misleading)"),
    ("abstain_rate_off_topic", "refused (off-topic)"),
    ("abstention_precision", "refusals that were justified"),
]


def arms_table(arms: dict[str, dict], split: str) -> str:
    summaries = {name: summarize_by_split(a["records"]).get(split, {}) for name, a in arms.items()}
    names = list(arms)
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label in ROWS:
        lines.append(f"| {label} | " + " | ".join(fmt(summaries[n].get(key)) for n in names) + " |")
    answered_bad = {n: sum(1 for r in a["records"] if r["split"] == split and r["kind"] != "answerable" and r["abstained"] is False)
                    for n, a in arms.items()}
    lines.append("| unanswerable questions answered anyway | " + " | ".join(str(answered_bad[n]) for n in names) + " |")
    return "\n".join(lines)


def curve_rows(gate: str, split: str, keep: list[float], drop: list[float], n: int = 40) -> list[dict]:
    pool = sorted(keep + drop)
    if not pool:
        return []
    lo, hi = pool[0], pool[-1]
    taus = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    return [{"gate": gate, "split": split, **row} for row in sweep(keep, drop, taus)]


def plot_curves(path, curves: list[dict], taus: dict[str, float | None]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gates = sorted({c["gate"] for c in curves})
    fig, axes = plt.subplots(1, len(gates), figsize=(5.2 * len(gates), 3.8), squeeze=False)
    for ax, gate in zip(axes[0], gates):
        for split, style in (("dev", "-"), ("test", "--")):
            rows = [c for c in curves if c["gate"] == gate and c["split"] == split]
            ax.plot([r["tau"] for r in rows], [r["answerable_kept"] for r in rows], style, color="tab:green", label=f"good answers kept ({split})")
            ax.plot([r["tau"] for r in rows], [r["unanswerable_refused"] for r in rows], style, color="tab:red", label=f"bad answers refused ({split})")
        if taus.get(gate) is not None:
            ax.axvline(taus[gate], color="gray", linestyle=":", label="no-harm threshold (dev)")
        ax.set_title(gate)
        ax.set_xlabel("threshold")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def first_attempt(plain: dict[str, list[dict]], signal: str, tau_ret: float | None) -> list[str]:
    """The first NLI version (weakest sentence, list lead-ins counted as claims), kept for comparison."""
    scores = {}
    for g in GENERATORS:
        path = results_dir() / f"nli_{g}_plain.json"
        if not path.exists():
            return []
        scores[g] = json.loads(path.read_text(encoding="utf-8"))
    good, bad = nli_pools(plain, scores, "dev")
    tau = best_threshold(good, bad)
    if tau is None or tau_ret is None:
        return []
    lines = ["| generator | answered correctly (test) | unanswerable refused (test) |", "|---|---|---|"]
    for g, label in GENERATORS.items():
        gated = apply_nli_gate(apply_retrieval_gate(plain[g], signal, tau_ret).records, scores[g], tau)
        s = summarize_by_split(gated.records)["test"]
        lines.append(f"| {label} | {fmt(s['correct_rate'])} | {fmt(s['abstain_rate'])} |")
    return [f"First NLI version (weakest sentence counted, list lead-ins treated as claims), threshold {fmt(tau)} chosen on dev, "
            "with the balanced score threshold:", ""] + lines


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.retrieval.index import load_all_chunks

    texts = {c.chunk_id: c.text for c in load_all_chunks()}
    plain = {g: load_run(f"{g}_plain") for g in GENERATORS}
    refusal = {g: load_run(f"{g}_refusal") for g in GENERATORS}

    signal_records = plain["qwen"]
    dev_an, test_an = signal_analysis(signal_records, "dev"), signal_analysis(signal_records, "test")
    signal = choose_signal(dev_an)
    taus = {"balanced": dev_an[signal]["tau"], "no_harm": dev_an[signal]["tau_no_harm"]}

    scorer = None
    nli_full: dict[str, dict] = {}
    for g in GENERATORS:
        path = results_dir() / f"nli3_{g}_plain.json"
        done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if any(r["id"] not in done for r in plain[g]) and scorer is None:
            from src.abstention.nli import NLIScorer

            scorer = NLIScorer()
        nli_full[g] = score_nli(f"{g}_plain", plain[g], scorer, texts)
    nli = {g: pick(v) for g, v in nli_full.items()}
    good_dev, bad_dev = nli_pools(plain, nli, "dev")
    good_test, bad_test = nli_pools(plain, nli, "test")
    tau_nli = threshold_keeping(good_dev, MIN_KEEP)
    bad_rows = [(g, r["id"], nli[g][r["id"]]) for g in GENERATORS for r in plain[g]
                if r["abstained"] is False and nli[g].get(r["id"]) is not None and is_bad(r)]

    out = ["# Phase 4 results: abstention", "",
           "Produced by `python -m src.evaluation.abstention_eval`. All thresholds are chosen on dev only. "
           f"Two operating points are used: balanced (equal weight on keeping good answers and refusing bad ones) and no-harm "
           f"(the strictest threshold that keeps at least {MIN_KEEP:.0%} of good dev answers).", "",
           "## Score threshold: which signal separates answerable from unanswerable?", "",
           "AUC is the chance that a random answerable question scores higher than a random unanswerable one (1.0 is perfect, 0.5 is chance).", "",
           "| signal | AUC dev | AUC test | dev: near-miss | dev: misleading | dev: off-topic | balanced threshold | no-harm threshold |", "|---|---|---|---|---|---|---|---|"]
    for s in SIGNALS:
        out.append(f"| {SIGNAL_LABEL[s]} | {fmt(dev_an[s]['auc'])} | {fmt(test_an[s]['auc'])} | "
                   + " | ".join(fmt(dev_an[s]["auc_by_kind"][k]) for k in KINDS)
                   + f" | {fmt(dev_an[s]['tau'])} | {fmt(dev_an[s]['tau_no_harm'])} |")
    out += ["", f"Chosen signal (higher dev AUC): **{SIGNAL_LABEL[signal]}**.", "",
            "## NLI check: does the evidence entail the answer?", "",
            "Each factual sentence of an answer is checked against the passages it cites. The score is the share of sentences entailed (probability at least 0.5). "
            "List lead-ins such as \"The metrics are:\" and refusal sentences are not counted. "
            "Passages are cut into overlapping 55-word windows and each sentence is matched to its best window.", "",
            f"- Good answers (answerable and judged correct): {len(good_dev)} on dev, {len(good_test)} on test.",
            f"- Bad answers (answers to unanswerable questions or not judged correct): {len(bad_dev)} on dev, {len(bad_test)} on test. "
            "Too few to tune or evaluate a threshold on bad answers, so the no-harm rule uses only the good answers.",
            f"- No-harm threshold from dev: {fmt(tau_nli)}. AUC on test: {fmt(auc(good_test, bad_test))}.",
            f"- Support score of every bad answer: " + (", ".join(f"{g}/{i} = {fmt(v)}" for g, i, v in bad_rows) or "none"), ""]
    fa = first_attempt(plain, signal, taus["balanced"])
    if fa:
        out += fa + [""]
    changes = []
    for g, label in GENERATORS.items():
        arms = build_arms(plain[g], refusal[g], signal, taus, nli[g], tau_nli)
        for split in ("test", "dev"):
            out += [f"## {label}: {split} split", "", arms_table(arms, split), ""]
        for name, a in arms.items():
            if a["changed"]:
                changes.append(f"- {label}, {name}: changed {len(a['changed'])} answers: " + ", ".join(a["changed"]))
    out += ["## Answers changed by each gate (question ids)", ""] + (changes or ["- none"]) + [""]

    curves = []
    for split in ("dev", "test"):
        rows = [r for r in signal_records if r["split"] == split and r.get(signal) is not None]
        curves += curve_rows("score threshold", split, [r[signal] for r in rows if r["kind"] == "answerable"],
                             [r[signal] for r in rows if r["kind"] != "answerable"])
    for split, (good, bad) in (("dev", (good_dev, bad_dev)), ("test", (good_test, bad_test))):
        curves += curve_rows("NLI check", split, good, bad)
    with (results_dir() / "abstention_curves.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gate", "split", "tau", "answerable_kept", "unanswerable_refused"])
        w.writeheader()
        w.writerows(curves)
    plot_curves(results_dir() / "abstention_curves.png", curves, {"score threshold": taus["no_harm"], "NLI check": tau_nli})
    text = "\n".join(out)
    (results_dir() / "phase4_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
