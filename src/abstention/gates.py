from __future__ import annotations

from dataclasses import dataclass


def auc(positive: list[float], negative: list[float]) -> float | None:
    """Chance that a random positive scores higher than a random negative (ties count half)."""
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def candidates(scores: list[float]) -> list[float]:
    u = sorted(set(scores))
    if not u:
        return []
    return [u[0] - 1e-6, *[(a + b) / 2 for a, b in zip(u, u[1:])], u[-1] + 1e-6]


def best_threshold(keep: list[float], drop: list[float]) -> float | None:
    """Threshold tau (refuse when score < tau) that maximises balanced accuracy: keep answerable, drop the rest.
    Among ties the lowest tau wins, which means fewer refusals."""
    if not keep or not drop:
        return None
    best, best_j = None, -2.0
    for tau in candidates(keep + drop):
        j = sum(s >= tau for s in keep) / len(keep) + sum(s < tau for s in drop) / len(drop) - 1
        if j > best_j + 1e-12:
            best, best_j = tau, j
    return best


def threshold_keeping(keep: list[float], min_keep: float = 0.94) -> float | None:
    """Strictest threshold that still keeps at least min_keep of the good items ("do no harm" operating point)."""
    if not keep:
        return None
    s = sorted(keep)
    allowed = int((1 - min_keep) * len(s) + 1e-9)
    return s[0] - 1e-6 if allowed == 0 else (s[allowed - 1] + s[allowed]) / 2


def sweep(keep: list[float], drop: list[float], taus: list[float]) -> list[dict]:
    return [
        {
            "tau": tau,
            "answerable_kept": sum(s >= tau for s in keep) / len(keep) if keep else None,
            "unanswerable_refused": sum(s < tau for s in drop) / len(drop) if drop else None,
        }
        for tau in taus
    ]


@dataclass
class Gated:
    records: list[dict]
    changed: list[str]


def refuse(record: dict, reason: str) -> dict:
    out = dict(record)
    out.update(
        abstained=True,
        correctness="incorrect" if record["kind"] == "answerable" else "not_applicable",
        supported_by_cited="not_applicable",
        supported_by_retrieved="not_applicable",
        n_valid_citations=0,
        gated_by=reason,
    )
    return out


def apply_retrieval_gate(records: list[dict], key: str, tau: float) -> Gated:
    """Refuse before answering when the best evidence score is below tau."""
    out, changed = [], []
    for r in records:
        if r.get(key) is not None and r[key] < tau and r["abstained"] is not True:
            out.append(refuse(r, "retrieval"))
            changed.append(r["id"])
        else:
            out.append(r)
    return Gated(out, changed)


def apply_nli_gate(records: list[dict], scores: dict[str, float | None], tau: float) -> Gated:
    """Refuse an answer whose weakest factual sentence is not entailed by its evidence."""
    out, changed = [], []
    for r in records:
        s = scores.get(r["id"])
        if s is not None and s < tau and r["abstained"] is False:
            out.append(refuse(r, "nli"))
            changed.append(r["id"])
        else:
            out.append(r)
    return Gated(out, changed)
