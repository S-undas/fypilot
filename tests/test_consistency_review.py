from src.consistency.review import describe, sample, score


def promise(i, doc, verdict):
    return {"doc": doc, "verdict": verdict, "section": "2 Vision", "page": "3", "promise": f"P{i}", "quote": f"q{i}"}


def conflict(i, doc, first, verdict):
    a = {"section": "5.2", "page": "1", "quote": f"a{i}"}
    return {"doc": doc, "first": first, "verdict": verdict, "a": a, "b": {**a, "section": "8.2", "quote": f"b{i}"}}


def data():
    promises = [promise(i, d, v) for i, (d, v) in enumerate((d, v) for d in ("D1", "D2") for v in ["shown"] * 6 + ["not_shown"] * 6)]
    conflicts = ([conflict(i, d, "conflict", "confirmed") for i, d in enumerate(["D1", "D2"] * 4)]
                 + [conflict(20, "D1", "conflict", "not_confirmed"), conflict(21, "D2", "consistent", "consistent"), conflict(22, "D1", "unrelated", "unrelated")])
    return promises, conflicts


def test_sample_is_deterministic_spread_and_skips_unrelated():
    promises, conflicts = data()
    s1 = sample(promises, conflicts, n_flagged=4, n_shown=2, n_conflict=4, n_dropped=1, n_consistent=1)
    s2 = sample(promises, conflicts, n_flagged=4, n_shown=2, n_conflict=4, n_dropped=1, n_consistent=1)
    assert [r.get("promise") or r["a"]["quote"] for r in s1] == [r.get("promise") or r["a"]["quote"] for r in s2]
    flagged = [r for r in s1 if r["kind"] == "promise" and r["group"] == "flagged"]
    assert len(flagged) == 4 and {r["doc"] for r in flagged} == {"D1", "D2"}
    assert sorted(r["group"] for r in s1 if r["kind"] == "conflict") == ["confirmed"] * 4 + ["consistent", "dropped"]


def test_describe_shows_both_places_of_a_conflict():
    where, text = describe({"kind": "conflict", "a": {"section": "5.2", "page": "1", "quote": "x"}, "b": {"section": "8.2", "page": "4", "quote": "y"}})
    assert "5.2" in where and "8.2" in where and '"x"' in text and '"y"' in text


def test_score_precision_and_leniency():
    key = {"R1": {"kind": "promise", "group": "flagged"}, "R2": {"kind": "promise", "group": "flagged"}, "R3": {"kind": "promise", "group": "shown"},
           "R4": {"kind": "conflict", "group": "confirmed"}, "R5": {"kind": "conflict", "group": "confirmed"}, "R6": {"kind": "conflict", "group": "dropped"}}
    rows = [{"qid": q, "human_label": v} for q, v in zip(key, ["n", "y", "n", "y", "p", "n"])]
    out = score(rows, key)
    assert out["promise"]["flag_precision"] == 0.5 and out["promise"]["shown_but_unsupported"] == 1.0
    assert out["conflict"]["conflict_precision"] == 0.5 and out["conflict"]["conflict_precision_lenient"] == 1.0
    assert out["conflict"]["real_among_dropped"] == 0.0 and out["conflict"]["real_among_consistent"] is None
