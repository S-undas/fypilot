from src.evaluation.judge import build_messages, judge, parse
from src.evaluation.metrics import summarize, summarize_by_split

GOOD = '{"abstained": false, "correctness": "correct", "supported_by_cited": "yes", "supported_by_retrieved": "yes"}'


def test_parse_accepts_plain_and_fenced_json():
    assert parse(GOOD)["correctness"] == "correct"
    assert parse("```json\n" + GOOD + "\n```")["abstained"] is False


def test_parse_rejects_bad_values_and_text():
    assert parse("no json here") is None
    assert parse(GOOD.replace('"correct"', '"great"')) is None
    assert parse(GOOD.replace("false", '"no"')) is None


def test_judge_falls_back_to_parse_error():
    class Bad:
        def complete(self, messages):
            return "I cannot grade this"

    out = judge(Bad(), "q", "a", "ans", [], [])
    assert out["correctness"] == "parse_error" and out["abstained"] is None


def test_prompt_marks_missing_reference():
    msgs = build_messages("q?", "", "answer", [(1, "cited text")], [(1, "cited text"), (2, "other")])
    body = msgs[1]["content"]
    assert "NONE" in body and "[2] other" in body and "Cited passages:\n[1] cited text" in body


def rec(kind, abstained, correctness="not_applicable", cited="yes", retrieved="yes", cites=1, gold=True, split="dev"):
    return {"kind": kind, "abstained": abstained, "correctness": correctness, "supported_by_cited": cited,
            "supported_by_retrieved": retrieved, "n_valid_citations": cites, "gold_in_top5": gold, "split": split}


def test_summarize_rates():
    records = [
        rec("answerable", False, "correct"),
        rec("answerable", False, "incorrect", cited="no", retrieved="no", cites=0),
        rec("answerable", True),
        rec("near_miss", True),
        rec("near_miss", False),
        rec("off_topic", True),
    ]
    s = summarize(records)
    assert s["n_answerable"] == 3 and s["correct_rate"] == 1 / 3
    assert s["false_refusal_rate"] == 1 / 3
    assert s["faithful_rate"] == 0.5 and s["citation_present_rate"] == 0.5 and s["citation_support_rate"] == 1.0
    assert s["abstain_rate"] == 2 / 3 and s["abstain_rate_near_miss"] == 0.5 and s["abstain_rate_off_topic"] == 1.0
    assert s["abstention_precision"] == 2 / 3


def test_summarize_by_split_and_empty():
    out = summarize_by_split([rec("answerable", False, "correct", split="dev"), rec("answerable", False, "partial", split="test")])
    assert set(out) == {"dev", "test", "all"} and out["all"]["n_answerable"] == 2
    assert summarize([])["correct_rate"] is None
