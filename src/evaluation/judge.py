from __future__ import annotations

import json
import re

SYSTEM = "You grade answers from a retrieval-augmented question answering system. Reply with one JSON object and nothing else."

SUPPORT = {"yes", "partial", "no", "not_applicable"}
CORRECT = {"correct", "partial", "incorrect", "not_applicable"}

INSTRUCTIONS = """Return a JSON object with exactly these keys:
"abstained": true if the main message of the system answer is that the documents do not state, report or contain the requested information, even if it then mentions related facts that do not answer the question. false if it gives the specific information the question asks for, whether that information is right or wrong. Example: question "What was X on topology B?", answer "The passages do not state X for B; they report X for topology A" means abstained is true.
"correctness": compare the system answer with the reference answer. "correct" if it conveys the key facts of the reference, "partial" if it misses key facts or adds wrong detail, "incorrect" if it contradicts or misses the reference, "not_applicable" if the reference is NONE.
"supported_by_cited": are all factual claims in the system answer supported by the cited passages? "yes", "partial", "no", or "not_applicable" if there are no cited passages or the answer abstains.
"supported_by_retrieved": the same check against all retrieved passages."""


def passages(texts: list[tuple[int, str]], limit: int = 1200) -> str:
    return "\n".join(f"[{n}] {t[:limit]}" for n, t in texts) or "(none)"


def build_messages(
    question: str,
    gold_answer: str,
    answer_text: str,
    cited: list[tuple[int, str]],
    retrieved: list[tuple[int, str]],
) -> list[dict]:
    reference = gold_answer or "NONE (the documents do not contain the answer to this question)"
    user = (
        f"Question: {question}\n\nReference answer: {reference}\n\nSystem answer: {answer_text}\n\n"
        f"Cited passages:\n{passages(cited)}\n\nAll retrieved passages:\n{passages(retrieved)}\n\n{INSTRUCTIONS}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    ok = (
        isinstance(d.get("abstained"), bool)
        and d.get("correctness") in CORRECT
        and d.get("supported_by_cited") in SUPPORT
        and d.get("supported_by_retrieved") in SUPPORT
    )
    return {k: d[k] for k in ("abstained", "correctness", "supported_by_cited", "supported_by_retrieved")} if ok else None


def judge(llm, question, gold_answer, answer_text, cited, retrieved) -> dict:
    text = llm.complete(build_messages(question, gold_answer, answer_text, cited, retrieved))
    parsed = parse(text)
    return parsed or {"abstained": None, "correctness": "parse_error", "supported_by_cited": "parse_error",
                      "supported_by_retrieved": "parse_error"}
