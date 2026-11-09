from __future__ import annotations

import json
import re

SYSTEM = "You check draft answers to defense-panel questions. Reply with one JSON object and nothing else."
LEVELS = {"yes", "partial", "no"}


def build_messages(question: str, answer_text: str, cited: list[tuple[int, str]]) -> list[dict]:
    passages = "\n".join(f"[{n}] {t[:1200]}" for n, t in cited) or "(none)"
    user = (
        f"Question: {question}\n\nDraft answer: {answer_text}\n\nCited passages:\n{passages}\n\n"
        'Return a JSON object with exactly these keys:\n'
        '"answers_question": does the draft answer actually address what the question asks? "yes", "partial" or "no".\n'
        '"supported": are the factual claims in the draft answer supported by the cited passages? "yes", "partial" or "no".'
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
    if d.get("answers_question") in LEVELS and d.get("supported") in LEVELS:
        return {"answers_question": d["answers_question"], "supported": d["supported"]}
    return None


def judge(llm, question: str, answer_text: str, cited: list[tuple[int, str]]) -> dict:
    parsed = parse(llm.complete(build_messages(question, answer_text, cited)))
    return parsed or {"answers_question": "parse_error", "supported": "parse_error"}
