from __future__ import annotations

import re

CITATION = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]|【[^】]*】")
LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
REFUSAL = re.compile(
    r"(do(es)? not|don't|doesn't|cannot|can't|no (information|mention|specific))[^.]{0,60}"
    r"(state|report|specify|provide|mention|contain|include|find|give|record|identify|list)"
    r"|\b(is|are|was|were) not (stated|reported|given|specified|provided|mentioned|listed|available)"
    r"|\bno [\w\s-]{0,40}\b(is|are|was|were) (given|reported|stated|listed|provided)\b",
    re.I,
)
SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
MIN_WORDS = 4


def split_claims(answer: str) -> list[str]:
    """Factual sentences of an answer: citations and markdown removed, refusal sentences and list lead-ins dropped."""
    text = SPACE_BEFORE_PUNCT.sub(r"\1", CITATION.sub("", answer).replace("**", "").replace("`", ""))
    claims = []
    for line in text.splitlines():
        line = LIST_MARK.sub("", line).strip()
        for sentence in SENTENCE_END.split(line):
            sentence = sentence.strip()
            if len(sentence.split()) >= MIN_WORDS and not sentence.endswith(":") and not REFUSAL.search(sentence):
                claims.append(sentence)
    return claims
