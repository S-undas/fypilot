from __future__ import annotations

import json
import re

from src.config import settings
from src.panel.categories import CATEGORIES
from src.retrieval.rerank import retrieve

MIN_WORDS, MAX_WORDS = 5, 40
SYSTEM = "You help students prepare for a final year project defense. You write the questions an evaluation panel would ask."
STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was", "were", "for", "on", "your", "you", "do", "does",
        "did", "how", "what", "why", "which", "this", "that", "with", "it", "be", "can", "would", "could"}


def build_messages(category: str, passages: list[str], n: int) -> list[dict]:
    spec = CATEGORIES[category]
    blocks = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    user = (
        f"Project passages:\n\n{blocks}\n\n"
        f"Category: {category}, meaning {spec['definition']}.\n\n"
        f"Write exactly {n} questions that a panel member would ask this student in this category. Rules:\n"
        "- One sentence of at most 25 words, asking about one thing only. No multi-part questions.\n"
        "- Each question must be specific to this project: mention its actual methods, components, data or claims.\n"
        "- Mix kinds: ask what the student did and why, and challenge at least one choice or result.\n"
        "- Each question must be different from the others.\n"
        "- Do not answer the questions.\n"
        f"Return only a JSON list of {n} strings."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_questions(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for q in items:
        if not isinstance(q, str):
            continue
        q = " ".join(q.split())
        if not MIN_WORDS <= len(q.split()) <= MAX_WORDS:
            continue
        out.append(q if q.endswith("?") else q.rstrip(".") + "?")
    return out


def words(q: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", q.lower()) if w not in STOP}


def similar(a: str, b: str, threshold: float = 0.6) -> bool:
    wa, wb = words(a), words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= threshold


def generate_questions(index, llm, reranker, doc: str, n: int | None = None) -> list[tuple[str, str]]:
    """(category, question) pairs for one document. Near-duplicates across categories are dropped."""
    n = n or settings.panel_questions_per_category
    out: list[tuple[str, str]] = []
    for category, spec in CATEGORIES.items():
        hits = retrieve(index, spec["query"], 6, settings.retrieval_mode, [doc], reranker)
        text = llm.complete(build_messages(category, [h.chunk.text for h in hits], n))
        kept = 0
        for q in parse_questions(text):
            if kept >= n or any(similar(q, prev) for _, prev in out):
                continue
            out.append((category, q))
            kept += 1
    return out
