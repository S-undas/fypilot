from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.qa.llm import LLM
from src.retrieval.index import Hit, Index

SYSTEM = (
    "You answer questions about final year project documents using only the numbered context passages. "
    "Cite the passages you use with their numbers in plain square brackets, for example [2] or [1][3]. "
    "Use no other citation format. "
    "Do not use outside knowledge."
)
REFUSAL = " If the passages do not contain the answer, say that you cannot find it in the documents."


@dataclass
class Citation:
    n: int
    doc_id: str
    section: str
    pages: str
    chunk_id: str

    def __str__(self) -> str:
        return f"[{self.n}] {self.doc_id}, {self.section}, p. {self.pages}" if self.pages else f"[{self.n}] {self.doc_id}, {self.section}"


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation]
    hits: list[Hit]
    invalid_refs: list[int] = field(default_factory=list)


def pages_of(hit: Hit) -> str:
    c = hit.chunk
    if not c.page_start:
        return ""
    return c.page_start if c.page_start == c.page_end else f"{c.page_start}-{c.page_end}"


def build_messages(question: str, hits: list[Hit], allow_refusal: bool = False) -> list[dict]:
    blocks = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        where = f"{c.doc_id}, {c.section}" + (f", p. {pages_of(h)}" if pages_of(h) else "")
        blocks.append(f"[{i}] ({where})\n{c.text}")
    user = "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM + (REFUSAL if allow_refusal else "")},
        {"role": "user", "content": user},
    ]


CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]|【(\d+)[^】]*】")


def parse_refs(text: str) -> list[int]:
    refs: list[int] = []
    for m in CITE.finditer(text):
        for n in re.split(r"\s*,\s*", m.group(1) or m.group(2)):
            if int(n) not in refs:
                refs.append(int(n))
    return refs


def answer(
    question: str,
    index: Index,
    llm: LLM,
    docs: list[str] | None = None,
    k: int | None = None,
    mode: str | None = None,
    allow_refusal: bool = False,
) -> Answer:
    hits = index.search(question, k, mode, docs)
    text = llm.complete(build_messages(question, hits, allow_refusal))
    citations, invalid = [], []
    for n in parse_refs(text):
        if 1 <= n <= len(hits):
            c = hits[n - 1].chunk
            citations.append(Citation(n, c.doc_id, c.section, pages_of(hits[n - 1]), c.chunk_id))
        else:
            invalid.append(n)
    return Answer(question, text, citations, hits, invalid)
