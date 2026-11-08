from __future__ import annotations

import csv
import random
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from src.ingestion.chunks import Chunk

KINDS = ("answerable", "near_miss", "misleading", "off_topic")


@dataclass
class EvalItem:
    id: str
    doc: str
    kind: str
    question: str
    gold_answer: str = ""
    gold_section: str = ""
    gold_page: str = ""
    evidence_quote: str = ""
    search_terms: str = ""
    note: str = ""
    split: str = ""
    verified: str = ""
    reviewer_note: str = ""

    @property
    def answerable(self) -> bool:
        return self.kind == "answerable"


COLUMNS = [f.name for f in fields(EvalItem)]


def load_items(path: Path) -> list[EvalItem]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [EvalItem(**{c: (row.get(c) or "").strip() for c in COLUMNS}) for row in csv.DictReader(f)]


def save_items(items: list[EvalItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(asdict(i) for i in items)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def page_in(chunk: Chunk, page: str) -> bool:
    if not page:
        return chunk.page_start is None
    lo, _, hi = page.partition("-")
    hi = hi or lo
    for p in (lo, hi):
        if p in (chunk.page_start, chunk.page_end):
            continue
        if all(x and x.isdigit() for x in (p, chunk.page_start, chunk.page_end)):
            if int(chunk.page_start) <= int(p) <= int(chunk.page_end):
                continue
        return False
    return True


def validate(items: list[EvalItem], chunks: list[Chunk]) -> list[str]:
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    problems: list[str] = []
    seen: set[str] = set()
    for it in items:
        tag = f"{it.id}"
        if it.id in seen:
            problems.append(f"{tag}: duplicate id")
        seen.add(it.id)
        if it.kind not in KINDS:
            problems.append(f"{tag}: unknown kind '{it.kind}'")
            continue
        if not it.question:
            problems.append(f"{tag}: empty question")
        if it.doc not in by_doc:
            problems.append(f"{tag}: unknown doc '{it.doc}'")
            continue
        doc_chunks = by_doc[it.doc]
        if it.answerable and not it.gold_answer:
            problems.append(f"{tag}: answerable item has no gold answer")
        if not it.answerable and it.gold_answer:
            problems.append(f"{tag}: unanswerable item has a gold answer")
        if it.kind != "off_topic":
            problems += check_evidence(it, doc_chunks)
        if not it.answerable:
            if not it.search_terms:
                problems.append(f"{tag}: unanswerable item needs search_terms")
            elif it.kind == "off_topic":
                text = norm(" ".join(c.text for c in doc_chunks))
                for term in re.split(r"\s*;\s*", it.search_terms):
                    if norm(term) and re.search(rf"\b{re.escape(norm(term))}\b", text):
                        problems.append(f"{tag}: off_topic term '{term}' appears in {it.doc}")
    return problems


def check_evidence(it: EvalItem, doc_chunks: list[Chunk]) -> list[str]:
    if not (it.gold_section and it.evidence_quote):
        return [f"{it.id}: needs gold_section and evidence_quote"]
    where = [c for c in doc_chunks if norm(c.section) == norm(it.gold_section)]
    if not where:
        return [f"{it.id}: no section '{it.gold_section}' in {it.doc}"]
    hits = [c for c in where if norm(it.evidence_quote) in norm(c.text)]
    if not hits:
        return [f"{it.id}: quote not found in section '{it.gold_section}'"]
    if not any(page_in(c, it.gold_page) for c in hits):
        pages = ", ".join(f"{c.page_start}-{c.page_end}" for c in hits)
        return [f"{it.id}: quote is on page {pages}, not {it.gold_page or 'none'}"]
    return []


def assign_splits(items: list[EvalItem], seed: int = 13) -> None:
    rng = random.Random(seed)
    for kind in KINDS:
        group = sorted((i for i in items if i.kind == kind), key=lambda i: i.id)
        rng.shuffle(group)
        for n, it in enumerate(group):
            it.split = "dev" if n % 2 == 0 else "test"
