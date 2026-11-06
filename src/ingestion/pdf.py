from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from src.ingestion.chunks import Chunk, build_chunks

NUM = re.compile(r"^((?:[A-Z]\.)?\d+(?:\.\d+)*)\.?\s+(\S.*)$")
CHAPTER = re.compile(r"^chapter\s+(\d+|[A-Z])\.?\s+(\S.*)$", re.I)
KEEP_FRONT = {"abstract", "executive summary"}
SKIP_FRONT = {
    "table of contents",
    "list of figures",
    "list of tables",
    "anti-plagiarism declaration",
    "author's declaration",
    "authors' declaration",
    "declaration",
}
MIN_OUTLINE_MATCHES = 10


@dataclass
class Row:
    text: str
    size: float
    bold: bool
    y: float
    block: int


@dataclass
class Heading:
    level: int
    number: str
    title: str


def norm(s: str) -> str:
    s = re.sub(r"[‘’�]", "'", s.lower().strip())
    return re.sub(r"\s+", " ", s)


def norm_title(s: str) -> str:
    s = re.sub(r"^chapter\s+(?:\d+|[a-z])\.?\s+", "", norm(s))
    while True:
        t = re.sub(r"^(?:[a-z]\.)?\d+(?:\.\d+)*\.?\s+", "", s)
        if t == s:
            return s
        s = t


def read_rows(page: pymupdf.Page) -> list[Row]:
    rows: list[Row] = []
    for bi, b in enumerate(page.get_text("dict")["blocks"]):
        if b["type"] != 0:
            continue
        groups: list[dict] = []
        for line in b["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in line["spans"]).strip()
            y, x = line["bbox"][1], line["bbox"][0]
            if groups and abs(y - groups[-1]["y"]) < 3 and x > groups[-1]["x"]:
                groups[-1]["text"] += " " + text
                groups[-1]["x"] = x
            else:
                s0 = spans[0]
                bold = "bold" in s0["font"].lower() or "medi" in s0["font"].lower() or bool(s0["flags"] & 16)
                groups.append({"y": y, "x": x, "text": text, "size": s0["size"], "bold": bold})
        rows += [Row(g["text"], g["size"], g["bold"], g["y"], bi) for g in groups]
    return rows


def strip_running(pages: list[list[Row]], heights: list[float]) -> list[list[Row]]:
    def in_margin(r: Row, h: float) -> bool:
        return r.y < 0.07 * h or r.y > 0.93 * h

    def key(t: str) -> str:
        return re.sub(r"\d+", "#", t.lower())

    counts: Counter[str] = Counter()
    for rows, h in zip(pages, heights):
        counts.update({key(r.text) for r in rows if in_margin(r, h)})
    out = []
    for rows, h in zip(pages, heights):
        keep = []
        for r in rows:
            if in_margin(r, h):
                t = r.text.strip()
                if re.fullmatch(r"[\divxlc]+", t.lower()) or CHAPTER.match(t):
                    continue
                if counts[key(t)] >= 2 and len(pages) >= 3:
                    continue
            keep.append(r)
        out.append(keep)
    return out


def split_number(text: str) -> tuple[str, str]:
    m = CHAPTER.match(text.strip())
    if m:
        return m.group(1), m.group(2)
    m = NUM.match(text.strip())
    return (m.group(1), m.group(2)) if m else ("", text.strip())


def outline_headings(doc: pymupdf.Document, pages: list[list[Row]]) -> tuple[dict[tuple[int, int], Heading], int, int]:
    found: dict[tuple[int, int], Heading] = {}
    cursor = (0, 0)
    total = matched = 0
    for level, title, pno in doc.get_toc(simple=True):
        t = norm_title(title)
        if t in SKIP_FRONT or not t:
            continue
        total += 1
        hit = None
        for pi in range(max(cursor[0], pno - 2), min(pno + 1, len(pages))):
            start = cursor[1] if pi == cursor[0] else 0
            for ri in range(start, len(pages[pi])):
                span = match_rows(pages[pi], ri, t)
                if span:
                    hit = (pi, ri, span)
                    break
            if hit:
                break
        if not hit:
            continue
        pi, ri, span = hit
        number, _ = split_number(pages[pi][ri].text)
        found[(pi, ri)] = Heading(level, number, title.strip())
        for k in range(1, span):
            found[(pi, ri + k)] = Heading(0, "", "")
        cursor = (pi, ri + span)
        matched += 1
    return found, matched, total


def match_rows(rows: list[Row], ri: int, title: str) -> int:
    """Number of rows (1 to 3) from ri that spell the heading title, else 0."""
    text = ""
    for k in range(3):
        if ri + k >= len(rows):
            return 0
        text = f"{text} {rows[ri + k].text}".strip()
        n = norm_title(text)
        if n == title:
            return k + 1
        if not title.startswith(n) or (k == 0 and len(n) < 12):
            return 0
    return 0


def font_headings(pages: list[list[Row]]) -> dict[tuple[int, int], Heading]:
    sizes: Counter[float] = Counter()
    for rows in pages:
        for r in rows:
            sizes[round(r.size, 1)] += len(r.text)
    if not sizes:
        return {}
    body = sizes.most_common(1)[0][0]
    found = {}
    for pi, rows in enumerate(pages):
        for ri, r in enumerate(rows):
            if r.bold and 1.25 * body <= r.size <= 1.5 * body and len(r.text) <= 80 and not r.text.endswith(":"):
                number, title = split_number(r.text)
                found[(pi, ri)] = Heading(1 + number.count(".") if number else 1, number, title)
    return found


def line_gap(rows: list[Row]) -> float:
    gaps = sorted(b.y - a.y for a, b in zip(rows, rows[1:]) if b.y - a.y > 1)
    return gaps[len(gaps) // 2] if gaps else 0.0


def to_paragraphs(rows: list[Row], gap: float) -> list[str]:
    paras: list[str] = []
    cur: list[str] = []
    prev_y = None
    for r in rows:
        t = re.sub(r"^[-•●◦▪]\s*", "- ", r.text)
        new_para = t.startswith("- ") or (prev_y is not None and gap and abs(r.y - prev_y) > 1.4 * gap)
        if cur and new_para:
            paras.append(" ".join(cur))
            cur = []
        if cur and cur[-1].endswith("-") and t[:1].islower():
            cur[-1] = cur[-1][:-1] + t
        else:
            cur.append(t)
        prev_y = r.y
    if cur:
        paras.append(" ".join(cur))
    return [re.sub(r"\s+", " ", p).strip() for p in paras if p.strip()]


def parse_pdf(path: Path, doc_id: str, doc_type: str, max_chars: int, overlap: int, min_chars: int = 0) -> tuple[list[Chunk], dict]:
    doc = pymupdf.open(path)
    raw = [read_rows(p) for p in doc]
    heights = [p.rect.height for p in doc]
    labels = [p.get_label() or str(i + 1) for i, p in enumerate(doc)]
    pages = strip_running(raw, heights)

    heads, matched, total = outline_headings(doc, pages)
    source = "outline"
    if matched < MIN_OUTLINE_MATCHES:
        heads, source = font_headings(pages), "font"
        matched, total = len(heads), len(heads)
    skip_rows = {k for k, v in heads.items() if v.level == 0}
    heads = {k: v for k, v in heads.items() if v.level > 0}
    first_body = min((pi for (pi, _), h in heads.items() if h.level == 1), default=0)

    sections: dict[str, list[tuple[str, int | None]]] = {}
    order: list[str] = []
    stack: dict[int, str] = {}
    front: str | None = None

    def add(section: str, text: str, page: int) -> None:
        if section not in sections:
            sections[section] = []
            order.append(section)
        sections[section].append((text, page))

    def flush_buf(buf: list[Row], gap: float, pi: int) -> None:
        current = stack.get(2) or stack.get(1)
        if current:
            for p in to_paragraphs(buf, gap):
                add(current, p, pi)
        buf.clear()

    for pi, rows in enumerate(pages):
        firsts = [norm(r.text) for r in raw[pi][:3]]
        if source == "outline" and pi < first_body:
            if any(f in SKIP_FRONT for f in firsts):
                front = None
                continue
            kept = next((f for f in firsts if f in KEEP_FRONT), None)
            if kept:
                front = kept.title()
            if front is None:
                continue
            body = [r for r in rows if norm(r.text) not in KEEP_FRONT]
            for p in to_paragraphs(body, line_gap(rows)):
                add(front, p, pi)
            continue
        if any(f in SKIP_FRONT for f in firsts):
            continue
        buf: list[Row] = []
        gap = line_gap(rows)
        for ri, r in enumerate(rows):
            if (pi, ri) in skip_rows:
                continue
            h = heads.get((pi, ri))
            if h is None:
                buf.append(r)
                continue
            flush_buf(buf, gap, pi)
            label = f"{h.number} {h.title}".strip()
            if h.level <= 2:
                stack = {k: v for k, v in stack.items() if k < h.level}
                stack[h.level] = label
            else:
                current = stack.get(2) or stack.get(1)
                if current:
                    add(current, label, pi)
        flush_buf(buf, gap, pi)

    ordered = [(s, sections[s]) for s in order if sections.get(s)]
    chunks = build_chunks(doc_id, doc_type, ordered, labels, max_chars, overlap, min_chars)
    stats = {"pages": len(doc), "source": source, "headings_matched": matched, "headings_total": total}
    return chunks, stats
