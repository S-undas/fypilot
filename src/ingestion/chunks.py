from __future__ import annotations

import re

from pydantic import BaseModel

SPLIT = re.compile(r"(?<=[.!?])\s+|\n")


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_type: str
    section: str
    page_start: str | None = None
    page_end: str | None = None
    text: str


def split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    size = 0
    for piece in (p for p in SPLIT.split(text) if p):
        while len(piece) > max_chars:
            if cur:
                out.append(" ".join(cur))
                cur, size = [], 0
            out.append(piece[:max_chars])
            piece = piece[max_chars:]
        if cur and size + len(piece) + 1 > max_chars:
            out.append(" ".join(cur))
            tail: list[str] = []
            n = 0
            for s in reversed(cur):
                if n + len(s) > overlap:
                    break
                tail.insert(0, s)
                n += len(s)
            cur, size = (tail, n) if n + len(piece) + 1 <= max_chars else ([], 0)
        cur.append(piece)
        size += len(piece) + 1
    if cur:
        out.append(" ".join(cur))
    return out


def pack(
    paras: list[tuple[str, int | None]], max_chars: int, overlap: int
) -> list[tuple[str, int | None, int | None]]:
    """Greedy-pack paragraphs into chunks; returns (text, first_page, last_page)."""
    out: list[tuple[str, int | None, int | None]] = []
    buf: list[str] = []
    pages: list[int] = []
    size = 0

    def flush() -> None:
        nonlocal buf, pages, size
        if buf:
            out.append(("\n".join(buf), min(pages) if pages else None, max(pages) if pages else None))
        buf, pages, size = [], [], 0

    for text, page in paras:
        pieces = split_long(text, max_chars, overlap) if len(text) > max_chars else [text]
        for piece in pieces:
            if buf and size + len(piece) + 1 > max_chars:
                flush()
            buf.append(piece)
            if page is not None:
                pages.append(page)
            size += len(piece) + 1
    flush()
    return out


def build_chunks(
    doc_id: str,
    doc_type: str,
    sections: list[tuple[str, list[tuple[str, int | None]]]],
    labels: list[str] | None,
    max_chars: int,
    overlap: int,
    min_chars: int = 0,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, paras in sections:
        for text, p0, p1 in pack(paras, max_chars, overlap):
            if len(text) < min_chars:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{len(chunks):04d}",
                    doc_id=doc_id,
                    doc_type=doc_type,
                    section=section,
                    page_start=labels[p0] if labels and p0 is not None else None,
                    page_end=labels[p1] if labels and p1 is not None else None,
                    text=text,
                )
            )
    return chunks
