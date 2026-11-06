from __future__ import annotations

import re
from pathlib import Path

from src.ingestion.chunks import Chunk, build_chunks

HEADING = re.compile(r"^(#{2,4})\s+(.*\S)\s*$")
LIST_ITEM = re.compile(r"^\s*(?:[-*]|\d+\.)\s")


def parse_markdown(path: Path, doc_id: str, doc_type: str, max_chars: int, overlap: int, min_chars: int = 0) -> list[Chunk]:
    sections: list[tuple[str, list[tuple[str, int | None]]]] = []
    heads = ["", "", ""]
    para: list[str] = []

    def label() -> str:
        return " > ".join(h for h in heads if h) or "Preamble"

    def add_section_if_new() -> list[tuple[str, int | None]]:
        if not sections or sections[-1][0] != label():
            sections.append((label(), []))
        return sections[-1][1]

    def flush() -> None:
        nonlocal para
        if para:
            add_section_if_new().append((" ".join(para) if not para[0].startswith("|") else "\n".join(para), None))
        para = []

    for line in path.read_text(encoding="utf-8").splitlines():
        m = HEADING.match(line)
        if m:
            flush()
            depth = len(m.group(1)) - 2
            heads[depth:] = [m.group(2)] + [""] * (2 - depth)
        elif line.startswith("# ") or line.startswith("Source:") or line.startswith("Extracted verbatim"):
            flush()
        elif not line.strip():
            flush()
        elif LIST_ITEM.match(line):
            flush()
            para.append(line.strip())
            flush()
        else:
            para.append(line.strip())
    flush()
    sections = [s for s in sections if s[1]]
    return build_chunks(doc_id, doc_type, sections, None, max_chars, overlap, min_chars)
