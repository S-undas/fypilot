from __future__ import annotations

import glob
import re
from dataclasses import dataclass

from src.config import settings
from src.ingestion.chunks import Chunk

FRONT = ("Abstract", "Executive Summary")
REF_LINE = re.compile(r"^(\d+\.?\s*)?(references|bibliography)$", re.I)
SKIP_TITLE = re.compile(r"^(list of|appendix)", re.I)
APPENDIX = re.compile(r"^[A-Z]\.\d")
CHAPTER_NUM = re.compile(r"^(\d+)(?:\.\d+)*\s")


@dataclass(frozen=True)
class Structure:
    titles: tuple[str, ...]
    chapters: dict[str, int]
    owner: dict[str, int | None]


@dataclass(frozen=True)
class Variant:
    kind: str = "intact"
    target: str | None = None


def chapter_of(chunk: Chunk) -> int | None:
    m = CHAPTER_NUM.match(chunk.section)
    return int(m.group(1)) if m else None


def chapter_owners(chunks: list[Chunk]) -> dict[str, int | None]:
    """Chapter of each chunk; unnumbered sections inside a chapter (a closing 'Conclusion') belong to the chapter before them."""
    cur: int | None = None
    out: dict[str, int | None] = {}
    for c in chunks:
        n = chapter_of(c)
        if n is not None:
            cur = n
        elif c.section in FRONT:
            cur = None
        elif APPENDIX.match(c.section):
            cur = None
        out[c.chunk_id] = n if n is not None else cur
    return out


def pdf_path(doc_id: str) -> str | None:
    found = glob.glob(str(settings.raw_dir / "**" / f"{doc_id}*.pdf"), recursive=True)
    return found[0] if found else None


def load_structure(doc_id: str, chunks: list[Chunk]) -> Structure:
    import pymupdf

    path = pdf_path(doc_id)
    sections = {c.section for c in chunks}
    titles = [t for t in FRONT if t in sections]
    chapters: dict[str, int] = {}
    has_ref = False
    if path:
        doc = pymupdf.open(path)
        names = [t[1].strip() for t in doc.get_toc() if t[0] == 1 and not SKIP_TITLE.match(t[1].strip())]
        chapters = {name: i for i, name in enumerate(names, 1)}
        half = len(doc) // 2
        has_ref = any(REF_LINE.match(line.strip()) for i, page in enumerate(doc) if i >= half for line in page.get_text().splitlines())
    return Structure(tuple(titles + list(chapters) + (["References"] if has_ref else [])), chapters, chapter_owners(chunks))


def apply_variant(structure: Structure, chunks: list[Chunk], variant: Variant) -> tuple[list[str], list[Chunk]]:
    titles, kept = list(structure.titles), chunks
    t = variant.target
    if variant.kind == "intact" or t is None:
        return titles, kept
    if t in structure.chapters:
        n = structure.chapters[t]
        kept = [c for c in chunks if structure.owner.get(c.chunk_id) != n]
    else:
        kept = [c for c in chunks if c.section != t]
    if variant.kind == "remove":
        titles = [x for x in titles if x != t]
    return titles, kept


def targets(structure: Structure, chunks: list[Chunk], requirements) -> dict[str, str]:
    """Requirement key -> outline title that can be removed or emptied (needs chunks to act on)."""
    out: dict[str, str] = {}
    for r in requirements:
        if r.pdf_only:
            continue
        for title in structure.titles:
            if r.matches(title) and (title in structure.chapters or title in FRONT):
                out[r.key] = title
                break
    return out


def chunks_in(structure: Structure, chunks: list[Chunk], pattern: str) -> list[Chunk]:
    """Chunks of the chapters whose title matches the pattern, in document order."""
    nums = {n for t, n in structure.chapters.items() if re.search(pattern, t, re.I)}
    return [c for c in chunks if structure.owner.get(c.chunk_id) in nums]


def apply_edits(chunks: list[Chunk], edits: dict[str, str] | None) -> list[Chunk]:
    """Copies of the chunks with replaced text, for planted-mistake tests. The index itself is never changed."""
    if not edits:
        return chunks
    return [c.model_copy(update={"text": edits[c.chunk_id]}) if c.chunk_id in edits else c for c in chunks]
