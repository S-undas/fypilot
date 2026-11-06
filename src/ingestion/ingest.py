from __future__ import annotations

import re
from pathlib import Path

from src.config import settings
from src.ingestion.chunks import Chunk
from src.ingestion.markdown import parse_markdown
from src.ingestion.pdf import parse_pdf


def doc_id_for(path: Path) -> str:
    m = re.match(r"^(F\d\d_\d+)", path.stem)
    return m.group(1) if m else path.stem


def ingest_file(path: Path) -> tuple[list[Chunk], dict]:
    doc_id, doc_type = doc_id_for(path), path.parent.name
    cfg = settings
    if path.suffix == ".pdf":
        return parse_pdf(path, doc_id, doc_type, cfg.chunk_max_chars, cfg.chunk_overlap_chars, cfg.chunk_min_chars)
    if path.suffix == ".md":
        return parse_markdown(path, doc_id, doc_type, cfg.chunk_max_chars, cfg.chunk_overlap_chars, cfg.chunk_min_chars), {}
    raise ValueError(f"unsupported file type: {path}")


def load_chunks(doc_id: str) -> list[Chunk]:
    path = settings.processed_dir / f"{doc_id}.jsonl"
    return [Chunk.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(settings.raw_dir.rglob("*")):
        if path.suffix not in {".pdf", ".md"}:
            continue
        chunks, stats = ingest_file(path)
        out = settings.processed_dir / f"{chunks[0].doc_id}.jsonl"
        out.write_text("\n".join(c.model_dump_json() for c in chunks) + "\n", encoding="utf-8")
        sizes = [len(c.text) for c in chunks]
        print(
            f"{chunks[0].doc_id:22} {len(chunks):4} chunks  "
            f"chars min/avg/max {min(sizes)}/{sum(sizes) // len(sizes)}/{max(sizes)}  {stats}"
        )


if __name__ == "__main__":
    main()
