from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.config import settings
from src.consistency import contradictions, missing, promises
from src.consistency.requirements import DOC_STAGES, DOC_TYPES, applicable
from src.consistency.structure import Structure, load_structure, pdf_path
from src.ingestion.chunks import Chunk
from src.ingestion.ingest import ingest_file
from src.panel.generate import generate_questions
from src.panel.run_panel import NOT_COVERED, panel_status, render_sheet
from src.qa.answer import answer

UPLOAD_DIR = settings.cache_dir / "uploads"
SUFFIXES = {".pdf", ".md"}


@dataclass
class Doc:
    id: str
    chunks: list[Chunk]
    type: str = "dev"
    stage: int = 2
    path: str | None = None
    demo: bool = False
    structure: Structure | None = field(default=None, repr=False)

    @property
    def is_pdf(self) -> bool:
        return bool(self.path) and self.path.lower().endswith(".pdf")


def safe_name(name: str) -> str:
    stem = Path(name).stem
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "document"
    return clean + Path(name).suffix.lower()


def save_upload(name: str, data: bytes) -> Path:
    suffix = Path(name).suffix.lower()
    if suffix not in SUFFIXES:
        raise ValueError("Only PDF and Markdown files are supported.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / safe_name(name)
    path.write_bytes(data)
    return path


def read_upload(name: str, data: bytes, doc_type: str, stage: int) -> Doc:
    path = save_upload(name, data)
    chunks, _ = ingest_file(path)
    if not chunks:
        raise ValueError("No text could be read from this file.")
    return Doc(id=chunks[0].doc_id, chunks=chunks, type=doc_type, stage=stage, path=str(path))


def demo_docs(index) -> list[Doc]:
    out = []
    for doc_id, doc_type in DOC_TYPES.items():
        chunks = [c for c in index.chunks if c.doc_id == doc_id]
        if chunks:
            out.append(Doc(id=doc_id, chunks=chunks, type=doc_type, stage=DOC_STAGES[doc_id], path=pdf_path(doc_id), demo=True))
    return out


def is_refusal(text: str) -> bool:
    return panel_status(text) == NOT_COVERED


def ask(question: str, index, doc: Doc, llm, reranker) -> dict:
    ans = answer(question, index, llm, [doc.id], allow_refusal=True, reranker=reranker)
    refused = is_refusal(ans.text)
    shown = ans.hits if refused else [ans.hits[c.n - 1] for c in ans.citations if c.n <= len(ans.hits)]
    return {
        "text": ans.text,
        "refused": refused,
        "sources": [str(c) for c in ans.citations],
        "passages": [{"label": f"{h.chunk.section}" + (f", p. {h.chunk.page_start}" if h.chunk.page_start else ""), "text": h.chunk.text} for h in shown],
    }


def panel(index, doc: Doc, llm, reranker, progress=None) -> list[dict]:
    questions = generate_questions(index, llm, reranker, doc.id)
    records = []
    for n, (category, question) in enumerate(questions, 1):
        ans = answer(question, index, llm, [doc.id], prompt="panel", reranker=reranker)
        records.append({"id": f"{doc.id}-{n:02d}", "doc": doc.id, "category": category, "question": question, "status": panel_status(ans.text),
                        "answer": ans.text, "sources": [str(c) for c in ans.citations]})
        if progress:
            progress(n, len(questions), question)
    return records


def panel_markdown(doc: Doc, records: list[dict]) -> str:
    return render_sheet(doc.id, records)


def structure_of(doc: Doc) -> Structure:
    if doc.structure is None:
        doc.structure = load_structure(doc.id, doc.chunks, doc.path if doc.is_pdf else None)
    return doc.structure


def required_parts(doc: Doc, index, reranker, llm, progress=None) -> list[dict]:
    return missing.check_doc(doc.id, doc.type, doc.stage, doc.chunks, structure_of(doc), index, reranker, llm, progress)


def promise_check(doc: Doc, index, reranker, llm, progress=None) -> list[dict]:
    return promises.promises_for(doc.id, doc.type, doc.chunks, structure_of(doc), index, reranker, llm, progress=progress)


def contradiction_check(doc: Doc, index, llm, progress=None) -> list[dict]:
    return contradictions.contradictions_for(doc.id, doc.type, doc.chunks, index.embedder, llm, progress)


def saved_rows(kind: str, doc_id: str) -> list[dict] | None:
    """Results saved by the command-line runs, shown instantly for the demo reports."""
    path = settings.eval_dir / "results" / f"{kind}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mine = [r for r in rows if r.get("doc") == doc_id and r.get("variant", "intact") == "intact"]
    return mine or None


def required_table(doc: Doc, rows: list[dict]) -> list[dict]:
    names = {r.key: r.name for r in applicable(doc.type, doc.stage)}
    return [{"required part": names.get(r["requirement"], r["requirement"]), "verdict": r["verdict"], "why": r["reason"],
             "evidence": "; ".join(r["evidence"])} for r in rows]


def promise_table(rows: list[dict]) -> list[dict]:
    order = {"contradicted": 0, "not_shown": 1, "shown": 2}
    return [{"verdict": r["verdict"], "promise": r["promise"], "made in": r["section"] + (f", p. {r['page']}" if r.get("page") else ""),
             "why": r["reason"], "evidence": "; ".join(r["evidence"])} for r in sorted(rows, key=lambda r: order.get(r["verdict"], 3))]


def conflict_table(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r["verdict"] != "confirmed":
            continue
        a, b = r["a"], r["b"]
        out.append({"statement A": f"{a['quote']} ({a['section']}, p. {a['page']})", "statement B": f"{b['quote']} ({b['section']}, p. {b['page']})", "why": r["reason"]})
    return out


def summary(rows: list[dict], kind: str) -> str:
    n = len(rows)
    if kind == "required":
        bad = sum(r["verdict"] != "present" for r in rows)
        return f"{n} required parts checked, {bad} missing or partial."
    if kind == "promises":
        bad = sum(r["verdict"] != "shown" for r in rows)
        return f"{n} promises found, {bad} not shown or contradicted."
    bad = sum(r["verdict"] == "confirmed" for r in rows)
    return f"{n} statement pairs compared, {bad} conflicts confirmed."
