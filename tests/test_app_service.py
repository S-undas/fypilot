import json

import pytest

from app import service
from app.service import Doc, ask, conflict_table, is_refusal, panel, promise_table, read_upload, required_table, safe_name, saved_rows, summary
from src.config import settings


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.last_ref = reply, None

    def complete(self, messages, **kw):
        return self.reply


def test_safe_name_keeps_extension_and_strips_odd_characters():
    assert safe_name("My Report (final).PDF") == "My_Report_final.pdf"
    assert safe_name("../../etc/passwd.md") == "passwd.md"


def test_upload_rejects_unsupported_types_and_reads_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "UPLOAD_DIR", tmp_path / "uploads")
    with pytest.raises(ValueError, match="PDF and Markdown"):
        service.save_upload("notes.docx", b"x")
    text = "# Intro\n\n" + "The system detects objects with a camera and a small model. " * 4 + "\n\n## Budget\n\n" + "The hardware budget is five thousand rupees in total. " * 3
    doc = read_upload("My Notes.md", text.encode(), "dev", 1)
    assert doc.id == "My_Notes" and doc.stage == 1 and doc.chunks and not doc.is_pdf and doc.path.endswith("My_Notes.md")
    with pytest.raises(ValueError, match="No text"):
        read_upload("empty.md", b"# Title only", "dev", 2)


def test_demo_docs_lists_only_known_documents_in_the_index(index, monkeypatch):
    monkeypatch.setattr(service, "DOC_TYPES", {"A": "rd", "Z": "dev"})
    monkeypatch.setattr(service, "DOC_STAGES", {"A": 1, "Z": 2})
    docs = service.demo_docs(index)
    assert [(d.id, d.type, d.stage, d.demo) for d in docs] == [("A", "rd", 1, True)]


def test_refusal_detection():
    assert is_refusal("I cannot find it in the documents.") and is_refusal("NOT COVERED")
    assert not is_refusal("The budget is 5000 rupees [1].")


def test_ask_returns_answer_sources_and_only_cited_passages(index):
    doc = Doc(id="B", chunks=[c for c in index.chunks if c.doc_id == "B"])
    out = ask("what is the hardware budget", index, doc, FakeLLM("The budget covers hardware [1]."), None)
    assert not out["refused"] and out["sources"] and out["sources"][0].startswith("[1] B,")
    assert len(out["passages"]) == 1 and "budget" in out["passages"][0]["text"]
    refused = ask("what is the hardware budget", index, doc, FakeLLM("I cannot find it in the documents."), None)
    assert refused["refused"] and refused["sources"] == [] and len(refused["passages"]) > 0


def test_panel_records_status_and_progress(index, monkeypatch):
    monkeypatch.setattr(service, "generate_questions", lambda *a, **k: [("ethics", "Why this choice?"), ("novelty", "What is new here?")])
    seen = []
    doc = Doc(id="A", chunks=[c for c in index.chunks if c.doc_id == "A"])
    records = panel(index, doc, FakeLLM("NOT COVERED"), None, lambda n, total, msg: seen.append((n, total)))
    assert [r["status"] for r in records] == ["NOT COVERED", "NOT COVERED"] and seen == [(1, 2), (2, 2)]
    assert "(ethics) Why this choice?" in service.panel_markdown(doc, records)


def test_saved_rows_filters_by_document_and_intact_variant(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    rows = [{"doc": "A", "variant": "intact", "x": 1}, {"doc": "A", "variant": "remove", "x": 2}, {"doc": "B", "x": 3}]
    (results / "missing_check.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(settings, "eval_dir", tmp_path)
    assert saved_rows("missing_check", "A") == [rows[0]] and saved_rows("missing_check", "Q") is None and saved_rows("nothing", "A") is None


def test_tables_and_summaries():
    doc = Doc(id="D", chunks=[], type="dev", stage=2)
    rows = [{"requirement": "abstract", "verdict": "present", "reason": "ok", "evidence": ["Abstract, p. 1"]},
            {"requirement": "user_manual", "verdict": "missing", "reason": "none", "evidence": []}]
    assert required_table(doc, rows)[1] == {"required part": "User manual", "verdict": "missing", "why": "none", "evidence": ""}
    assert summary(rows, "required") == "2 required parts checked, 1 missing or partial."
    promises = [{"verdict": "shown", "promise": "A", "section": "1", "page": "2", "reason": "", "evidence": []},
                {"verdict": "contradicted", "promise": "B", "section": "2", "page": "", "reason": "x", "evidence": ["p"]}]
    table = promise_table(promises)
    assert [r["promise"] for r in table] == ["B", "A"] and table[0]["made in"] == "2" and summary(promises, "promises") == "2 promises found, 1 not shown or contradicted."
    a, b = {"quote": "0.4", "section": "5", "page": "1"}, {"quote": "25%", "section": "8", "page": "3"}
    pairs = [{"verdict": "confirmed", "a": a, "b": b, "reason": "differ"}, {"verdict": "not_confirmed", "a": a, "b": b, "reason": "same"}]
    assert len(conflict_table(pairs)) == 1 and summary(pairs, "conflicts") == "2 statement pairs compared, 1 conflicts confirmed."
