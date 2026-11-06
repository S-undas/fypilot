from pathlib import Path

import pymupdf

from src.ingestion.chunks import pack, split_long
from src.ingestion.ingest import doc_id_for
from src.ingestion.markdown import parse_markdown
from src.ingestion.pdf import norm_title, parse_pdf, split_number


def make_pdf(path: Path, pages, toc=None, labels=None):
    doc = pymupdf.open()
    for rows in pages:
        page = doc.new_page()
        y = 80
        for text, size, bold in rows:
            page.insert_text((72, y), text, fontsize=size, fontname="hebo" if bold else "helv")
            y += size * 1.7
    if toc:
        doc.set_toc(toc)
    if labels:
        doc.set_page_labels(labels)
    doc.save(path)


def test_split_long_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} ends here." for i in range(60))
    parts = split_long(text, 300, 60)
    assert len(parts) > 1
    assert all(len(p) <= 300 for p in parts)
    assert parts[0].split(". ")[-1] in parts[1]


def test_pack_merges_small_paragraphs_and_tracks_pages():
    out = pack([("a" * 100, 3), ("b" * 100, 4), ("c" * 100, 4)], 250, 20)
    assert len(out) == 2
    assert out[0][1:] == (3, 4)
    assert out[1][1:] == (4, 4)


def test_heading_number_helpers():
    assert split_number("1.2 Intended Audience") == ("1.2", "Intended Audience")
    assert split_number("Chapter 3 Methods") == ("3", "Methods")
    assert split_number("A.1.2 Survey") == ("A.1.2", "Survey")
    assert norm_title("7.3.1 1. Uploading Topologies") == "uploading topologies"


def test_doc_id_from_filename():
    assert doc_id_for(Path("F25_044_blind_aid.pdf")) == "F25_044"
    assert doc_id_for(Path("proposal_x.pdf")) == "proposal_x"


def test_font_fallback_pdf(tmp_path):
    body = [("Some body text for the paragraph. " * 3, 11, False)] * 3
    pages = [
        [("My Project Title", 20, True), ("Group A", 11, False)],
        [("Abstract", 16, True), *body, ("Scope", 16, True), *body],
    ]
    path = tmp_path / "p.pdf"
    make_pdf(path, pages)
    chunks, stats = parse_pdf(path, "p", "proposals", 1500, 150, 20)
    assert stats["source"] == "font"
    assert [c.section for c in chunks] == ["Abstract", "Scope"]
    assert "Group A" not in " ".join(c.text for c in chunks)
    assert chunks[0].page_start == "2"


def test_outline_pdf_skips_front_matter_and_strips_headers(tmp_path):
    body = [("Body sentence about the topic here.", 11, False)] * 4
    pages = [
        [("Project Title", 20, True), ("Student Name 22L-0000", 11, False)],
        [("Abstract", 14, True), ("This abstract explains the project aims.", 11, False)],
        [("Table of Contents", 14, True), ("1.1 Topic ........ 4", 11, False)],
        [("Chapter 1 Introduction", 14, True), *body],
    ]
    toc = [[1, "Introduction", 4]]
    for i in range(1, 12):
        pages.append([(f"1.{i} Topic{i}", 14, True), *body])
        toc.append([2, f"Topic{i}", 4 + i])
    labels = [
        {"startpage": 0, "style": "r", "firstpagenum": 1},
        {"startpage": 3, "style": "D", "firstpagenum": 1},
    ]
    path = tmp_path / "r.pdf"
    make_pdf(path, pages, toc=toc, labels=labels)
    doc = pymupdf.open(path)
    for p in doc:
        if p.number >= 3:
            p.insert_text((72, 30), "CHAPTER 1. INTRODUCTION", fontsize=10)
    doc.saveIncr()
    doc.close()

    chunks, stats = parse_pdf(path, "r", "reports", 1500, 150, 20)
    sections = [c.section for c in chunks]
    assert stats["source"] == "outline" and stats["headings_matched"] == stats["headings_total"] == 12
    assert sections[0] == "Abstract" and chunks[0].page_start == "ii"
    assert "1.3 Topic3" in sections
    topic3 = next(c for c in chunks if c.section == "1.3 Topic3")
    assert topic3.page_start == "4"
    joined = " ".join(c.text for c in chunks)
    assert "Student Name" not in joined and "........" not in joined
    assert "CHAPTER 1. INTRODUCTION" not in joined


def test_markdown_sections_and_lists(tmp_path):
    md = tmp_path / "h.md"
    md.write_text(
        "# Title\n\nSource: x\n\n## Tab\n\nIntro text here that is long enough to keep.\n\n"
        "### Sub\n\n- first item is a complete bullet point\n- second item is another bullet point\n\n"
        "#### Deep\n\nDeep paragraph text that should be kept in its own section.\n",
        encoding="utf-8",
    )
    chunks = parse_markdown(md, "h", "handbook", 1500, 150, 20)
    assert [c.section for c in chunks] == ["Tab", "Tab > Sub", "Tab > Sub > Deep"]
    assert chunks[1].text.count("\n") == 1
    assert all(c.page_start is None for c in chunks)
