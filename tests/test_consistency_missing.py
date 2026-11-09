import json

from src.consistency.missing import check, heading_verdict, parse, render_sheet
from src.consistency.requirements import DOC_TYPES, REQUIREMENTS, applicable
from src.consistency.structure import Structure, Variant, apply_variant, chapter_owners, targets
from src.ingestion.chunks import Chunk


def chunk(i, section):
    return Chunk(chunk_id=f"D:{i}", doc_id="D", doc_type="reports", section=section, page_start="1", page_end="1", text=f"text of {section}")


SECTIONS = ["Abstract", "1 Introduction", "4 Software Requirement Specifications", "4.8 Interface", "Conclusion", "5 Design", "A.1 Survey"]
CHUNKS = [chunk(i, s) for i, s in enumerate(SECTIONS)]


def structure():
    chapters = {"Introduction": 1, "Software Requirement Specifications": 4, "Design": 5}
    return Structure(("Abstract", *chapters), chapters, chapter_owners(CHUNKS))


def test_requirements_depend_on_project_type():
    rd, dev = {r.key for r in applicable("rd")}, {r.key for r in applicable("dev")}
    assert {"methodology", "experiments"} <= rd and not {"methodology", "experiments"} & dev
    assert set(DOC_TYPES.values()) == {"rd", "dev"} and len({r.key for r in REQUIREMENTS}) == len(REQUIREMENTS)


def test_heading_rules_match_titles_not_numbers():
    by_key = {r.key: r for r in REQUIREMENTS}
    assert heading_verdict(by_key["design"], ["Introduction", "High-Level and Low-Level Design"]) == "present"
    assert heading_verdict(by_key["user_manual"], ["Introduction", "Conclusions"]) == "missing"
    assert by_key["introduction"].matches("Introduction") and not by_key["introduction"].matches("6.1 Introduction")


def test_chapter_owners_keep_closing_section_with_its_chapter():
    owner = chapter_owners(CHUNKS)
    assert owner["D:0"] is None and owner["D:4"] == 4 and owner["D:6"] is None


def test_remove_and_empty_variants():
    s = structure()
    titles, kept = apply_variant(s, CHUNKS, Variant("remove", "Software Requirement Specifications"))
    assert "Software Requirement Specifications" not in titles
    assert [c.section for c in kept] == ["Abstract", "1 Introduction", "5 Design", "A.1 Survey"]
    titles, kept = apply_variant(s, CHUNKS, Variant("empty", "Software Requirement Specifications"))
    assert "Software Requirement Specifications" in titles and len(kept) == 4
    titles, kept = apply_variant(s, CHUNKS, Variant("remove", "Abstract"))
    assert "Abstract" not in titles and len(kept) == len(CHUNKS) - 1
    assert apply_variant(s, CHUNKS, Variant()) == (list(s.titles), CHUNKS)


def test_targets_skip_pdf_only_and_missing_titles():
    found = targets(structure(), CHUNKS, applicable("dev"))
    assert found["abstract"] == "Abstract" and found["srs"] == "Software Requirement Specifications"
    assert "references" not in found and "user_manual" not in found


def test_parse_verdict():
    assert parse('{"verdict": "partial", "evidence": [1, "x", 3], "reason": "thin"}') == {"verdict": "partial", "evidence": [1, 3], "reason": "thin"}
    assert parse('{"verdict": "maybe"}') is None and parse("no json") is None


class FakeLLM:
    def __init__(self, verdict):
        self.verdict, self.prompts = verdict, []

    def complete(self, messages):
        self.prompts.append(messages[-1]["content"])
        return json.dumps({"verdict": self.verdict, "evidence": [1], "reason": "ok"})


def test_check_uses_chapter_passages_and_cites_evidence(index):
    title = "Proposed Approach and Methodology"
    s = Structure((title,), {title: 1}, {c.chunk_id: (1 if c.section == "1 Methods" else None) for c in index.chunks})
    req = next(r for r in REQUIREMENTS if r.key == "methodology")
    llm = FakeLLM("present")
    out = check(req, "A", [title], s, [c for c in index.chunks if c.doc_id == "A"], index, None, llm)
    assert out["verdict"] == "present" and out["heading"] == "present" and out["evidence"] == ["1 Methods, p. 1"]
    assert "Sections inside the matching chapter: 1 Methods" in llm.prompts[0]


def test_check_references_needs_no_llm(index):
    req = next(r for r in REQUIREMENTS if r.key == "references")
    s = Structure(("Introduction", "References"), {}, {})
    assert check(req, "A", ["Introduction", "References"], s, [], index, None, None)["verdict"] == "present"
    assert check(req, "A", ["Introduction"], s, [], index, None, None)["verdict"] == "missing"


def test_render_sheet_counts_gaps():
    rows = [{"doc": "F25_044", "requirement": "abstract", "verdict": "present", "reason": "", "evidence": ["Abstract, p. 1"]},
            {"doc": "F25_044", "requirement": "conclusion", "verdict": "missing", "reason": "none", "evidence": []}]
    md = render_sheet("F25_044", rows)
    assert "2 required parts checked. 1 are missing or partial." in md and "**Conclusion and future work**: missing (none)" in md
