from src.evaluation.dataset import EvalItem, assign_splits, load_items, page_in, save_items, validate
from src.ingestion.chunks import Chunk


def chunk(section, text, p0="3", p1="4", doc="D1"):
    return Chunk(chunk_id=f"{doc}:{section}", doc_id=doc, doc_type="reports", section=section,
                 page_start=p0, page_end=p1, text=text)


CHUNKS = [
    chunk("2.1 Scope", "The system runs on a Raspberry Pi 4 with a camera module."),
    chunk("2.2 Goals", "We want real time alerts for obstacles."),
]


def good():
    return EvalItem(id="q1", doc="D1", kind="answerable", question="What hardware?",
                    gold_answer="A Raspberry Pi 4", gold_section="2.1 Scope", gold_page="4",
                    evidence_quote="runs on a Raspberry Pi 4")


def test_valid_answerable_passes():
    assert validate([good()], CHUNKS) == []


def test_catches_wrong_quote_section_page_and_doc():
    bad_quote = good(); bad_quote.id = "a"; bad_quote.evidence_quote = "runs on a Jetson"
    bad_section = good(); bad_section.id = "b"; bad_section.gold_section = "9.9 Nope"
    bad_page = good(); bad_page.id = "c"; bad_page.gold_page = "7"
    bad_doc = good(); bad_doc.id = "d"; bad_doc.doc = "NOPE"
    problems = validate([bad_quote, bad_section, bad_page, bad_doc], CHUNKS)
    assert len(problems) == 4
    assert any("quote not found" in p for p in problems)
    assert any("no section" in p for p in problems)
    assert any("not 7" in p for p in problems)
    assert any("unknown doc" in p for p in problems)


def test_off_topic_term_must_be_absent():
    ok = EvalItem(id="o1", doc="D1", kind="off_topic", question="Which GPU?", search_terms="gpu; cuda")
    bad = EvalItem(id="o2", doc="D1", kind="off_topic", question="Which camera?", search_terms="camera module")
    problems = validate([ok, bad], CHUNKS)
    assert problems == ["o2: off_topic term 'camera module' appears in D1"]


def test_unanswerable_rules():
    no_terms = EvalItem(id="n1", doc="D1", kind="off_topic", question="x?")
    with_answer = EvalItem(id="n2", doc="D1", kind="off_topic", question="x?", gold_answer="y", search_terms="zzz")
    problems = validate([no_terms, with_answer], CHUNKS)
    assert "n1: unanswerable item needs search_terms" in problems
    assert "n2: unanswerable item has a gold answer" in problems


def test_near_miss_needs_related_evidence():
    nm = EvalItem(id="m1", doc="D1", kind="near_miss", question="How many cameras?", search_terms="number of cameras",
                  gold_section="2.1 Scope", gold_page="3", evidence_quote="a camera module")
    assert validate([nm], CHUNKS) == []


def test_page_in_handles_ranges_and_roman_labels():
    c = chunk("s", "t", "21", "22")
    assert page_in(c, "21") and page_in(c, "22") and page_in(c, "21-22") and not page_in(c, "23")
    r = chunk("s", "t", "iv", "iv")
    assert page_in(r, "iv") and not page_in(r, "v")
    md = chunk("s", "t", None, None)
    assert page_in(md, "") and not page_in(md, "3")


def test_split_is_balanced_and_deterministic(tmp_path):
    items = [EvalItem(id=f"a{i}", doc="D1", kind="answerable", question="q") for i in range(10)]
    items += [EvalItem(id=f"o{i}", doc="D1", kind="off_topic", question="q", search_terms="z") for i in range(6)]
    assign_splits(items)
    first = [i.split for i in items]
    assign_splits(items)
    assert first == [i.split for i in items]
    assert sum(i.split == "dev" for i in items if i.kind == "answerable") == 5
    path = tmp_path / "q.csv"
    save_items(items, path)
    assert [i.id for i in load_items(path)] == [i.id for i in items]
