from __future__ import annotations

import html
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from app import service
from src.config import settings
from src.qa.llm import ProvidersUnavailable

if (Path(os.environ["HF_HOME"]) / "hub").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
settings.llm_chain_rounds = 2
settings.llm_max_retries = 2
settings.llm_cooldown_s = 15.0

st.set_page_config(page_title="FYPilot", page_icon="🧭", layout="wide")

TYPES = {"Development-style report": "dev", "Research-style report (has experiments)": "rd"}
STAGES = {"Final report after the second semester": 2, "Report after the first semester": 1}
GAP_KINDS = [
    ("required", "Required parts", "What the handbook says a report must contain.", "missing_check"),
    ("promises", "Promises without evidence", "Things the report promises that it never backs up with results.", "promises_check"),
    ("conflicts", "Contradictions", "Statements in different places that disagree, such as two different numbers.", "contradictions_check"),
]
PROVIDER_NAMES = {"cerebras": "Cerebras", "groq": "Groq", "gemini": "Gemini"}
VERDICT_KIND = {"present": "ok", "shown": "ok", "partial": "warn", "not_shown": "warn", "missing": "bad", "contradicted": "bad"}
CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1080px;}
[data-testid="stSidebar"] {background: #f3f5f9;}
h1, h2, h3 {letter-spacing: -0.01em;}
.hero h1 {margin: 0 0 .15rem 0; font-size: 2.3rem;}
.hero p {margin: 0 0 .9rem 0; color: #5b6475; font-size: 1.02rem;}
.badge {display: inline-block; padding: 2px 11px; margin: 0 6px 4px 0; border-radius: 999px; font-size: .78rem; font-weight: 600; line-height: 1.6;}
.badge.ok {background: #e3f4e8; color: #17692f;}
.badge.warn {background: #fff1da; color: #8a5200;}
.badge.bad {background: #fde6e6; color: #a31b1b;}
.badge.info {background: #e4ecfd; color: #1d4aa8;}
.badge.muted {background: #eceff4; color: #4a5468;}
.brand {font-size: 1.5rem; font-weight: 700; margin-bottom: .1rem;}
.tagline {color: #5b6475; font-size: .9rem; margin-bottom: 1rem;}
.stTabs [data-baseweb="tab-list"] {gap: 1.4rem;}
.stTabs [data-baseweb="tab"] {font-weight: 600;}
div[data-testid="stMetric"] {background: #f3f5f9; border-radius: 12px; padding: .7rem 1rem;}
</style>
"""


def show_limit(exc: Exception) -> None:
    if not isinstance(exc, ProvidersUnavailable):
        st.error(f"Something went wrong: {exc}")
        return
    st.error("The language models did not answer just now.")
    st.markdown("\n".join(f"- {PROVIDER_NAMES.get(p, p)}: {why}" for p, why in exc.reasons.items()))
    st.caption("These are free tiers. A short rate limit clears in a minute, and daily limits reset each day.")


def badge(text: str, kind: str = "muted") -> str:
    return f"<span class='badge {kind}'>{html.escape(text)}</span>"


@st.cache_resource(show_spinner="Loading the search and ranking models. This takes about half a minute the first time...")
def resources():
    from src.qa.llm import LLM
    from src.retrieval.embedder import Embedder
    from src.retrieval.index import Index
    from src.retrieval.rerank import Reranker

    embedder, reranker, demo = Embedder(), Reranker(), None
    if (settings.index_dir / "meta.json").exists():
        try:
            demo = Index.load(settings.index_dir, embedder)
            reranker.rerank("warm up", demo.search("warm up", 2))
        except RuntimeError:
            demo = None
    return embedder, reranker, LLM(settings.consistency_chain), demo


def progress_bar(label: str):
    bar = st.progress(0.0, text=label)

    def update(n: int, total: int, message: str) -> None:
        bar.progress(n / max(total, 1), text=f"{label}: {n} of {total}. {message[:80]}")

    return bar, update


def sidebar(embedder, demo_index):
    from src.retrieval.index import Index

    st.session_state.setdefault("uploads", {})
    docs = service.demo_docs(demo_index) if demo_index is not None else []
    with st.sidebar:
        st.markdown("<div class='brand'>🧭 FYPilot</div><div class='tagline'>Defense preparation for final year projects</div>", unsafe_allow_html=True)
        lookup = {d.id: (d, demo_index) for d in docs}
        lookup.update(st.session_state.uploads)
        choice = st.selectbox("Document", list(lookup), disabled=not lookup, help="Questions and checks use only this document.") if lookup else None
        with st.expander("Add your own report", expanded=not lookup):
            file = st.file_uploader("PDF or Markdown", type=["pdf", "md"], label_visibility="collapsed")
            kind = st.radio("Project type", list(TYPES))
            stage = st.radio("Stage of the report", list(STAGES))
            if st.button("Read and index", type="primary", disabled=file is None, width="stretch"):
                try:
                    with st.spinner("Reading and indexing the document..."):
                        doc = service.read_upload(file.name, file.getvalue(), TYPES[kind], STAGES[stage])
                        st.session_state.uploads[doc.id] = (doc, Index.build(doc.chunks, embedder))
                    st.success(f"Read {len(doc.chunks)} sections of {doc.id}. Pick it in the Document list.")
                except ValueError as exc:
                    st.error(str(exc))
        st.caption("Answers come from free-tier models. If one has reached its daily limit the next one is used, so a busy day can be slow.")
    if choice is None:
        return None, None
    return lookup[choice]


def ask_tab(doc, index, llm, reranker):
    with st.form("ask_form", border=False):
        question = st.text_input("Question", placeholder="Which dataset was used for training?", label_visibility="collapsed")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted and question.strip():
        try:
            with st.spinner("Searching and answering..."):
                st.session_state[f"ask_{doc.id}"] = service.ask(question, index, doc, llm, reranker)
        except RuntimeError as exc:
            show_limit(exc)
    out = st.session_state.get(f"ask_{doc.id}")
    if not out:
        st.caption("Ask anything the document should answer. If it does not say, you get a clear \"not in the document\" instead of a guess.")
        return
    with st.container(border=True):
        st.markdown(badge("Not in the document", "warn") if out["refused"] else badge("Answered from the document", "ok"), unsafe_allow_html=True)
        st.markdown(out["text"])
        if out["sources"]:
            st.caption(("Closest passages, not an answer: " if out["refused"] else "Sources: ") + "; ".join(out["sources"]))
    if out["passages"]:
        with st.expander("Show the closest passages" if out["refused"] else "Show the passages"):
            for p in out["passages"]:
                st.markdown(f"**{p['label']}**")
                st.write(p["text"])


def panel_tab(doc, index, llm, reranker):
    st.write("Likely panel questions by category, answered from the document. A question marked NOT COVERED is a weak spot to prepare.")
    key = f"panel_{doc.id}"
    if st.button("Generate panel questions", type="primary"):
        bar, update = progress_bar("Questions")
        try:
            st.session_state[key] = service.panel(index, doc, llm, reranker, update)
        except RuntimeError as exc:
            show_limit(exc)
        bar.empty()
    records = st.session_state.get(key)
    if not records:
        return
    weak = [r for r in records if r["status"] == "NOT COVERED"]
    cols = st.columns(3)
    cols[0].metric("Questions", len(records))
    cols[1].metric("Covered by the document", len(records) - len(weak))
    cols[2].metric("Weak spots", len(weak))
    if weak:
        with st.container(border=True):
            st.markdown("**Weak spots to prepare**")
            for r in weak:
                st.markdown(f"- ({r['category']}) {r['question']}")
    st.download_button("Download the prep sheet", service.panel_markdown(doc, records), file_name=f"{doc.id}_prep_sheet.md")
    for category in dict.fromkeys(r["category"] for r in records):
        group = [r for r in records if r["category"] == category]
        gaps = sum(r["status"] == "NOT COVERED" for r in group)
        with st.expander(f"{category.capitalize()}  ({len(group)} questions, {gaps} not covered)"):
            for r in group:
                with st.container(border=True):
                    st.markdown(f"**{r['question']}**")
                    if r["status"] == "NOT COVERED":
                        st.markdown(badge("Not covered", "warn"), unsafe_allow_html=True)
                    else:
                        st.markdown(badge("Covered", "ok"), unsafe_allow_html=True)
                        st.write(r["answer"])
                        st.caption("Sources: " + "; ".join(r["sources"]))


def run_gap(kind: str, doc, index, reranker, llm):
    bar, update = progress_bar("Checking")
    if kind == "required":
        rows = service.required_parts(doc, index, reranker, llm, update)
    elif kind == "promises":
        rows = service.promise_check(doc, index, reranker, llm, update)
    else:
        rows = service.contradiction_check(doc, index, llm, update)
    bar.empty()
    return rows


def verdict_style(value: str) -> str:
    colors = {"ok": "#e3f4e8", "warn": "#fff1da", "bad": "#fde6e6"}
    kind = VERDICT_KIND.get(value)
    return f"background-color: {colors[kind]}; font-weight: 600" if kind else ""


def show_gap(kind: str, doc, rows):
    st.caption(service.summary(rows, kind))
    table = {"required": lambda: service.required_table(doc, rows), "promises": lambda: service.promise_table(rows),
             "conflicts": lambda: service.conflict_table(rows)}[kind]()
    if not table:
        st.write("Nothing to report.")
        return
    frame = pd.DataFrame(table)
    shown = frame.style.map(verdict_style, subset=["verdict"]) if "verdict" in frame else frame
    st.dataframe(shown, width="stretch", hide_index=True)


def gap_tab(doc, index, llm, reranker):
    st.write("Three checks: required parts from the handbook, promises the report never backs up, and statements that contradict each other.")
    if not doc.is_pdf:
        st.info("The gap checks read the chapter structure of a PDF report, so they are not available for Markdown files.")
        return
    st.caption("Required parts take about a minute. Promises take several minutes. Contradictions send hundreds of requests and can take an hour on free tiers.")
    for kind, title, blurb, saved in GAP_KINDS:
        key = f"gap_{kind}_{doc.id}"
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(blurb)
            cols = st.columns([1, 1, 3])
            saved_rows = service.saved_rows(saved, doc.id) if doc.demo else None
            if saved_rows and cols[0].button("Show saved results", key=f"saved_{key}"):
                st.session_state[key] = saved_rows
            if cols[1].button("Run the check", key=f"run_{key}", type="primary"):
                try:
                    st.session_state[key] = run_gap(kind, doc, index, reranker, llm)
                except RuntimeError as exc:
                    show_limit(exc)
            if st.session_state.get(key):
                show_gap(kind, doc, st.session_state[key])


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    embedder, reranker, llm, demo_index = resources()
    doc, index = sidebar(embedder, demo_index)
    st.markdown("<div class='hero'><h1>FYPilot</h1><p>Ask questions, prepare for panel questions and find gaps in a final year project report.</p></div>",
                unsafe_allow_html=True)
    if doc is None:
        st.info("Add a report in the sidebar to begin.")
        return
    kind = "Research" if doc.type == "rd" else "Development"
    st.markdown(badge(doc.id, "info") + badge(f"{kind}-style", "muted") + badge(f"Stage {doc.stage}", "muted") + badge(f"{len(doc.chunks)} sections", "muted"),
                unsafe_allow_html=True)
    ask, panel, gaps = st.tabs(["Ask", "Panel prep", "Gap report"])
    with ask:
        ask_tab(doc, index, llm, reranker)
    with panel:
        panel_tab(doc, index, llm, reranker)
    with gaps:
        gap_tab(doc, index, llm, reranker)


main()
