from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from app import service
from src.config import settings

if (Path(os.environ["HF_HOME"]) / "hub").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
settings.llm_chain_rounds = 1
settings.llm_max_retries = 2

st.set_page_config(page_title="FYPilot", layout="wide")
LIMIT_MESSAGE = ("The language model services did not answer. They are free tiers with daily limits, or the network blocked them. "
                 "Try again in a little while, or tomorrow.")

TYPES = {"Development-style report": "dev", "Research-style report (has experiments)": "rd"}
STAGES = {"Final report after the second semester": 2, "Report after the first semester": 1}
GAP_KINDS = [("required", "Required parts", "missing_check"), ("promises", "Promises without evidence", "promises_check"),
             ("conflicts", "Contradictions", "contradictions_check")]


@st.cache_resource(show_spinner="Loading the search and ranking models...")
def resources():
    from src.qa.llm import LLM
    from src.retrieval.embedder import Embedder
    from src.retrieval.index import Index
    from src.retrieval.rerank import Reranker

    embedder = Embedder()
    demo = None
    if (settings.index_dir / "meta.json").exists():
        try:
            demo = Index.load(settings.index_dir, embedder)
        except RuntimeError:
            demo = None
    return embedder, Reranker(), LLM(settings.consistency_chain), demo


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
        st.header("FYPilot")
        st.caption("Questions, panel prep and gap checks for a final year project report. Answers use only the document you pick.")
        with st.expander("Add your own report", expanded=not docs):
            file = st.file_uploader("PDF or Markdown", type=["pdf", "md"])
            kind = st.radio("Project type", list(TYPES))
            stage = st.radio("Stage of the report", list(STAGES))
            if st.button("Read and index", disabled=file is None):
                try:
                    with st.spinner("Reading and indexing the document..."):
                        doc = service.read_upload(file.name, file.getvalue(), TYPES[kind], STAGES[stage])
                        st.session_state.uploads[doc.id] = (doc, Index.build(doc.chunks, embedder))
                    st.success(f"Read {len(doc.chunks)} sections of {doc.id}.")
                except ValueError as exc:
                    st.error(str(exc))
        lookup = {d.id: (d, demo_index) for d in docs}
        lookup.update({k: v for k, v in st.session_state.uploads.items()})
        if not lookup:
            st.info("Add a report to start.")
            return None, None
        choice = st.selectbox("Document", list(lookup))
        st.caption("Free-tier models. If one has reached its daily limit, the next one is used, so a busy day can be slow.")
    return lookup[choice]


def ask_tab(doc, index, llm, reranker):
    question = st.text_input("Ask a question about this document", placeholder="Which dataset was used for training?")
    if st.button("Ask", disabled=not question.strip()):
        try:
            with st.spinner("Searching and answering..."):
                st.session_state[f"ask_{doc.id}"] = service.ask(question, index, doc, llm, reranker)
        except RuntimeError:
            st.error(LIMIT_MESSAGE)
    out = st.session_state.get(f"ask_{doc.id}")
    if not out:
        return
    if out["refused"]:
        st.warning("This is not answered in the document. Nothing is guessed.")
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
    if st.button("Generate panel questions"):
        bar, update = progress_bar("Questions")
        try:
            st.session_state[key] = service.panel(index, doc, llm, reranker, update)
        except RuntimeError:
            st.error(LIMIT_MESSAGE)
        bar.empty()
    records = st.session_state.get(key)
    if not records:
        return
    weak = [r for r in records if r["status"] == "NOT COVERED"]
    st.subheader(f"{len(weak)} of {len(records)} questions are not covered")
    for r in weak:
        st.markdown(f"- ({r['category']}) {r['question']}")
    for category in dict.fromkeys(r["category"] for r in records):
        with st.expander(category.capitalize()):
            for r in (r for r in records if r["category"] == category):
                st.markdown(f"**{r['question']}**")
                if r["status"] == "NOT COVERED":
                    st.warning("NOT COVERED")
                else:
                    st.write(r["answer"])
                    st.caption("Sources: " + "; ".join(r["sources"]))
    st.download_button("Download the prep sheet", service.panel_markdown(doc, records), file_name=f"{doc.id}_prep_sheet.md")


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


def show_gap(kind: str, doc, rows):
    st.caption(service.summary(rows, kind))
    table = {"required": lambda: service.required_table(doc, rows), "promises": lambda: service.promise_table(rows),
             "conflicts": lambda: service.conflict_table(rows)}[kind]()
    if table:
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.write("Nothing to report.")


def gap_tab(doc, index, llm, reranker):
    st.write("Three checks: required parts from the handbook, promises the report never backs up, and statements that contradict each other.")
    if not doc.is_pdf:
        st.info("The gap checks read the chapter structure of a PDF report, so they are not available for Markdown files.")
        return
    st.caption("Required parts take about a minute. Promises take several minutes. Contradictions send hundreds of requests and can take an hour on free tiers.")
    for kind, title, saved in GAP_KINDS:
        key = f"gap_{kind}_{doc.id}"
        st.subheader(title)
        cols = st.columns(2)
        saved_rows = service.saved_rows(saved, doc.id) if doc.demo else None
        if saved_rows and cols[0].button("Show saved results", key=f"saved_{key}"):
            st.session_state[key] = saved_rows
        if cols[1].button("Run the check", key=f"run_{key}"):
            try:
                st.session_state[key] = run_gap(kind, doc, index, reranker, llm)
            except RuntimeError:
                st.error(LIMIT_MESSAGE)
        if st.session_state.get(key):
            show_gap(kind, doc, st.session_state[key])


def main() -> None:
    embedder, reranker, llm, demo_index = resources()
    doc, index = sidebar(embedder, demo_index)
    st.title("FYPilot")
    if doc is None:
        st.write("Add a report in the sidebar to begin.")
        return
    st.caption(f"{doc.id}: {len(doc.chunks)} sections. {'Research' if doc.type == 'rd' else 'Development'}-style, stage {doc.stage}.")
    ask, panel, gaps = st.tabs(["Ask", "Panel prep", "Gap report"])
    with ask:
        ask_tab(doc, index, llm, reranker)
    with panel:
        panel_tab(doc, index, llm, reranker)
    with gaps:
        gap_tab(doc, index, llm, reranker)


main()
