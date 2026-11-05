# FYPilot: Project Specification

## 1. Overview

FYPilot is a retrieval-augmented generation (RAG) assistant that helps students prepare for final year project (FYP) evaluations. A student uploads their FYP documents (PDF/DOCX), optionally together with the FYP handbook or guidelines. The system answers questions about those documents, anticipates what an evaluation panel will ask, finds weaknesses in the documents, and refuses to answer when the documents do not support an answer.

**Course category:** RAG Bot for a Niche Domain (NLP course project).

**Core idea:** every output is grounded in the student's own documents and carries a citation (document, section, page). If the evidence is missing, the system says so instead of guessing.

## 2. Problem Statement

Students preparing for an FYP defense usually do not know:

- which questions the panel is likely to ask,
- whether their own proposal and report actually answer those questions,
- whether the proposal promises things the report never delivers,
- whether any required item from the handbook is missing.

General-purpose chatbots tend to fabricate plausible answers. FYPilot is designed to be trustworthy first: grounded, cited, and willing to abstain.

## 3. System Description

**Input:** FYP documents (PDF/DOCX), for example a proposal, progress report, and final report, plus optionally the FYP handbook/guidelines.

**Output:** four capabilities, described below.

### (A) Grounded Q&A with citations

The student asks a question about their project. The system retrieves relevant chunks from the uploaded documents and generates an answer. Each claim in the answer is cited as **document + section + page**.

### (B) Panel-question generation

The system generates questions an evaluator would likely ask, grouped by category:

- motivation
- novelty
- methodology
- evaluation
- feasibility
- limitations
- ethics

Each question comes with a **draft answer retrieved from the student's own documents**. If the documents do not cover the question, it is marked **"NOT COVERED"**. This tells the student exactly where their documents are weak.

### (C) Gap and consistency checking

1. **Promised but unevidenced claims.** For example, the proposal commits to evaluating X but the report never evaluates X.
2. **Contradictions** between documents or between sections (for example, different dataset sizes or different reported results).
3. **Missing required items.** Requirements from the handbook/guidelines that no uploaded document addresses.

### (D) Abstention

When the documents do not support an answer, the system says so instead of guessing. Implemented through two checks:

1. **Retrieval-score thresholds.** If the best retrieved evidence scores below a threshold, abstain.
2. **NLI entailment check.** A natural language inference model checks whether the retrieved evidence actually entails the answer or claim. If it is not entailed, abstain or flag it.

## 4. Proposed Architecture

```
Documents (PDF/DOCX) 
   -> Ingestion (parse, clean, chunk; keep doc / section / page metadata)
   -> Embedding + vector store
   -> Retrieval (top-k, optional reranking)
   -> Abstention gate (score threshold + NLI)
   -> LLM generation with citations
   -> Features: Q&A (A), panel questions (B), gap/consistency checks (C)
   -> Streamlit UI
```

### Proposed starting stack (open to change)

| Part | Choice |
|---|---|
| Orchestration | LangChain (or LlamaIndex) |
| Embeddings | `BAAI/bge-small-en-v1.5` (Hugging Face) |
| Vector store | Chroma or FAISS |
| NLI model | `cross-encoder/nli-deberta-v3-base` |
| LLM | OpenAI API (or a local model via Ollama) |
| Parsing | PyMuPDF, python-docx |
| UI | Streamlit |

## 5. Evaluation Plan

The course requires demonstrating methodology and results, not only a working interface. Planned evaluation:

1. **Test set** (hand-built by the team, kept small):
   - answerable questions with gold answers and gold source locations,
   - unanswerable questions (the documents do not contain the answer),
   - for component (C), documents with deliberately planted gaps and contradictions.
2. **Comparisons / ablations:**
   - plain RAG (no abstention) vs. RAG + score threshold vs. RAG + threshold + NLI,
   - retrieval variants (for example, with and without reranking).
3. **Metrics:**
   - retrieval: recall@k, hit rate on the gold section/page,
   - answer quality: correctness against gold answers, faithfulness to the retrieved evidence,
   - citation accuracy: whether cited document/section/page really supports the claim,
   - abstention: precision and recall of abstaining on unanswerable questions, false-refusal rate on answerable ones,
   - gap/contradiction detection: precision and recall on planted issues.
4. **Error analysis:** examine failure cases (wrong retrieval, hallucinated answers, wrongful abstentions) and discuss limitations.

## 6. Scope and Constraints (from the course guidelines)

- Do **not** train models from scratch. Use pretrained models from Hugging Face or APIs.
- Keep the document set **small, clean, and manageable**. Do not spend time collecting or scraping data.
- Google Colab or Kaggle may be used for GPU work. Streamlit or Gradio may be used for the interface.
- The grade rewards **NLP methodology and results**, not a fancy interface.
- Project ideas must not duplicate another group's idea. Register the idea in the shared Excel sheet.
- Groups of 2 to 3 students.

## 7. Deliverables

1. Project idea registered in the shared Excel sheet.
2. Final code, evaluation, and presentation.

## 8. Planned Milestones

1. Repository scaffold and sample document set.
2. Ingestion with section and page metadata.
3. Retrieval and cited Q&A (A).
4. Evaluation set and baseline results.
5. Abstention (D) and ablation results.
6. Panel-question generation (B).
7. Gap and consistency checking (C).
8. Streamlit UI, final evaluation, presentation.

## 9. Open Decisions

- Which LLM to use (API vs. local), and the budget for API calls.
- Which sample FYP documents to use (real reports with permission, or public ones).
- Chunking strategy (by section vs. fixed size) and how section headings are detected.
- How contradictions between documents are detected (claim extraction + NLI is the starting idea).
