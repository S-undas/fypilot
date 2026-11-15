# FYPilot

A retrieval-augmented assistant for final year project (FYP) defense preparation. It reads a student's report and:

- answers questions with citations (document, section, page), and refuses when the report does not say;
- writes the questions a defense panel is likely to ask, with draft answers or NOT COVERED;
- checks the report for missing required parts, promises it never backs up, and statements that contradict each other.

Built for a university NLP course (RAG bot for a niche domain). No model is trained: it uses pretrained models for search and ranking, and hosted language models through an OpenAI-compatible API.

## Results in one paragraph
On 57 hand-verified questions, hybrid search plus a reranker puts the right passage first for 84% of answerable questions, and the system answered 38 of 38 answerable questions and refused 19 of 19 unanswerable ones. A refusal instruction in the prompt is the best abstention method; a score threshold and an NLI check did not help. See [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md) for the comparison, error analysis and limits, and [docs/PHASES.md](docs/PHASES.md) for what each phase built.

## Setup
Python 3.11, CPU only.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add the keys you have:

```
CEREBRAS_API_KEY=
GROQ_API_KEY=
GEMINI_API_KEY=
```

Models are tried in this order: Cerebras, then Groq (the same Qwen model), then Gemini as a last fallback. One working key is enough. Search and ranking models download from Hugging Face on first use and are cached in `.hf_cache/`.

## Data
The reports are student work and are not in this repository. `data/` and `index/` are gitignored. Put documents in `data/raw/`:

```
data/raw/reports/
data/raw/proposals/
data/raw/handbook/
```

Then build the chunks and the search index:

```
.venv\Scripts\python -m src.ingestion.ingest
.venv\Scripts\python -m src.retrieval.index
```

## Run the app
```
run_app.bat
```

or `.venv\Scripts\python -m streamlit run app/main.py`. It opens at http://localhost:8501. Stop it with Ctrl+C in its window.

The sidebar lists the four demo reports from the saved index. Add your own PDF or Markdown report there. The gap checks need a PDF. Free-tier models have daily limits, and the contradictions check on a new report makes hundreds of calls, so it is slow.

## Commands
```
.venv\Scripts\python -m src.qa.cli "How many meetings with the supervisor are required?" --doc fyp_page
.venv\Scripts\python -m src.evaluation.final                    final comparison and error analysis
.venv\Scripts\python -m src.consistency.report                  Phase 6 tables
.venv\Scripts\python -m pytest -q                               tests
```

The full list of commands for every phase is at the end of [docs/PHASES.md](docs/PHASES.md).

## Layout
```
app/          Streamlit app (main.py) and its logic (service.py)
src/ingestion/    PDF and Markdown parsing, chunks with section and page
src/retrieval/    embeddings, FAISS and BM25 index, reranker
src/qa/           cited answers and the model chain
src/abstention/   score threshold and NLI check
src/panel/        panel question generation
src/consistency/  required parts, promises, contradictions
src/evaluation/   metrics, ablations, final report
src/config.py     models, thresholds and limits in one place
eval/             evaluation results (questions and labels stay local)
docs/             spec, results and phase notes
tests/            pytest tests
```

## Limits
- Small test sets (38 answerable and 19 unanswerable questions), and a small document set: seven documents (four reports, two proposals and the handbook) from one university.
- The gap checks are screening aids: about half of their flags are not real gaps, and they miss some real ones.
- The language models are free tiers with daily limits, so heavy runs can be slow or fail until the limit resets.
