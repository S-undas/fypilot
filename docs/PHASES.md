# FYPilot: Phases and Progress

What was built in each phase, with the key decisions and numbers. `PROJECT_SPEC.md` describes what the system should do. This file records what was actually done.

Pipeline: documents -> ingestion -> chunks with metadata -> retrieval -> LLM answer with citations -> abstention gate, panel questions, gap and consistency checks -> evaluation -> UI.

## Dataset
- 4 FYP final reports: 2 research-style (with an experimental results chapter) and 2 development-style.
- 2 short project proposals (3 pages each).
- The FYP handbook, extracted from the university FYP guidelines web page into one Markdown file (34 sections).
- Documents are identified by ID (F25_044, F25_051, F25_181, F25_182, two proposals, `fyp_page`), never by student name.
- All documents stay local (`data/` is gitignored) because the reports are student work.
- Nothing is trained. The documents are only indexed and queried. They also serve as the test material for evaluation.

## Stack used, compared with the original spec
| Part | Spec proposed | Used |
|---|---|---|
| Orchestration | LangChain | None. Plain Python, so scores, citations and ablation switches stay visible |
| Embeddings | bge-small-en-v1.5 | Same |
| Reranker | Optional | `cross-encoder/ms-marco-MiniLM-L-6-v2` on the top 20 hybrid results |
| Vector store | Chroma or FAISS | FAISS (flat index), plus BM25 |
| LLM | OpenAI API | Free-tier OpenAI-compatible APIs (see Phase 2) |
| Parsing | PyMuPDF, python-docx | PyMuPDF for PDF, a Markdown reader for the handbook. DOCX not needed |
| NLI | nli-deberta-v3-base | Same, used in Phase 4 |
| UI | Streamlit | Used in Phase 7 |

Runs on CPU only. Python 3.11, virtual environment in `.venv/`.

## Phase 0: Setup (done)
- Folder layout, central settings in `src/config.py` (models, chunk sizes, k, thresholds, provider limits), requirements, tests.
- Thresholds are left unset on purpose. They are tuned on a dev split later, not guessed.
- API keys live in `.env` (gitignored). Only `.env.example` is committed.

## Phase 1: Ingestion (done)
Turns each document into chunks that keep document, section and printed page.
- **PDF reports:** headings come from the PDF's built-in outline, matched to text rows. Page labels use the printed page number, not the viewer's.
- **PDFs without an outline** (the proposals): headings detected from font size and boldness.
- **Skipped on purpose:** title page, declarations (student names), table of contents, lists of figures and tables, running headers. Abstract and Executive Summary are kept.
- **Handbook:** bold subheads become headings, giving paths like `FYP - Rules and Instructions > FYP Grades`.
- **Chunking:** one chunk per section, split at 1500 characters with 150 overlap, chunks under 40 characters dropped.
- **Result:** 7 documents became 506 chunks, saved as JSONL in `data/processed/`.
- **Checks:** 388 of 389 outline headings matched to text. All 938 chunk start and end points were confirmed on the page they cite. The font-based fallback recovers every level 1 and 2 heading (recall 1.00, precision 0.86 to 0.97 against the outline).
- **Known limits:** text inside figures is not extracted, tables flatten to plain text, proposals cite PDF page numbers because they have no printed ones.

## Phase 2: Retrieval and cited answers (done)
**Retrieval**
- Chunks are embedded as `section title + text` with `BAAI/bge-small-en-v1.5` and stored in a FAISS index.
- Three modes: dense, BM25, and hybrid (reciprocal rank fusion). Each result carries both scores. Search can be limited to one document.
- A cross-encoder reranker was evaluated in Phase 3 and is now part of the pipeline.

**LLM layer** (OpenAI-compatible endpoints, one client)
| Role | Provider | Model |
|---|---|---|
| Main | Cerebras | `qwen-3.8-27b` |
| Fallback (same model) | Groq | `qwen/qwen3.8-27b` |
| Judge for evaluation | Google Gemini | `gemini-3.5-flash-lite` |

- Why these: all three have free tiers and work quickly. The fallback is the same model, so failover does not change answers. The judge is a different model family, which reduces self-grading bias.
- A smoke comparison with `gpt-oss-120b` showed Qwen about ten times faster under our rate limits and more consistent in citation format. Phase 3 confirmed the choice on gold answers.
- The wrapper paces calls by each model's requests and tokens per minute, retries on rate limits, switches provider with a cooldown, and caches every response on disk, so reruns cost nothing.

**Cited answers**
- Retrieved passages are numbered in the prompt. The model cites `[n]`, and each reference is mapped back to document, section and page. Invalid references are flagged.
- The baseline prompt has no explicit refusal instruction, so abstention can be measured separately.
- Observation: both models already refuse obvious off-topic questions unprompted. Phase 3 measures how far this goes.

## Phase 3: Evaluation set and baseline (done)
Goal: measure plain retrieval and answering before adding abstention.

**Done: the evaluation set** (57 questions)
| Kind | Count | Meaning |
|---|---|---|
| answerable | 38 | The document gives a clear answer |
| near_miss | 10 | The topic is covered, the specific detail is not |
| misleading | 5 | Related but different information exists (for example a target instead of a measured value) |
| off_topic | 4 | The question belongs to another project |

- Questions and gold answers were drafted from the documents. Each carries a verbatim evidence quote, and a script fills in the gold section and page and checks the quote exists there. For off-topic questions a script confirms the key terms are absent from the document.
- Every question was then checked by a person against the real PDFs. Rows that were ambiguous or wrong were replaced. This is stated in the write-up as LLM-drafted and human-verified.
- Split: assigned by a seeded script, about half dev and half test within each kind (29 dev, 28 test). Thresholds and model choice are tuned on dev only. Final numbers are reported on test.
- Evaluation scope: one document at a time, as a student would upload their own project. A smaller run across all documents is an extra check.
- The question file stays local because it quotes the reports.

**Results** (produced by `python -m src.evaluation.report`; a plain-language walkthrough is in `PHASE3_RESULTS.md`)
- **Retrieval comparison** (38 answerable questions, search inside the right document): dense, BM25, hybrid, and hybrid followed by a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`, rereads the top 20 results). Hybrid with reranking is best on both splits: MRR 0.879 and right passage in the top 5 for 94.7% of questions, against 0.789 and 86.8% for plain hybrid. It was chosen by a dev-only rule (best MRR on dev). With 19 questions per split, a one-question difference moves recall by about 5 points, so treat gaps as indicative.
- **End-to-end effect:** the two answerable questions that plain hybrid retrieval missed were wrongly refused. With reranking the right passage is found, and both are answered correctly.
- **Baseline** (Qwen 3.8 27B, top 5 passages, hybrid with reranking): 38 of 38 answerable questions answered correctly and 19 of 19 unanswerable questions refused, with every answer supported by the retrieved passages. Without reranking: 94.7% correct and 94.7% refused.
- **Prompt:** a bare prompt without a grounding instruction behaves almost the same (89.5% refused). Qwen is cautious by nature.
- **Model comparison on dev:** `gpt-oss-120b` ties on correctness without reranking (89.5%) but refuses only 60% of unanswerable questions (it hallucinated on near-miss and misleading ones) and is about 10 times slower. Qwen is the main model. `gpt-oss-120b` stays as a less cautious second generator for Phase 4.
- **Ceiling effect:** with 38 and 19 questions, a perfect score still leaves a wide margin (38 of 38 is consistent with a true rate of about 91% or higher). The set cannot separate small differences, which is why Phase 4 also uses the less cautious generator.
- **Judge check:** the first prompt mislabeled refusals that were followed by related facts. The prompt was sharpened and a text-rule cross-check added. A person then labeled 20 answers blind (including the ones the checks had flagged). The judge agreed with the person on 19 of 20 refuse-or-answer labels (kappa 0.90) and on all 11 correctness labels. Before review it was 18 of 20 (kappa 0.80): one human label was corrected after reading the answer again, and the change is recorded in the sheet. The remaining disagreement is a judge error on a plain refusal that listed related facts. The judge's support-by-passage labels (faithfulness and citation support) were not checked by a person, so those scores are unvalidated.

## Phase 4: Abstention (done)
Goal: refuse when the documents do not support an answer, and measure whether each method helps.

**Built**
- Score threshold: refuse before answering when the best evidence score is low. Two signals were compared (dense similarity and the reranker score), and thresholds are set on dev only at two operating points (balanced, and "no-harm", which keeps about 94% of good dev answers).
- NLI check (`cross-encoder/nli-deberta-v3-base`): each factual sentence of an answer must be entailed by the passage windows it cites. Three versions were tried and all are reported.
- Comparison on both generators (Qwen 3.8 27B and `gpt-oss-120b`): plain, prompt refusal, score threshold, prompt refusal plus threshold, and threshold plus NLI.
- Outputs: tables, curves (`abstention_curves.png`) and a summary produced by `python -m src.evaluation.abstention_eval`.

**Results** (a plain-language walkthrough is in `PHASE4_RESULTS.md`)
- A refusal instruction in the prompt fixed the only hallucinations on test (two, from `gpt-oss-120b`) with no loss of correct answers. Qwen already refused every unanswerable question.
- The reranker score separates answerable from unanswerable questions far better than dense similarity (AUC 0.80 and 0.87 against 0.60 and 0.66). As a gate it costs 21% to 42% of correct answers on test, and a threshold tuned on 19 questions was unstable between dev and test.
- The NLI check did not help: it refused most correct answers in version 1, and after fixes it still scored a quarter of correct answers as unsupported while letting a misleading answer through.
- Setup note: model downloads and caches live inside the project folder (`.hf_cache`), not on the system drive.

## Phase 5: Panel questions (done)
Goal: likely panel questions per category, each answered from the student's documents or marked NOT COVERED.

**Built**
- Question generation per category (motivation, novelty, methodology, evaluation, feasibility, limitations, ethics): the passages most relevant to the category go to the model, which writes three short, single-topic questions specific to the project, including at least one that challenges a choice or result. Near-duplicates are dropped.
- Answering with hybrid search, reranking and a cited answer. A dedicated prompt asks the model to reply exactly "NOT COVERED" when the passages do not answer.
- A prep sheet per document in Markdown and CSV: a weak-spots list, then questions by category with draft answers and sources.
- Evaluation: a baseline of fixed generic questions, an LLM judge over covered answers, and a blind human check of 36 sampled questions.

**Results** (a plain-language walkthrough is in `PHASE5_RESULTS.md`)
- Document-specific questions were marked NOT COVERED 34% of the time against 4% for generic questions, so questions built from the document expose gaps that generic ones cannot. Rates are highest for ethics and feasibility.
- Against a person's labels, NOT COVERED precision was 75% (88% if partly covered counts as not covered) and recall 40% (35%). When the system flags a question it is usually right, but it misses many gaps, mostly "why did you choose X over Y" and "how did you validate Z" questions that it answers from related facts.
- 30 of 36 sampled questions were rated as sensible panel questions.

## Phase 6: Gap and consistency checks (done)
Goal: find what a panel would catch in a report: missing required parts, promises with no evidence, and statements that contradict each other.

**Built**
- Required parts: the handbook's report requirements as a list by chapter name, split by project type and by stage. Each part is checked by retrieving the matching chapter and asking the model for present, partial or missing with citations. References is checked directly in the PDF. A plain heading rule is the baseline.
- Promises: promises and claims are extracted from the introduction, vision and methodology chapters (each must quote the report word for word), then looked up in the implementation, test, results and conclusion chapters and judged shown, not shown or contradicted.
- Contradictions: specific facts are extracted from every passage, paired when they carry different numbers or names, judged by a first check, and confirmed by a second check that reads both full passages.
- Evaluation: a removal test (delete or empty each required chapter), planted mistakes (changed numbers and invented promises, approved by a person), and a blind human check of 34 flags.
- Model chain: payment errors fall through to the next provider, timeouts are retried, and the whole chain is retried after a cooldown. Gemini is a last fallback for these checks only.
- The model judges these checks, not NLI, because NLI did not help in Phase 4.

**Results** (a plain-language walkthrough is in `PHASE6_RESULTS.md`)
- Required parts: 45 present, 3 partial, 2 missing across the four reports. Both "missing" flags were confirmed by a person. Deleted chapters were flagged 44 of 44 times, emptied chapters 44 of 44, against 0 of 44 for the heading rule.
- Promises: 225 found, 63% shown. A person judged 5 of 10 flags right and 0 of 5 "shown" promises unsupported.
- Contradictions: 1,661 pairs compared, 29 confirmed. A person judged 7 of 12 confirmed pairs real.
- Planted mistakes: 13 of 14 invented promises and 7 of 12 changed numbers were found. Most contradiction misses happen before the judge, when facts are picked out.

## Phase 7: Interface and final write-up (done)
Goal: a simple interface over the pipeline, and the final results with an error analysis. A README and slides were left out of scope.

**Built**
- Final results: one report that puts plain RAG against RAG with abstention in a single table, and an error analysis of the question answering and the panel questions, produced by `python -m src.evaluation.final` from saved runs. A plain-language version is in `FINAL_RESULTS.md`.
- Streamlit app (`app/main.py`, logic in `app/service.py`) with three views over one chosen document:
  - Ask: a cited answer, or a clear "not answered in the document" when the evidence is missing. It uses hybrid search, the reranker and the refusal instruction, the best combination from Phase 4.
  - Panel prep: questions by category with draft answers or NOT COVERED, a weak-spots list and a downloadable prep sheet.
  - Gap report: required parts, promises and contradictions. Saved results load instantly for the four demo reports. The checks can also be run on any PDF, with a progress bar.
- A report can be added as PDF or Markdown. It is read, split and indexed on the CPU with no model calls. The gap checks need the chapter structure of a PDF, so they are not offered for Markdown.
- The three Phase 6 checkers now run on a single document with an optional progress callback, so the app and the command line share the same code.

**Notes**
- The app sets the Hugging Face offline switch when the models are already cached, because a blocked connection to the hub otherwise adds minutes of retries to the first question.
- In the app the model chain fails fast with a plain message instead of waiting out the daily-limit cooldowns used in batch runs.
- Timing on this machine: once the models are loaded a question took about 2.5 seconds. The first question after the app starts is slower while the models load, and adding a report took about 40 seconds (35 of them indexing). A panel prep sheet and the gap checks make many model calls, so their time depends on the free-tier limits of the day.
- Run it with `streamlit run app/main.py`.

## Limitations to state
- Small corpus, and all reports follow one university template, so heading detection and tuned thresholds may not transfer to very different documents.
- The judge is an LLM, so its grading has some bias. A hand-labeled sample measures it.
- Free-tier models and rate limits, CPU only. Daily caps changed which model answered part of the Phase 6 runs.
- Phase 6 settings were adjusted on the same four reports they are scored on, and the human samples are small.
- Evaluation questions were drafted with an LLM and verified by a person.

## Running it
```
python -m src.ingestion.ingest                      # build chunks in data/processed
python -m src.retrieval.index                       # build the search index
python -m src.retrieval.cli "question" --doc F25_044
python -m src.qa.cli "question" --doc F25_044       # cited answer (needs API keys)
python -m src.evaluation.retrieval_eval             # compare search methods on the eval set
python -m src.evaluation.run_baseline --name NAME --rerank   # answer and judge every question
python -m src.evaluation.report NAME [NAME ...]     # build the results tables
python -m src.evaluation.abstention_eval            # compare refusal methods (needs the four Phase 4 runs)
python -m src.panel.run_panel --mode doc            # panel questions per document (use --mode generic for the baseline)
python -m src.panel.check make                      # blind sheet for the human check
python -m src.panel.report                          # build the Phase 5 results tables
python -m src.consistency.missing --mode check      # required parts per report (--mode perturb for the removal test)
python -m src.consistency.promises                  # promises and their evidence
python -m src.consistency.contradictions            # statement pairs that conflict
python -m src.consistency.plant make                # planted mistakes to approve, then: plant evaluate
python -m src.consistency.review make               # blind sheet for the human check, then: review score
python -m src.consistency.report                    # build the Phase 6 results tables
python -m src.evaluation.final                      # build the final comparison and error analysis
streamlit run app/main.py                           # the app
python -m pytest -q
```
