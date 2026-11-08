# Phase 3 Results: What We Measured and What It Shows

Phase 3 measures how well the system works before any abstention features are added. All numbers below come from `python -m src.evaluation.report` and the runs behind it. Nothing is estimated by hand.

## How the measurement works
1. **Question set:** 57 questions with known answers and locations. 38 are answerable, 19 are not (near-miss, misleading or off-topic). Each was checked by a person against the real documents. The set is split into dev (to choose settings) and test (to report), about half each.
2. **Retrieval test:** for each answerable question, search inside the correct document and check where the passage holding the evidence lands. No language model is involved.
3. **Answer test:** take the top 5 passages, ask the model to answer with citations, then let a second model (the judge) grade the answer.
4. **Judge check:** a person labeled 20 answers blind and we measured how often the judge agreed.

## 1. Which search method finds the right passage?
All 38 answerable questions.

| Search | Right passage ranked first | Right passage in top 5 | MRR |
|---|---|---|---|
| Dense (meaning) | 55% | 87% | 0.677 |
| BM25 (keywords) | 61% | 95% | 0.750 |
| Hybrid (both merged) | 71% | 87% | 0.789 |
| Hybrid + reranker | **84%** | **95%** | **0.879** |

- Each method alone has a weakness: dense misses exact terms and numbers, BM25 misses paraphrases. Merging them improves ordering but not the top-5 coverage.
- The reranker rereads the top 20 hybrid results against the question and reorders them. It was best on both dev and test, so it became part of the pipeline.
- Misses concentrate in the longest report and in short single-fact questions, such as one value in a table row.
- Each split has 19 questions, so one question moves a score by about 5 points. Read these as indicative.

## 2. How well does the system answer and refuse?
Model: Qwen 3.8 27B, top 5 passages, search inside the right document.

| | Hybrid | Hybrid + reranker | Bare prompt, hybrid |
|---|---|---|---|
| Answerable questions answered correctly | 94.7% | **100%** | 94.7% |
| Answerable questions wrongly refused | 5.3% | **0%** | 5.3% |
| Unanswerable questions refused | 94.7% | **100%** | 89.5% |
| Right passage in top 5 | 86.8% | 94.7% | 86.8% |

- Both wrong refusals without the reranker happened because retrieval missed the right passage. With it, the passage is found and both are answered correctly. Better retrieval directly removed the false refusals.
- Every answer was judged supported by the retrieved passages, with citations present and supporting. These two scores are unvalidated, because only refusal and correctness labels were checked by a person.
- The bare prompt (no instruction to stay within the passages) behaves almost the same. Qwen is cautious by nature, so it refuses nearly everything unanswerable on its own.

## 3. Which model should generate answers?
Dev split only, hybrid retrieval.

| | Qwen 3.8 27B | gpt-oss-120b |
|---|---|---|
| Answered correctly | 89.5% | 89.5% |
| Wrongly refused | 10.5% | 0% |
| Unanswerable refused | 100% | 60% |
| Citations support the answer | 100% | 94.7% |
| Time per question | about 1 second | about 10 seconds |

- Correctness ties, so the choice rests on the rest. Qwen refuses unanswerable questions reliably and is about ten times faster, so it is the main model.
- `gpt-oss-120b` hallucinated on near-miss and misleading questions: it invented a "high test accuracy" for a model whose accuracy is never reported, and it claimed a reduction "compared with human defenders" that the report only states against a random baseline. It is kept as a less cautious second generator for Phase 4, where an abstention gate has real mistakes to catch.

## 4. Can the judge be trusted?
A person labeled 20 answers blind. The sample deliberately included answers the checks had flagged.

| Label | Labels compared | Agreement | Kappa |
|---|---|---|---|
| Refused or answered | 20 | 95% | 0.90 |
| Correct, partial, incorrect | 11 | 100% | 1.00 |

- Kappa measures agreement beyond chance. Above 0.8 is strong.
- Before review the refusal agreement was 90% (kappa 0.80). One human label was corrected after rereading the answer, and the change is recorded.
- The judge's first prompt wrongly marked refusals as answers when the refusal listed related facts. The prompt was sharpened, and one such error remains.

## Conclusions
- Retrieval quality drives answer quality. A reranker fixed the only answerable questions the system got wrong.
- The cautious model already refuses almost every unanswerable question, which leaves little room for an abstention gate. Phase 4 therefore also tests the less cautious model.
- Pinned for the rest of the project: hybrid search with reranking, Qwen 3.8 27B as main model, Gemini 3.5 Flash Lite as judge.

## Limits
- Small question set (38 and 19). A perfect score on it is still consistent with a true rate of about 91% or higher.
- Questions were drafted with an LLM and verified by a person.
- The judge is an LLM. Its faithfulness and citation labels were not checked by a person.
- All documents follow one university template.
