# Phase 4 results: abstention

Produced by `python -m src.evaluation.abstention_eval`. All thresholds are chosen on dev only. Two operating points are used: balanced (equal weight on keeping good answers and refusing bad ones) and no-harm (the strictest threshold that keeps at least 94% of good dev answers).

## Score threshold: which signal separates answerable from unanswerable?

AUC is the chance that a random answerable question scores higher than a random unanswerable one (1.0 is perfect, 0.5 is chance).

| signal | AUC dev | AUC test | dev: near-miss | dev: misleading | dev: off-topic | balanced threshold | no-harm threshold |
|---|---|---|---|---|---|---|---|
| best dense similarity in the top 5 | 0.600 | 0.655 | 0.589 | 0.386 | 0.947 | 0.708 | 0.611 |
| reranker score of the top passage | 0.800 | 0.865 | 0.705 | 0.825 | 1.000 | 4.428 | -0.919 |

Chosen signal (higher dev AUC): **reranker score of the top passage**.

## NLI check: does the evidence entail the answer?

Each factual sentence of an answer is checked against the passages it cites. The score is the share of sentences entailed (probability at least 0.5). List lead-ins such as "The metrics are:" and refusal sentences are not counted. Passages are cut into overlapping 55-word windows and each sentence is matched to its best window.

- Good answers (answerable and judged correct): 38 on dev, 36 on test.
- Bad answers (answers to unanswerable questions or not judged correct): 1 on dev, 2 on test. Too few to tune or evaluate a threshold on bad answers, so the no-harm rule uses only the good answers.
- No-harm threshold from dev: 0.000. AUC on test: 0.903.
- Support score of every bad answer: gptoss/F044-N2 = 0.000, gptoss/F051-N2 = 1.000, gptoss/HB-N3 = 0.000

First NLI version (weakest sentence counted, list lead-ins treated as claims), threshold 0.109 chosen on dev, with the balanced score threshold:

| generator | answered correctly (test) | unanswerable refused (test) |
|---|---|---|
| Qwen 3.8 27B | 0.421 | 1.000 |
| gpt-oss-120b | 0.105 | 1.000 |

## Qwen 3.8 27B: test split

| metric | plain | prompt refusal | + threshold (balanced) | + threshold (no-harm) | prompt refusal + threshold (no-harm) | + threshold + NLI (no-harm) |
|---|---|---|---|---|---|---|
| answerable questions | 19 | 19 | 19 | 19 | 19 | 19 |
| answered correctly | 1.000 | 1.000 | 0.579 | 0.789 | 0.789 | 0.789 |
| wrongly refused | 0.000 | 0.000 | 0.421 | 0.211 | 0.211 | 0.211 |
| unanswerable questions | 9 | 9 | 9 | 9 | 9 | 9 |
| refused (all unanswerable) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| refused (near-miss) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| refused (misleading) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| refused (off-topic) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| refusals that were justified | 1.000 | 1.000 | 0.529 | 0.692 | 0.692 | 0.692 |
| unanswerable questions answered anyway | 0 | 0 | 0 | 0 | 0 | 0 |

## Qwen 3.8 27B: dev split

| metric | plain | prompt refusal | + threshold (balanced) | + threshold (no-harm) | prompt refusal + threshold (no-harm) | + threshold + NLI (no-harm) |
|---|---|---|---|---|---|---|
| answerable questions | 19 | 19 | 19 | 19 | 19 | 19 |
| answered correctly | 1.000 | 1.000 | 0.632 | 0.947 | 0.947 | 0.947 |
| wrongly refused | 0.000 | 0.000 | 0.368 | 0.053 | 0.053 | 0.053 |
| unanswerable questions | 10 | 10 | 10 | 10 | 10 | 10 |
| refused (all unanswerable) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| refused (near-miss) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| refused (misleading) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| refused (off-topic) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| refusals that were justified | 1.000 | 1.000 | 0.588 | 0.909 | 0.909 | 0.909 |
| unanswerable questions answered anyway | 0 | 0 | 0 | 0 | 0 | 0 |

## gpt-oss-120b: test split

| metric | plain | prompt refusal | + threshold (balanced) | + threshold (no-harm) | prompt refusal + threshold (no-harm) | + threshold + NLI (no-harm) |
|---|---|---|---|---|---|---|
| answerable questions | 19 | 19 | 19 | 19 | 19 | 19 |
| answered correctly | 1.000 | 1.000 | 0.579 | 0.789 | 0.789 | 0.789 |
| wrongly refused | 0.000 | 0.000 | 0.421 | 0.211 | 0.211 | 0.211 |
| unanswerable questions | 9 | 9 | 9 | 9 | 9 | 9 |
| refused (all unanswerable) | 0.778 | 1.000 | 1.000 | 0.889 | 1.000 | 0.889 |
| refused (near-miss) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| refused (misleading) | 0.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.500 |
| refused (off-topic) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| refusals that were justified | 1.000 | 1.000 | 0.529 | 0.667 | 0.692 | 0.667 |
| unanswerable questions answered anyway | 2 | 0 | 0 | 1 | 0 | 1 |

## gpt-oss-120b: dev split

| metric | plain | prompt refusal | + threshold (balanced) | + threshold (no-harm) | prompt refusal + threshold (no-harm) | + threshold + NLI (no-harm) |
|---|---|---|---|---|---|---|
| answerable questions | 19 | 19 | 19 | 19 | 19 | 19 |
| answered correctly | 1.000 | 1.000 | 0.632 | 0.947 | 0.947 | 0.947 |
| wrongly refused | 0.000 | 0.000 | 0.368 | 0.053 | 0.053 | 0.053 |
| unanswerable questions | 10 | 10 | 10 | 10 | 10 | 10 |
| refused (all unanswerable) | 0.900 | 0.900 | 1.000 | 0.900 | 0.900 | 0.900 |
| refused (near-miss) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| refused (misleading) | 0.667 | 0.667 | 1.000 | 0.667 | 0.667 | 0.667 |
| refused (off-topic) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| refusals that were justified | 1.000 | 1.000 | 0.588 | 0.900 | 0.900 | 0.900 |
| unanswerable questions answered anyway | 1 | 1 | 0 | 1 | 1 | 1 |

## Answers changed by each gate (question ids)

- Qwen 3.8 27B, + threshold (balanced): changed 15 answers: F044-A1, F044-A4, F182-A1, F182-A2, F182-A3, F182-A6, F182-A7, F182-A8, F051-A1, F051-A6, F051-A7, F051-A8, F181-A4, F181-A8, HB-A5
- Qwen 3.8 27B, + threshold (no-harm): changed 5 answers: F044-A4, F182-A1, F182-A3, F182-A7, F051-A8
- Qwen 3.8 27B, prompt refusal + threshold (no-harm): changed 5 answers: F044-A4, F182-A1, F182-A3, F182-A7, F051-A8
- gpt-oss-120b, + threshold (balanced): changed 18 answers: F044-A1, F044-A4, F044-N2, F182-A1, F182-A2, F182-A3, F182-A6, F182-A7, F182-A8, F051-A1, F051-A6, F051-A7, F051-A8, F051-N2, F181-A4, F181-A8, HB-A5, HB-N3
- gpt-oss-120b, + threshold (no-harm): changed 6 answers: F044-A4, F182-A1, F182-A3, F182-A7, F051-A8, HB-N3
- gpt-oss-120b, prompt refusal + threshold (no-harm): changed 5 answers: F044-A4, F182-A1, F182-A3, F182-A7, F051-A8
