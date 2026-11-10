# Final results: plain RAG against RAG with abstention, and where it fails

Produced by `python -m src.evaluation.final` from the saved runs of Phases 3 to 5. No model is called.

## Plain RAG against RAG with abstention (all 57 questions: 38 answerable, 19 unanswerable)

| | Qwen, bare prompt | Qwen, grounded prompt | Qwen, grounded + reranker (plain RAG) | Qwen, + refusal instruction | gpt-oss, plain RAG | gpt-oss, + refusal instruction |
|---|---|---|---|---|---|---|
| answerable questions answered correctly | 36 of 38 | 36 of 38 | 38 of 38 | 38 of 38 | 38 of 38 | 38 of 38 |
| answerable questions wrongly refused | 2 of 38 | 2 of 38 | 0 of 38 | 0 of 38 | 0 of 38 | 0 of 38 |
| unanswerable questions refused | 17 of 19 | 18 of 19 | 19 of 19 | 19 of 19 | 16 of 19 | 18 of 19 |
| unanswerable questions answered anyway | 2 of 19 | 1 of 19 | 0 of 19 | 0 of 19 | 3 of 19 | 1 of 19 |

## Refusal methods on the test split: Qwen, test split (19 answerable, 9 unanswerable)

| | plain | prompt refusal | + threshold (balanced) | + threshold (no-harm) | prompt refusal + threshold (no-harm) | + threshold + NLI (no-harm) |
|---|---|---|---|---|---|---|
| answered correctly | 1.000 | 1.000 | 0.579 | 0.789 | 0.789 | 0.789 |
| wrongly refused | 0.000 | 0.000 | 0.421 | 0.211 | 0.211 | 0.211 |
| refused (all unanswerable) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| unanswerable questions answered anyway | 0 | 0 | 0 | 0 | 0 | 0 |

## Refusal methods on the test split: gpt-oss, test split (19 answerable, 9 unanswerable)

| | plain | prompt refusal | + threshold (balanced) | + threshold (no-harm) | prompt refusal + threshold (no-harm) | + threshold + NLI (no-harm) |
|---|---|---|---|---|---|---|
| answered correctly | 1.000 | 1.000 | 0.579 | 0.789 | 0.789 | 0.789 |
| wrongly refused | 0.000 | 0.000 | 0.421 | 0.211 | 0.211 | 0.211 |
| refused (all unanswerable) | 0.778 | 1.000 | 1.000 | 0.889 | 1.000 | 0.889 |
| unanswerable questions answered anyway | 2 | 0 | 0 | 1 | 0 | 1 |

## Where the question answering fails

| run | right passage missed, then refused | right passage missed, then wrong answer | wrong answer with the right passage | unanswerable answered anyway (near-miss / misleading / off-topic) |
|---|---|---|---|---|
| Qwen, bare prompt | 2 | 0 | 0 | 2 / 0 / 0 |
| Qwen, grounded prompt | 2 | 0 | 0 | 1 / 0 / 0 |
| Qwen, grounded + reranker (plain RAG) | 0 | 0 | 0 | 0 / 0 / 0 |
| Qwen, + refusal instruction | 0 | 0 | 0 | 0 / 0 / 0 |
| gpt-oss, plain RAG | 0 | 0 | 0 | 0 / 3 / 0 |
| gpt-oss, + refusal instruction | 0 | 0 | 0 | 0 / 1 / 0 |

## Where the panel questions fail (document-specific questions a person labeled)

| the person said | system said COVERED | system said NOT COVERED |
|---|---|---|
| covered | 3 | 1 |
| partly covered | 4 | 1 |
| not covered | 9 | 6 |

Gaps the system missed (answered, but the person said not covered): 9. By question type, why or justify: 6, validate or test: 2, other: 1.
