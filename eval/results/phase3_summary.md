# Phase 3 results

Produced by `python -m src.evaluation.report`. Judge labels come from an LLM judge, and its agreement with a person is reported at the end.

## Retrieval comparison (answerable questions, search inside the right document)

How to read: recall@k is the share of questions whose right passage is in the top k results. mrr is the average of 1 divided by the rank of the right passage (1 is best). section_hit@5 means a passage from the right section is in the top 5.

**dev**

| search | n | recall@1 | recall@3 | recall@5 | recall@10 | section_hit@5 | mrr |
|---|---|---|---|---|---|---|---|
| dense | 19 | 0.526 | 0.789 | 0.895 | 0.895 | 0.895 | 0.67 |
| bm25 | 19 | 0.579 | 0.789 | 0.895 | 0.895 | 0.895 | 0.702 |
| hybrid | 19 | 0.684 | 0.842 | 0.842 | 0.895 | 0.842 | 0.762 |
| hybrid+rerank | 19 | 0.789 | 0.842 | 0.947 | 0.947 | 0.947 | 0.839 |

**test**

| search | n | recall@1 | recall@3 | recall@5 | recall@10 | section_hit@5 | mrr |
|---|---|---|---|---|---|---|---|
| dense | 19 | 0.579 | 0.789 | 0.842 | 0.895 | 0.842 | 0.683 |
| bm25 | 19 | 0.632 | 1.0 | 1.0 | 1.0 | 1.0 | 0.798 |
| hybrid | 19 | 0.737 | 0.895 | 0.895 | 0.895 | 0.895 | 0.816 |
| hybrid+rerank | 19 | 0.895 | 0.947 | 0.947 | 1.0 | 0.947 | 0.918 |

**all**

| search | n | recall@1 | recall@3 | recall@5 | recall@10 | section_hit@5 | mrr |
|---|---|---|---|---|---|---|---|
| dense | 38 | 0.553 | 0.789 | 0.868 | 0.895 | 0.868 | 0.677 |
| bm25 | 38 | 0.605 | 0.895 | 0.947 | 0.947 | 0.947 | 0.75 |
| hybrid | 38 | 0.711 | 0.868 | 0.868 | 0.895 | 0.868 | 0.789 |
| hybrid+rerank | 38 | 0.842 | 0.895 | 0.947 | 0.974 | 0.947 | 0.879 |

## Answers and refusals: dev

How to read: answered correctly and wrongly refused are shares of answerable questions. The refused rows are shares of unanswerable questions the system declined to answer. Labels come from an LLM judge, whose agreement with a person is reported at the end.

| metric | qwen_main | qwen_rerank | qwen_bare | gptoss_dev |
|---|---|---|---|---|
| answerable questions | 19 | 19 | 19 | 19 |
| answered correctly | 0.895 | 1.000 | 0.895 | 0.895 |
| wrongly refused | 0.105 | 0.000 | 0.105 | 0.000 |
| right passage in top 5 | 0.842 | 0.947 | 0.842 | 0.842 |
| answer supported by retrieved passages | 1.000 | 1.000 | 1.000 | 1.000 |
| citations support the answer | 1.000 | 1.000 | 1.000 | 0.947 |
| unanswerable questions | 10 | 10 | 10 | 10 |
| refused (all unanswerable) | 1.000 | 1.000 | 1.000 | 0.600 |
| refused (near-miss) | 1.000 | 1.000 | 1.000 | 0.600 |
| refused (misleading) | 1.000 | 1.000 | 1.000 | 0.333 |
| refused (off-topic) | 1.000 | 1.000 | 1.000 | 1.000 |
| refusals that were justified | 0.833 | 1.000 | 0.833 | 1.000 |

## Answers and refusals: test

How to read: answered correctly and wrongly refused are shares of answerable questions. The refused rows are shares of unanswerable questions the system declined to answer. Labels come from an LLM judge, whose agreement with a person is reported at the end.

| metric | qwen_main | qwen_rerank | qwen_bare |
|---|---|---|---|
| answerable questions | 19 | 19 | 19 |
| answered correctly | 1.000 | 1.000 | 1.000 |
| wrongly refused | 0.000 | 0.000 | 0.000 |
| right passage in top 5 | 0.895 | 0.947 | 0.895 |
| answer supported by retrieved passages | 1.000 | 1.000 | 1.000 |
| citations support the answer | 1.000 | 1.000 | 1.000 |
| unanswerable questions | 9 | 9 | 9 |
| refused (all unanswerable) | 0.889 | 1.000 | 0.778 |
| refused (near-miss) | 0.800 | 1.000 | 0.600 |
| refused (misleading) | 1.000 | 1.000 | 1.000 |
| refused (off-topic) | 1.000 | 1.000 | 1.000 |
| refusals that were justified | 1.000 | 1.000 | 1.000 |

## Answers and refusals: all

How to read: answered correctly and wrongly refused are shares of answerable questions. The refused rows are shares of unanswerable questions the system declined to answer. Labels come from an LLM judge, whose agreement with a person is reported at the end.

| metric | qwen_main | qwen_rerank | qwen_bare | gptoss_dev |
|---|---|---|---|---|
| answerable questions | 38 | 38 | 38 | 19 |
| answered correctly | 0.947 | 1.000 | 0.947 | 0.895 |
| wrongly refused | 0.053 | 0.000 | 0.053 | 0.000 |
| right passage in top 5 | 0.868 | 0.947 | 0.868 | 0.842 |
| answer supported by retrieved passages | 1.000 | 1.000 | 1.000 | 1.000 |
| citations support the answer | 1.000 | 1.000 | 1.000 | 0.947 |
| unanswerable questions | 19 | 19 | 19 | 10 |
| refused (all unanswerable) | 0.947 | 1.000 | 0.895 | 0.600 |
| refused (near-miss) | 0.900 | 1.000 | 0.800 | 0.600 |
| refused (misleading) | 1.000 | 1.000 | 1.000 | 0.333 |
| refused (off-topic) | 1.000 | 1.000 | 1.000 | 1.000 |
| refusals that were justified | 0.900 | 1.000 | 0.895 | 1.000 |

## Judge agreement with human labels (qwen_main)

| label | labeled by a person | agreement | kappa |
|---|---|---|---|
| refused or answered | 20 | 0.950 | 0.900 |
| correct, partial or incorrect | 11 | 1.000 | 1.000 |
