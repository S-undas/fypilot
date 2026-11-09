# Phase 5 results: panel questions

Produced by `python -m src.panel.report`. Questions are generated per document and category, answered from the document, and marked NOT COVERED when the passages do not answer them.

## Document-specific vs generic questions

| | document-specific questions | generic questions |
|---|---|---|
| questions | 83 | 84 |
| marked NOT COVERED | 34% (28 of 83) | 4% (3 of 84) |
| mean similarity to the document (specificity) | 0.740 | 0.624 |
| covered answers that address the question (judge) | 96% (53 of 55) | 99% (80 of 81) |
| covered answers supported by their sources (judge) | 96% (53 of 55) | 99% (80 of 81) |

## Document-specific questions by category

| category | questions | not covered |
|---|---|---|
| motivation | 12 | 8% (1 of 12) |
| novelty | 12 | 8% (1 of 12) |
| methodology | 12 | 33% (4 of 12) |
| evaluation | 12 | 42% (5 of 12) |
| feasibility | 11 | 64% (7 of 11) |
| limitations | 12 | 17% (2 of 12) |
| ethics | 12 | 67% (8 of 12) |

## Document-specific questions by document

| doc | questions | not covered |
|---|---|---|
| F25_044 | 21 | 33% (7 of 21) |
| F25_051 | 21 | 38% (8 of 21) |
| F25_181 | 21 | 29% (6 of 21) |
| F25_182 | 20 | 35% (7 of 20) |

## Human check of coverage status

A person looked up each sampled question in the documents without seeing the system's status.

| | document-specific | generic | all |
|---|---|---|---|
| questions labeled | 24 | 12 | 36 |
| marked NOT COVERED by the system | 8 | 0 | 8 |
| not covered according to the person | 15 | 1 | 16 |
| NOT COVERED precision | 0.750 | - | 0.750 |
| NOT COVERED recall | 0.400 | 0.000 | 0.375 |
| precision if partly covered counts as not covered | 0.875 | - | 0.875 |
| recall if partly covered counts as not covered | 0.350 | 0.000 | 0.269 |
| wrongly marked NOT COVERED | 2 | 0 | 2 |
| wrongly marked covered | 9 | 1 | 10 |
| partly covered (counted as covered) | 5 | 5 | 10 |
| mean relevance rating (1 to 3) | 2.875 | 2.667 | 2.806 |
