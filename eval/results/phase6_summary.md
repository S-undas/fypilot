# Phase 6 results: gap and consistency checks

Produced by `python -m src.consistency.report`. Each report is checked against the handbook's required parts, for promises it never backs up, and for statements that contradict each other.

## Missing required parts

| document | type | stage | checked | present | partial | missing |
|---|---|---|---|---|---|---|
| F25_044 | rd | 2 | 14 | 13 | 0 | 1 |
| F25_182 | rd | 2 | 14 | 12 | 1 | 1 |
| F25_051 | dev | 2 | 12 | 10 | 2 | 0 |
| F25_181 | dev | 1 | 10 | 10 | 0 | 0 |

Removal test: each required chapter was deleted from a copy of the report, or emptied while its heading stayed.

| test | cases | plain heading rule flags it | retrieval + model flags it | model says missing |
|---|---|---|---|---|
| chapter deleted | 44 | 100% (44 of 44) | 100% (44 of 44) | 64% (28 of 44) |
| heading kept, content removed | 44 | 0% (0 of 44) | 100% (44 of 44) | 89% (39 of 44) |

## Promises without evidence

| document | type | stage | checked | shown | not_shown | contradicted |
|---|---|---|---|---|---|---|
| F25_044 | rd | 2 | 73 | 42 | 27 | 4 |
| F25_182 | rd | 2 | 58 | 38 | 19 | 1 |
| F25_051 | dev | 2 | 51 | 41 | 9 | 1 |
| F25_181 | dev | 1 | 43 | 21 | 13 | 9 |

## Contradictions inside a report

| document | pairs compared | flagged by first check | confirmed by second check |
|---|---|---|---|
| F25_044 | 413 | 14 | 4 |
| F25_182 | 414 | 16 | 2 |
| F25_051 | 424 | 11 | 6 |
| F25_181 | 410 | 41 | 17 |

Pairs judged by: cerebras/qwen-3.8-27b (801), groq/qwen/qwen3.8-27b (520), gemini/gemini-3.5-flash-lite (340).

## Human check

A person labeled a sample of the flags without seeing the system's verdict.

| measure | value |
|---|---|
| promise flags that were right (10 labeled) | 50% (lenient 50%) |
| promises marked shown that were not backed up (5 labeled) | 0% |
| conflicts confirmed that were real (12 labeled) | 58% (lenient 67%) |
| conflicts dropped by the second check that were real (4 labeled) | 0% |
| pairs judged consistent that were real conflicts (3 labeled) | 0% |

## Planted mistakes

Mistakes were inserted into copies of the extracted text, then the checkers were run on the copies.

| kind | planted | found |
|---|---|---|
| unproven promise (extracted by the system) | 14 | 93% (13 of 14) |
| unproven promise (extracted and flagged) | 14 | 93% (13 of 14) |
| contradiction (number changed in one place) | 12 | 58% (7 of 12) |
