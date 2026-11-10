# Final Results: Plain RAG Against RAG With Abstention, and Where It Fails

This brings the results of Phases 1 to 6 into one place and adds the error analysis. The comparison tables come from `python -m src.evaluation.final`, which reads the saved runs and calls no model. The other numbers come from the phase reports named in each row.

## At a glance
| Component | What was measured | Result | Details |
|---|---|---|---|
| Reading documents | Heading and page accuracy | 506 chunks from 7 documents. 388 of 389 outline headings matched. All 938 start and end points were on the page they cite. | `PHASES.md`, Phase 1 |
| Finding passages | Was the right passage retrieved, 38 answerable questions | Hybrid search plus reranker: right passage first for 84.2%, in the top 5 for 94.7%. Plain hybrid: 71.1% and 86.8%. Dense only: 55.3% and 86.8%. | `PHASE3_RESULTS.md` |
| Cited answers | Correct answers and refusals, 57 questions | 38 of 38 answerable answered correctly and 19 of 19 unanswerable refused (Qwen with reranker). | `PHASE3_RESULTS.md` |
| Grader | Agreement with a person on 20 answers | 19 of 20 on refuse or answer, 11 of 11 on correctness. | `PHASE3_RESULTS.md` |
| Refusing | Plain RAG against four refusal methods | A refusal instruction helps at no cost. A score threshold costs 21% to 42% of correct answers. An NLI check did not help. | `PHASE4_RESULTS.md` |
| Panel questions | Marked NOT COVERED, checked by a person | 34% of document-specific questions against 4% of generic ones. When it says NOT COVERED it is right 75% of the time, but it finds only 40% of the gaps. | `PHASE5_RESULTS.md` |
| Gap and consistency checks | Missing parts, promises, contradictions | Missing parts flagged correctly in the removal test. Promise flags right 50% of the time, contradiction flags 58%. 93% of planted promises and 58% of planted number changes found. | `PHASE6_RESULTS.md` |

## Plain RAG against RAG with abstention
This is the comparison the course asks for. "Plain RAG" is the grounded prompt with hybrid search and the reranker, and no refusal instruction. All 57 questions are used: 38 answerable and 19 unanswerable.

| | Qwen, bare prompt | Qwen, grounded prompt | Qwen, plain RAG | Qwen, + refusal instruction | gpt-oss, plain RAG | gpt-oss, + refusal instruction |
|---|---|---|---|---|---|---|
| answerable, answered correctly | 36 of 38 | 36 of 38 | 38 of 38 | 38 of 38 | 38 of 38 | 38 of 38 |
| answerable, wrongly refused | 2 of 38 | 2 of 38 | 0 of 38 | 0 of 38 | 0 of 38 | 0 of 38 |
| unanswerable, refused | 17 of 19 | 18 of 19 | 19 of 19 | 19 of 19 | 16 of 19 | 18 of 19 |
| unanswerable, answered anyway | 2 of 19 | 1 of 19 | 0 of 19 | 0 of 19 | 3 of 19 | 1 of 19 |

- The reranker is the biggest single improvement for Qwen. It removes the two wrongly refused answerable questions, because the right passage now reaches the model.
- Qwen is cautious by nature, so a refusal instruction cannot improve on a perfect 19 of 19. The less cautious gpt-oss answers 3 of 19 unanswerable questions, and the instruction cuts that to 1.
- The score threshold and the NLI check are on the test split in `PHASE4_RESULTS.md`. Neither beat the plain instruction, and the threshold lost correct answers (down to 58% to 79% answered correctly on the test split, from 100%).

## Where it fails
**Question answering.**

| run | right passage missed, then refused | right passage missed, then wrong answer | wrong answer with the right passage | unanswerable answered anyway (near-miss / misleading / off-topic) |
|---|---|---|---|---|
| Qwen, bare prompt | 2 | 0 | 0 | 2 / 0 / 0 |
| Qwen, grounded prompt | 2 | 0 | 0 | 1 / 0 / 0 |
| Qwen, plain RAG | 0 | 0 | 0 | 0 / 0 / 0 |
| Qwen, + refusal instruction | 0 | 0 | 0 | 0 / 0 / 0 |
| gpt-oss, plain RAG | 0 | 0 | 0 | 0 / 3 / 0 |
| gpt-oss, + refusal instruction | 0 | 0 | 0 | 0 / 1 / 0 |

- **Retrieval failures** are the main error for Qwen. When the right passage is not in the top 5, Qwen refuses, which is the safe behavior. It is a lost answer, not a wrong one.
- **Hallucinations** come from questions that look answerable. gpt-oss answers the "misleading" kind, where related but different information exists, such as a target value in place of a measured one. A "near-miss" slipped through only for Qwen when the reranker was not used (twice with the bare prompt, once with the grounded prompt).
- **Off-topic questions** were never answered by any run.

**Panel questions.**

| the person said | system said COVERED | system said NOT COVERED |
|---|---|---|
| covered | 3 | 1 |
| partly covered | 4 | 1 |
| not covered | 9 | 6 |

The system missed 9 gaps, answering a question the person judged not covered. By question type, 6 were "why" or "justify" questions, 2 were "how did you validate or test" questions, and 1 was another kind. The model restates the facts around a choice and answers as if the reason were given.

**Gap and consistency checks.** The 5 planted number changes that were missed split into 3 that were never captured as facts, 1 that formed no candidate pair, and 1 that was judged not a conflict. The errors that matter most come before the judge.

## What the results say
- **Does abstention help?** Only where the model is not already cautious. A refusal instruction fixed gpt-oss's hallucinations at no cost. A score threshold and an NLI check cost correct answers or missed misleading answers, so we recommend the instruction alone.
- **What limits quality?** Retrieval for answers, justification questions for panel prep, and fact extraction for the gap checks. In each case the later stage works when it gets the right input.
- **Trust.** A refusal, a NOT COVERED label and a "shown" verdict can be relied on. A missing answer does not prove the document is complete, and a flag from the gap checks needs a human look.

## Limits
- Small test sets: 38 answerable and 19 unanswerable questions, a few dozen human labels in Phases 5 and 6. One question moves a rate by about 5 points, and several rates sit at 100%, which hides small differences.
- Four reports from one university template, and the handbook of that university. Results may not transfer to other templates, and the course goal was not to generalize.
- The questions were drafted with a model and checked by a person. The grader is a model, validated on 20 answers. Its labels for faithfulness and citation support were not checked by a person.
- Thresholds and several Phase 6 settings were tuned on the same documents they are scored on.
- Free-tier models with daily limits. Part of the Phase 6 contradiction judging ran on a different model from the rest.
- Heading detection is tuned to the template, and text inside figures is not extracted.
