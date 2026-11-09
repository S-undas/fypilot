# Phase 5 Results: Panel Questions

Phase 5 generates the questions an evaluation panel is likely to ask about a student's project, grouped by category, and answers each one from the student's own documents or marks it NOT COVERED. The NOT COVERED questions show the student where the documents are weak. All numbers come from `python -m src.panel.report` and the runs behind it.

## How it works
1. For each of seven categories (motivation, novelty, methodology, evaluation, feasibility, limitations, ethics), find the passages most relevant to the category and ask the model for three short, single-topic questions specific to the project, including at least one that challenges a choice or result.
2. Answer each question with hybrid search, reranking and a cited answer. The prompt tells the model to reply exactly "NOT COVERED" when the passages do not answer.
3. Write a prep sheet per document: a weak-spots list, then the questions by category with draft answers and sources.

Four reports were used: 83 document-specific questions (one near-duplicate was dropped) and a baseline of 21 fixed generic questions per report (84 in total).

## Finding 1: questions built from the document expose gaps that generic questions cannot
| | Document-specific | Generic |
|---|---|---|
| Questions | 83 | 84 |
| Marked NOT COVERED | **34%** (28) | **4%** (3) |
| Mean similarity to the document | 0.74 | 0.62 |

- Generic questions such as "What are the main limitations?" can always be loosely answered from something in the report, so almost nothing is flagged.
- Document-specific questions ask about this project's actual choices and claims, so they find what the report leaves out.
- NOT COVERED rates by category, document-specific: ethics 67%, feasibility 64%, evaluation 42%, methodology 33%, limitations 17%, motivation 8%, novelty 8%. This fits typical reports, which say little about ethics and feasibility.

## Finding 2: NOT COVERED is usually right, but the system misses many gaps
A person checked 36 sampled questions (24 document-specific, 12 generic) against the reports without seeing the system's status.

| Document-specific questions | Partly covered counts as covered | Partly covered counts as not covered |
|---|---|---|
| Precision (when the system says NOT COVERED, is it right?) | 75% (6 of 8) | 88% (7 of 8) |
| Recall (how many real gaps did it find?) | 40% (6 of 15) | 35% (7 of 20) |

- When it flags a question, it is usually right. It missed most gaps, though: of 15 document-specific questions the person judged not covered, the system answered 9 anyway.
- The misses are mostly open-ended "why did you choose X over Y" and "how did you validate Z" questions. The report states what was chosen, so the model restates those facts and answers as if the question were covered, while the justification or validation is not in the documents.
- This is a harder problem than the factual questions of Phases 3 and 4, where the same model refused reliably. A question that asks for a reason invites a plausible answer built from related facts.
- The LLM judge did not catch this: it rated 96% of the covered document-specific answers as addressing the question. The person's labels are the reliable measure.

## Finding 3: the questions themselves are good
- 30 of the 36 sampled questions were rated 3 (a sensible question a real panel could ask), 5 were rated 2 and 1 was rated 1.
- Mean relevance: 2.88 for document-specific and 2.67 for generic questions.

## Conclusions
- Generating questions from the documents is clearly better than generic questions at surfacing weak spots.
- The NOT COVERED label can be trusted when present, but its absence does not prove coverage, especially for justification questions.
- A natural next step, not done here, is a stricter check for "why" and "how did you validate" questions: require that the passages state the reason or the validation itself, not only the facts around it.

## Limits
- Small human sample: 36 questions, labeled by one person.
- The boundary between "partly covered" and "not covered" is a judgment call. Results are shown both ways.
- Four reports that share one university template.
- Generated questions can carry a premise the report does not support. The relevance rating handles this only partly.
