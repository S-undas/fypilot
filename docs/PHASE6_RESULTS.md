# Phase 6 Results: Gap and Consistency Checks

Phase 6 checks a report the way a careful panel member would: does it contain every part the handbook requires, does it back up what it promises, and does it contradict itself. All numbers come from `python -m src.consistency.report` and the runs behind it.

## How it works
1. **Missing required parts.** The handbook's report requirements were turned into a list by chapter name, split by project type (research-style reports need a methodology chapter and an experimental results chapter, development-style do not) and by stage (a first-semester report does not need a user manual or test metrics). For each required part the system reads the report's own contents list, pulls the opening passages of the matching chapter plus the best search results, and asks the model for present, partial or missing with citations. The References page is checked directly in the PDF, because it is not in the searchable text.
2. **Promises without evidence.** The model lists the promises and claims in the introduction, vision and methodology chapters, and each must quote the report word for word or it is dropped. For each promise, search runs over the implementation, test, results and conclusion chapters and the model decides: shown, not shown or contradicted.
3. **Contradictions.** The model lists specific facts (numbers, tool and model names) from every passage. Facts about the same thing in different places are paired when they carry different numbers or names, and a first model check asks whether they conflict. Only pairs flagged there go through a second check that reads both full passages.

Four reports were used: 50 required-part checks, 225 promises and 1,661 statement pairs. Two of the reports are research-style and two are development-style. F25_181 is a first-semester report, which its own text states, so it is held to the first-semester requirements.

## Finding 1: missing parts are found reliably, and emptied chapters are what a heading rule cannot see
| | present | partial | missing |
|---|---|---|---|
| All four reports (50 checks) | 45 | 3 | 2 |

- The two "missing" flags are real: F25_182 has no user manual and F25_044 has no final conclusion for the whole report, only per-chapter conclusions. A person checked all five flags: both "missing" flags were right, and one of three "partial" flags was right. The two wrong "partial" flags were on chapters that clearly exist.
- A "partial" verdict means "go and look", not "this is a gap". The model sees only a few passages of a long chapter and sometimes concludes the proof is not there.

**Removal test.** Each required chapter was taken out of a copy of the report, or emptied while its heading was kept. This needs no labeling, because the answer is known by construction.

| test | cases | plain heading rule flags it | retrieval + model flags it | model says missing |
|---|---|---|---|---|
| chapter deleted | 44 | 100% (44 of 44) | 100% (44 of 44) | 64% (28 of 44) |
| heading kept, content removed | 44 | 0% (0 of 44) | 100% (44 of 44) | 89% (39 of 44) |

- A plain "is there a heading" rule catches deleted chapters just as well, so the model adds nothing there.
- It earns its place on emptied chapters, where the heading is still there and the rule sees nothing.

## Finding 2: promises are found, but a flag is right only about half the time
| | promises | shown | not shown | contradicted |
|---|---|---|---|---|
| All four reports | 225 | 142 | 68 | 15 |

- 63% of promises were backed up. Real gaps include ultrasonic distance alerts in F25_044 that are only connected and never measured, and Docker, JWT and expert validation in F25_182 that are promised and never shown.
- F25_181 has 9 "contradicted" promises, nearly all because its own conclusion says the AI and story parts were deferred to the second semester. That is correct for a first-semester report, but it shows what a panel would ask about.
- Many flags are about business projections ("scale into a global hub", "monetize through subscriptions"), which no report would evidence. These are counted as flags and lower precision.

**Human check.** A person labeled a sample without seeing the system's verdict.

| | result |
|---|---|
| Promise flags that were right | 50% (5 of 10) |
| Promises marked shown that were not backed up | 0% (0 of 5), one was only partly backed up |

- A "shown" verdict can be trusted. A flag needs a human look, because about half are not real gaps.

## Finding 3: contradictions are real but sparse, and some are the same issue repeated
| | pairs compared | flagged by first check | confirmed by second check |
|---|---|---|---|
| All four reports | 1,661 | 82 | 29 |

- Real conflicts found include a confidence cutoff of 0.4 in one section and 25% in another, two accuracy figures for the same cited paper, a 16 GB and a 4 GB RAM requirement, graphics card requirements that disagree, and Unreal Engine 5.3 against 5.6 across test cases.
- Several confirmed pairs are one issue reported many times, for example the Unreal Engine version in F25_181. So 29 pairs are fewer distinct problems.
- Not every flag is a conflict: related datasets with different sizes and an export-format list that grew were judged conflicts by the system and not by the person.

**Human check.**

| | result |
|---|---|
| Confirmed conflicts that were real | 58% (7 of 12), or 67% counting "partly" |
| Pairs dropped by the second check that were real conflicts | 0% (0 of 4) |
| Pairs judged consistent that were real conflicts | 0% (0 of 3) |

- The second check is not throwing away real conflicts, but those two samples are tiny and prove little either way.

## Finding 4: planted mistakes show what the checkers catch and where they miss
Mistakes were put into copies of the extracted text. A person approved the list: 26 of 28 were kept, and the two dropped were test-case version labels that were not claims.

| kind | planted | found |
|---|---|---|
| Invented promise with a target number, never evidenced | 14 | 93% (13 of 14) |
| Number changed in one of two places that agree | 12 | 58% (7 of 12) |

- Promises are found well. The one miss was never extracted as a promise.
- Contradictions are found about half the time. Tracing the 5 misses: 3 were never captured as facts (a code snippet, a table cell, a number inside a long sentence), 1 produced no candidate pair, and 1 was judged not a conflict. So most of the loss happens before the judge, in picking out facts.

## Conclusions
- Missing parts: dependable for "missing", and the model is worthwhile for emptied chapters. Treat "partial" as a prompt to look.
- Promises: trust "shown", and review flags.
- Contradictions: useful, with a false-alarm rate near 40% and a miss rate near 40% on planted numeric conflicts. It is a screening aid for a human, not a verdict.

## Limits
- Small human samples (10, 5, 12, 4 and 3 labeled) from one person. Rates move a lot with one row.
- Only about half of the candidate statement pairs were compared (1,661 of 3,437): the top 400 per report by similarity plus every pair where both statements carry numbers and share wording. At most 6 facts are taken from a passage.
- Conflicts need a number or a name on both sides to be considered. A wording conflict without either is not found.
- Settings such as the number of opening passages and the number-overlap rule were adjusted after seeing results on these same four reports, with no held-out set. The overlap rule was added because a known 0.4 versus 25% conflict was not being compared.
- The contradictions run was judged by three models because free-tier limits changed during the run: Qwen on Cerebras for 801 pairs, the same Qwen on Groq for 520 and Gemini Flash Lite for 340. The other checks were mostly run on the first two.
- The planted mistakes are artificial. The sites and the invented sentences were generated by code and a model, and a person approved them rather than writing them.
- Only the handbook's requirements were checked. Sub-sections of the report template are not.
- Four reports from one university template.
