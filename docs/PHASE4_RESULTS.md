# Phase 4 Results: Abstention

Phase 4 asks whether the system can be made to refuse when the documents do not support an answer, and whether each refusal method helps or hurts. All numbers come from `python -m src.evaluation.abstention_eval` and the runs behind it. Thresholds were chosen on the dev split only, and the tables below show the test split.

## What was compared
Each method was applied to two generators: Qwen 3.8 27B (cautious) and `gpt-oss-120b` (less cautious). All use hybrid search with reranking and the top 5 passages.

| Method | What it does |
|---|---|
| Plain | Prompt without any grounding or refusal instruction |
| Prompt refusal | Adds an instruction to say so when the passages do not contain the answer |
| Score threshold | Refuses before answering when the best evidence score is below a threshold |
| NLI check | After answering, checks that each sentence of the answer is entailed by the cited passages, and refuses if not |

The score threshold has two operating points, both set on dev. Balanced weighs keeping good answers and refusing bad ones equally. No-harm is the strictest threshold that still keeps about 94% of good dev answers.

## Finding 1: a refusal instruction in the prompt is the best fix
Test split, 19 answerable and 9 unanswerable questions.

| | Qwen plain | Qwen prompt refusal | gpt-oss plain | gpt-oss prompt refusal |
|---|---|---|---|---|
| Answerable answered correctly | 100% | 100% | 100% | 100% |
| Unanswerable refused | 100% | 100% | 78% | **100%** |
| Unanswerable answered anyway | 0 | 0 | 2 | **0** |

- Qwen already refuses every unanswerable question, so there is nothing for any method to fix.
- `gpt-oss-120b` answered two misleading questions on test (it used a related number as if it answered the question). One added sentence in the prompt removed both, at no cost to correct answers.
- On dev, one misleading question is still answered by `gpt-oss-120b` with the refusal prompt. A prompt instruction is not a guarantee.

## Finding 2: the score threshold works, but it costs correct answers
Which signal separates answerable from unanswerable questions? AUC is the chance that a random answerable question scores higher than a random unanswerable one (1.0 is perfect, 0.5 is chance).

| Signal | AUC dev | AUC test | Misleading (dev) | Off-topic (dev) |
|---|---|---|---|---|
| Best dense similarity in the top 5 | 0.60 | 0.66 | 0.39 | 0.95 |
| Reranker score of the top passage | **0.80** | **0.87** | 0.83 | 1.00 |

- The reranker score is a much better signal than raw similarity. Dense similarity cannot detect misleading questions at all (below 0.5), because they sit close to relevant text.
- Both signals separate off-topic questions well.

What happens when the reranker score is used as a gate (test split):

| | Plain | Balanced threshold | No-harm threshold |
|---|---|---|---|
| Qwen: answerable answered correctly | 100% | 58% | 79% |
| Qwen: unanswerable refused | 100% | 100% | 100% |
| gpt-oss: answerable answered correctly | 100% | 58% | 79% |
| gpt-oss: unanswerable refused | 78% | 100% | 89% |

- The balanced threshold refuses every unanswerable question but also refuses 42% of answerable ones. That is a poor trade.
- The no-harm threshold lost one of 19 answerable questions on dev by design, but lost four of 19 (21%) on test. A threshold tuned on 19 questions does not transfer reliably.
- The gate refuses answerable questions whose best passage scores low, typically answers that sit in tables or short lists.
- Combining the refusal prompt with the no-harm gate gives `gpt-oss-120b` 100% refusal on test, but with the same 21% loss of correct answers. The prompt alone already achieved 100%.

## Finding 3: the NLI check did not help
NLI means natural language inference: a model that says whether one text entails another. Here it checks each sentence of an answer against the cited passages. Three versions were tried, and each change was made after inspecting why correct answers scored badly.

| Version | Change | Result |
|---|---|---|
| 1 | Weakest sentence must be entailed | Refused most correct answers (test: only 42% of Qwen's and 11% of gpt-oss's correct answers survived) |
| 2 | Dropped list lead-ins ("The metrics are:") and scored the share of sentences entailed | No longer harmful, but 28 of 74 correct answers still scored zero, so no usable threshold |
| 3 | Matched each sentence to short 55-word windows of the passage | Better separation (test AUC 0.83 to 0.90) but 21 of 74 correct answers still scored zero |

- NLI models are trained on short sentence pairs. A fact sitting in a long list or a flattened table row, such as a hardware requirement, is often scored as unsupported even when it is plainly there.
- Misleading answers can pass. One bad answer scored a perfect 1.0: it reported a real 52% reduction but named the wrong comparison group. The numbers are in the passage, so the sentence looks supported.
- There were only three bad answers in total across both generators, so no threshold could be tuned on them. The no-harm rule used the good answers only, and it came out at zero, which means the gate refuses nothing.

## Conclusions
- The cheapest effective method is the refusal instruction in the prompt. It fixed the hallucinations on test with no loss of correct answers.
- The score threshold with the reranker score is a usable, model-independent safety net for off-topic questions, but its threshold is unstable on a small set, and the price is lost correct answers.
- The NLI check, as implemented, adds nothing here and is not recommended. This is a negative result worth reporting.
- Both generators were already strong. The set has very few bad answers to catch, which limits how much any method can show.

## Limits
- Small set: 19 answerable and 9 or 10 unanswerable questions per split. One question moves a rate by 5 to 10 points.
- Only three bad answers exist in total, so NLI results are descriptive, not conclusive.
- The NLI method was changed twice after seeing failures. Thresholds were still chosen on dev only, and all versions are reported.
- Judge labels come from an LLM judge checked against a person on a sample.
