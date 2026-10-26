# Pitfalls

This is not a usage guide. It is a record of the failures that changed the
design.

The short version: I wanted a small memory kernel, then spent repeated runs and
tokens learning that most of the problem was not the retrieval algorithm. If
these notes save someone else one full evaluation cycle, they did their job.

## If you remember one thing

**Controlling what goes into the model matters more than tuning the retriever.**

Retrieval was already good enough (`all@10` 94.7%); the real limit was how much
boilerplate was in the context and whether the decisive sentence survived
truncation.

## 1. Context budget

| Characters per session | Correct | Tokens/question | Cost |
|---|---:|---:|---:|
| 4,000 | 101/133 | 9,932 | ¥1.68 |
| 12,000 | 114/133 | 23,823 | ¥3.53 |

A targeted check was more direct: on 17 wrong answers, changing only the
character limit fixed 12 immediately.

But 12,000 characters made tokens 2.4× larger. The real fix was the next
section, not a larger budget.

## 2. Assistant replies are 88% of the corpus

Across the whole corpus, user turns are only **12%** of characters; assistant
turns are **88%**, mostly "Congratulations! Here are five suggestions..."
boilerplate. It does not help answer the question and dilutes attention.

Clipping assistant replies to 200 characters:

| | Correct | Tokens/question | Cost |
|---|---:|---:|---:|
| Full text | 114/133 | 23,823 | ¥3.53 |
| **Assistant clipped to 200** | **123/133** | **6,743** | **¥1.18** |

**Tokens fell 3.5× while accuracy rose 6.8 points.** Less noise beat more
context.

## 3. "I don't know" is not always a model failure

At 4,000 characters, 13 questions refused; at 12,000, only 4 did.

The cost signal was stronger: refused answers used **2,970 completion tokens**,
while correct answers used **551** (5.4×). The model does not give up
immediately; it searches, fails, and then says it cannot find the answer.

When you see "I don't know", first check whether the material reached the
context. Do not blame the model by default.

## 4. Fusion: concatenation starves the lexical arm

The first fusion put vector hits first and appended lexical hits. If the vector
arm returned 60 hits and `limit` was 60, lexical had no room. "Hybrid" was
vector-only in disguise; the identical numbers from both arms were the clue.

Pure RRF (k=60) was worse: recall@3 fell from 79.9% to **60.1%**.

The downstream aggregator takes `max()` per session. RRF emits a total order
where the first and last items differ by a tiny amount; the magnitude needed to
choose the best exchange per session is gone.

The final design: **min-max scale each arm, then blend by weight** (vector 1.0 /
lexical 0.5). The lexical arm can contribute, and the magnitude survives.

## 5. Rerank scores can destroy the ordering

This was self-inflicted: cross-encoder logits (roughly −11..+11) and fused
scores (0..2) were returned in one list. A downstream re-sort by `score` pushed
negative-scoring head items below positive-scoring tail items.

Recall@3 fell from 79.9% to **31.7%** — visibly nonsensical.

The fix: emit **rank-monotone scores** (`1/rank`), so a downstream max/sort
cannot invert head and tail.

## 6. Thinking is on by default and consumes max_tokens

DeepSeek reasoning models enable thinking with effort=high by default. The
chain of thought and the answer share one `max_tokens` budget. With 4,000
tokens, about 6% of questions spent the whole budget on thinking and returned
empty content. It looked like a refusal; it was a budget failure.

Retries made it worse: one retry took 34 seconds and 5× the tokens because the
whole context was resent.

Rule: **budget the first attempt generously** (8,000+; this project used
16,000). Do not rely on retries to repair a budget error.

## 7. Silent drops are worse than errors

When an intention's `due` failed to parse, the code did `continue`: no error,
no counter, `no_change` returned. The caller could not see that a commitment
had been dropped.

A related bug: expired intentions were rendered as `expired` but never written
back. `list(statuses=("active",))` still returned them after 20 days, so counts
and dashboards were wrong.

Lesson: **if a field exists, fill it.** A `dropped_intentions` field that is
always 0 is worse than no field; it implies the problem is handled.

## 8. Time parsing boundaries

- Chinese and English relative-time expressions such as `in 2 hours` and
  `3 days later` used to return `None`.
- Relative time must be parsed against the **session date**
  (`parse_due(text, now=session_date)`), otherwise the computed date is wrong.

## 9. Event time is not session time

The conversation says "I took a baking class yesterday" or "my 10th outdoor
run", while the dataset provides only a session date.

Storing only the session date forces the model to use it as the event date.
Typical consequence: two events were 81 days apart, but the gold answer was 15
weeks. The model's arithmetic was exact; the anchor was wrong.

**This is a storage-layer problem, not a model arithmetic problem.** I have not
verified that converting relative expressions to absolute dates at write time
fixes this class; that needs another experiment. What is certain is that a
session date alone is not enough.

## 10. Two storage traps

- Episode ids use `uuid5(user:session:minute:text)`: the same session, minute and
  text collide, and a batch upsert raises `DuplicateIDError` in Chroma.
- Some restricted environments silently block `rm -rf`. The old index survived,
  the new collection used 384 dimensions while the old one used 1536, and every
  query failed with a shape mismatch. Use a cleanup method that raises
  (`shutil.rmtree`) so failure is visible.

## 11. Evaluation methodology costs more than code bugs

- **Wrong file for attribution:** analysing retrieval errors from an old
  configuration turned "7 retrieval problems" into "16" and pointed in the wrong
  direction. Check the artifact version before changing the criterion.
- **Same-model judging is optimistic:** rule-based judging (numeric tolerance +
  word coverage) on the same answers gave 123/133 vs 115/133; self-judging was
  about 6 points higher.
- **Prompting cannot recover data that is not there:** adding "convert yesterday
  to an absolute date" changed none of 20 questions because the date was never
  in the context.

## 12. Current score (honest version)

LongMemEval_s, 500 questions, MergedMemory episode layer with in-library
reranking, k=10 sessions, 12,000 chars per session, assistant replies clipped to
200 chars, thinking enabled:

| Judge | Correct | Accuracy |
|---|---:|---:|
| LLM self-judge (same model as answerer) | 431/500 | **86.2%** |
| Rule-based lower bound | — | **78.4%** |

The temporal subset is 119/133 = 89.5% of the 500-question run.
