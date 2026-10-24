# LongMemEval: temporal-reasoning subset

This is the first external benchmark run for the episode layer. It is **not** a
full LongMemEval result: only the `temporal-reasoning` subset is evaluated, and
the answerer/judge is `deepseek-flash`, not GPT-4o. Do not compare the accuracy
column directly with published GPT-4o numbers.

## Scope

- Dataset: `longmemeval_s.json`
- Subset: `question_type == "temporal-reasoning"` — 133 questions
- Memory unit: one exchange (user + assistant) = one episode
- Retrieval unit: session, aggregated from episode hits
- Answerer and judge: `deepseek-flash`, temperature 0, same model
- Self-judging is a documented scoring limitation
- 20-question runs are smoke/timing runs, not a second benchmark

## Accuracy

| Questions | Correct | Accuracy |
|---:|---:|---:|
| 133 (full temporal-reasoning subset) | 101 | 75.9% |
| 20 (first run) | 17 | 85.0% |
| 20 (timed run) | 16 | 80.0% |

The two 20-question runs differ by one question. That is ±1 judge jitter, not a
configuration difference.

## Token use

| Metric | 20 timed | 20 first |
|---|---:|---:|
| Total tokens | 217,934 | 228,185 |
| Prompt | 193,302 | 195,228 |
| Completion | 24,632 | 32,957 |
| Per question | 10,896 | 11,409 |
| Per-question split | prompt 9,665 + completion 1,232 | — |
| LLM calls | 41 = 2.05/question | 42 = 2.1/question |

Each question uses two calls: one answer, one judge. Retrieval uses a local
embedding model, so retrieval contributes 0 tokens.

## Time

| Metric | Value |
|---|---:|
| 20 questions, 2 concurrent | 154 s |
| Per question | 7.7 s |
| First index build, 32,930 episodes | 1,259.9 s (~21 min) |
| Reuse existing index | 2.5 s |

## Retrieval quality (133 questions, session level)

| Arm | recall@3 | recall@5 | recall@10 | recall@20 | all@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| merged (facade) | 75.6% | 86.1% | 92.5% | 97.2% | 88.0% | 0.831 |
| lex | 77.2% | 83.9% | 91.9% | 96.6% | 82.7% | 0.856 |
| chroma | 75.6% | 86.1% | 92.5% | 97.2% | 88.0% | 0.831 |
| hybrid | 80.8% | 87.8% | 94.8% | 97.5% | 91.0% | 0.878 |
| hybrid+rerank (chunk level) | **86.8%** | **91.7%** | 94.8% | 97.5% | 91.0% | **0.935** |

The same 20-question smoke set reached `recall@3 = 91.2%`, `recall@5 = 97.5%`,
`recall@10 = 100%` with the hybrid+rerank arm. That is a smoke-set number, not a
replacement for the 133-question result.

## Key configuration

| Item | Choice | Why it matters |
|---|---|---|
| Memory unit | one exchange = one episode | session-level retrieval would be truncated by the 512-token limit; exchange keeps the whole local turn |
| Vector store | Chroma, 32,930 episodes | reusable index; second build 2.5 s |
| Embedding | `multi-qa-MiniLM-L6-cos-v1`, 384d, local | English QA retrieval model; hard 512-token limit |
| Fusion | lexical + vector, session aggregate, RRF(k=60) | one arm is not enough: recall@5 lex 83.9%, vector 86.1%, hybrid 87.8% |
| Rerank | `ms-marco-MiniLM-L-6-v2`, exchange-level chunks | reranker also truncates at 512 tokens; scoring a whole session loses most of it |
| Context budget | top-10 sessions × 4,000 chars | largest lever: k=5 ceiling 78.9%, k=10 ceiling 91.0% |
| Answer/judge | `deepseek-flash`, temperature 0, same model | self-judge bias is a scoring limitation |
| Write path | `MergedMemory.record_turn` | benchmark writes through the facade, not the store directly |
| Retrieval path | `MergedMemory.episodes_search` | merged arm uses the facade's own retrieval |

Other implementation details, not part of the conclusion: `depth=60`,
`top_n=20`, `max_chunks=6`, concurrency 6/2, retry budget, timezone
`Asia/Shanghai`.

## The episode hook

The benchmark uses the episode hook end to end:

- writes go through `MergedMemory.record_turn` (not `EpisodeStore` directly);
- retrieval goes through `MergedMemory.episodes_search`;
- `build_context` / `inject` opens the episode section on retrospective and
  temporal queries.

The hook is bilingual. Current marker coverage on the 133-question subset is
**132/133**; the only question that does not open the episode branch is a
weekday habit ("What time do I wake up on Tuesdays and Thursdays?"), which
belongs to the rhythm/routine layer rather than episode retrieval.

Retrieval-only smoke after the hook change reproduced the same orders:

```text
20 questions:  hybrid+rerank recall@3 91.2%, gate 20/20
133 questions: hybrid+rerank recall@3 86.8%, gate 132/133
```

Chinese and English time expressions both reach the same path:

```text
yesterday / last Friday / next Monday / in 3 days / 2 weeks ago
```

Chinese equivalents: `昨天 / 上周五 / 下周三 / 三天后`.

## What not to claim

- Not full LongMemEval.
- Not GPT-4o answerer/judge.
- Not a statistical result from 20 questions.
- `merged` retrieval is not the same as `hybrid+rerank`; the merged arm is the
  facade's own path, the other arms isolate retrieval components.
- The 133-question retrieval table is evidence retrieval, not end-to-end answer
  accuracy.
