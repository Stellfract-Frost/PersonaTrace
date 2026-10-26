# LongMemEval: 500-question result

Primary external result for the episode layer. This is `LongMemEval_s` with all
six question types, 500 questions in one run.

The answerer and judge are both `deepseek-flash` at temperature 0. This is a
self-judge setup, not GPT-4o; do not compare the accuracy column directly with
published GPT-4o numbers.

## Accuracy

| Question type | Questions | Correct | Accuracy | tok/question |
|---|---:|---:|---:|---:|
| single-session-assistant | 56 | 55 | 98.2% | 21,591 |
| single-session-user | 70 | 67 | 95.7% | 6,367 |
| temporal-reasoning | 133 | 119 | 89.5% | 6,901 |
| knowledge-update | 78 | 67 | 85.9% | 6,426 |
| multi-session | 133 | 105 | 78.9% | 7,116 |
| single-session-preference | 30 | 18 | 60.0% | 25,022 |
| **total** | **500** | **431** | **86.2%** | **8,467** |

- LLM self-judge: **431/500 = 86.2%**
- Rule-judge lower bound: **78.4%**
- Cost: **≈ ¥5.50** (input ¥1/M cache-miss, output ¥4/M)

## Reproduce

The configuration below is the default for the harness; the only flags you need
are the dataset selection and the per-session budget.

```bash
PERSONA_LLM_MODEL=deepseek-flash python scripts/eval_longmemeval.py \
  --question-type all \
  --chroma data/chroma_lme --reuse-index \
  --arms merged+rerank --answerer llm \
  --workers 6 --context-k 10 --session-chars 12000 --assistant-chars 200 \
  --results data/lme_500.json
```

- `--question-type all` selects all six question types (500 questions).
- `--reuse-index` skips the embedding step after the first build.
- `--session-chars 12000 --assistant-chars 200` set the context budget.

## Key configuration

| Item | Choice | Why it matters |
|---|---|---|
| Dataset | `LongMemEval_s`, 500 questions | six question types, not a temporal-only slice |
| Memory unit | one user+assistant exchange = one episode | fits the encoder's 512-token limit; whole sessions do not |
| Vector store | Chroma, 123,922 episodes | index reuse avoids a full rebuild |
| Embedding | `multi-qa-MiniLM-L6-cos-v1`, 384d, local | English QA retrieval model; 512-token limit |
| Fusion | per-arm min-max scaling, vector 1.0 / lexical 0.5 | concatenation starved the lexical arm; pure RRF lost magnitude |
| Rerank | `ms-marco-MiniLM-L-6-v2`, exchange-level chunks | cross-encoder has the same 512-token limit |
| Session budget | top-10 sessions × 12,000 chars | 4,000 chars truncated most sessions |
| Assistant replies | kept in full for `single-session-assistant` and `single-session-preference`; clipped to 200 chars for the other four types | the answer lives in the assistant's own words for those two types |
| Answer prompt | recommendation questions must name the remembered preference they build on | grounds the answer in the user's history instead of generic advice |
| Answer/judge | `deepseek-flash`, temperature 0, thinking on | self-judge bias is a scoring limitation |
| Write path | `MergedMemory.record_turn` | benchmark writes through the facade |
| Retrieval path | `MergedMemory.episodes_search` | the merged arm uses the facade's own retrieval |

Other details: `depth=60`, `top_n=20`, `max_chunks=6`, answer `max_tokens=16,000`,
timezone `Asia/Shanghai`.

## The episode hook

The benchmark uses the episode path end to end:

- writes go through `MergedMemory.record_turn`, not `EpisodeStore` directly;
- retrieval goes through `MergedMemory.episodes_search`;
- `build_context` / `inject` is the host-facing retrospective hook.

`has_retrospective` fires on 161/500 questions. The benchmark calls
`episodes_search` directly, so the gate is not on the measured retrieval path;
it is a host-integration check, not a benchmark filter.

Chinese and English time expressions reach the same parser. The English forms:

```text
yesterday / last Friday / next Monday / in 3 days / 2 hours later
```

Chinese equivalents are accepted by the same parser.

## What not to claim

- This is `LongMemEval_s`, not every LongMemEval split.
- The answerer/judge is `deepseek-flash`, not GPT-4o.
- Self-judging is looser than an independent judge; the rule-based 78.4% is a
  lower bound, not a replacement.
- The 86.2% is a result from this harness/configuration, not a claim that the
  memory layer generalizes to every assistant or every domain.
