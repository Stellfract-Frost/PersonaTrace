# Design notes: what failed, what was measured, and what changed

This is the engineering log behind PersonaTrace. It is intentionally about
tradeoffs and failure modes, not a feature list. The short version: retrieval
was not failing because embeddings were weak. It was failing because the query
had no way to say *what kind of answer it wanted*.

## 1. The failure that started this

The test query was:

```text
我那个 deadline 是几号
```

The store contained `周五要演示`. The answer was "I don't have that deadline."
Three independent channels failed at once:

| Channel | What it measured | Why it failed |
|---|---|---|
| lexical IDF | shared character bigrams | `deadline` is Latin; no Chinese label shares it |
| embedding cosine | semantic similarity | a 4–8 character label is a noisy embedding unit; "deadline date" requires inference, not synonym matching |
| structured time | `due_at_ts` | the extractor never wrote a due timestamp for `周五要演示` |

The last one was the useful discovery: the sentence already contained a
structured slot. The model did not need to be smarter; the write path needed to
put `周五` into a typed field.

## 2. Measure first

Fixture: 200 beliefs, 50 queries, with distractor clusters and several
time-intent questions. Metrics: recall@3 and MRR. All methods use the same rows
and the same embedding model. This is a directional gate, not a statistically
meaningful benchmark.

Local run with `BAAI/bge-small-zh-v1.5`:

| Method | recall@3 | MRR |
|---|---|---|
| v2 native, no pinned | 2.0% | 0.020 |
| v2 native, pinned | 2.0% | 0.020 |
| v1 string only | 61.0% | 0.657 |
| v1 + bge | 73.0% | 0.737 |
| outer lexical | 61.0% | 0.637 |
| outer + bge | 74.0% | 0.770 |
| **outer + bge + time facet** | **76.0%** | **0.760** |

The earlier 12/10 fixture scored 90.0%. That number was too easy: 12 rows and
10 queries gave the time bucket almost no distractors. On 200/50, the gain from
the time facet is +2.0 points recall but -0.010 MRR: the bucket can promote a
correct due item above an equally correct semantic item. Both numbers are real;
the larger fixture is the one to quote.

## 3. What changed

### 3.1 17 categories, two tracks

v1 had 23 categories spread across work, persona, cross and ephemeral logs. v2
had 9. The merge keeps 17 beliefs:

```text
work:    explicit_facts expertise projects goals likes dislikes constraints
persona: identity traits interests style_notes behavior_patterns
         decision_style interpersonal boundaries habit state
```

Logs (`observation_log`, `uncertain`, `low_confidence`, `timeline`,
`analysis_log`, `corrections`) stay logs. They are not beliefs: they must not
enter retention, pinned budgets or the profile prompt.

### 3.2 Typed time

A single `due_at_ts` float cannot distinguish:

- a future deadline (`下周三交周报`)
- a past event (`上周五去了海边`)
- a recurring habit (`每天十点睡`)
- a fuzzy window (`年底前换工作`)

The kernel now carries `time_kind` and `time_raw` alongside `due_at_ts`.
`parse_due()` resolves common Chinese expressions in code; the model only
supplies the surface phrase. Past expressions are refused before they can
become deadlines.

### 3.3 Outer retrieval

`MemoryV2.recall()` is still a kernel primitive: it has no query text and uses a
constant similarity. The merged path does not call it. Instead:

```text
sim   = label is a substring of query ? 1.0
      : 0.5 * vector_norm + 0.5 * lexical
score = (0.15 + 0.85 * sim) * (0.35 + 0.65 * retention)
```

- lexical uses candidate-set IDF character bigrams; no overlap returns 0.5
  (neutral), not 0;
- vector uses candidate-set z-score + tanh, so 0.5 means "average", not
  "irrelevant";
- pinned categories are always kept;
- cross-category items come from the unselected tail, not ranks 16–18.

### 3.4 Time facet is a conditional bypass

`time_intent` includes `几号 / 哪天 / 什么时候 / 截止 / deadline / ddl / 日程 /
安排 / 到期 / 之前 / 之后`. `TIME_HABIT` (`平时 / 每天 / 一般 / 习惯 / 通常 /
总是`) disables the facet. The bucket is unioned with vector results, not used
as a replacement. It holds at most two items.

### 3.5 Beliefs and intentions are different memory types

Long-term preference and a one-off commitment must not share a table:

| | belief | intention |
|---|---|---|
| storage | `MemoryV2` | `IntentStore` (SQLite) |
| time | retention + `as_of` | `due_at` + grace + status |
| identity | `uuid5(user:track:category:label)` | `uuid5(user:title:date)` |
| disappearance | archive at low retention | `done/cancelled/expired` |

A one-off "后天要上课" stays important even if it was mentioned once.
`importance` is a write-time axis; `hits` does not raise it. No due date means
an explicit TTL, not an invented deadline.

### 3.6 SQLite is the default, not Redis

The merged facade uses the kernel's SQLite backend by default. Redis is
optional. A single-user local install should not need a server. `MergedMemory`
can store beliefs and intentions in one SQLite file.

### 3.7 Write source is a gate

`MergedMemory.remember()` rejects empty user messages and sources outside
`direct / bridge / console`. A proactive turn (`attention`) must not write
memory; otherwise the assistant's own words become "facts about the user."

### 3.8 Conservative write-time dedup

`MemoryV2` is idempotent only when the label is exactly the same. v1 merged
similar labels through embeddings; the merged facade does a smaller, explicit
check before writing:

- normalize possessives and common intensifiers;
- normalize a few synonym pairs (`偏好 → 喜欢`, `在赶 → 在做`);
- compare labels in the same category with `SequenceMatcher` at `>= 0.86`;
- never merge two items with different typed time slots.

This catches `喜欢深色主题` / `偏好深色主题` and `名字叫小陌` / `我的名字叫小陌`,
while keeping `喜欢猫` / `喜欢猫毛` and `周五要演示` / `下周三演示` separate. It
does not create an alias table: a low-lexical-overlap paraphrase can still
produce a second record.

## 4. Pitfalls (the useful part)

1. **A single `due` float is not a time model.** It cannot carry past,
   recurring, or range semantics.
2. **Category is not a time gate.** `state` can contain "周五要演示"; only
   typed time plus a parser catches it.
3. **Do not put tasks in the belief table.** A task has a lifecycle; retention
   is the wrong clock.
4. **Do not merge tasks by label.** "这周三上课" and "下周三上课" are two
   records.
5. **Pinned can eat the budget.** With three pinned items and `k=4`, only one
   related slot remains. The time facet is a separate section for this reason.
6. **RRF over a degenerate lexical channel is worse than no fusion.** A channel
   that returns the same ranking for every query adds noise.
7. **Vector z-score is a scale fix, not a quality fix.** It makes cosine scores
   comparable inside a candidate set; it does not make a weak embedding good.
8. **`平时几点睡` is not a deadline query.** Negative habit terms must disable
   the time bucket.
9. **`上周五` must not become today.** Past expressions return `kind="past"`;
   the write path refuses them.
10. **Do not run v1 `_decay_all` and v2 retention together.** Two clocks produce
    two confidences and an unexplainable rank.
11. **Read raw Chroma rows for migration.** `PersonaManager.load()` applies the
    old decay before migration; that would decay twice.
12. **Label idempotency is not semantic dedup.** `uuid5(user:...:label)`
    deduplicates repeated labels; the merged write path adds a conservative
    lexical check. It is not an alias table, so a low-overlap paraphrase can
    still create a second record.
13. **Pydantic retries need the error text.** Retrying with the same prompt just
    repeats the same invalid JSON. Feed the `ValidationError` back.
14. **An unparseable due must not become a 24h TTL.** Skip it or ask; do not
    silently invent a deadline.
15. **The hidden hint needs the host's tool names.** `memory_view` does not
    exist in this package; use `memory_recall / memory_state`.
16. **Truncation must preserve the hint.** Compute the hint first, then cut the
    body. Otherwise the model sees a partial list and no "there is more" signal.
17. **SQLite key prefixes and TTL are easy to double-prefix.** Test restart and
    expiry, not just inserts.
18. **Fifty queries are still small.** The 200/50 fixture is harder than the
    original 12/10 one, but it cannot support a claim like "23 points better"
    without a held-out set.

## 5. Deliberately not done

- no UI, persona or proactive messaging;
- no multi-tenant/auth model;
- no graph unification of beliefs and intentions yet;
- no full semantic alias merge; dedup is lexical and conservative;
- no 100+ query benchmark or CI gate;
- no Redis requirement; SQLite is enough for a local single user.

## 6. Reproduce

```bash
python -m pytest -q
python scripts/eval_recall.py
python scripts/migrate_v1.py --db ./personatrace.db          # dry run
python scripts/migrate_v1.py --db ./personatrace.db --apply
```

The evaluation script prints the actual numbers; it does not hard-code the
table above. If the embedding model is unavailable, it falls back to a hashing
embedder and labels the run accordingly.
