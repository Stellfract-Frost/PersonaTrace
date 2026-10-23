# Temporal memory kernel

`personatrace.memory_v2` is the storage and lifecycle kernel. It does not call
an LLM, does not build prompts, and does not depend on a vector database.

## Belief

```text
id            deterministic uuid5(user, track, category, label)
user_id       single-user by default, carried through every key
track         work | persona
category      profile category
label         short index (the embedding surface)
confidence    extraction confidence, not importance
stability     days until retention reaches 1/e
last_support  timestamp of the last supporting evidence
valid_from    when the belief became true
invalid_at    when it stopped being true (kept, not deleted)
status        active | deprecated | contested
hits          number of supporting observations
dimension     identity | need | predict
reasoning     why it was stored; read only after a hit
due_at_ts     optional deadline/event timestamp
time_kind     none | due | past | range | recurring
time_raw      original time expression
evidence      L0 evidence ids
```

`uuid5` makes writes idempotent: the same belief lands on the same record.
The tradeoff is that a paraphrase is a new label and therefore a new record.

## Retention

```text
R(t) = exp(-(now - last_support) / stability)
effective_confidence = confidence * R(t)
```

Each new supporting evidence increases stability:

```text
S' = S * (1 + k * (1 - R))
```

The closer the belief is to being forgotten, the larger the gain. A review
immediately after writing adds almost nothing. Stability is capped at 20× the
category's initial value so a frequently repeated item cannot become immortal.

Initial stability is category-specific:

| category | days |
|---|---|
| explicit_facts | 520 |
| identity | 300 |
| constraints | 180 |
| expertise | 240 |
| likes / dislikes | 90 |
| projects | 60 |
| goals | 45 |
| state | 10 |
| traits | 300 |
| interests | 120 |
| style_notes | 180 |
| behavior_patterns | 120 |
| decision_style | 240 |
| interpersonal | 180 |
| boundaries | 365 |
| habit | 90 |

Below `R < 0.30`, a belief is archived: it remains in storage but is not
returned by normal reads. `gc()` can reclaim very old, rarely supported
beliefs; archive is not deletion.

## Two time axes

`valid_from` / `invalid_at` describe when a fact was true. `last_support`
describes when the system learned or reinforced it. Corrected beliefs are
invalidated, not removed, so `state(as_of=...)` can answer "what did we think
in September" without replaying the whole store.

## Three read layers

| Layer | Unit | When it is read |
|---|---|---|
| index | label + category + retention | every query |
| context | reasoning | only for selected items |
| evidence | original sentence | only for selected items, first N |

The scoring path never touches raw evidence. This keeps the embedding surface
small and makes the expensive read proportional to the number of hits.

## Write path

1. Resolve the belief id from `(user, track, category, label)`.
2. Acquire a per-user lock with a fencing token.
3. Re-read the belief and check the version.
4. Append evidence id; reinforce stability; increment hits.
5. Write the hash and timeline index.
6. Release the lock.

Concurrent writers either replay or fail visibly; they do not silently
overwrite each other.

## Backends

`build_backend()` chooses, in order:

1. Redis, if the `redis` package is installed and a URL is provided;
2. SQLite, at `db_path` / `PERSONA_DB` / `~/.personatrace/memory.db`;
3. in-process memory, for tests.

SQLite is the default for a local single-user install. Redis is optional and
exists for multi-process sharing.

## What the kernel does not do

- It does not extract beliefs from text; `extract` / `MergedMemory` do that.
- It does not decide which query intent is active; `recall` / `facet` do that.
- It does not manage short-term tasks; `IntentStore` does that.
- It does not own prompts, tone, UI, or proactive messages.

Those boundaries are deliberate. The kernel stays small enough to test without
an LLM, a network, or a running server.
