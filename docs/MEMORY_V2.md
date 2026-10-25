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
recorded_at   when this version was written (transaction time)
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

`valid_from` / `invalid_at` describe when a fact was true. `recorded_at`
describes when that version was written to the store (transaction time).
`last_support` is when the fact was learned or reinforced.

Every belief write appends a version to `history:{user}:{belief}` (a zset scored by
`recorded_at`, with one snapshot hash per version). `state(as_of=T)` selects the
latest version that was already known at `T`, then applies the validity window
at `T`. This is snapshot semantics: a reinforcement or correction made later
cannot rewrite what the store believed earlier. Passing
`known_as_of=now` asks the different question "what do we currently know about
`T`", which is valid-time-only retrieval.

Corrections are invalidations, not deletions. A second invalidation may shorten
the validity window but never extends it, and the history keeps both versions.
A later observation of the same label reopens the validity window only when its
timestamp is at or after `invalid_at`; the gap between the correction and the
revival stays empty in snapshots. Pass `revive=False` to `observe` to keep a
corrected belief closed.

Due and typed-time fields use a sentinel: omitting them preserves the previous
value, while an explicit `null` / `"none"` / empty string clears it. This is
what stops an old deadline from sticking forever after a belief is re-written
without one.

Older databases without history get one base version from the current row on
the next write; reads before that first write use the legacy row only when its
`recorded_at` is not in the future.

## Retrieval

`recall(query=...)` has its own lexical channel: candidate-set IDF over
character bigrams, with label substring matches promoted to 1.0. Vector cosine
is optional and is averaged with lexical when both are present. The score is
`(0.15 + 0.85 * similarity) * (0.35 + 0.65 * retention)`, so relevance leads
and freshness modulates it. Absent any query or vector, similarity stays at the
neutral 0.5 and ordering falls back to retention. This gives the kernel a
usable retrieval baseline (66.0% recall@3 on the 200/50 fixture) without an
embedding model; the outer `recall.rank` layer remains for time facets and
cross-category selection.

## Three read layers

| Layer | Unit | When it is read |
|---|---|---|
| index | label + category + retention | every query |
| context | reasoning | only for selected items |
| evidence | original sentence | only for selected items, first N |

The scoring path never touches raw evidence. This keeps the embedding surface
small and makes the expensive read proportional to the number of hits. L0 keys
are scoped by belief (`evidence:{user}:{belief}:{evidence}`), so two beliefs
from one turn can share an episode id without overwriting each other's
`belief_id` or raw sentence. Legacy keys are still readable.

## Write path

1. Resolve the belief id from `(user, track, category, label)`.
2. Acquire a per-user lock with a fencing token.
3. Re-read the belief and its version.
4. Append evidence id; reinforce stability; increment hits.
5. Write the hash with an atomic compare-and-swap on the version
   (`cas_hash`), append a history version, and update the timeline index.
6. Release the lock with a compare-and-delete, so a token whose TTL expired
   cannot delete the next writer's lock.

If the CAS loses, the writer re-reads and replays the observation; it never
overwrites the newer version with a stale copy. The CAS is a backend primitive:
in-process has a lock, SQLite uses `BEGIN IMMEDIATE`, Redis uses a Lua script.
The lock is an optimization; correctness comes from the CAS.

## Backends

`build_backend()` chooses, in order:

1. Redis, if the `redis` package is installed and a URL is provided;
2. SQLite, at `db_path` / `PERSONA_DB` / `~/.personatrace/memory.db`;
3. in-process memory, for tests.

SQLite is the default for a local single-user install. Redis is optional and
exists for multi-process sharing. If SQLite cannot be opened, the kernel still
fails open to in-process memory so the host keeps running, but it logs a
WARNING, records `last_backend_warning()`, and `personatrace-doctor` exits
non-zero for that unexpected fallback (explicit `PERSONA_DB=off` stays a normal
test-only mode). `MergedMemory` resolves intentions, episodes and routines to
the same durable path as the belief backend; omitting `sqlite_path` no longer
leaves three stores in `:memory:`.

## What the kernel does not do

- It does not extract beliefs from text; `extract` / `MergedMemory` do that.
- It does not decide which query intent is active; `facet` does that.
- It does not manage short-term tasks; `IntentStore` does that.
- It does not own prompts, tone, UI, or proactive messages.
- It does not require an embedding model; lexical retrieval is built in.

Those boundaries are deliberate. The kernel stays small enough to test without
an LLM, a network, or a running server.
