# Routine memory: what others do, and what to copy

Research note for the 1.6.0 episode/routine layer. This is a design comparison,
not a benchmark claim.

## TL;DR

- **The quality jump is the embedding model, not Chroma.** On the local 200/50
  fixture, lexical-only retrieval is 61.0% recall@3; adding bge embeddings is
  74.0%. Chroma with the same vectors does not add ranking quality at this
  scale; it adds persistence, ANN, metadata filtering and operations.
- **Do not hardcode actions like `class`.** Separate three things: *what*
  happened (free-text action), *when* it recurs (calendar recurrence), and
  *whether to ask* (probe policy). Only the last two need generic code.
- **Use the iCalendar recurrence model as the reference.** RFC 5545 already
  defines weekly recurrence, time zones, exception dates and extra dates:
  `RRULE`, `EXDATE`, `RDATE`, `TZID`. It has no notion of "class", which is the
  point.
- **Episodes are not beliefs.** Zep/Graphiti's episode node, with `MENTIONS`
  edges and a reference time, is a useful provenance model; LangMem's
  semantic/episodic/procedural split is a useful boundary.
- **Proactive prediction is hard.** The ProactiveBench paper reports F1 66.47%
  for a fine-tuned proactive model. This supports a conservative, in-turn,
  explicit-feedback design rather than a background "guess and send" agent.

## 1. Vector DB vs no vector DB

Three different questions are often mixed together:

| Setup | What changes | Local 200/50 recall@3 |
|---|---|---:|
| lexical only | no embedding model | 61.0% |
| bge embeddings, exact cosine in memory | embedding model, no vector DB | 74.0% |
| bge embeddings + Chroma | same vectors, ANN/persistence | ~74% at this scale |

The first row is the real quality gap. The second and third rows use the same
vectors; at 200 beliefs, exact brute-force cosine is cheap and Chroma's HNSW is
approximate by design. Chroma's docs describe its single-node index as HNSW
approximate nearest-neighbor search. So:

- for a few hundred labels, compute exact cosine; do not add an ANN dependency
  for accuracy;
- for a growing episode archive or multi-process service, Chroma/Qdrant makes
  sense for persistence, filtering and scale;
- Chroma should be an **index**, never the source of truth for `status`,
  `valid_until`, `exdate` or probe state.

In this repo the split is:

```text
SQLite  = ledger: belief / intention / episode / routine state
Chroma  = optional index: label or episode candidate ids
rank()  = final scoring: lexical + vector + retention + time
```

## 2. How other systems model this

### Mem0

Mem0's docs describe the flow as: the application sends conversation turns to
`add`, then calls `search` before the next model request; the application
decides which returned memories to put in the prompt. Its value is extraction
and retrieval, not a recurrence engine. There is no `class` concept; memories
are text/metadata.

Source: [How Mem0 works](https://docs.mem0.ai/core-concepts/how-it-works)

What to copy: on-demand retrieval, not replaying the full transcript.
What not to copy: Mem0 does not solve weekly routines; that still needs a
calendar/state layer.

### Zep / Graphiti

Graphiti calls an episode "a single data ingestion event"; the episode is a
node and extracted entities are connected via `MENTIONS`. Episodes carry a
reference time and support provenance queries. This is a good model for the
turn layer: raw turn = episode, beliefs = entities/edges, evidence = link.

Source: [Adding Episodes — Zep](https://help.getzep.com/graphiti/core-concepts/adding-episodes)

What to copy: episode as provenance; reference time; "why did we believe this?"
What not to copy: a full temporal knowledge graph for a single-user memory
layer. SQLite episode rows plus belief ids are enough for 1.6.0.

### LangMem

LangMem separates semantic (facts), episodic (past experiences) and procedural
(system behavior) memory. Its memory manager prompts an LLM to expand or
consolidate state; storage is optional and pluggable. The docs explicitly say
the best memory systems are application-specific.

Source: [Long-term Memory in LLM Applications — LangMem](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)

What to copy: the semantic/episodic boundary; extraction/consolidation can be
LLM-driven, without an action enum.
What not to copy: prompt optimization and procedural rewriting; out of scope.

### Letta / MemGPT

Letta exposes memory blocks and archival memory, with the agent managing what
to keep in context versus archive. It is a general agent-memory system, not a
calendar/routine engine.

Source: [Context hierarchy — Letta](https://docs.letta.com/v1-sdk/memory/context-hierarchy)

What to copy: tiered context (hot context vs archival).
What not to copy: agent-managed memory edits as the only write path; this repo
has explicit schemas and lifecycle rules.

### Proactive agents

The ICLR 2025 Proactive Agent paper collects real human activities, labels
proactive task predictions as accepted/rejected, trains a reward model, and
reports an F1 of 66.47% for proactive assistance. That is a useful warning:
proactive prediction is not solved, so the memory layer should provide a
candidate and let the host decide, instead of sending messages on its own.

Source: [Proactive Agent (arXiv:2410.12361)](https://arxiv.org/abs/2410.12361)

What to copy: explicit accepted/rejected feedback; conservative triggering.
What not to copy: background autonomous task initiation.

## 3. The recurrence model

RFC 5545 (iCalendar) already has the vocabulary:

```text
RRULE:FREQ=WEEKLY;BYDAY=TU;UNTIL=20261220T235959Z
EXDATE;TZID=Asia/Shanghai:20261006T190000
RDATE;TZID=Asia/Shanghai:20261008T190000
```

- `RRULE`: weekly, weekdays, interval, until/count;
- `EXDATE`: holidays / cancelled classes;
- `RDATE`: makeup classes;
- `TZID`: local time zone;
- `DTSTART`: first occurrence.

Source: [RFC 5545 — iCalendar](https://www.rfc-editor.org/rfc/rfc5545.html)

For 1.6.0, store normalized fields (`weekdays`, `local_time`, `interval`,
`valid_from`, `valid_until`, `exdates`, `rdates`) for indexing, and serialize
to RRULE for portability. A full RRULE parser is not required on day one.

### One-off vs weekly

```text
"class tomorrow"
  → one-off Intention
  → Routine candidate anchored to the same weekday next week
  → only after explicit confirmation does it become weekly

"class every Tuesday at 19:00"
  → weekly Routine directly (explicit recurrence)

"no class today"
  → skipped_today, not a denial of the weekly routine

"no more Tuesday classes"
  → denied / ended
```

This avoids both failure modes: treating every one-off as a permanent routine,
and failing to learn a weekly class that was only stated once.

## 4. Generic routine schema

```text
Routine
  id
  user_id
  action              # free text: "attend class", "gym", "therapy"
  local_time          # 19:00
  weekdays            # [2] or [1,3,5]
  interval            # 1 = every week, 2 = odd/even weeks
  anchor_date         # for interval/odd-even
  timezone
  valid_from
  valid_until         # semester end; nullable
  exdates             # holidays / cancelled
  rdates              # makeup
  status              # candidate | confirmed | trusted | paused | dormant
  confidence
  observations / confirms / denies / unanswered
  last_observed_at / last_asked_at / last_confirmed_at
  evidence_episode_ids
  sensitivity         # optional; no probing for sensitive topics
```

There is no `action_type == "class"` branch. `action` is whatever the host or
user wrote. The only generic policy is:

```text
probe only if:
  status in (candidate, confirmed)
  and now is inside the local window
  and today's weekday matches
  and today is not an EXDATE
  and the user has sent a turn
  and this occurrence has not been asked already
```

## 5. What I would copy

1. Episode as provenance (Zep/Graphiti), but in SQLite.
2. Semantic/episodic split (LangMem), but with explicit schemas.
3. Calendar recurrence (RFC 5545), but normalized fields internally.
4. On-demand retrieval (Mem0), not transcript replay.
5. Conservative proactive feedback (ProactiveBench), not autonomous sending.
6. Chroma as an optional candidate index, not a ledger.

## 6. What I would not copy

- A full temporal graph for 1.6.0; too much for the current scope.
- LLM-only memory writes without a deterministic schema.
- Background scheduled messages.
- Action/verb keyword lists for routine detection.
- Vector DB as the source of truth for routine state.
