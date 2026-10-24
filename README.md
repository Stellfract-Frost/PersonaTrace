# PersonaTrace

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-ready-4B5563)](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/docs/MCP.md)

**A memory layer for what is still true, what is due, and why it was stored.**

PersonaTrace is not a chatbot or an agent harness. It is a small memory kernel
that a host can call. The host owns the model, the prompt,
the UI, the tone, and any proactive behavior.

The unit of memory is not a fact. It is **a fact that holds for a while**:
it has a validity window, a retention curve, evidence, and optionally a due
date. A separate intention store handles one-off commitments whose lifetime is
measured in hours or days, not in months.

## What it remembers

| Type | Examples | Lifetime |
|---|---|---|
| Belief | preferences, identity, constraints, projects, goals, habits | retention curve per category; archived, not deleted |
| Intention | "class the day after tomorrow", "submit the report Friday" | due + grace; `done / cancelled / expired` |
| Episode | the raw user/assistant turn | SQLite; time-window + lexical retrospective search |
| Routine | "attend class every Tuesday at 19:00" | weekly recurrence + term/exceptions + probe state |
| Evidence | the sentence a belief came from | kept with the belief; read only after a hit |
| Mention | "someone said he is preparing for an exam" | separate from first-hand beliefs; decays faster |

The merged path uses **17 belief categories** across two tracks:

```text
work:    explicit_facts expertise projects goals likes dislikes constraints
persona: identity traits interests style_notes behavior_patterns
         decision_style interpersonal boundaries habit state
```

## How retrieval works

```text
query
  ├─ pinned: boundaries / constraints / identity
  ├─ semantic: lexical IDF + vector cosine (candidate-set normalized)
  ├─ time facet: due/range rows, only for time-intent questions
  ├─ cross-category: items from categories not already selected
  └─ evidence on demand: reasoning + original sentence for the first hits
```

`score = (0.15 + 0.85 * similarity) * (0.35 + 0.65 * retention)`

A time question is not answered by a better embedding. It is answered by
putting the date into a typed field and querying that field. The facet is a
conditional bypass, not a second retriever.

## Install

```bash
# Recommended: Chroma + an embedding model for semantic memory
pip install "personatrace[vector]"

# Minimal install: lexical/time fallback only
pip install personatrace
```

Python 3.10+. The core belief kernel uses the standard library. The merged
facade uses Pydantic; the default local store is SQLite. Redis is optional.

### Recommended Chroma episode index

SQLite owns the episode rows and all state. Chroma is the recommended candidate
index for semantic retrospective search; install it unless you explicitly want
the lexical/time-only fallback. The embedding model is what makes semantic recall
work; Chroma is the index, not the source of truth.

```python
from personatrace.chroma_index import ChromaEpisodeIndex

index = ChromaEpisodeIndex(
    path="./persona_data/chroma",
    embed_fn=embed_fn,          # list[str] -> list[list[float]]
)
memory = MergedMemory(
    user_id="alice",
    sqlite_path="./personatrace.db",
    vector_index=index,
)
```

If Chroma is unavailable, `EpisodeStore` falls back to lexical + time search.
That fallback is intentionally usable, but it is weaker for paraphrases and
larger episode archives. The v1 `persona_items` collection is separate; episodes
use `episode_turns`.

## Quick start: structured memory

The host extracts beliefs/intentions; PersonaTrace stores, ranks and expires
them. This path does not call an LLM.

```python
from personatrace import MergedMemory

memory = MergedMemory(
    user_id="alice",
    sqlite_path="./personatrace.db",   # one file for the whole memory
)

memory.write(
    beliefs=[{
        "category": "identity",
        "label": "Name is Mo",
        "confidence": 0.9,
        "reasoning": "user introduced themselves",
    }],
    intentions=[{
        "action": "attend class",
        "due": "2026-10-25",
        "importance": 0.9,
        "confidence": 0.8,
    }],
    source="direct",
)

context = memory.build_context("When is my deadline?")
print(context.agenda_lines)
print(memory.inject("When is my deadline?"))
```

- `build_context()` returns structured data and has no side effects.
- `inject()` is an optional prompt adapter; the host can render the pack itself.
- `source` must be one of `direct / bridge / console`; proactive turns must not
  write user facts.

### Turn and routine

```python
turn = memory.record_turn("I have class every Tuesday at 19:00", session_id="s1")
memory.label_routine(
    "attend class",
    time="19:00",
    weekdays=[2],              # Monday=1 ... Sunday=7
    anchor_date="2026-10-20",
    valid_until="2026-12-20",  # semester end
    status="candidate",
)
pack = memory.build_context("Are we talking?")
print(pack.routine_probes)     # hints only; the host decides whether to ask
```

A routine is weekly, not daily. One observation creates a candidate; explicit
`label_routine` or a confirmation makes it weekly. `skipped_today`, `denied`
and silence are separate states.

## MCP

The server exposes thirteen tools. The merged tools use SQLite by default, so
no Redis or server setup is required.

```json
{
  "mcpServers": {
    "personatrace": { "command": "personatrace-mcp" }
  }
}
```

| Tool | Purpose |
|---|---|
| `memory_state` | current or historical beliefs |
| `memory_recall` | pinned + ranked beliefs, with hidden count |
| `memory_observe` | write one evidence-backed belief |
| `memory_correct` | invalidate a belief without deleting history |
| `memory_predict` | next action from the behavior sequence |
| `memory_remember` | write already-extracted beliefs / intentions |
| `memory_context` | structured pinned/dated/related/cross/agenda pack |
| `memory_agenda` | read-only short-term agenda |
| `memory_turn` | store one raw turn; returns an episode id |
| `memory_episodes` | retrospective raw-turn search by topic and time |
| `memory_routine_probe` | read-only in-turn routine probe candidates |
| `memory_routine_feedback` | confirmed / denied / skipped_today / unanswered / ended |
| `memory_routine_label` | explicit weekly routine with term/exceptions |

Every call has an 8s timeout and returns an error result instead of raising.
A failing memory must never take the host's conversation down.

See [MCP.md](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/docs/MCP.md).

## Evaluation

A 200-belief, 50-query fixture is included. It is a directional gate, not a
benchmark. The earlier 12/10 fixture scored 90.0%; it was too easy.

```text
method                      recall@3     MRR
v2 native (no pinned)            2.0%   0.020
v2 native (pinned)               2.0%   0.020
v1 string                       61.0%   0.657
v1 + bge                        73.0%   0.737
outer + bge                     74.0%   0.770
outer + bge + time facet        76.0%   0.760
```

The useful comparison is `outer + bge` → `+ time facet`: +2.0 recall but
-0.010 MRR. The time bucket can promote a due item above an equally correct
semantic item. The larger fixture is the number to quote.

```bash
python scripts/eval_recall.py
python scripts/eval_recall.py --no-embed
```

### LongMemEval temporal-reasoning subset

First external run, not the full benchmark and not GPT-4o:

```text
133 questions (temporal-reasoning)     101 correct   75.9%
20 questions (first run)                 17 correct   85.0%
20 questions (timed run)                 16 correct   80.0%
retrieval hybrid+rerank (133)           recall@3 86.8%, MRR 0.935
build 32,930 episodes                   1,259.9 s; reuse 2.5 s
```

Answerer and judge are both `deepseek-flash` at temperature 0, so self-judge
bias applies. The bilingual episode hook opens on 132/133 questions (20/20 on
the smoke set). The merged-context gate and the retrieval numbers are explained
in [LONGMEMEVAL.md](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/docs/LONGMEMEVAL.md).

## Design notes

The full failure analysis, measurements, tradeoffs and pitfalls are in
[DESIGN_NOTES.md](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/docs/DESIGN_NOTES.md).
The temporal kernel is described in
[MEMORY_V2.md](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/docs/MEMORY_V2.md).
The routine/vector research note is
[ROUTINE_MEMORY_RESEARCH.md](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/docs/ROUTINE_MEMORY_RESEARCH.md).

## What is shipped

| Area | Status |
|---|---|
| v2 kernel: retention, bi-temporal state, evidence, CAS guard | shipped |
| SQLite persistence by default | shipped |
| 17-category belief contract + Chinese/English typed time | shipped |
| outer retrieval + time facet + cross-category | shipped |
| short-term intentions with importance / TTL / anti-nag | shipped |
| conservative write-time dedup | shipped |
| `MergedMemory` facade + MCP tools | shipped |
| episodes + retrospective search | shipped (1.6.3) |
| weekly routines + in-turn probes | shipped (1.6.3) |
| v1 → v2 migration script | shipped; dry-run by default |
| graph memory | experimental, not wired into default retrieval |
| `PersonaManager` default path switched to merged | not yet (M3) |

Known limits:

- single user; no multi-tenant or auth;
- no UI, persona, or proactive messaging;
- dedup is lexical and conservative; a low-overlap paraphrase can still create a second record;
- 50 queries are still a small evaluation set;
- routines are weekly fields, not a full RFC 5545 RRULE parser yet;
- episode retrieval falls back to lexical/time when Chroma is unavailable; Chroma requires an embedding function
- routine probes are hints; the host must ask and must report feedback;
- there is no background scheduler and no automatic proactive message;
- `PersonaManager` remains the legacy v1 pipeline; use `MergedMemory` or MCP
  for the merged path.

## License

MIT
