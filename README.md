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

## Two goals, one repo

This project has two goals. The first is a lightweight memory kernel that stays
inspectable — one SQLite file, no write-time LLM calls, no graph database.

The second is to write down what actually went wrong while building it. Every
budget, fusion rule and state transition here was paid for with a failed run,
and most of those failures were not obvious from the code. They are recorded in
[docs/PITFALLS.md](docs/PITFALLS.md) so the next person does not pay for them again.

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
# Omitting sqlite_path is also durable now: beliefs, intentions, episodes and
# routines all follow the default backend path.

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

Every tool call runs in one isolated worker process with an 8s deadline. A
timeout kills the worker, so a slow call cannot keep writing behind the error
result or exhaust a shared thread pool; the next call starts a fresh worker.
In-memory mode (`PERSONA_DB=off`) runs in-process and is tests-only. JSON-RPC
batch arrays are accepted, and malformed messages return an error instead of
crashing the stdio loop. Both tool families use one `PERSONA_USER_ID` and one
store, so `memory_observe` is visible to `memory_context`.

See [MCP.md](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/docs/MCP.md).

## Evaluation

### LongMemEval_s: 500 questions

Primary external result for the merged episode layer. The answerer and judge are
both `deepseek-flash` at temperature 0. This is not a GPT-4o number.

| Question type | Questions | Correct | Accuracy | tok/question |
|---|---:|---:|---:|---:|
| single-session-assistant | 56 | 55 | 98.2% | 21,591 |
| single-session-user | 70 | 67 | 95.7% | 6,367 |
| temporal-reasoning | 133 | 119 | 89.5% | 6,901 |
| knowledge-update | 78 | 67 | 85.9% | 6,426 |
| multi-session | 133 | 105 | 78.9% | 7,116 |
| single-session-preference | 30 | 18 | 60.0% | 25,022 |
| **total** | **500** | **431** | **86.2%** | **8,467** |

```text
LLM self-judge            86.2%  (431/500)
rule-judge lower bound    78.4%
cost                      about ¥5.50
temporal subset           89.5%  (119/133)
```

The result file is `data/lme_500_final.json`. Details, configuration and the
hook note are in [docs/LONGMEMEVAL.md](docs/LONGMEMEVAL.md).

### Local regression fixture

The 200-belief / 50-query fixture is a deterministic CI gate, not a benchmark.
It is intentionally kept out of the main result: [docs/LOCAL_EVAL.md](docs/LOCAL_EVAL.md).

```bash
python scripts/eval_recall.py --no-embed --fail-under 0.60 --gate-method "outer lex"
```

## Design notes

The full failure analysis, measurements, tradeoffs and pitfalls are in
[DESIGN_NOTES.md](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/docs/DESIGN_NOTES.md).
The temporal kernel is described in
[MEMORY_V2.md](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/docs/MEMORY_V2.md).
The routine/vector research note is
[ROUTINE_MEMORY_RESEARCH.md](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/docs/ROUTINE_MEMORY_RESEARCH.md).
Release history is in [CHANGELOG.md](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/CHANGELOG.md);
the v1 → 2.x import/behavior changes are in
[UPGRADING.md](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/docs/UPGRADING.md).

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
| episodes + retrospective search | shipped |
| optional cross-encoder reranking inside `episodes_search` | shipped |
| weekly routines + in-turn probes | shipped |
| v1 → v2 migration script | shipped; dry-run by default |
| runtime v1 pipeline | removed in 2.0; migration shim only |
| CI: tests on Windows/Linux + recall gate | shipped |

Known limits:

- single user; no multi-tenant or auth;
- no UI, persona, or proactive messaging;
- dedup is lexical and conservative; numbers/months must match, and a low-overlap paraphrase can still create a second record;
- the 50-query local fixture is a CI gate, not a benchmark; LongMemEval is the external result;
- routines are weekly fields, not a full RFC 5545 RRULE parser yet;
- episode retrieval falls back to lexical/time when Chroma is unavailable; Chroma requires an embedding function.
- routine probes are hints; the host must ask and must report feedback;
- there is no background scheduler and no automatic proactive message;
- old Chroma profiles must be migrated with `scripts/migrate_v1.py`; the
  runtime v1 pipeline is no longer shipped.

## License

MIT
