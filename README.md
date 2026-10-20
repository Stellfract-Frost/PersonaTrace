# PersonaTrace

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-ready-4B5563)](MCP.md)

**A memory layer that knows what is still true.**

[中文说明](README.zh-CN.md)

Models have no sense of time. They will happily treat a preference you dropped in July as current, keep both sides of a contradiction, and never tell you which sentence a conclusion came from. One turn hides it. Weeks of planning exposes it.

So the unit here is not a fact. It is **a fact that holds for a while** — with a start date, a shelf life, and the evidence that put it there.

**Scope: memory only.** Someone else builds the assistant. This is the part that remembers. No UI, no persona, no proactive messaging — those belong to the host.

---

## Three assumptions this is built on

**1. What is missing is time, not storage.**
Keeping text was never the hard part. Answering *"is this still true, and how long has it been true?"* is. Every belief carries `valid_from`, `invalid_at`, `retention` and `due_in`.

**2. Behaviour tells you who someone is.**
What people say is filtered. What they *do* — when they show up, how often, at which hour, on which app — is not. Rhythm is sampled passively and promoted to a belief only after it repeats.

**3. A label is a compression, and compression loses context.**
`goals / ship V1.0` says nothing about *"demo is Friday, missing it breaks the contract."* Labels are the index, not the memory.

---

## What it does

### Retrieval is two-stage: index first, read second

| Layer | Unit | Size | When it is read |
|---|---|---|---|
| index | `label` (10–20 tokens) | hundreds | every query |
| context | `reasoning` — why it was stored | one line | hits only |
| evidence | the original sentence | 1–4k chars | hits only, first N |

Scoring never touches raw text. Only after an item is selected does `expand=N` go back for *why it was stored* and *what was actually said*.

The cheap part stays cheap: the embedding surface is labels (hundreds × 15 tokens), not conversation chunks (thousands × 200). The accurate part stays accurate: the interpreter sees the original sentence instead of a flattened tag.

### Beliefs expire — Ebbinghaus, per category

`R(t) = exp(-t / S)`. `S` is stability in days, set per category: facts live for years, `state` lives for days. Each new supporting evidence lengthens `S` with the spacing effect (`S' = S·(1 + k·(1−R))`) — repeating something just heard adds almost nothing, repeating something about to be forgotten adds a lot. Below `R < 0.3`, a belief is archived: kept, no longer retrieved.

### Two time axes

`valid_from` / `invalid_at` are independent of when the fact was learned. Corrected beliefs are invalidated, **not deleted**, so `state(as_of=...)` can still answer *"what did we think in September"*.

### Behaviour becomes a belief

Activity is sampled passively — on Windows via `UserAssist` + `Recent` (ROT13-decoded keys, FILETIME epoch, no daemon needed) — bucketed into 7×24 counters, and promoted only when the pattern recurs on **≥3 distinct days** with ±1 hour tolerance. Promotion consumes the observation, so nothing is promoted twice; a rhythm not re-confirmed within 30 days is dropped.

### The primary key is the identity

`uuid5(user:track:category:label)` — the same belief always lands on the same record, so writes are idempotent and the profile can be rebuilt from the store. Observation-type records fold their timestamp into the key, so *the same sentence at two different times* stays two observations.

### 15 relevant, 3 from other tracks

Retrieval returns 15 items that answer the question, plus **3 from categories that were not selected** — the "unasked but related" recall that makes an answer feel like it came from someone who knows you. Deliberately not rank 16–18.

### Writes stay off the critical path

Two agents, one job each: the **responder** only replies, the **detective** only writes memory. The detective runs on a thread pool after the reply, throttled by profile maturity (120s cold, 600s warm). Deep mode is the same pipeline with a stronger prompt and a bigger budget — not a third reasoning layer.

---

## How it compares

| | Stores | Cannot answer |
|---|---|---|
| **mem0** | extracted facts + vector index | "is this still true?" |
| **Zep / Graphiti** | temporal knowledge graph, validity windows | "what is this person's rhythm?" |
| **Letta** | context tiers the model pages itself | "when should this expire?" |
| **Memobase** | profile + event streams | "what did the behaviour imply?" |
| **PersonaTrace** | beliefs with shelf life + evidence + rhythm | — |

mem0's 2026 rewrite dropped UPDATE/DELETE in favour of append-only extraction with temporal reasoning at retrieval. Same conclusion reached from the other side: **append evidence, judge at read time.** The difference is that conclusions live in their own layer here, so the prompt never carries raw history.

---

## Plug it in

### MCP (no adapter code)

```json
{
  "mcpServers": {
    "personatrace": { "command": "python", "args": ["mcp_server.py"] }
  }
}
```

| tool | when it fires |
|---|---|
| `memory_state` | "what do we know about this person, now or back then" |
| `memory_recall` | needs background; pinned + ranked, plus how many were hidden |
| `memory_observe` | after a turn — idempotent, same label never duplicates |
| `memory_correct` | user says "you remembered that wrong" |
| `memory_predict` | next action, from the behaviour sequence |

Standard library only. Every call has an 8s timeout and returns an error *result* instead of raising — a failing memory must never take the host's conversation down.

### Python SDK

```python
from personatrace import PersonaConfig, PersonaManager, LLMClient
from sentence_transformers import SentenceTransformer

cfg = PersonaConfig(llm_api_key="sk-...", llm_base_url="https://api.openai.com/v1", llm_model="gpt-4o")
emb = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
mgr = PersonaManager(cfg, llm=LLMClient(cfg), embed_fn=lambda t: emb.encode(t).tolist())

mgr.ingest_conversation(user_msg="...", ai_reply="...", user_id="alice")
print(mgr.format_profile(user_id="alice"))
```

### FastAPI

`/v1/observe` · `/v1/state?as_of=` · `/v1/recall?expand=` · `/v1/behavior` · `/v1/behavior/predict` · `/v1/beliefs/{id}/invalidate` · `/v1/health`

---

## Status

| Built | Next |
|---|---|
| retention decay computed at read time (no cron scan) | two-way evidence query (evidence → which beliefs) |
| bi-temporal state + `as_of` | MySQL persistence (evidence / belief / events) |
| two-stage retrieval (index → read) | `as_of` wired into the chat prompt |
| rhythm pipeline: passive sampling → ≥3-day gate → consume | intent routing (facts / advice / recall / self) |
| behaviour sequence + next-action estimate | quantify the accuracy cost of dropping instant analysis |
| cross-category association (15 + 3) | |
| MCP server · FastAPI layer · idempotent writes | |

```
$ pytest test_memory_v2.py test_mcp_server.py -q
27 passed
```

Single user by design; `user_id` is threaded through so it stays additive.

---

## Not doing

- **UI, persona, proactive messages.** The host owns how it feels.
- **Multi-tenant / auth.** Not needed for one person; it would be decoration.
- **Parameter-level memory (LoRA).** Unverifiable at this scale.
- **Self-hosted ANN tuning.** A few hundred beliefs — brute-force ranking is simpler and more accurate.

---

## License

MIT
