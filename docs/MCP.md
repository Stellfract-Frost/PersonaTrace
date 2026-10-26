# MCP

PersonaTrace exposes a memory API, not an assistant. Hosts decide when to call
it, how to render the result, and whether to speak. The MCP layer returns
structured data only; it does not write prompts or take over the conversation.

## 1. Connect

```json
{
  "mcpServers": {
    "personatrace": {
      "command": "personatrace-mcp"
    }
  }
}
```

If the package is not installed, use the repository entry:

```json
{
  "mcpServers": {
    "personatrace": {
      "command": "python",
      "args": ["-m", "personatrace.mcp_server"]
    }
  }
}
```

The default store is SQLite (`~/.personatrace/memory.db`). No Redis, MySQL,
API key, or network is required. Set `PERSONA_DB` to another path to move it;
`PERSONA_DB=off` uses an in-process store for tests only.

## 2. Tools

| Tool | Purpose |
|---|---|
| `memory_state` | beliefs at the current or a historical moment |
| `memory_recall` | pinned + ranked beliefs, with hidden count |
| `memory_observe` | write one evidence-backed belief |
| `memory_correct` | invalidate a belief; history is kept |
| `memory_predict` | next action from the behavior sequence |
| `memory_remember` | write already-extracted beliefs / intentions; no LLM call |
| `memory_context` | structured `pinned / dated / related / cross / agenda_lines` pack |
| `memory_agenda` | read-only short-term agenda with due, importance and bucket |
| `memory_turn` | store one raw turn; returns an `episode_id`; no extraction |
| `memory_episodes` | retrospective raw-turn search by topic and time range |
| `memory_routine_probe` | read-only weekly-routine probe candidates for this turn |
| `memory_routine_feedback` | `confirmed / denied / skipped_today / unanswered / ended` |
| `memory_routine_label` | explicit weekly routine with weekday/time/term/exceptions |

### Fields that matter

- `retention`: how much of the belief is still fresh. `R = exp(-t/S)`.
- `as_of` / `known_as_of`: valid time and transaction time. With only `as_of`,
  the result is a true snapshot of what was known then.
- `staleness`: time since support / stability. Above 1 means more than half a
  shelf life has passed.
- `due_in`: days until due (negative = overdue). This is the planning field.
- `time_kind` / `time_raw`: typed time (`due / past / range / recurring`) and the
  original expression. On `memory_observe`, omitting these fields preserves the
  old value; passing `null` / `"none"` / `""` clears it.
- `revive`: `memory_observe` reopens a corrected belief by default when the new
  observation is at or after its invalidation time; pass `false` to keep it
  closed.
- `hidden`: how many related items were squeezed out by the budget.
- `agenda_lines`: short-term intentions; `[proactive]` marks the one item whose
  time bucket changed and may be raised proactively.
- `episodes`: raw turns selected only for retrospective queries (`last time`,
  `previously`, `remember`).
- `routine_probes`: in-turn probe candidates; the host decides whether to ask.
  A probe is never sent by the server.

## 3. Minimal sequence

```json
{"jsonrpc":"2.0","id":1,"method":"initialize"}
{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
{"jsonrpc":"2.0","id":3,"method":"tools/call",
 "params":{"name":"memory_remember","arguments":{
   "beliefs":[{"category":"identity","label":"Name is Mo"}],
   "intentions":[{"action":"attend class","due":"2026-10-25","importance":0.9}],
   "source":"direct"
 }}}
{"jsonrpc":"2.0","id":4,"method":"tools/call",
 "params":{"name":"memory_context","arguments":{"query":"When is my deadline?"}}}
```

Protocol: newline-delimited JSON-RPC 2.0 over stdio. Notifications (no `id`)
produce no response.

## 4. Contract

| Contract | Implementation |
|---|---|
| Core protocol has no SDK dependency | stdlib JSON-RPC loop |
| Merged tools are optional | lazy `MergedMemory` import; Pydantic + SQLite |
| Failure does not take the host down | one isolated worker process, 8s deadline, kill on expiry; exceptions become `isError` results |
| Batch-safe | JSON-RPC arrays are processed one by one; malformed/non-object messages return an error |
| Wire is UTF-8 | stdin/stdout are reconfigured explicitly, independent of the host locale |
| Internal errors stay internal | generic `-32603` on the wire; traceback goes to stderr |
| One user namespace | both tool families use `PERSONA_USER_ID` and the same store |
| No conversation pollution | tools return structured data; no prompt mutation |
| No scope creep | no proactive messaging, no persona, no UI |

The old five tools (`state / recall / observe / correct / predict`) and the
merged tools now resolve to the same `MemoryV2` instance/namespace. The merged
tools add intentions, episodes and routines in the same SQLite file.

`as_of` is snapshot time by default: it selects what was known at that moment.
Pass `known_as_of` (for example, now) to ask what the current store knows about
a historical date.

In-memory mode (`PERSONA_DB=off`) runs in-process because a child process cannot
share it; durable mode uses the isolated worker and a real deadline.

## 5. Tests

```bash
python -m pytest test_mcp_server.py -q
python -m pytest test_memory_v2.py -q
```

`test_stdio_end_to_end` starts a real subprocess and walks
`initialize → tools/list → observe → state`. The merged tests cover
`remember → context → agenda` through both the handler and a subprocess.
`test_worker_timeout_is_real_and_restartable` proves a sleeping tool is killed
at the deadline and the next call starts a fresh worker; the batch tests send a
JSON array through stdin and assert the process still exits 0.
