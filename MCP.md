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

### Fields that matter

- `retention`: how much of the belief is still fresh. `R = exp(-t/S)`.
- `staleness`: time since support / stability. Above 1 means more than half a
  shelf life has passed.
- `due_in`: days until due (negative = overdue). This is the planning field.
- `time_kind` / `time_raw`: typed time (`due / past / range / recurring`) and the
  original expression.
- `hidden`: how many related items were squeezed out by the budget.
- `agenda_lines`: short-term intentions; `[需主动]` marks the one item whose
  time bucket changed and may be raised proactively.

## 3. Minimal sequence

```json
{"jsonrpc":"2.0","id":1,"method":"initialize"}
{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
{"jsonrpc":"2.0","id":3,"method":"tools/call",
 "params":{"name":"memory_remember","arguments":{
   "beliefs":[{"category":"identity","label":"名字叫小陌"}],
   "intentions":[{"action":"上课","due":"后天","importance":0.9}],
   "source":"direct"
 }}}
{"jsonrpc":"2.0","id":4,"method":"tools/call",
 "params":{"name":"memory_context","arguments":{"query":"我那个 deadline 是几号"}}}
```

Protocol: newline-delimited JSON-RPC 2.0 over stdio. Notifications (no `id`)
produce no response.

## 4. Contract

| Contract | Implementation |
|---|---|
| Core protocol has no SDK dependency | stdlib JSON-RPC loop |
| Merged tools are optional | lazy `MergedMemory` import; Pydantic + SQLite |
| Failure does not take the host down | 8s timeout per call; exceptions become `isError` results |
| No conversation pollution | tools return structured data; no prompt mutation |
| No scope creep | no proactive messaging, no persona, no UI |

The old five tools (`state / recall / observe / correct / predict`) still work
against `MemoryV2`. The three merged tools use `MergedMemory` and share one
SQLite file for beliefs and intentions.

## 5. Tests

```bash
python -m pytest test_mcp_server.py -q
python -m pytest test_memory_v2.py -q
```

`test_stdio_end_to_end` starts a real subprocess and walks
`initialize → tools/list → observe → state`. The merged tests cover
`remember → context → agenda` through both the handler and a subprocess.
