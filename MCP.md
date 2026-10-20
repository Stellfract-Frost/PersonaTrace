# MCP 接入（给宿主用的标准出口）

PersonaTrace 只做一件事：**记住并说清「现在什么还成立、什么已经到期」**。它不做陪伴、不做 UI、不主动找你——那些是宿主的事。

MCP server 是它对外唯一的标准出口。任何支持 MCP 的宿主（Claude Desktop、Cursor、自研 agent）贴上配置就能用，不需要读源码、不需要写适配层。

---

## 1. 挂上（一行配置）

```json
{
  "mcpServers": {
    "personatrace": {
      "command": "python",
      "args": ["C:/Users/NewAdmin/OneDrive/Desktop/goal/mcp_server.py"]
    }
  }
}
```

放在宿主的 MCP 配置文件里（Claude Desktop 是 `claude_desktop_config.json`，Cursor 是 `.cursor/mcp.json`）。

**不需要** Redis、MySQL、API Key、网络。默认落 SQLite（`~/.personatrace/memory.db`），重启不丢；装了 redis 包且给 `PERSONA_DB` 指向 Redis 时自动切过去共享；`PERSONA_DB=off` 则退回进程内（仅测试用）。

---

## 2. 五个工具

| 工具 | 什么时候宿主会调它 | 返回什么 |
|---|---|---|
| `memory_state` | 开场、需要判断"这个人现在什么状态" | 结论列表，每条带 `confidence / retention / staleness / due_in / needs_review` |
| `memory_recall` | 需要背景信息时 | 约束与身份类**强制入选**，其余按 `相关度 × 保持率` 排序，附 `hidden` 条数 |
| `memory_observe` | 一段对话结束、或读到一份文件 | `hits / stability_days / retention`（幂等：同一条不会变成两条） |
| `memory_correct` | 用户说"你记错了" | 标记失效，**不删除**（时间线仍可回溯） |
| `memory_predict` | 需要预判下一步动作 | bigram 转移候选 + 支持度 |

### 为什么这几个字段重要

- `retention`（保持率）：这条结论还剩多少"新鲜度"。艾宾浩斯 `R = exp(-t/S)`，`S` 随证据强化而增长。
- `staleness`：距上次证据支撑 / 稳定度。**> 1 说明已经过了半个周期**，该轻用。
- `due_in`：还有几天到期（负数=已超期）。**这是"规划"真正需要的字段**——不是"过了几天"，而是"还剩几天"。
- `needs_review`：到期临近或保持率过低，提示宿主先确认再引用。
- `hidden`：有多少条相关记忆被预算挤掉了。告诉模型"我知道的比给你的多"，比假装知道全部更安全。

---

## 3. 一个最小调用序列

```json
{"jsonrpc":"2.0","id":1,"method":"initialize"}
{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
{"jsonrpc":"2.0","id":3,"method":"tools/call",
 "params":{"name":"memory_observe","arguments":{"category":"goals","label":"交付 V1.0","track":"work"}}}
{"jsonrpc":"2.0","id":4,"method":"tools/call",
 "params":{"name":"memory_state","arguments":{}}}
```

协议：**newline-delimited JSON-RPC 2.0 over stdio**。通知（无 `id`）不返回任何东西。

自己验证：

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize"}' | python mcp_server.py
```

---

## 4. 契约硬化（为什么宿主敢接）

| 契约 | 实现 |
|---|---|
| **零依赖** | 只依赖 Python 标准库 + `memory_v2.py`；没有官方 SDK、没有 fastmcp |
| **失败不拖垮宿主** | 每个工具调用 8 秒超时；任何异常转成 `isError` 文本结果，**不抛给宿主** |
| **不污染对话** | 工具只返回结构化数据，不写 prompt、不改宿主上下文 |
| **数据拿得走** | `export_counterpart_skill` 导出可复用 Skill；建议再加纯 JSON 导出 |
| **不越界** | 不做主动性、不做情绪、不做人格——这些是宿主的产品选择 |

---

## 5. 自测

```bash
python -m pytest test_mcp_server.py -q     # 10 tests: protocol + real subprocess end-to-end
python -m pytest test_memory_v2.py -q      # 17 tests: kernel algorithms
```

其中 `test_stdio_end_to_end` 会真实起一个 `mcp_server.py` 子进程，走 stdin/stdout 完成 initialize → tools/list → observe → state 全链路。
