# PersonaTrace

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-ready-4B5563)](MCP.md)

**一个知道「现在还成立什么」的记忆层。**

[English](README.md)

模型没有时间概念。它会把三个月前你已经放弃的偏好当成现状，会把一对矛盾的两句话同时记住，也不会告诉你某个结论是从哪句话推出来的。一轮对话看不出来，做几周规划就会崩。

所以这里的记忆单位不是「事实」，而是**在某段时间内成立的事实**——有起始时间、有保质期、有把它放进去的那句话。

**范围：只做记忆。** 助手由别人来搭，这里只管「记得住」。没有 UI、没有人格、不会主动找你——那些是宿主的事。

---

## 三条前提

**1. 缺的不是存储，是时间。**
把文字存下来从来不难。难的是回答「这条现在还成立吗、成立了多久」。所以每条信条都带 `valid_from`、`invalid_at`、`retention`、`due_in`。

**2. 行为说明一个人是谁。**
人说的话是过滤过的，做的事不是——什么时候出现、多久来一次、几点、在用什么应用。节律是被动采样出来的，不是问出来的。

**3. 标签是一种压缩，压缩必然丢语境。**
`goals / 交付 V1.0` 完全看不出「周五要演示，做不完就违约」。所以标签只当索引，不当记忆本身。

---

## 它做了什么

### 检索是两级的：先查索引，再读原文

| 层 | 单位 | 体量 | 什么时候读 |
|---|---|---|---|
| 索引 | `label`（10–20 token） | 几百条 | 每次查询都读 |
| 语境 | `reasoning`（为什么存这条） | 一句 | 只有命中的 |
| 证据 | 原始语句 | 1–4k 字 | 只有命中的、只取前 N 条 |

打分阶段完全不碰原文。只有选中之后，`expand=N` 才回表去取「为什么存」和「当时到底说了什么」。

便宜的地方一直便宜：向量化面是标签（几百条 × 15 token），不是对话切片（几千条 × 200 token）。准确的地方一直准确：模型看到的是原句，不是压平的标签。

### 信条会过期（艾宾浩斯，按类别给稳定度）

`R(t) = exp(-t / S)`。`S` 是稳定度（天），按类别设定：事实能活几年，`state` 只能活几天。每来一条支持证据，`S` 按间隔效应变长（`S' = S·(1 + k·(1−R))`）——刚说完就重复几乎没用，快忘的时候重复才真正加固。`R < 0.3` 进归档：留着，但不再取用。

### 两条时间轴

`valid_from` / `invalid_at` 和「什么时候知道的」是两件事。被纠正的信条只打失效标记、**不删除**，所以 `state(as_of=...)` 还能回答「我们九月的时候是怎么认为的」。

### 行为升格为信条

行为被动采样（Windows 上是 `UserAssist` + `Recent`：ROT13 键名、FILETIME 纪元，不需要守护进程），按小时分桶，**跨 ≥3 个不同日期**、±1 小时容差才升格。升格即消费，同一条观察不会重复升格；30 天没再确认的节律会被删掉。

### 主键就是身份

`uuid5(user:track:category:label)`——同一条结论永远落在同一条记录上，所以写入天然幂等，整份画像可以从存储里重建。观察类记录把时间戳并进主键，于是「同一句话在两个时间说」始终是两条。

### 15 条定点 + 3 条跨轨道

检索给你 15 条答得上问题的，再加 **3 条来自本次没选中的类别**——就是那种「没被问、但相关」的回忆，让回答听起来像认识你的人说的。刻意不走 rank 16–18。

### 写入不在关键路径上

两个 Agent，各干一件事：**助手**只负责回复，**侦探**只负责写记忆。侦探在回复之后丢进线程池，按画像成熟度节流（冷启动 120s、成型后 600s）。深度模式是同一条链路换个更强的 prompt 和更大的预算，不是第三层推理。

---

## 和同类比

| | 存什么 | 答不了什么 |
|---|---|---|
| **mem0** | 抽取的事实 + 向量索引 | 「这条现在还成立吗」 |
| **Zep / Graphiti** | 时序知识图谱、有效期窗口 | 「这个人的节律是什么」 |
| **Letta** | 模型自己换页的上下文分层 | 「这条什么时候该过期」 |
| **Memobase** | 画像 + 事件流 | 「行为暗示了什么」 |
| **PersonaTrace** | 带保质期的信条 + 证据 + 节律 | — |

mem0 在 2026 年把 UPDATE/DELETE 砍掉，改成只追加 + 检索期做时间推理。同一个结论，从另一边得出的：**证据只追加，判断放到读取时。** 差别在于这里结论单独有一层，prompt 不必背原始历史。

---

## 怎么接

### MCP（不用写适配层）

```json
{
  "mcpServers": {
    "personatrace": { "command": "python", "args": ["mcp_server.py"] }
  }
}
```

| 工具 | 什么时候会被调 |
|---|---|
| `memory_state` | 「这个人现在 / 当时是什么状态」 |
| `memory_recall` | 需要背景：必选项 + 排序结果 + 被隐藏的条数 |
| `memory_observe` | 一轮结束后写入，幂等，同一条不会重复 |
| `memory_correct` | 用户说「你记错了」 |
| `memory_predict` | 根据行为序列预测下一步 |

只用标准库。每次调用 8 秒超时，出错返回一个错误**结果**而不是抛异常——记忆挂了不能把宿主的对话带崩。

### Python SDK

```python
from persona import PersonaConfig, PersonaManager, LLMClient
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

## 现状

| 已完成 | 下一步 |
|---|---|
| 读时计算保持率，不需要定时全表扫 | 证据双向查询（证据 → 它支撑了哪些结论） |
| 双时态 + `as_of` 回溯 | MySQL 落盘（证据 / 信条 / 事件） |
| 两级检索（索引 → 回表） | `as_of` 接进对话 prompt |
| 节律链路：被动采样 → ≥3 天门槛 → 升格即消费 | 意图路由（事实 / 建议 / 回忆 / 自我认知） |
| 行为序列 + 下一步预测 | 量化「放弃即时分析」带来的精度损失 |
| 跨类别联想（15 + 3） | |
| MCP server · FastAPI 层 · 幂等写入 | |

```
$ pytest test_memory_v2.py test_mcp_server.py -q
27 passed
```

单用户，是设计如此；`user_id` 一直贯通，以后要加多租户是加法不是重写。

---

## 明确不做

- **UI、人格、主动消息。** 那是宿主的事。
- **多租户 / 鉴权。** 一个人用不需要，加了是装饰。
- **LoRA 参数化记忆。** 这个规模验证不了，写进去也是摆设。
- **自建 ANN 调优。** 几百条信条，暴力排序更简单也更准。

---

## License

MIT
