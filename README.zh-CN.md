# PersonaTrace

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-ready-4B5563)](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/docs/MCP.md)

**一个记得「现在还成立什么、什么快到期、为什么记它」的记忆层。**

[English](README.md)

PersonaTrace 不是聊天机器人，也不是 agent harness。
它只做记忆：宿主负责模型、prompt、UI、语气和主动行为。

记忆的单位不是「事实」，而是**在某段时间内成立的事实**：有生效时间、保持率、
证据，以及可选的到期时间。一次性承诺单独放在 intention 里，生命周期按天/小时算，
不按长期画像的 retention 算。

## 它记什么

| 类型 | 例子 | 生命周期 |
|---|---|---|
| Belief | 偏好、身份、约束、项目、目标、习惯 | 按类别算保持率；归档不删除 |
| Intention | 后天要上课、周五交报告 | due + grace；`done / cancelled / expired` |
| Evidence | 结论对应的原话 | 命中后才回表读 |
| Episode | 原始 user/assistant turn | SQLite；按时间和字面回溯 |
| Routine | 每周二 19:00 上课 | weekly recurrence + 学期/例外 + probe 状态 |
| Mention | 别人说他最近在考研 | 和第一人称事实分开，衰减更快 |

合并路径使用 **17 类 belief**，两轨：

```text
work:    explicit_facts expertise projects goals likes dislikes constraints
persona: identity traits interests style_notes behavior_patterns
         decision_style interpersonal boundaries habit state
```

## 检索

```text
query
  ├─ pinned: boundaries / constraints / identity
  ├─ semantic: lexical IDF + vector cosine (candidate-set normalized)
  ├─ time facet: due/range rows for time-intent queries
  ├─ cross-category: items from categories not already selected
  └─ evidence on demand: reasoning + original sentence for the first hits
```

`score = (0.15 + 0.85 * similarity) * (0.35 + 0.65 * retention)`

时间问题不是靠更强的 embedding 解决的，而是靠把日期写进 typed 字段、再查这个字段。
时间桶是条件旁路，不是第二个检索器。

## 安装

```bash
# Recommended: Chroma + an embedding model
pip install "personatrace[vector]"

# Minimal install: lexical/time fallback only
pip install personatrace
```

Python 3.10+。belief 内核只用标准库；合并 facade 用 Pydantic；默认 SQLite，
不需要 Redis。

### 推荐装上 Chroma episode 索引

SQLite 是 episode 的 source of truth；Chroma 是推荐的语义候选索引。除非你明确
只想要字面/时间兜底，否则建议安装 `vector` extra。真正提供语义能力的是
embedding 模型，Chroma 负责索引，不负责状态。

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

Chroma 不可用时自动退回字面 + 时间检索。这个兜底是刻意保留的，但对复述和更大的
episode 库会更弱。v1 的 `persona_items` collection 不变；episode 使用独立的
`episode_turns` collection。

## 快速开始：结构化写入

宿主负责抽取，PersonaTrace 负责存、排序、过期；这条路径不调用 LLM。

```python
from personatrace import MergedMemory

memory = MergedMemory(
    user_id="alice",
    sqlite_path="./personatrace.db",   # one file for beliefs and intentions
)
# 不传 sqlite_path 也是持久的：belief / intention / episode / routine
# 都会落到默认后端路径，而不是 belief 持久、其余三者在内存。

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

- `build_context()` 返回结构化数据，无副作用；
- `inject()` 只是可选的 prompt 适配器，宿主可以自己渲染；
- `source` 只允许 `direct / bridge / console`，主动回合不能写用户事实。

### Turn 和 routine

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
print(pack.routine_probes)     # hint only; the host decides whether to ask
```

Routine 是 weekly，不是 daily。一次观察只生成 candidate；明确标注或确认后才成为
weekly。`skipped_today`、`denied`、沉默是三种不同状态。

## MCP

```json
{
  "mcpServers": {
    "personatrace": { "command": "personatrace-mcp" }
  }
}
```

十三个工具：`memory_state / memory_recall / memory_observe / memory_correct /
memory_predict / memory_remember / memory_context / memory_agenda /
memory_turn / memory_episodes / memory_routine_probe / memory_routine_feedback /
memory_routine_label`。
合并工具默认 SQLite，不需要 Redis。每次调用在独立 worker 进程里跑，8 秒到期
直接杀进程：不会出现「返回超时但后台还在写」，也不会把线程池占满；下一次调用
重新拉起 worker。`PERSONA_DB=off` 的进程内模式只用于测试。JSON-RPC batch
数组会被逐条处理，坏消息返回错误而不是打挂 stdio 循环。两族工具共用同一个
`PERSONA_USER_ID` 和同一个库，`memory_observe` 写的东西 `memory_context` 能看到。

## 评测

200 条信条、50 条查询，只看方向，不当统计结论。之前 12/10 的 90.0% 太容易：

```text
method                      recall@3     MRR
v2 native (query, no pinned)    66.0%   0.707
v2 native (pinned)               2.0%   0.020
outer lex                       61.0%   0.637
outer + bge                     74.0%   0.770
outer + bge + time facet        76.0%   0.760
```

`v2 native (query)` 是内核自己的字面检索，不套外层；`pinned` 是压力样例：
`k=3` 时有四十条 pinned，预算全被 pinned 吃掉，所以仍是 2.0%。
`outer + bge` → `+ time facet`：recall +2.0，MRR -0.010。时间桶可能把一条正确的
due 记录排到同样正确的语义记录前面；引用大 fixture 的数字。CI 用确定性的
`outer lex` 通道卡 60%。

```bash
python scripts/eval_recall.py
# CI 闸门（确定性的字面通道，不需要 embedding 模型）
python scripts/eval_recall.py --no-embed --fail-under 0.60 --gate-method "outer lex"
```

### LongMemEval temporal-reasoning 子集

第一次外部评测，不是完整 LongMemEval，也不是 GPT-4o：

```text
133 questions (temporal-reasoning)   101 correct   75.9%
20 questions (first run)              17 correct   85.0%
20 questions (timed run)              16 correct   80.0%
hybrid+rerank retrieval (133)         recall@3 86.8%, MRR 0.935
index build 32,930 episodes           1,259.9 s; reuse 2.5 s
```

作答和判分都是 `deepseek-flash`，temperature 0，存在同模型自判偏差。双语
episode hook 在 133 题中命中 132 题（20 题 smoke 为 20/20）；合并上下文和检索
数字见
[LONGMEMEVAL.md](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/docs/LONGMEMEVAL.md)。

## 设计笔记

失败分析、实测、取舍和踩坑在
[DESIGN_NOTES.md](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/docs/DESIGN_NOTES.md)；
时序内核在
[MEMORY_V2.md](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/docs/MEMORY_V2.md)。

## 现状与边界

| 模块 | 状态 |
|---|---|
| v2 内核：保持率、双时态、证据、CAS | 已发布 |
| SQLite 默认持久化 | 已发布 |
| 17 类 belief + 中英 typed time | 已发布 |
| 外层检索 + 时间桶 + 跨类别 | 已发布 |
| 短期 intention：importance / TTL / anti-nag | 已发布 |
| 保守的词面去重 | 已发布 |
| `MergedMemory` + MCP | 已发布 |
| episode + 回溯检索 | 已发布（1.6.5） |
| weekly routine + in-turn probe | 已发布（1.6.5） |
| v1 → v2 迁移脚本 | 有；默认 dry-run |
| 运行时 v1 | 2.0 已移除；只保留迁移 shim |
| CI：Windows/Linux 测试 + recall 闸门 | 已发布 |

去重是词面且保守的；低重叠的复述仍可能产生第二条记录。50 条查询也仍然是小样本。
Routine 目前是 weekly 字段，不是完整 RFC 5545 RRULE parser；probe 只是 hint，
Chroma 不可用时 episode 检索退回字面 + 时间；Chroma 需要外部 embed_fn。
宿主必须自己问并回传 feedback；没有后台定时器，也不会自动主动发消息。

明确不做：UI、人格、主动消息、多租户、聊天平台插件本体。

## License

MIT
