# PersonaTrace

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-ready-4B5563)](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/MCP.md)

**一个记得「现在还成立什么、什么快到期、为什么记它」的记忆层。**

[English](README.md)

PersonaTrace 不是聊天机器人，不是 AstrBot 插件，也不是 agent harness。
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
  ├─ pinned：boundaries / constraints / identity
  ├─ 语义：字面 IDF + 向量余弦（候选集内归一）
  ├─ 时间桶：只对时间意图查询 due/range
  ├─ 跨类别：从没入选的类别里补
  └─ 回表：前几条读 reasoning + 原话
```

`score = (0.15 + 0.85 * similarity) * (0.35 + 0.65 * retention)`

时间问题不是靠更强的 embedding 解决的，而是靠把日期写进 typed 字段、再查这个字段。
时间桶是条件旁路，不是第二个检索器。

## 安装

```bash
pip install personatrace
pip install "personatrace[llm,vector]"   # 可选：LLM / embedding
```

Python 3.10+。belief 内核只用标准库；合并 facade 用 Pydantic；默认 SQLite，
不需要 Redis。

## 快速开始：结构化写入

宿主负责抽取，PersonaTrace 负责存、排序、过期；这条路径不调用 LLM。

```python
from personatrace import MergedMemory

memory = MergedMemory(
    user_id="alice",
    sqlite_path="./personatrace.db",   # belief 和 intention 同一个文件
)

memory.write(
    beliefs=[{
        "category": "identity",
        "label": "名字叫小陌",
        "confidence": 0.9,
        "reasoning": "用户自我介绍",
    }],
    intentions=[{
        "action": "上课",
        "due": "后天",
        "importance": 0.9,
        "confidence": 0.8,
    }],
    source="direct",
)

context = memory.build_context("我那个 deadline 是几号")
print(context.agenda_lines)
print(memory.inject("我那个 deadline 是几号"))
```

- `build_context()` 返回结构化数据，无副作用；
- `inject()` 只是可选的 prompt 适配器，宿主可以自己渲染；
- `source` 只允许 `direct / bridge / console`，主动回合不能写用户事实。

## MCP

```json
{
  "mcpServers": {
    "personatrace": { "command": "personatrace-mcp" }
  }
}
```

八个工具：`memory_state / memory_recall / memory_observe / memory_correct /
memory_predict / memory_remember / memory_context / memory_agenda`。
合并工具默认 SQLite，不需要 Redis。每次调用 8 秒超时，出错返回错误结果，
不抛异常拖垮宿主的对话。

## 评测

12 条信条、10 条查询，只看方向，不当统计结论：

```text
v2 native (no pinned)      35.0% / 0.333
v2 native (pinned)         23.3% / 0.283
v1 string                  53.3% / 0.583
v1 + bge                   76.7% / 0.833
outer + bge                76.7% / 0.833
outer + bge + time facet   90.0% / 0.933
```

真正有价值的是 `outer + bge` → `+ time facet`：提升来自 typed time，不是换模型。

```bash
python scripts/eval_recall.py
```

## 设计笔记

失败分析、实测、取舍和踩坑在
[DESIGN_NOTES.md](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/DESIGN_NOTES.md)；
时序内核在
[MEMORY_V2.md](https://github.com/Stellfract-Frost/PersonaTrace/blob/main/MEMORY_V2.md)。

## 现状与边界

| 模块 | 状态 |
|---|---|
| v2 内核：保持率、双时态、证据、CAS | 已发布 |
| SQLite 默认持久化 | 已发布 |
| 17 类 belief + 中文 typed time | 已发布 |
| 外层检索 + 时间桶 + 跨类别 | 已发布 |
| 短期 intention：importance / TTL / anti-nag | 已发布 |
| `MergedMemory` + MCP | 已发布（M2） |
| v1 → v2 迁移脚本 | 有；默认 dry-run |
| graph memory | 实验性，未接默认检索 |
| `PersonaManager` 默认切到合并路径 | 未做（M3） |

明确不做：UI、人格、主动消息、多租户、AstrBot 插件本体。
这个仓库不是 AstrBot 插件；插件在独立仓库，不属于这个包。

## License

MIT
