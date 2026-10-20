# PersonaTrace × AstrBot

给 AstrBot 用的时序记忆层。它不是"再挂几个工具"，而是把记忆接到宿主的两条腿上：

- **每轮读**：`on_llm_request` 把 2–3 条最该被想起的记忆塞进当前轮（`extra_user_content_parts`），不依赖模型愿不愿意调工具。
- **每轮写**：`after_message_sent` 把这一轮丢进线程池，用宿主自己的模型做抽取，按 `allow()` 节流后写入。

所以第一句话就可能"她记得你"，而不是等冷启动。

## 安装

1. 把本目录改名/放到 AstrBot 的 `data/plugins/astrbot_plugin_personatrace/`。
2. 安装核心包（尚未发布 PyPI，直接从仓库装）：
   ```bash
   pip install "personatrace @ git+https://github.com/Stellfract-Frost/PersonaTrace"
   ```
   或在本仓库根目录 `pip install -e .`。
   不需要 chromadb、sentence-transformers、Redis——插件只用 `memory_v2` + `extractor`。
3. 在 WebUI 重载插件。

## 适配的平台与身份键

- 目标形态：**AstrBot 插件 + aiocqhttp（OneBot v11 反向 WebSocket）**，例如 SnowLuma / NapCat 这类实现。
  平台名固定为 `aiocqhttp`，与 OneBot v11 适配器一致。
- 身份键：`aiocqhttp:<QQ号>`。私聊与群聊里**同一个人共用一份**；`scope=per_group` 时改用
  `aiocqhttp:group:<群号>`（整群共享一份关系记忆）。
- **不依赖 FastAPI、也不依赖 MCP**：插件只 import `personatrace` 的核心模块，零第三方依赖，
  不需要起 uvicorn，也不需要配 MCP server。
- 管理员命令要求 AstrBot 配置里的 `admins_id` 含你的 QQ 号；若设置了唤醒前缀，命令需带上。

## 命令（仅管理员）

| 命令 | 作用 |
|---|---|
| `/memory <key>` | 查看某个人记住了什么（不传 key 则查自己） |
| `/del <key>` | 彻底删除某个人的记忆，回执写明删了几条 |

`key` 形如 `aiocqhttp:1001`（平台:发送者）。删除不可逆——`invalidate` 才是留痕的"纠正"。

## 配置

见 `_conf_schema.json`：注入条数/字数、粒度（`per_user` 默认 / `per_group`）、冷启动与成型后的抽取间隔、单轮写入上限。

## 已知限制

- **后端默认落 SQLite**：`<AstrBot 数据目录>/plugin_data/astrbot_plugin_personatrace/memory.db`，重启不丢；装了 `redis` 且配置可用时优先用 Redis（多进程共享）。
- **不带向量去重**：靠 `uuid5(user:track:category:label)` 主键 + hits 强化，字面不同但同义的结论仍会分两条。
- `persona` 轨的部分类别（`traits` / `style_notes` / `interpersonal` / `boundaries`）在 `CATEGORY_STABILITY` 里没有定义，会落到 60 天默认稳定度。
- 记忆里的原文证据最多保留 50 条/结论（`MAX_EVIDENCE`）。

## 为什么不走 MCP

MCP 只提供工具，工具是否被调用由模型决定；而"她记得我"需要无条件发生。MCP 仍然保留，用于"用户主动问起时深挖"和跨宿主（Claude Desktop / Cursor）。
