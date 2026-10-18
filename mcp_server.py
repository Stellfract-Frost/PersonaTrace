"""PersonaTrace MCP server —— 零依赖 stdio 实现（newline-delimited JSON-RPC 2.0）。

为什么自己实现而不依赖官方 SDK：
1. 插件契约第一条是「零配置可跑」——多一个必须安装的包，就多一道墙；
2. 整个协议本机要用到的只有 initialize / tools/list / tools/call / ping 四个方法，
   加起来不到 200 行，比引一个 SDK 更容易审。

对外暴露四个工具（宿主模型自己决定什么时候调，不需要把画像全量塞进上下文）：
  memory_state     → 此刻（或某个历史时刻）它认为你是什么状态
  memory_recall    → 按查询取用：约束与身份强制入选，其余按相关度×保持率排序
  memory_observe   → 写入一条观察（证据 + 结论），幂等
  memory_correct   → 失效一条结论（不删除，保留历史）
  memory_predict   → 根据行为序列预测下一步动作

契约硬化（插件第三条「失败不拖垮宿主」）：
  - 所有工具调用带超时，超时返回 isError 内容而不是抛异常；
  - 任何内部异常都转成工具结果里的错误文本，宿主只会看到"这条记忆没取到"，对话不受影响。

启动：python mcp_server.py
"""

from __future__ import annotations

import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable, Dict, Optional

from memory_v2 import MemoryV2

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "personatrace-memory", "version": "1.0.0"}
DEFAULT_TIMEOUT = 8.0

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mcp_tool")
_mem: Optional[MemoryV2] = None


def get_mem() -> MemoryV2:
    global _mem
    if _mem is None:
        _mem = MemoryV2()
    return _mem


def _text(payload: Any, is_error: bool = False) -> Dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2, default=str)}],
        "isError": is_error,
    }


def _guarded(fn: Callable[..., Any], timeout: float = DEFAULT_TIMEOUT) -> Callable[..., Dict[str, Any]]:
    """超时 + 异常兜底：Memory 出错绝不能让宿主的对话崩。"""

    def wrapper(args: Dict[str, Any]) -> Dict[str, Any]:
        future = _executor.submit(fn, args)
        try:
            return _text(future.result(timeout=timeout))
        except FutureTimeout:
            future.cancel()
            return _text({"error": "memory_timeout", "detail": f"超过 {timeout}s，已放弃本次记忆操作"}, True)
        except Exception as exc:  # noqa: BLE001
            return _text({"error": "memory_failed", "detail": str(exc)}, True)

    return wrapper


# ---------------------------------------------------------------- 工具实现

def _state(args: Dict[str, Any]) -> Any:
    import time

    as_of = args.get("as_of")
    if isinstance(as_of, str):  # 允许 ISO 字符串，宿主更容易传
        from datetime import datetime
        as_of = datetime.fromisoformat(as_of).timestamp()
    rows = get_mem().state(
        as_of=float(as_of) if as_of else None,
        include_archived=bool(args.get("include_archived")),
        include_invalidated=bool(args.get("include_invalidated")),
        expand=int(args.get("expand", 0)),
    )
    return {
        "as_of": as_of or time.time(),
        "count": len(rows),
        "beliefs": rows[: int(args.get("limit", 30))],
    }


def _recall(args: Dict[str, Any]) -> Any:
    query = args.get("query", "")
    result = get_mem().recall(as_of=args.get("as_of"), k=int(args.get("k", 10)),
                              expand=int(args.get("expand", 0)))
    # 无向量时按标签字面命中做一层轻量排序，保证工具在最小依赖下也有意义
    if query:
        ql = query.lower()
        shown = sorted(result["shown"], key=lambda x: (ql not in x["label"].lower(), -x["effective_confidence"]))
    else:
        shown = result["shown"]
    return {
        "query": query,
        "pinned": result["pinned_count"],
        "hidden": result["hidden_count"],
        "items": shown,
        "note": "pinned = 约束与身份类，永远入选；hidden = 相关但被挤出预算的条数",
    }


def _observe(args: Dict[str, Any]) -> Any:
    belief = get_mem().observe(
        args["category"], args["label"],
        track=args.get("track", "work"),
        confidence=float(args.get("confidence", 0.5)),
        evidence_id=args.get("evidence_id"),
        dimension=args.get("dimension", "identity"),
        reasoning=args.get("reasoning", ""),
        evidence_text=args.get("evidence_text", ""),
        due_at_ts=args.get("due_at_ts"),
    )
    return {
        "id": belief.id,
        "hits": belief.hits,
        "stability_days": round(belief.stability, 2),
        "retention": round(belief.retention_now(), 3),
        "effective_confidence": belief.effective_confidence(),
    }


def _correct(args: Dict[str, Any]) -> Any:
    get_mem().invalidate(args["belief_id"], status=args.get("status", "deprecated"))
    return {"status": "invalidated", "belief_id": args["belief_id"], "note": "不删除，时间线上仍可回溯"}


def _predict(args: Dict[str, Any]) -> Any:
    return {"predictions": get_mem().predict_next(int(args.get("window", 30)))}


HANDLERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "memory_state": _guarded(_state),
    "memory_recall": _guarded(_recall),
    "memory_observe": _guarded(_observe),
    "memory_correct": _guarded(_correct),
    "memory_predict": _guarded(_predict),
}

TOOLS = [
    {
        "name": "memory_state",
        "description": "读取某人此刻的记忆状态（或 as_of 指定的历史时刻）。返回带保留率、失效日、是否需要复核的结论列表。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "as_of": {"type": ["number", "string"], "description": "UNIX 秒或 ISO 时间；不传=现在"},
                "include_archived": {"type": "boolean", "description": "是否包含已归档（保留率过低）的条目"},
                "include_invalidated": {"type": "boolean", "description": "是否包含已失效（被纠正/取代）的条目，调试用"},
                "limit": {"type": "integer", "default": 30},
                "expand": {"type": "integer", "minimum": 0, "maximum": 5, "default": 0,
                           "description": "回表深度：0=只返回标签层；>0=附带 reasoning 与 N 条原文片段"},
            },
        },
    },
    {
        "name": "memory_recall",
        "description": "按查询取用记忆：约束与身份类强制入选，其余按相关度×保持率排序，返回被隐藏的条数。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户当前的问题或话题"},
                "as_of": {"type": ["number", "string"]},
                "k": {"type": "integer", "default": 10},
                "expand": {"type": "integer", "minimum": 0, "maximum": 5, "default": 0,
                           "description": "回表深度：命中后才读原文，0 表示不做深浅读"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_observe",
        "description": "写入一条关于用户的观察。相同 label 重复写入会累计 hits 并延长稳定度（幂等，不产生副本）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "examples": ["goals", "likes", "constraints", "identity", "state"]},
                "label": {"type": "string", "minLength": 2},
                "track": {"type": "string", "enum": ["work", "persona", "cross"], "default": "work"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                "dimension": {"type": "string", "enum": ["identity", "need", "predict"], "default": "identity"},
                "evidence_id": {"type": "string"},
                "reasoning": {"type": "string", "description": "为什么存这条：深读时作为语境带回，避免标签扁平化"},
                "evidence_text": {"type": "string", "description": "原始语句片段，落 L0，检索打分不使用"},
                "due_at_ts": {"type": "number", "description": "UNIX 秒；用于'还有几天到期'"},
            },
            "required": ["category", "label"],
        },
    },
    {
        "name": "memory_correct",
        "description": "把一条结论标记为失效（不删除，保留时间线）。用户说'你记错了'时调用。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "belief_id": {"type": "string"},
                "status": {"type": "string", "enum": ["deprecated", "contested"], "default": "deprecated"},
            },
            "required": ["belief_id"],
        },
    },
    {
        "name": "memory_predict",
        "description": "根据最近行为序列预测下一步动作（bigram 转移），返回候选与支持度。",
        "inputSchema": {
            "type": "object",
            "properties": {"window": {"type": "integer", "default": 30}},
        },
    },
]


# ---------------------------------------------------------------- 协议

def handle(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """纯函数：一条 JSON-RPC 请求进，一条响应出（通知返回 None）。"""
    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if msg_id is None and method.startswith("notifications/"):
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        handler = HANDLERS.get(name)
        if handler is None:
            return {"jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": -32602, "message": f"unknown tool: {name}"}}
        return {"jsonrpc": "2.0", "id": msg_id, "result": handler(args)}

    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": -32601, "message": f"method not found: {method}"}}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                         "error": {"code": -32700, "message": "parse error"}}) + "\n")
            sys.stdout.flush()
            continue

        try:
            resp = handle(msg)
        except Exception:  # noqa: BLE001 - 协议层最后一道兜底
            resp = {"jsonrpc": "2.0", "id": msg.get("id"),
                    "error": {"code": -32603, "message": traceback.format_exc(limit=2)}}
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
