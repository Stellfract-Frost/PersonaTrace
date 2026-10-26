"""PersonaTrace MCP server —— 零依赖 stdio 实现（newline-delimited JSON-RPC 2.0）。

为什么自己实现而不依赖官方 SDK：
1. 插件契约第一条是「零配置可跑」——多一个必须安装的包，就多一道墙；
2. 整个协议本机要用到的只有 initialize / tools/list / tools/call / ping 四个方法，
   加起来不到 200 行，比引一个 SDK 更容易审。

对外暴露十三个工具（宿主模型自己决定什么时候调，不需要把画像全量塞进上下文）：
  memory_state     → 此刻（或某个历史时刻）它认为你是什么状态
  memory_recall    → 按查询取用：约束与身份强制入选，其余按相关度×保持率排序
  memory_observe   → 写入一条观察（证据 + 结论），幂等
  memory_correct   → 失效一条结论（不删除，保留历史）
  memory_predict   → 根据行为序列预测下一步动作
  memory_remember  → 写入已抽取的 beliefs / intentions（不调用 LLM）
  memory_context   → 结构化读取：pinned / dated / related / cross / 短期任务
  memory_agenda    → 只读短期任务议程
  memory_turn      → 只存一轮原始对话，返回 episode_id
  memory_episodes  → 按时间和字面回溯原始 turn
  memory_routine_probe    → 只读 weekly routine 的 in-turn probe 候选
  memory_routine_feedback → confirmed / denied / skipped_today / unanswered / ended
  memory_routine_label    → 显式标注 weekly routine（无动作词表）

协议层仍然零依赖；merged 工具按需加载 Pydantic + SQLite。

契约硬化（插件第三条「失败不拖垮宿主」）：
  - 所有工具调用带超时，超时返回 isError 内容而不是抛异常；
  - 任何内部异常都转成工具结果里的错误文本，宿主只会看到"这条记忆没取到"，对话不受影响。

启动：personatrace-mcp / python -m personatrace.mcp_server
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, Optional

try:  # 作为包导入：pip install 之后 / python -m personatrace.mcp_server
    from .memory_v2 import DEFAULT_DB_PATH, MemoryV2
except ImportError:  # 直接以脚本方式运行：python personatrace/mcp_server.py
    from memory_v2 import DEFAULT_DB_PATH, MemoryV2  # type: ignore[no-redef]

PROTOCOL_VERSION = "2024-11-05"
SUPPORTED_PROTOCOL_VERSIONS = frozenset({PROTOCOL_VERSION})
SERVER_INFO = {"name": "personatrace-memory", "version": "2.1.2"}
DEFAULT_TIMEOUT = 8.0

_mem: Optional[MemoryV2] = None
_merged = None


def get_merged():
    """Lazy merged facade; SQLite by default, no Redis required.

    One user id for every tool family. The old five tools used to instantiate
    ``MemoryV2()`` with the kernel default (``default_user``) while the merged
    tools read ``PERSONA_USER_ID`` (default ``default``). Both wrote to the
    same file but into different key namespaces, so ``memory_observe`` was
    invisible to ``memory_context``. ``get_mem`` now returns this store.
    """
    global _merged
    if _merged is None:
        try:
            from .merged_memory import MergedMemory
        except ImportError:
            from merged_memory import MergedMemory  # type: ignore[no-redef]
        raw_db = (os.environ.get("PERSONA_DB") or "").strip()
        # "off" is the documented test-only in-memory mode. Passing it as a
        # filename used to create a literal file named "off"; normalize it to
        # SQLite's in-memory database for the merged facade.
        if raw_db.lower() in {"off", "none", "memory", "memory://", ":memory:"}:
            db = ":memory:"
        else:
            db = raw_db or DEFAULT_DB_PATH
        user_id = (os.environ.get("PERSONA_USER_ID") or "default").strip() or "default"
        tz = os.environ.get("PERSONA_TZ") or "Asia/Shanghai"
        _merged = MergedMemory(user_id=user_id, sqlite_path=db, tz=tz)
    return _merged


def get_mem() -> MemoryV2:
    """Return the one belief kernel shared with the merged tools."""
    global _mem
    if _mem is None:
        _mem = get_merged().memory
    return _mem


def _text(payload: Any, is_error: bool = False) -> Dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2, default=str)}],
        "isError": is_error,
    }


def _guarded(fn: Callable[..., Any]) -> Callable[..., Dict[str, Any]]:
    """异常兜底：Memory 出错绝不能让宿主的对话崩。

    There is deliberately no thread-based timeout here. A Python thread cannot
    be cancelled, and ``Future.cancel()`` on a running task is a no-op; the old
    implementation returned ``memory_timeout`` while the write kept running and
    four such calls permanently filled a 4-worker pool. The real deadline is
    enforced by the isolated worker in ``_ProcessRunner``; direct handler calls
    (tests / embedding) are synchronous and have no fake deadline to advertise.
    """

    def wrapper(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return _text(fn(args))
        except Exception as exc:  # noqa: BLE001
            return _text({"error": "memory_failed", "detail": str(exc)}, True)

    return wrapper


class _ProcessRunner:
    """One long-lived worker process with a genuinely enforceable deadline.

    MCP stdio is sequential: one request is read, one response is written. A
    single worker is enough, and because the worker is a child process it can
    be killed on timeout. The next call starts a fresh worker. This removes
    both the fake cancellation and the thread-pool leak.

    The worker is a ``subprocess`` rather than ``multiprocessing``: anonymous
    OS pipes work in confined environments where multiprocessing's named pipes
    are denied, and the worker needs no pickling of live objects.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.timeout = max(0.05, float(timeout))
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    @staticmethod
    def _readline(proc: subprocess.Popen, timeout: float) -> Optional[str]:
        """Read one line without blocking the caller past ``timeout``."""
        lines: "queue.Queue[str]" = queue.Queue(maxsize=1)

        def _read() -> None:
            try:
                assert proc.stdout is not None
                lines.put(proc.stdout.readline())
            except Exception:
                lines.put("")

        threading.Thread(target=_read, name="mcp_worker_reader", daemon=True).start()
        try:
            return lines.get(timeout=timeout)
        except queue.Empty:
            return None

    def _start(self) -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "personatrace.mcp_server", "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1, env=env,
        )
        # Startup is not tool execution: give the interpreter a separate,
        # generous budget so a 0.3s test deadline does not fail on imports.
        ready = self._readline(self._proc, max(10.0, self.timeout))
        if not ready:
            self._kill()
            raise RuntimeError("memory worker did not start")
        try:
            envelope = json.loads(ready)
        except json.JSONDecodeError as exc:
            self._kill()
            raise RuntimeError(f"memory worker ready line invalid: {exc}") from exc
        if not envelope.get("ready"):
            self._kill()
            raise RuntimeError(f"memory worker failed to start: {envelope!r}")

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=2.0)
        except Exception:
            pass
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            self._kill()

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            try:
                if self._proc is None or self._proc.poll() is not None:
                    self._kill()
                    self._start()
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write(json.dumps([name, args], ensure_ascii=False, default=str) + "\n")
                self._proc.stdin.flush()
            except Exception as exc:  # noqa: BLE001
                self._kill()
                return _text({"error": "memory_worker_failed", "detail": str(exc)}, True)

            proc = self._proc
            assert proc is not None
            line = self._readline(proc, self.timeout)
            if line is None:
                self._kill()
                return _text({
                    "error": "memory_timeout",
                    "detail": f"超过 {self.timeout}s；隔离进程已终止，本次操作不会在后台继续",
                    "cancelled": True,
                }, True)

            if not line:
                self._kill()
                return _text({"error": "memory_worker_failed", "detail": "worker exited without a response"}, True)
            try:
                envelope = json.loads(line)
            except json.JSONDecodeError:
                self._kill()
                return _text({"error": "memory_worker_failed", "detail": "worker returned invalid JSON"}, True)
            if envelope.get("ok"):
                return envelope.get("result")
            return _text(envelope.get("error") or {"error": "memory_worker_failed"}, True)


def _worker_enabled() -> bool:
    """Process isolation needs a durable store; in-memory mode is test-only."""
    if os.environ.get("PERSONA_MCP_INPROCESS", "").strip().lower() in {"1", "true", "yes"}:
        return False
    target = (os.environ.get("PERSONA_DB") or "").strip().lower()
    return target not in {"off", "none", "memory", "memory://", ":memory:"}


def _worker_stdio() -> int:
    """Child entry point: newline-JSON request/response over anonymous pipes."""
    for stream in (sys.stdin, sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    real_stdout = sys.stdout
    real_stdout.write(json.dumps({"ready": True}) + "\n")
    real_stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            name, args = request
            if not isinstance(name, str) or not isinstance(args, dict):
                raise ValueError("request must be [name, args]")
        except Exception as exc:  # noqa: BLE001
            real_stdout.write(json.dumps(
                {"ok": False, "error": {"error": "bad_request", "detail": str(exc)}},
                ensure_ascii=False,
            ) + "\n")
            real_stdout.flush()
            continue

        fn = RAW_HANDLERS.get(name)
        if fn is None:
            real_stdout.write(json.dumps(
                {"ok": False, "error": {"error": "unknown_tool", "detail": name}},
                ensure_ascii=False,
            ) + "\n")
            real_stdout.flush()
            continue

        # A library print must not corrupt the JSON stream.
        sys.stdout = sys.stderr
        try:
            result = _text(fn(args))
            response = {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001
            response = {"ok": False, "error": {"error": "memory_failed", "detail": str(exc)}}
        finally:
            sys.stdout = real_stdout
        real_stdout.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
        real_stdout.flush()
    return 0


# ---------------------------------------------------------------- 工具实现

def _as_ts(value: Any, default: Optional[float] = None) -> Optional[float]:
    """把 as_of 统一成 UNIX 秒：接受数字或 ISO 字符串。

    之前 _state 自己转、_recall 直接透传，于是 recall 传 ISO 会在
    `valid_from_ts > as_of` 那里 TypeError：float 与 str 不能比较。
    """
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"as_of 不是合法时间：{value!r}") from exc
        if parsed.tzinfo is None:
            # A date without an offset must use the same zone as the store;
            # otherwise "2026-10-25" means a different day in the host's zone.
            tz_name = os.environ.get("PERSONA_TZ") or "Asia/Shanghai"
            try:
                parsed = parsed.replace(tzinfo=ZoneInfo(tz_name))
            except Exception:
                parsed = parsed.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        return parsed.timestamp()
    raise ValueError(f"as_of 类型不支持：{type(value).__name__}")


def _state(args: Dict[str, Any]) -> Any:
    import time

    as_of = _as_ts(args.get("as_of"))
    known_as_of = _as_ts(args.get("known_as_of"))
    rows = get_mem().state(
        as_of=as_of,
        include_archived=bool(args.get("include_archived")),
        include_invalidated=bool(args.get("include_invalidated")),
        expand=int(args.get("expand", 0)),
        known_as_of=known_as_of,
    )
    now = time.time()
    return {
        "as_of": as_of if as_of is not None else now,
        "known_as_of": known_as_of if known_as_of is not None else (as_of if as_of is not None else now),
        "count": len(rows),
        "beliefs": rows[: int(args.get("limit", 30))],
    }


def _recall(args: Dict[str, Any]) -> Any:
    query = str(args.get("query") or "")
    result = get_mem().recall(
        as_of=_as_ts(args.get("as_of")),
        k=int(args.get("k", 10)),
        expand=int(args.get("expand", 0)),
        query=query,
        known_as_of=_as_ts(args.get("known_as_of")),
    )
    return {
        "query": query,
        "pinned": result["pinned_count"],
        "hidden": result["hidden_count"],
        "items": result["shown"],
        "note": "pinned = 约束与身份类，永远入选；hidden = 相关但被挤出预算的条数",
    }


def _observe(args: Dict[str, Any]) -> Any:
    # Only forward fields the caller actually supplied. Omitted typed-time
    # fields preserve the old value; explicit null / "none" / "" clears it.
    typed: Dict[str, Any] = {}
    for key in ("due_at_ts", "time_kind", "time_raw"):
        if key in args:
            typed[key] = args[key]
    if "revive" in args:
        typed["revive"] = bool(args["revive"])
    belief = get_mem().observe(
        args["category"], args["label"],
        track=args.get("track", "work"),
        confidence=float(args.get("confidence", 0.5)),
        evidence_id=args.get("evidence_id"),
        dimension=args.get("dimension", "identity"),
        reasoning=args.get("reasoning", ""),
        evidence_text=args.get("evidence_text", ""),
        ts=_as_ts(args.get("now")),
        **typed,
    )
    return {
        "id": belief.id,
        "hits": belief.hits,
        "stability_days": round(belief.stability, 2),
        "retention": round(belief.retention_now(), 3),
        "effective_confidence": belief.effective_confidence(),
    }


def _correct(args: Dict[str, Any]) -> Any:
    ok = get_mem().invalidate(
        args["belief_id"],
        ts=_as_ts(args.get("ts")),
        status=args.get("status", "deprecated"),
        recorded_at=_as_ts(args.get("recorded_at")),
    )
    if not ok:
        # 之前静默返回「已失效」，模型会据此宣称纠正成功，实际什么都没发生。
        raise ValueError(f"belief not found: {args['belief_id']}")
    return {"status": "invalidated", "belief_id": args["belief_id"], "note": "不删除，时间线上仍可回溯"}


def _predict(args: Dict[str, Any]) -> Any:
    return {"predictions": get_mem().predict_next(int(args.get("window", 30)))}


def _now_arg(args: Dict[str, Any]) -> Optional[datetime]:
    value = args.get("now")
    if value is None or value == "":
        return None
    from zoneinfo import ZoneInfo

    tz_name = os.environ.get("PERSONA_TZ") or "Asia/Shanghai"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("Asia/Shanghai")
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz)
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"now 不是合法时间：{value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    return parsed


def _merged_remember(args: Dict[str, Any]) -> Any:
    return get_merged().write(
        args.get("beliefs") or [],
        args.get("intentions") or [],
        source=str(args.get("source") or "direct"),
        now=_now_arg(args),
        episode_id=str(args.get("episode_id") or ""),
    )


def _merged_turn(args: Dict[str, Any]) -> Any:
    return get_merged().record_turn(
        str(args.get("user_msg") or ""),
        str(args.get("ai_reply") or ""),
        source=str(args.get("source") or "direct"),
        session_id=str(args.get("session_id") or ""),
        now=_now_arg(args),
    )


def _merged_episodes(args: Dict[str, Any]) -> Any:
    return {
        "items": get_merged().episodes_search(
            str(args.get("query") or ""),
            after=args.get("after"),
            before=args.get("before"),
            limit=int(args.get("limit", 3)),
        )
    }


def _merged_routine_probe(args: Dict[str, Any]) -> Any:
    return {"items": get_merged().routine_probes(
        now=_now_arg(args), limit=int(args.get("limit", 3)),
    )}


def _merged_routine_feedback(args: Dict[str, Any]) -> Any:
    return get_merged().routine_feedback(
        str(args.get("probe_id") or ""),
        str(args.get("result") or ""),
        now=_now_arg(args),
        note=str(args.get("note") or ""),
    )


def _merged_routine_label(args: Dict[str, Any]) -> Any:
    return get_merged().label_routine(
        str(args.get("action") or ""),
        time=str(args.get("time") or "09:00"),
        weekdays=args.get("weekdays") or (),
        interval=int(args.get("interval", 1)),
        anchor_date=args.get("anchor_date"),
        valid_from=args.get("valid_from"),
        valid_until=args.get("valid_until"),
        exdates=args.get("exdates") or (),
        rdates=args.get("rdates") or (),
        status=str(args.get("status") or "candidate"),
        confidence=float(args.get("confidence", 0.55)),
        allow_probe=bool(args.get("allow_probe", True)),
        sensitivity=str(args.get("sensitivity") or ""),
        note=str(args.get("note") or ""),
        now=_now_arg(args),
    )


def _merged_context(args: Dict[str, Any]) -> Any:
    return get_merged().build_context(
        str(args.get("query") or ""),
        k=int(args.get("k", 4)),
        now=_now_arg(args),
    ).to_dict()


def _merged_agenda(args: Dict[str, Any]) -> Any:
    return get_merged().agenda(
        now=_now_arg(args),
        max_lines=int(args.get("max_lines", 6)),
    )


def _debug_sleep(args: Dict[str, Any]) -> Any:
    """Test-only hook for the real-deadline worker; never listed in TOOLS."""
    import time

    time.sleep(float(args.get("seconds", 5)))
    return {"slept": True}


RAW_HANDLERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "memory_state": _state,
    "memory_recall": _recall,
    "memory_observe": _observe,
    "memory_correct": _correct,
    "memory_predict": _predict,
    "memory_remember": _merged_remember,
    "memory_context": _merged_context,
    "memory_agenda": _merged_agenda,
    "memory_turn": _merged_turn,
    "memory_episodes": _merged_episodes,
    "memory_routine_probe": _merged_routine_probe,
    "memory_routine_feedback": _merged_routine_feedback,
    "memory_routine_label": _merged_routine_label,
    "__sleep__": _debug_sleep,
}

HANDLERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    name: _guarded(fn) for name, fn in RAW_HANDLERS.items() if not name.startswith("__")
}

TOOLS = [
    {
        "name": "memory_state",
        "description": "读取某人此刻的记忆状态（或 as_of 指定的历史时刻）。返回带保留率、失效日、是否需要复核的结论列表。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "as_of": {"type": ["number", "string"], "description": "UNIX 秒或 ISO 时间；不传=现在"},
                "known_as_of": {"type": ["number", "string"],
                                "description": "事务时间：当时系统知道什么。不传=与 as_of 相同（真快照）；传 now=用今天的知识回看过去"},
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
                "query": {"type": "string", "description": "用户当前的问题或话题；内核用字面 IDF 检索"},
                "as_of": {"type": ["number", "string"]},
                "known_as_of": {"type": ["number", "string"],
                                "description": "事务时间；不传=与 as_of 相同（真快照）"},
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
                "due_at_ts": {"type": ["number", "null"], "description": "UNIX 秒；传 null 清除旧截止时间，不传则保留"},
                "time_kind": {"type": "string", "enum": ["none", "due", "past", "range", "recurring"],
                              "description": "传 none 清除旧时间类型；不传则保留"},
                "time_raw": {"type": "string", "description": "时间原话；传空串清除，不传则保留"},
                "revive": {"type": "boolean", "default": True,
                           "description": "失效后再次观察到同一 label 时是否重开有效期"},
                "now": {"type": ["number", "string"],
                        "description": "写入时间（UNIX 秒或 ISO）；不传=现在"},
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
                "ts": {"type": ["number", "string"], "description": "有效时间：这条事实从何时起不再成立"},
                "recorded_at": {"type": ["number", "string"], "description": "事务时间：何时记录这次纠正；默认现在"},
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
    {
        "name": "memory_remember",
        "description": "写入已经抽取好的 beliefs / intentions。不调用 LLM；source 必须是 direct/bridge/console。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "beliefs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "label": {"type": "string", "minLength": 2},
                            "confidence": {"type": "number"},
                            "reasoning": {"type": "string"},
                            "evidence_text": {"type": "string"},
                            "due": {"type": "string"},
                        },
                        "required": ["category", "label"],
                    },
                },
                "intentions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "minLength": 2},
                            "detail": {"type": "string"},
                            "due": {"type": "string"},
                            "ttl_hours": {"type": "number"},
                            "importance": {"type": "number"},
                            "confidence": {"type": "number"},
                            "needs_confirmation": {"type": "boolean"},
                            "recurrence": {"type": "string"},
                        },
                        "required": ["action"],
                    },
                },
                "source": {"type": "string", "enum": ["direct", "bridge", "console"], "default": "direct"},
                "episode_id": {"type": "string", "description": "可选：把写入挂到 memory_turn 返回的 episode"},
                "now": {"type": ["number", "string"], "description": "UNIX 秒或 ISO；不传=现在"},
            },
        },
    },
    {
        "name": "memory_context",
        "description": "按查询返回结构化上下文：pinned / dated / related / cross / 短期任务。不拼 prompt，不改变 anti-nag 状态。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 4},
                "now": {"type": ["number", "string"]},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_agenda",
        "description": "返回短期任务议程（due / importance / bucket / status）及隐藏条数，不改状态。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_lines": {"type": "integer", "default": 6},
                "now": {"type": ["number", "string"]},
            },
        },
    },
    {
        "name": "memory_turn",
        "description": "只存一轮原始对话，不做抽取。返回 episode_id，供 memory_remember / memory_episodes 链接。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_msg": {"type": "string", "minLength": 1},
                "ai_reply": {"type": "string", "default": ""},
                "session_id": {"type": "string", "default": ""},
                "source": {"type": "string", "enum": ["direct", "bridge", "console"], "default": "direct"},
                "now": {"type": ["number", "string"]},
            },
            "required": ["user_msg"],
        },
    },
    {
        "name": "memory_episodes",
        "description": "回溯检索原始 turn：按字面 + 时间范围，返回 user/assistant 原文与关联的 belief/意向 id。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "default": ""},
                "after": {"type": ["number", "string"]},
                "before": {"type": ["number", "string"]},
                "limit": {"type": "integer", "default": 3},
            },
        },
    },
    {
        "name": "memory_routine_probe",
        "description": "只读返回当前时间窗口内的 routine probe 候选；不发送、不标记已问。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "now": {"type": ["number", "string"]},
                "limit": {"type": "integer", "default": 3},
            },
        },
    },
    {
        "name": "memory_routine_feedback",
        "description": "回传 probe 结果：confirmed / denied / skipped_today / unanswered / ended。沉默不是 confirmed。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "probe_id": {"type": "string"},
                "result": {"type": "string", "enum": ["confirmed", "denied", "skipped_today", "unanswered", "ended"]},
                "note": {"type": "string", "default": ""},
                "now": {"type": ["number", "string"]},
            },
            "required": ["probe_id", "result"],
        },
    },
    {
        "name": "memory_routine_label",
        "description": "显式标注 weekly routine：自由 action + weekday/time + 单双周/学期/例外。无动作词表。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "minLength": 1},
                "time": {"type": "string", "default": "09:00"},
                "weekdays": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 7}},
                "interval": {"type": "integer", "enum": [1, 2], "default": 1},
                "anchor_date": {"type": ["number", "string"]},
                "valid_from": {"type": ["number", "string"]},
                "valid_until": {"type": ["number", "string"]},
                "exdates": {"type": "array", "items": {"type": ["number", "string"]}},
                "rdates": {"type": "array", "items": {"type": ["number", "string"]}},
                "status": {"type": "string", "enum": ["candidate", "confirmed", "trusted", "paused", "ended"], "default": "candidate"},
                "confidence": {"type": "number", "default": 0.55},
                "allow_probe": {"type": "boolean", "default": True},
                "sensitivity": {"type": "string", "default": ""},
                "note": {"type": "string", "default": ""},
                "now": {"type": ["number", "string"]},
            },
            "required": ["action"],
        },
    },
]


# ---------------------------------------------------------------- 协议

def _error(msg_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _handle_single(msg: Any, runner: Optional[_ProcessRunner] = None) -> Optional[Dict[str, Any]]:
    """One JSON-RPC message in, one response out (notification -> None).

    A non-object message is a JSON-RPC error, not an ``AttributeError`` that
    escapes and kills the stdio loop. Batch arrays are handled by
    ``_handle_batch`` before this function is called.
    """
    if not isinstance(msg, dict):
        return _error(None, -32600, f"invalid request: expected object, got {type(msg).__name__}")

    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params") or {}
    if not isinstance(method, str):
        return _error(msg_id, -32600, "invalid request: method must be a string")
    if not isinstance(params, dict):
        return _error(msg_id, -32602, "invalid params: expected object")

    if msg_id is None and method.startswith("notifications/"):
        return None

    if method == "initialize":
        requested = params.get("protocolVersion")
        version = (
            requested
            if isinstance(requested, str) and requested in SUPPORTED_PROTOCOL_VERSIONS
            else PROTOCOL_VERSION
        )
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {
                "protocolVersion": version,
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
        if not isinstance(args, dict):
            return _error(msg_id, -32602, "invalid params: arguments must be an object")
        if name not in HANDLERS:
            return _error(msg_id, -32602, f"unknown tool: {name}")
        if runner is not None:
            result = runner.call(str(name), args)
        else:
            result = HANDLERS[name](args)
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    return _error(msg_id, -32601, f"method not found: {method}")


def handle(msg: Any, runner: Optional[_ProcessRunner] = None) -> Optional[Dict[str, Any]]:
    """Handle one message. Kept as a pure-ish entry point for tests/embedding."""
    return _handle_single(msg, runner)


def _safe_handle(msg: Any, runner: Optional[_ProcessRunner] = None) -> Optional[Dict[str, Any]]:
    try:
        return _handle_single(msg, runner)
    except Exception:  # noqa: BLE001 - protocol last resort
        # Keep file paths and internals off the JSON-RPC wire; stderr is the
        # host's log, stdout is the protocol stream.
        traceback.print_exc(file=sys.stderr, limit=8)
        sys.stderr.flush()
        msg_id = msg.get("id") if isinstance(msg, dict) else None
        return _error(msg_id, -32603, "internal error; see server stderr")


def _handle_batch(messages: Any, runner: Optional[_ProcessRunner] = None) -> Any:
    """JSON-RPC batch support. A batch must not crash the stdio loop.

    MCP transports normally send single objects, but any host/proxy may send a
    JSON array. The old loop passed the list to ``msg.get`` and then crashed
    while building the error response, taking the whole process down.
    """
    if not isinstance(messages, list):
        return _error(None, -32600, "invalid request: expected object or batch array")
    if not messages:
        return _error(None, -32600, "invalid request: empty batch")
    responses = [resp for resp in (_safe_handle(m, runner) for m in messages) if resp is not None]
    return responses or None


def main() -> int:
    if "--worker" in sys.argv[1:]:
        return _worker_stdio()

    for stream in (sys.stdin, sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    runner: Optional[_ProcessRunner] = None
    if _worker_enabled():
        runner = _ProcessRunner()
    else:
        sys.stderr.write(
            "[personatrace] in-process MCP mode: durable timeouts are disabled. "
            "Set PERSONA_DB to a file to enable the isolated worker.\n"
        )
        sys.stderr.flush()

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                resp = _error(None, -32700, "parse error")
            else:
                if isinstance(msg, list):
                    resp = _handle_batch(msg, runner)
                else:
                    resp = _safe_handle(msg, runner)

            if resp is not None:
                try:
                    payload = json.dumps(resp, ensure_ascii=False, default=str)
                except (TypeError, ValueError) as exc:
                    payload = json.dumps(_error(None, -32603, f"response encoding failed: {exc}"))
                sys.stdout.write(payload + "\n")
                sys.stdout.flush()
    finally:
        if runner is not None:
            runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
