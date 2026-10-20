"""PersonaTrace × AstrBot：把时序记忆接进宿主的两条腿（写 + 读）。

设计原则（对应 README 的三层结构）：
- 核心不动：memory_v2 一行不改，宿主相关全部收在 adapter.py。
- 读在主路径：on_llm_request 往 extra_user_content_parts 塞一段短记忆，不依赖模型调工具。
- 写不在关键路径：after_message_sent 丢线程池，按 allow() 节流。
- 失败不拖垮宿主：任何异常只记日志，绝不让聊天中断。
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from astrbot.api import star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest

try:  # 核心包没装时不要让整个插件加载失败，而是给它一个能说清楚的降级态
    from . import adapter

    _IMPORT_ERROR: Optional[BaseException] = None
except Exception as _exc:  # pragma: no cover - 取决于部署环境
    adapter = None  # type: ignore[assignment]
    _IMPORT_ERROR = _exc

try:  # AstrBot 较新版本才有 TextPart；老版本退回 dict
    from astrbot.core.agent.message import TextPart

    def _text_part(text: str):
        return TextPart(text=text)

except Exception:  # pragma: no cover - 取决于宿主版本

    def _text_part(text: str):
        return {"type": "text", "text": text}


try:
    from astrbot.core import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("personatrace")


PLUGIN_NAME = "astrbot_plugin_personatrace"
_INSTALL_HINT = (
    "记忆后端未安装：pip install \"personatrace @ git+https://github.com/Stellfract-Frost/PersonaTrace\""
)

DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "inject_max_items": 3,
    "inject_max_chars": 240,
    "scope": "per_user",            # per_user | per_group
    "window_cold_sec": 120,         # 画像未成型时的抽取间隔
    "window_warm_sec": 600,         # 成型后的抽取间隔
    "cold_belief_threshold": 10,    # 少于这么多条算"没成型"
    "max_per_turn": 8,              # 单轮最多写入几条（防止模型一次吐太多）
}


class Main(star.Star):
    def __init__(self, context: star.Context, config: Optional[dict] = None) -> None:
        self.context = context
        self.cfg = {**DEFAULTS, **(config or {})}
        self._mems: Dict[str, Any] = {}
        self._extractors: Dict[str, Any] = {}
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pt-ingest")
        self._lock = threading.Lock()          # _mems / _extractors 会被事件循环和线程池同时访问
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._backend = None
        self._data_dir = self._resolve_data_dir()

    # ------------------------------------------------------------ 基础设施

    def _resolve_data_dir(self):
        try:
            from astrbot.core.star.star_tools import StarTools

            return StarTools.get_data_dir(PLUGIN_NAME)
        except Exception:
            import pathlib

            return pathlib.Path.cwd() / "data"

    def _shared_backend(self):
        """后端只建一次：SQLite 落在 AstrBot 的插件数据目录，重启不丢。"""
        if self._backend is None:
            from personatrace.memory_v2 import SQLiteBackend, build_backend

            db_path = str(self._data_dir / "memory.db")
            try:
                self._backend = SQLiteBackend(db_path)
                logger.info(f"[personatrace] backend = sqlite ({db_path})")
            except Exception as exc:
                self._backend, mode = build_backend()
                logger.warning(f"[personatrace] sqlite 不可用（{exc}），退回 {mode}")
        return self._backend

    def _mem(self, key: str):
        from personatrace.memory_v2 import MemoryV2

        with self._lock:
            mem = self._mems.get(key)
            if mem is None:
                mem = MemoryV2(backend=self._shared_backend(), user_id=adapter.normalize_key(key))
                self._mems[key] = mem
            return mem

    def _identity(self, event: AstrMessageEvent) -> str:
        platform = ""
        sender_id = ""
        group_id = ""
        for name in ("get_platform_name", "get_platform_id"):
            fn = getattr(event, name, None)
            if callable(fn):
                try:
                    platform = str(fn() or "")
                    if platform:
                        break
                except Exception:
                    pass
        for name in ("get_sender_id",):
            fn = getattr(event, name, None)
            if callable(fn):
                try:
                    sender_id = str(fn() or "")
                except Exception:
                    pass
        for name in ("get_group_id",):
            fn = getattr(event, name, None)
            if callable(fn):
                try:
                    group_id = str(fn() or "")
                except Exception:
                    pass
        if not platform:
            umo = str(getattr(event, "unified_msg_origin", "") or "")
            platform = umo.split(":")[0] if umo else "unknown"
        return adapter.key_from_event_parts(
            platform, sender_id, group_id, scope=str(self.cfg.get("scope", "per_user"))
        )

    # ------------------------------------------------------------ 读：每轮注入

    @filter.on_llm_request()
    async def inject_memory(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        if not self.cfg.get("enabled", True) or _IMPORT_ERROR is not None:
            return
        if event.get_extra("_pt_injected"):
            return                              # agent 回路会多次请求模型，每轮只注入一次
        try:
            self._loop = asyncio.get_running_loop()
            mem = self._mem(self._identity(event))
            rows = mem.state()
            text = adapter.render_memory(
                rows,
                max_items=int(self.cfg.get("inject_max_items", 3)),
                max_chars=int(self.cfg.get("inject_max_chars", 240)),
            )
            if not text:
                return                      # 冷启动：库里没东西，什么都不加
            req.extra_user_content_parts = list(req.extra_user_content_parts or [])
            req.extra_user_content_parts.append(_text_part(text))
            event.set_extra("_pt_injected", True)
        except Exception as exc:            # 记忆坏了不能影响聊天
            logger.warning(f"[personatrace] inject failed: {exc}")

    # ------------------------------------------------------------ 写：回复后异步抽取

    @filter.on_llm_response()
    async def capture_reply(self, event: AstrMessageEvent, response: LLMResponse) -> None:
        try:
            self._loop = asyncio.get_running_loop()
            event.set_extra("_pt_reply", getattr(response, "completion_text", "") or "")
        except Exception:
            pass

    @filter.after_message_sent()
    async def remember(self, event: AstrMessageEvent) -> None:
        if not self.cfg.get("enabled", True) or _IMPORT_ERROR is not None:
            return
        try:
            if self._loop is None:
                self._loop = asyncio.get_running_loop()
            reply = str(event.get_extra("_pt_reply") or "").strip()
            user_msg = str(getattr(event, "message_str", "") or "").strip()
            if not user_msg:
                return
            if user_msg.startswith(("/", "／")):     # 命令不记
                return
            key = self._identity(event)
            mem = self._mem(key)
            cold = mem.stats().get("beliefs", 0) < int(self.cfg.get("cold_belief_threshold", 10))
            window = int(self.cfg.get("window_cold_sec", 120) if cold else self.cfg.get("window_warm_sec", 600))
            if not mem.allow("extract", window):
                return                              # 节流：别每条消息都跑一次抽取
            # 用会话自己的 provider：多配置文件/多模型时，umo=None 会挑到别的模型
            provider = await self.context.get_using_provider_async(event.unified_msg_origin)
            if provider is None:
                return
            self._pool.submit(self._ingest_blocking, key, user_msg, reply, provider)
        except Exception as exc:
            logger.warning(f"[personatrace] schedule ingest failed: {exc}")

    def _ingest_blocking(self, key: str, user_msg: str, reply: str, provider: Any = None) -> None:
        """线程池里跑：同步抽取 + 写 MemoryV2。所有异常就地吞掉。"""
        try:
            mem = self._mem(key)
            with self._lock:
                cached = self._extractors.get(key)
            if cached is not None and cached[0] is provider:
                extractor = cached[1]
            else:
                if provider is None or self._loop is None:
                    return
                from personatrace.config import PersonaConfig
                from personatrace.extractor import PersonaExtractor

                llm = adapter.HostLLM(provider, self._loop)
                extractor = PersonaExtractor(PersonaConfig(), llm)
                with self._lock:
                    self._extractors[key] = (provider, extractor)
            bridge = adapter.ExtractBridge(mem, extractor, max_per_turn=int(self.cfg.get("max_per_turn", 8)))
            written = bridge.ingest(user_msg, reply)
            if written:
                logger.info(f"[personatrace] {key}: +{written}")
        except Exception as exc:
            logger.warning(f"[personatrace] ingest failed for {key}: {exc}")

    # ------------------------------------------------------------ 管理命令（仅管理员）

    @filter.command("memory")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def show_memory(self, event: AstrMessageEvent):
        """查看某个人记住了什么：/memory <key>"""
        if _IMPORT_ERROR is not None:
            yield event.plain_result(_INSTALL_HINT)
            return
        try:
            arg = self._arg(event)
            key = adapter.normalize_key(arg) if arg else self._identity(event)
        except ValueError as exc:
            yield event.plain_result(f"key 不合法：{exc}")
            return
        try:
            mem = self._mem(key)
            rows = mem.state()
            stats = mem.stats()
            if not rows:
                yield event.plain_result(f"{key}：0 条记忆（后端 {stats.get('backend')}）")
                return
            lines = adapter.pick_lines(rows, max_items=8)
            body = "\n".join(f"- {x}" for x in lines)
            more = f"\n……共 {len(rows)} 条" if len(rows) > len(lines) else f"\n共 {len(rows)} 条"
            yield event.plain_result(f"{key}：\n{body}{more}\n后端 {stats.get('backend')}")
        except Exception as exc:
            yield event.plain_result(f"读取失败：{exc}")

    @filter.command("del")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def delete_memory(self, event: AstrMessageEvent):
        """删除某个人的全部记忆（不可逆）：/del <key>"""
        if _IMPORT_ERROR is not None:
            yield event.plain_result(_INSTALL_HINT)
            return
        arg = self._arg(event)
        if not arg:
            yield event.plain_result("用法：/del <key>")
            return
        try:
            key = adapter.normalize_key(arg)
            removed = self._mem(key).purge()
        except ValueError as exc:
            yield event.plain_result(f"key 不合法：{exc}")
            return
        except Exception as exc:
            yield event.plain_result(f"删除失败：{exc}")
            return
        with self._lock:
            self._mems.pop(key, None)
            self._extractors.pop(key, None)
        yield event.plain_result(f"已删除 {key} 的 {removed} 条记忆（不可恢复）")

    @staticmethod
    def _arg(event: AstrMessageEvent) -> str:
        text = str(getattr(event, "message_str", "") or "").strip()
        parts = text.replace("／", "/").split(maxsplit=1)
        return parts[1].strip() if len(parts) == 2 else ""

    # ------------------------------------------------------------ 卸载

    async def terminate(self) -> None:
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:      # Python 3.8 兼容
            self._pool.shutdown(wait=False)
        logger.info("[personatrace] plugin unloaded")
