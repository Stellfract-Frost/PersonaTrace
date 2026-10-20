"""AstrBot 适配层：把宿主的概念翻译成 PersonaTrace 的概念。

这一层刻意不 import astrbot —— 它只处理纯数据，能在没有 AstrBot 的环境里单测。
主机的身份、注入时机、渲染形态全部在这里收口，核心内核不感知宿主。
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from personatrace.llm_client import extract_json

_KEY_RE = re.compile(r"^[A-Za-z0-9_.:@\-]{1,64}$")


# ---------------------------------------------------------------- 身份


def identity_key(platform: str, sender_id: str) -> str:
    """把（平台, 发送者）压成一个稳定的记忆键。

    例如 aiocqhttp + 12345 -> "aiocqhttp:12345"。
    用平台前缀而不是裸 QQ 号：Telegram 用户没有 QQ 号，裸号会全塌成一个用户。
    """
    platform = (platform or "unknown").strip().lower()
    sender_id = str(sender_id or "").strip()
    if not sender_id:
        raise ValueError("sender_id is empty; refusing to build a shared memory key")
    key = f"{platform}:{sender_id}"
    if not _KEY_RE.match(key):
        # 平台 id 里出现奇怪字符时兜底：只保留安全字符，避免污染命名空间
        key = re.sub(r"[^A-Za-z0-9_.:@\-]", "_", key)[:64]
    return key


def key_from_event_parts(platform: str, sender_id: str, group_id: str = "", scope: str = "per_user") -> str:
    """scope=per_user（默认，一人一份）或 per_group（整群共享一份关系记忆）。"""
    if scope == "per_group" and group_id:
        return identity_key(platform, f"group:{group_id}")
    return identity_key(platform, sender_id)


def normalize_key(raw: str) -> str:
    """管理员输入 /memory <key> 时的校验（防止空 key 或脏 key 打到别人库上）。"""
    key = (raw or "").strip()
    if not key:
        raise ValueError("key 为空")
    if not _KEY_RE.match(key):
        raise ValueError(f"key 不合法：{key!r}（只允许字母数字和 _.:@-）")
    return key


# ---------------------------------------------------------------- 渲染


def _due_text(due_in: Optional[float]) -> str:
    if due_in is None:
        return ""
    if due_in < 0:
        return "已过期"
    if due_in < 1:
        return "今天到期"
    return f"还剩 {int(due_in)} 天"


def _staleness_text(staleness: Optional[float]) -> str:
    if staleness is None or staleness < 1:
        return ""
    return f"{int(staleness * 30)} 天没再提"


def _rank(row: Dict[str, Any]) -> tuple:
    """先按紧迫度，再按有效置信度。故意不按字母序，避免每次都注入同几条。"""
    due = row.get("due_in")
    urgency = 0 if due is not None and due <= 7 else (1 if row.get("needs_review") else 2)
    return (urgency, -(row.get("effective_confidence") or 0.0))


def pick_lines(rows: Iterable[Dict[str, Any]], max_items: int = 3) -> List[str]:
    items: List[str] = []
    for row in sorted(rows, key=_rank):
        label = (row.get("label") or "").strip()
        if not label:
            continue
        suffix = "，".join(x for x in (_due_text(row.get("due_in")), _staleness_text(row.get("staleness"))) if x)
        items.append(f"{label}（{suffix}）" if suffix else label)
        if len(items) >= max_items:
            break
    return items


def render_memory(
    rows: Iterable[Dict[str, Any]],
    max_items: int = 3,
    max_chars: int = 240,
    header: str = "（关于这个人的记忆，供参考，别逐条复述）",
) -> str:
    """把记忆压成一段可以直接塞进对话的短文本；没有可用的就返回空串。"""
    lines = pick_lines(rows, max_items=max_items)
    if not lines:
        return ""
    body = "；".join(lines)
    text = f"{header}{body}"
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text


# ---------------------------------------------------------------- 抽取（复用宿主模型）


class HostLLM:
    """把 AstrBot 的 provider 伪装成 PersonaTrace 的 LLMClient（鸭子类型）。

    extractor 是同步的、跑在线程池里；provider.text_chat 是 async。
    所以这里用 run_coroutine_threadsafe 把调用丢回主事件循环，线程安全的桥。
    """

    def __init__(self, provider: Any, loop: asyncio.AbstractEventLoop, timeout: float = 90.0) -> None:
        self.provider = provider
        self.loop = loop
        self.timeout = timeout

    async def _call(self, prompt: str, model: Optional[str]) -> Any:
        return await self.provider.text_chat(prompt=prompt, contexts=[], system_prompt="", model=model)

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 4096, model: Optional[str] = None) -> str:
        prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
        fut = asyncio.run_coroutine_threadsafe(self._call(prompt, model), self.loop)
        resp = fut.result(timeout=self.timeout)
        return getattr(resp, "completion_text", "") or ""

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 8192,
        model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return extract_json(self.chat(messages, max_tokens=max_tokens, model=model))


class ExtractBridge:
    """一轮对话结束后，把抽取结果写进 MemoryV2。

    注意：这里刻意不走 persona_manager —— 那条链路写的是另一套存储（ProfileData + chromadb），
    和 MemoryV2 从来没连通。适配层只认 MemoryV2，插件因此不需要 chromadb / sentence-transformers。
    """

    def __init__(self, mem: Any, extractor: Any, max_per_turn: int = 8) -> None:
        self.mem = mem
        self.extractor = extractor
        self.max_per_turn = max_per_turn

    def ingest(self, user_msg: str, ai_reply: str, history: str = "") -> int:
        result = self.extractor.extract_from_conversation(
            user_msg=user_msg,
            ai_reply=ai_reply,
            history_context=history,
            now=datetime.now(),
            vector_store=None,      # 不带向量库：靠 uuid5 主键 + hits 强化做去重
            user_id=self.mem.user_id,
        )
        if not result:
            return 0
        written = 0
        for track, block in (
            ("work", result.get("work_updates") or {}),
            ("persona", result.get("persona_updates") or {}),
        ):
            if not isinstance(block, dict):
                continue
            for category, items in block.items():
                for item in items or []:
                    if written >= self.max_per_turn:
                        return written
                    if isinstance(item, dict):
                        label = str(item.get("label") or "").strip()
                        confidence = float(item.get("confidence") or 0.5)
                        reasoning = str(item.get("reasoning") or "")
                    else:
                        label, confidence, reasoning = str(item).strip(), 0.5, ""
                    if len(label) < 2:
                        continue
                    self.mem.observe(
                        category,
                        label,
                        track=track,
                        confidence=max(0.0, min(1.0, confidence)),
                        reasoning=reasoning,
                        evidence_text=(user_msg or "")[:1000],
                    )
                    written += 1
        return written
