"""契约测试：用假 astrbot 模块跑插件骨架，不需要装 AstrBot。

验证四件事：
1. on_llm_request 只在有记忆时注入，且注入体不超过预算；
2. 两个人（不同 sender_id）互不可见；
3. /memory /del 的行为（含删除后清零）；
4. 注入顺序：快到期的排在前面。
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))      # 让 `astrbot` 这个包名可用


def _install_fake_astrbot() -> None:
    def passthrough(*a, **kw):
        def deco(fn):
            return fn

        return deco

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api_star = types.ModuleType("astrbot.api.star")
    api_event = types.ModuleType("astrbot.api.event")
    api_provider = types.ModuleType("astrbot.api.provider")
    core = types.ModuleType("astrbot.core")
    core_star_tools = types.ModuleType("astrbot.core.star.star_tools")

    class Star:
        pass

    class Context:
        pass

    class AstrMessageEvent:
        pass

    api_star.Star = Star
    api_star.Context = Context
    api_event.AstrMessageEvent = AstrMessageEvent

    flt = types.SimpleNamespace(
        command=passthrough,
        on_llm_request=passthrough,
        on_llm_response=passthrough,
        after_message_sent=passthrough,
        permission_type=passthrough,
        PermissionType=types.SimpleNamespace(ADMIN="admin", MEMBER="member"),
    )
    api_event.filter = flt

    class LLMResponse:
        def __init__(self, completion_text: str = "") -> None:
            self.completion_text = completion_text

    class ProviderRequest:
        def __init__(self) -> None:
            self.system_prompt = ""
            self.extra_user_content_parts = []

    api_provider.LLMResponse = LLMResponse
    api_provider.ProviderRequest = ProviderRequest
    core.logger = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)

    core_star_tools.StarTools = types.SimpleNamespace(get_data_dir=lambda name=None: HERE / "_data")

    api.star = api_star
    api.event = api_event
    api.provider = api_provider
    astrbot.api = api
    astrbot.core = core
    core.star = types.ModuleType("astrbot.core.star")
    core.star.star_tools = core_star_tools

    for name, mod in {
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.star": api_star,
        "astrbot.api.event": api_event,
        "astrbot.api.provider": api_provider,
        "astrbot.core": core,
        "astrbot.core.star": core.star,
        "astrbot.core.star.star_tools": core_star_tools,
    }.items():
        sys.modules[name] = mod


_install_fake_astrbot()

import astrbot_plugin_personatrace.main as plugin_main          # noqa: E402
from astrbot_plugin_personatrace import adapter                 # noqa: E402


class FakeEvent:
    def __init__(self, sender: str, text: str = "", platform: str = "aiocqhttp", group: str = "") -> None:
        self._sender = sender
        self._platform = platform
        self._group = group
        self.message_str = text
        self.unified_msg_origin = f"{platform}:FriendMessage:{sender}"
        self._extra = {}

    def get_sender_id(self):
        return self._sender

    def get_platform_name(self):
        return self._platform

    def get_group_id(self):
        return self._group

    def set_extra(self, k, v):
        self._extra[k] = v

    def get_extra(self, k):
        return self._extra.get(k)

    def plain_result(self, text):
        return text


def req():
    return plugin_main.ProviderRequest()


def make_plugin():
    """契约测试不落盘：注入内存后端，持久化由 test_persistence.py 单独覆盖。"""
    from personatrace.memory_v2 import MemoryBackend

    ctx = types.SimpleNamespace(get_using_provider=lambda umo=None: None)
    plugin = plugin_main.Main(context=ctx, config={"inject_max_items": 3})
    plugin._backend = MemoryBackend()
    return plugin


def test_cold_start_injects_nothing():
    p = make_plugin()
    e = FakeEvent("1001", "你好")
    r = req()
    asyncio.run(p.inject_memory(e, r))
    assert r.extra_user_content_parts == [], "空库时不该注入任何东西"


def test_inject_and_budget():
    p = make_plugin()
    e = FakeEvent("1001")
    key = adapter.identity_key("aiocqhttp", "1001")
    mem = p._mem(key)
    mem.observe("goals", "交付驻车集 V1.0", track="work", confidence=0.8, due_at_ts=__import__("time").time() + 2 * 86400)
    mem.observe("likes", "偏好深色主题", track="work")
    r = req()
    asyncio.run(p.inject_memory(e, r))
    assert len(r.extra_user_content_parts) == 1
    part = r.extra_user_content_parts[0]
    text = part["text"] if isinstance(part, dict) else part.text
    assert "记忆" in text and "驻车集" in text
    assert len(text) <= 240, f"注入超预算: {len(text)}"
    assert text.index("驻车集") < text.index("深色主题"), "快到期的应排前面"


def test_two_users_isolated():
    p = make_plugin()
    a, b = FakeEvent("1001"), FakeEvent("2002")
    p._mem(adapter.identity_key("aiocqhttp", "1001")).observe("goals", "A 的私事", track="work")
    ra, rb = req(), req()
    asyncio.run(p.inject_memory(a, ra))
    asyncio.run(p.inject_memory(b, rb))
    assert ra.extra_user_content_parts, "A 应看到自己的记忆"
    assert rb.extra_user_content_parts == [], "B 不该看到 A 的记忆"


def test_memory_and_del_commands():
    p = make_plugin()
    key = adapter.identity_key("aiocqhttp", "1001")
    p._mem(key).observe("goals", "在准备四级", track="work")

    async def run(coro):
        return [x async for x in coro]

    out = asyncio.run(run(p.show_memory(FakeEvent("1001", f"/memory {key}"))))
    assert out and "四级" in out[0] and "1 条" in out[0]

    out = asyncio.run(run(p.delete_memory(FakeEvent("1001", f"/del {key}"))))
    assert "已删除" in out[0] and "1 条" in out[0]

    out = asyncio.run(run(p.show_memory(FakeEvent("1001", f"/memory {key}"))))
    assert "0 条" in out[0]


def test_bad_key_rejected():
    p = make_plugin()

    async def run(coro):
        return [x async for x in coro]

    out = asyncio.run(run(p.delete_memory(FakeEvent("1001", "/del ../etc"))))
    assert "不合法" in out[0]


def test_inject_only_once_per_turn():
    """agent 回路会多次请求模型；同一轮只该注入一次，否则记忆会重复堆进上下文。"""
    p = make_plugin()
    e = FakeEvent("1001")
    p._mem(adapter.identity_key("aiocqhttp", "1001")).observe("goals", "一次就够", track="work")
    r1, r2 = req(), req()
    asyncio.run(p.inject_memory(e, r1))
    asyncio.run(p.inject_memory(e, r2))
    assert len(r1.extra_user_content_parts) == 1
    assert r2.extra_user_content_parts == [], "第二轮不该重复注入"


def test_per_group_scope():
    p = make_plugin()
    p.cfg["scope"] = "per_group"
    a = FakeEvent("1001", group="g1")
    b = FakeEvent("2002", group="g1")
    c = FakeEvent("1001", group="g2")
    assert p._identity(a) == p._identity(b), "同群应按群共享"
    assert p._identity(a) != p._identity(c)
    p.cfg["scope"] = "per_user"
    assert p._identity(a) != p._identity(b), "per_user 下应按人分开"


def test_extract_bridge_writes_into_memory_v2():
    """写入腿：抽取结果 → MemoryV2（不经过 persona_manager 那套存储）。"""
    from personatrace.memory_v2 import MemoryBackend, MemoryV2

    class FakeExtractor:
        def extract_from_conversation(self, **kwargs):
            return {
                "work_updates": {
                    "goals": [{"label": "在准备四级", "confidence": 0.8, "reasoning": "用户自己说的"}],
                    "likes": [{"label": "偏好深色主题"}],
                },
                "persona_updates": {"identity": [{"label": "计算机专业学生"}]},
            }

    mem = MemoryV2(backend=MemoryBackend(), user_id="aiocqhttp:1001")
    bridge = adapter.ExtractBridge(mem, FakeExtractor())
    n = bridge.ingest("我在准备四级，专业课也忙", "加油")
    rows = mem.state()
    assert n == 3 and len(rows) == 3, (n, rows)
    tracks = {r["category"]: r.get("track") for r in rows}
    assert tracks.get("goals") == "work" and tracks.get("identity") == "persona"


def test_host_llm_bridges_sync_call_to_async_provider():
    """抽取跑在线程池里，但 provider 是 async —— 这条桥必须通。"""
    import threading

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()

    class FakeProvider:
        async def text_chat(self, prompt=None, contexts=None, system_prompt=None, model=None):
            assert "hi" in (prompt or "")
            return types.SimpleNamespace(completion_text='{"ok": 1}')

    try:
        llm = adapter.HostLLM(FakeProvider(), loop, timeout=5)
        assert llm.chat_json([{"role": "user", "content": "hi"}]) == {"ok": 1}
    finally:
        loop.call_soon_threadsafe(loop.stop)


if __name__ == "__main__":
    for fn in (
        test_cold_start_injects_nothing,
        test_inject_and_budget,
        test_two_users_isolated,
        test_memory_and_del_commands,
        test_bad_key_rejected,
        test_extract_bridge_writes_into_memory_v2,
        test_host_llm_bridges_sync_call_to_async_provider,
    ):
        fn()
        print(f"[PASS] {fn.__name__}")
