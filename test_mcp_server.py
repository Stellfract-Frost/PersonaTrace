"""MCP server 测试：协议层单测 + 真实子进程端到端。

跑法：D:/Anaconda3/python.exe -m pytest test_mcp_server.py -q
"""

import json
import os
import subprocess
import sys

import pytest

from personatrace import mcp_server
from personatrace.intentions import IntentStore
from personatrace.memory_v2 import MemoryBackend, MemoryV2
from personatrace.merged_memory import MergedMemory


@pytest.fixture(autouse=True)
def fresh_mem():
    mcp_server._mem = MemoryV2(backend=MemoryBackend(), user_id="mcp_user")
    mcp_server._merged = MergedMemory(
        memory=MemoryV2(backend=MemoryBackend(), user_id="mcp_user"),
        intents=IntentStore(":memory:", default_tz="Asia/Shanghai"),
        user_id="mcp_user",
    )
    yield
    mcp_server._mem = None
    mcp_server._merged = None


def call(name, **args):
    return mcp_server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": args},
    })


def payload(resp):
    return json.loads(resp["result"]["content"][0]["text"])


def test_initialize_and_capabilities():
    resp = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert resp["result"]["protocolVersion"]
    assert resp["result"]["capabilities"]["tools"] is not None
    assert resp["result"]["serverInfo"]["name"] == "personatrace-memory"


def test_notification_has_no_response():
    assert mcp_server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_tools_list():
    resp = mcp_server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in resp["result"]["tools"]}
    assert names == {
        "memory_state", "memory_recall", "memory_observe", "memory_correct",
        "memory_predict", "memory_remember", "memory_context", "memory_agenda",
        "memory_turn", "memory_episodes", "memory_routine_probe",
        "memory_routine_feedback", "memory_routine_label",
    }
    for t in resp["result"]["tools"]:
        assert t["inputSchema"]["type"] == "object"


def test_observe_then_state():
    r = payload(call("memory_observe", category="goals", label="交付 V1.0", track="work"))
    assert r["hits"] == 1 and 0 < r["retention"] <= 1

    r = payload(call("memory_observe", category="goals", label="交付 V1.0", track="work"))
    assert r["hits"] == 2                                  # 幂等：同一条，不是副本

    state = payload(call("memory_state"))
    assert state["count"] == 1


def test_recall_pins_constraints():
    call("memory_observe", category="constraints", label="不接受远程办公", track="work")
    call("memory_observe", category="likes", label="喜欢深色主题", track="work")
    r = payload(call("memory_recall", query="远程"))
    assert r["pinned"] == 1
    labels = [x["label"] for x in r["items"]]
    assert "不接受远程办公" in labels


def test_correct_keeps_history():
    bid = payload(call("memory_observe", category="goals", label="学 Rust", track="work"))["id"]
    call("memory_correct", belief_id=bid)
    assert payload(call("memory_state"))["count"] == 0
    assert payload(call("memory_state", include_invalidated=True))["count"] == 1   # 没删，只是失效


def test_predict():
    import time
    m = mcp_server.get_mem()
    t = time.time() - 60
    for i, k in enumerate(["app", "ask", "app", "ask", "app", "ask"]):
        m.record_behavior(k, "debug", ts=t + i)
    r = payload(call("memory_predict"))
    assert r["predictions"][0]["next"] == "app"




def test_merged_remember_context_agenda():
    now = "2026-10-23T10:00:00+08:00"
    r = payload(call(
        "memory_remember",
        beliefs=[{"category": "identity", "label": "名字叫小陌", "confidence": 0.9}],
        intentions=[{"action": "上课", "due": "后天", "importance": 0.9}],
        now=now,
    ))
    assert r["beliefs"] == 1
    assert r["intentions"] == 1

    context = payload(call("memory_context", query="我那个 deadline 是几号", now=now))
    assert context["agenda_lines"]
    assert any("上课" in line for line in context["agenda_lines"])
    assert any(row["label"] == "名字叫小陌" for row in context["pinned"])

    agenda = payload(call("memory_agenda", now=now))
    assert agenda["items"][0]["title"] == "上课"


def test_merged_remember_source_gate():
    r = payload(call(
        "memory_remember",
        beliefs=[{"category": "identity", "label": "名字叫小陌"}],
        source="attention",
    ))
    assert r["status"] == "skipped"
    assert payload(call("memory_context", query="名字"))["pinned"] == []




def test_merged_turn_episodes_link_beliefs():
    now = "2026-10-27T10:00:00+08:00"
    turn = payload(call("memory_turn", user_msg="上次说每周二上课", now=now))
    assert turn["episode_id"]

    written = payload(call(
        "memory_remember",
        beliefs=[{"category": "identity", "label": "名字叫小陌"}],
        episode_id=turn["episode_id"],
        now=now,
    ))
    assert written["episode_id"] == turn["episode_id"]

    episodes = payload(call("memory_episodes", query="上课", now=now))["items"]
    assert episodes
    assert episodes[0]["user_msg"] == "上次说每周二上课"
    assert written["belief_ids"][0] in episodes[0]["belief_ids"]


def test_routine_label_probe_feedback():
    now = "2026-10-27T19:05:00+08:00"
    labeled = payload(call(
        "memory_routine_label",
        action="attend class", time="19:00", weekdays=[2],
        anchor_date="2026-10-20", status="candidate", now=now,
    ))
    assert labeled["weekdays"] == [2]
    assert labeled["status"] == "candidate"

    probes = payload(call("memory_routine_probe", now=now))["items"]
    assert probes
    assert probes[0]["action"] == "attend class"

    result = payload(call(
        "memory_routine_feedback", probe_id=probes[0]["probe_id"],
        result="confirmed", now=now,
    ))
    assert result["status"] == "confirmed"
    assert payload(call("memory_routine_probe", now=now))["items"] == []


def test_unknown_tool_and_method():
    resp = call("memory_nope")
    assert resp["error"]["code"] == -32602
    resp = mcp_server.handle({"jsonrpc": "2.0", "id": 9, "method": "nope"})
    assert resp["error"]["code"] == -32601


def test_tool_failure_does_not_raise():
    """插件契约：记忆出错只能变成 isError 结果，不能把宿主搞崩。"""
    def boom(_args):
        raise RuntimeError("backend down")

    mcp_server.HANDLERS["__boom__"] = mcp_server._guarded(boom)
    try:
        resp = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": "__boom__", "arguments": {}}})
        assert resp["result"]["isError"] is True
        assert "backend down" in resp["result"]["content"][0]["text"]
    finally:
        mcp_server.HANDLERS.pop("__boom__", None)


def test_stdio_end_to_end():
    """真实起进程：走 stdin/stdout，验证协议能跑通。"""
    here = os.path.dirname(os.path.abspath(__file__))
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "memory_observe",
                    "arguments": {"category": "likes", "label": "用 Vim", "track": "work"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "memory_state", "arguments": {}}},
    ]
    stdin = "\n".join(json.dumps(r, ensure_ascii=False) for r in requests) + "\n"
    proc = subprocess.run(
        [sys.executable, "mcp_server.py"],
        input=stdin, capture_output=True, text=True, timeout=60, cwd=here,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    lines = [json.loads(x) for x in proc.stdout.strip().splitlines() if x.strip()]
    assert len(lines) == 4                      # 通知不产生响应
    assert lines[0]["result"]["serverInfo"]["name"] == "personatrace-memory"
    assert len(lines[1]["result"]["tools"]) == 13
    assert json.loads(lines[3]["result"]["content"][0]["text"])["count"] == 1
