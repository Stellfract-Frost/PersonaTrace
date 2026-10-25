"""MCP server 测试：协议层单测 + 真实子进程端到端。

跑法：D:/Anaconda3/python.exe -m pytest test_mcp_server.py -q
"""

import json
import os
import subprocess
import sys
import time

import pytest

from personatrace import mcp_server
from personatrace.intentions import IntentStore
from personatrace.memory_v2 import MemoryBackend, MemoryV2
from personatrace.merged_memory import MergedMemory


@pytest.fixture(autouse=True)
def fresh_mem():
    memory = MemoryV2(backend=MemoryBackend(), user_id="mcp_user")
    mcp_server._mem = memory
    mcp_server._merged = MergedMemory(
        memory=memory,
        intents=IntentStore(":memory:", default_tz="Asia/Shanghai"),
        user_id="mcp_user",
    )
    yield
    mcp_server._mem = None
    mcp_server._merged = None


def _workspace_db(name):
    """A DB under the repo (sandbox-safe on Windows), not the user's home."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, name)
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except OSError:
            pass
    return path


def call(name, **args):
    return mcp_server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": args},
    })


def payload(resp):
    return json.loads(resp["result"]["content"][0]["text"])


def test_initialize_and_capabilities():
    resp = mcp_server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": mcp_server.PROTOCOL_VERSION},
    })
    assert resp["result"]["protocolVersion"] == mcp_server.PROTOCOL_VERSION
    assert resp["result"]["capabilities"]["tools"] is not None
    assert resp["result"]["serverInfo"]["name"] == "personatrace-memory"

    newer = mcp_server.handle({
        "jsonrpc": "2.0", "id": 2, "method": "initialize",
        "params": {"protocolVersion": "2099-01-01"},
    })
    assert newer["result"]["protocolVersion"] == mcp_server.PROTOCOL_VERSION


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


def test_old_and_merged_tools_share_one_namespace():
    """memory_observe / memory_state / memory_context must see the same user."""
    payload(call("memory_observe", category="likes", label="喜欢单一命名空间", track="work"))
    assert payload(call("memory_state"))["count"] == 1

    context = payload(call("memory_context", query="单一命名空间"))
    labels = [row["label"] for row in context["pinned"] + context["related"]]
    assert "喜欢单一命名空间" in labels

    payload(call("memory_remember", beliefs=[{"category": "identity", "label": "名字叫小陌"}]))
    state = payload(call("memory_state"))
    labels = [row["label"] for row in state["beliefs"]]
    assert "名字叫小陌" in labels


def test_batch_messages_are_handled_without_crashing():
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "memory_observe",
                    "arguments": {"category": "likes", "label": "批处理写入"}}},
    ]
    responses = mcp_server._handle_batch(batch)
    assert [r["id"] for r in responses] == [1, 2]
    assert responses[1]["result"]["isError"] is False

    # Non-object / empty batches are protocol errors, not AttributeErrors.
    assert mcp_server._handle_batch([])["error"]["code"] == -32600
    assert mcp_server._handle_batch([1, "x"])[0]["error"]["code"] == -32600


def test_batch_stdio_does_not_kill_the_process():
    db = _workspace_db("_test_mcp_batch.db")
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, PYTHONPATH=here, PERSONA_DB=db,
               PERSONA_USER_ID="stdio_batch")
    proc = subprocess.run(
        [sys.executable, "-m", "personatrace.mcp_server"],
        input=json.dumps(batch, ensure_ascii=False) + "\n",
        capture_output=True, text=True, encoding="utf-8", timeout=60, env=env,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    responses = json.loads(proc.stdout)
    assert [r["id"] for r in responses] == [1, 2]


def test_worker_timeout_is_real_and_restartable(monkeypatch):
    db = _workspace_db("_test_mcp_timeout.db")
    monkeypatch.setenv("PERSONA_DB", db)
    monkeypatch.setenv("PERSONA_USER_ID", "timeout_user")
    runner = mcp_server._ProcessRunner(timeout=1.0)
    try:
        started = time.monotonic()
        result = runner.call("__sleep__", {"seconds": 5})
        elapsed = time.monotonic() - started
        assert result["isError"] is True
        assert "memory_timeout" in result["content"][0]["text"]
        assert elapsed < 3.0, "timeout must terminate the worker, not wait for sleep"

        again = runner.call("memory_state", {})
        assert again["isError"] is False
        assert "count" in json.loads(again["content"][0]["text"])
    finally:
        runner.close()


def test_mcp_naive_iso_uses_the_store_timezone(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    monkeypatch.setenv("PERSONA_TZ", "Asia/Shanghai")
    expected = datetime(2026, 10, 25, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
    assert mcp_server._as_ts("2026-10-25") == expected
    assert mcp_server._as_ts("2026-10-25T00:00:00+08:00") == expected
    assert mcp_server._as_ts("2026-10-24T16:00:00Z") == expected


def test_in_memory_mode_does_not_create_an_off_file(monkeypatch, tmp_path):
    monkeypatch.setenv("PERSONA_DB", "off")
    monkeypatch.chdir(tmp_path)
    mcp_server._merged = None
    mcp_server._mem = None

    written = payload(call("memory_observe", category="likes", label="内存模式"))
    assert written["hits"] == 1
    assert payload(call("memory_state"))["count"] == 1
    assert not (tmp_path / "off").exists()


def test_observe_typed_time_preserves_clears_and_revives():
    now = "2026-10-27T10:00:00+08:00"
    later = "2026-10-27T11:00:00+08:00"
    due = 1893456000  # 2030-01-01, far outside the test date

    first = payload(call(
        "memory_observe", category="goals", label="带截止的观察", track="work",
        due_at_ts=due, time_kind="due", time_raw="2030 年", now=now,
    ))
    # Omitting typed-time fields preserves the old value.
    payload(call("memory_observe", category="goals", label="带截止的观察",
                 track="work", now=later))
    row = payload(call("memory_state", as_of="2026-10-27T10:30:00+08:00"))["beliefs"][0]
    assert row["due_at_ts"] == due
    assert row["time_kind"] == "due"

    # Explicit null / "none" / "" clears it.
    payload(call(
        "memory_observe", category="goals", label="带截止的观察", track="work",
        due_at_ts=None, time_kind="none", time_raw="", now="2026-10-27T11:30:00+08:00",
    ))
    row = payload(call("memory_state", as_of="2026-10-27T11:45:00+08:00"))["beliefs"][0]
    assert row["due_at_ts"] is None
    assert row["time_kind"] == "none"

    payload(call("memory_correct", belief_id=first["id"],
                 ts="2026-10-27T12:00:00+08:00",
                 recorded_at="2026-10-27T12:00:00+08:00"))
    payload(call("memory_observe", category="goals", label="带截止的观察",
                 track="work", now="2026-10-27T13:00:00+08:00"))
    state = payload(call("memory_state", as_of="2026-10-27T13:30:00+08:00"))
    assert [row["label"] for row in state["beliefs"]] == ["带截止的观察"]


def test_stdio_stdout_is_utf8_even_with_gbk_locale():
    """The pipe must be UTF-8 even when the host decodes with a GBK locale."""
    here = os.path.dirname(os.path.abspath(__file__))
    db = _workspace_db("_test_mcp_utf8.db")
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    stdin = ("\n".join(json.dumps(r, ensure_ascii=False) for r in requests) + "\n").encode("utf-8")
    env = dict(os.environ, PYTHONPATH=here, PERSONA_DB=db,
               PERSONA_USER_ID="utf8_user", PYTHONIOENCODING="gbk")
    proc = subprocess.run(
        [sys.executable, "-m", "personatrace.mcp_server"],
        input=stdin, capture_output=True, timeout=60, env=env,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    text = proc.stdout.decode("utf-8")          # strict; no locale involvement
    responses = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert [r["id"] for r in responses] == [1, 2]
    assert any("description" in tool for tool in responses[1]["result"]["tools"])


def test_internal_error_does_not_leak_traceback(monkeypatch):
    def boom(_msg, _runner=None):
        raise RuntimeError("secret internal path /tmp/private")

    monkeypatch.setattr(mcp_server, "_handle_single", boom)
    resp = mcp_server._safe_handle({"jsonrpc": "2.0", "id": 7, "method": "ping"})
    message = resp["error"]["message"]
    assert resp["error"]["code"] == -32603
    assert "Traceback" not in message
    assert "secret internal path" not in message


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
    db = _workspace_db("_test_mcp_stdio.db")
    env = dict(os.environ, PYTHONPATH=here, PERSONA_DB=db,
               PERSONA_USER_ID="stdio_user")
    proc = subprocess.run(
        [sys.executable, "-m", "personatrace.mcp_server"],
        input=stdin, capture_output=True, text=True, encoding="utf-8",
        timeout=60, cwd=here, env=env,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    lines = [json.loads(x) for x in proc.stdout.strip().splitlines() if x.strip()]
    assert len(lines) == 4                      # 通知不产生响应
    assert lines[0]["result"]["serverInfo"]["name"] == "personatrace-memory"
    assert len(lines[1]["result"]["tools"]) == 13
    assert json.loads(lines[3]["result"]["content"][0]["text"])["count"] == 1
