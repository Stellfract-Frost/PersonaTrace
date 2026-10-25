"""P1 tests for the merged write/read facade."""

import copy
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from personatrace.intentions import IntentStore
from personatrace.memory_v2 import MemoryBackend, MemoryV2
from personatrace.merged_memory import MergedMemory

TZ = "Asia/Shanghai"
Z = ZoneInfo(TZ)
NOW = datetime(2026, 10, 23, 10, 0, tzinfo=Z)


class FakeLLM:
    def __init__(self, payloads):
        self.payloads = [copy.deepcopy(p) for p in payloads]
        self.calls = 0

    def chat_json(self, messages, max_tokens=1600):
        self.calls += 1
        if not self.payloads:
            return None
        return self.payloads.pop(0)


def make(payloads):
    mem = MemoryV2(backend=MemoryBackend(), user_id="u")
    intents = IntentStore(":memory:", default_tz=TZ, proactive_cooldown_s=0)
    return MergedMemory(memory=mem, llm=FakeLLM(payloads), intents=intents, user_id="u", tz=TZ)


def test_remember_writes_belief_and_intention():
    mm = make([{
        "beliefs": [{
            "category": "identity", "label": "名字叫小陌",
            "confidence": 0.9, "reasoning": "用户自我介绍",
        }],
        "intentions": [{
            "action": "上课", "detail": "后天要上课",
            "due": "后天", "importance": 0.9, "confidence": 0.8,
        }],
    }])
    result = mm.remember("我叫小陌，后天要上课", now=NOW, force=True)
    assert result["status"] == "success"
    assert result["beliefs"] == 1
    assert result["intentions"] == 1

    labels = [row["label"] for row in mm.memory.state(as_of=NOW.timestamp())]
    assert "名字叫小陌" in labels

    intents = mm.intents.list("u")
    assert len(intents) == 1
    assert intents[0].title == "上课"
    assert intents[0].importance == 0.9
    assert intents[0].due_at == datetime(2026, 10, 25, tzinfo=Z).timestamp()


def test_one_off_importance_is_not_frequency():
    mm = make([{
        "beliefs": [],
        "intentions": [{
            "action": "上课", "due": "后天", "importance": 0.9,
            "confidence": 0.8,
        }],
    }])
    mm.remember("后天要上课", now=NOW, force=True)
    item = mm.intents.list("u")[0]
    assert item.importance == 0.9
    assert len(mm.intents.list("u")) == 1


def test_source_gate_and_empty_user():
    mm = make([{"beliefs": [], "intentions": [{"action": "上课", "due": "后天"}]}])
    assert mm.remember("后天要上课", now=NOW, source="attention", force=True)["status"] == "skipped"
    assert mm.memory.stats()["beliefs"] == 0
    assert mm.intents.list("u") == []

    assert mm.remember("", now=NOW, force=True)["status"] == "skipped"


def test_unparseable_due_is_not_silently_turned_into_ttl():
    mm = make([{
        "beliefs": [],
        "intentions": [{"action": "周末爬山", "due": "周末", "importance": 0.7}],
    }])
    result = mm.remember("周末去爬山", now=NOW, force=True)
    assert result["intentions"] == 0
    assert mm.intents.list("u") == []


def test_no_due_intention_gets_explicit_ttl_and_expires():
    mm = make([{
        "beliefs": [],
        "intentions": [{"action": "买牛奶", "importance": 0.7}],
    }])
    mm.remember("提醒我买牛奶", now=NOW, force=True)
    item = mm.intents.list("u")[0]
    assert item.due_at is not None
    assert abs(item.due_at - (NOW.timestamp() + 86400)) < 5

    agenda = mm.intents.render_agenda("u", now=NOW)
    assert "买牛奶" in agenda
    assert mm.intents.render_agenda("u", now=NOW + timedelta(days=1, seconds=1)) == ""


def test_pydantic_error_is_retried():
    mm = make([
        {"beliefs": [{"category": "not_a_category", "label": "坏数据"}], "intentions": []},
        {"beliefs": [{"category": "identity", "label": "名字叫小陌"}], "intentions": []},
    ])
    result = mm.remember("我叫小陌", now=NOW, force=True)
    assert result["beliefs"] == 1
    assert mm.llm.calls == 2


def test_inject_includes_agenda_and_dated_belief():
    mm = make([{
        "beliefs": [{
            "category": "goals", "label": "周五要演示",
            "confidence": 0.9, "due": "周五", "reasoning": "客户演示",
        }],
        "intentions": [{
            "action": "上课", "due": "后天", "importance": 0.9,
            "confidence": 0.8,
        }],
    }])
    mm.remember("周五要演示，后天要上课", now=NOW, force=True)
    text = mm.inject("我那个 deadline 是几号", now=NOW)
    assert "【短期任务】" in text
    assert "上课" in text
    assert "【和日期有关的】" in text
    assert "周五要演示" in text
    assert "截止" in text


def test_build_context_enriches_only_selected_rows():
    mm = make([])
    mm.write(beliefs=[{
        "category": "identity", "label": "名字叫小陌",
        "reasoning": "用户自我介绍", "evidence_text": "我叫小陌",
    }], now=NOW)
    pack = mm.build_context("我叫什么", now=NOW)
    assert pack.pinned
    row = pack.pinned[0]
    assert row["reasoning"] == "用户自我介绍"
    assert "我叫小陌" in row["evidence_texts"]


def test_build_context_returns_structured_sections():
    mm = make([{
        "beliefs": [{"category": "identity", "label": "名字叫小陌"}],
        "intentions": [],
    }])
    mm.remember("我叫小陌", now=NOW, force=True)
    pack = mm.build_context("我叫什么", now=NOW)
    assert any(row["label"] == "名字叫小陌" for row in pack.pinned)
    assert isinstance(pack.agenda_lines, list)


def test_sqlite_path_persists_beliefs_and_intents(tmp_path):
    path = str(tmp_path / "merged.db")
    llm = FakeLLM([{
        "beliefs": [{"category": "identity", "label": "名字叫小陌"}],
        "intentions": [{"action": "上课", "due": "后天", "importance": 0.9}],
    }])
    first = MergedMemory(llm=llm, user_id="u", tz=TZ, sqlite_path=path)
    first.remember("我叫小陌，后天要上课", now=NOW, force=True)

    second = MergedMemory(llm=FakeLLM([]), user_id="u", tz=TZ, sqlite_path=path)
    assert any(row["label"] == "名字叫小陌" for row in second.memory.state(as_of=NOW.timestamp()))
    assert [item.title for item in second.intents.list("u")] == ["上课"]


def test_hidden_hint_survives_truncation():
    mm = make([])
    for index in range(6):
        mm.memory.observe("likes", f"喜欢东西{index}", track="work", ts=NOW.timestamp())
    text = mm.inject("随便", now=NOW, k=2, max_chars=80)
    assert "另有" in text
    assert "memory_recall" in text
    assert len(text) >= 20

def test_default_constructor_uses_one_durable_file(tmp_path, monkeypatch):
    db = tmp_path / "one.db"
    monkeypatch.setenv("PERSONA_DB", str(db))
    first = MergedMemory(user_id="alice")
    first.write(
        beliefs=[{"category": "identity", "label": "名字叫小陌"}],
        intentions=[{"action": "交报告", "due": "后天"}],
        source="direct",
        now=NOW,
    )
    first.record_turn("上次说要交报告", "收到", session_id="s1", now=NOW)
    first.close()

    second = MergedMemory(user_id="alice")
    try:
        assert [row["label"] for row in second.memory.state(as_of=NOW.timestamp())] == ["名字叫小陌"]
        assert [item.title for item in second.intents.list("alice")] == ["交报告"]
        assert second.episodes_search("报告", limit=2)[0]["user_msg"] == "上次说要交报告"
    finally:
        second.close()


def test_write_does_not_silently_truncate_to_four():
    mm = make([])
    result = mm.write(
        beliefs=[{"category": "likes", "label": f"喜欢测试项{index}"} for index in range(6)],
        intentions=[{"action": f"任务{index}", "importance": 0.5} for index in range(6)],
        now=NOW,
    )
    assert result["requested_beliefs"] == 6
    assert result["requested_intentions"] == 6
    assert result["beliefs"] == 6
    assert result["intentions"] == 6
    assert result["distinct_beliefs"] == 6
    assert result["distinct_intentions"] == 6
    assert result["dropped_beliefs"] == 0
    assert result["dropped_intentions"] == 0
    assert len(mm.memory.state(as_of=NOW.timestamp())) == 6
    assert len(mm.intents.list("u")) == 6


def test_write_dedups_paraphrased_preference():
    mm = make([])
    mm.write(beliefs=[{"category": "likes", "label": "喜欢深色主题"}], now=NOW)
    mm.write(beliefs=[{"category": "likes", "label": "偏好深色主题"}], now=NOW)

    rows = mm.memory.state(as_of=NOW.timestamp())
    assert [(row["label"], row["hits"]) for row in rows] == [("喜欢深色主题", 2)]


def test_write_dedups_possessive_variant():
    mm = make([])
    mm.write(beliefs=[{"category": "identity", "label": "名字叫小陌"}], now=NOW)
    mm.write(beliefs=[{"category": "identity", "label": "我的名字叫小陌"}], now=NOW)
    assert [row["label"] for row in mm.memory.state(as_of=NOW.timestamp())] == ["名字叫小陌"]


def test_write_does_not_merge_different_deadlines():
    mm = make([])
    mm.write(beliefs=[{"category": "goals", "label": "周五要演示", "due": "周五"}], now=NOW)
    mm.write(beliefs=[{"category": "goals", "label": "下周三演示", "due": "下周三"}], now=NOW)
    assert len(mm.memory.state(as_of=NOW.timestamp())) == 2


def test_write_does_not_merge_similar_but_different_labels():
    mm = make([])
    mm.write(beliefs=[{"category": "likes", "label": "喜欢猫"}], now=NOW)
    mm.write(beliefs=[{"category": "likes", "label": "喜欢猫毛"}], now=NOW)
    assert len(mm.memory.state(as_of=NOW.timestamp())) == 2


def test_write_does_not_merge_one_character_month_change():
    """Short labels need near-identity; 0.88 is not a paraphrase."""
    mm = make([])
    mm.write(beliefs=[{"category": "goals", "label": "计划明年三月去日本东京旅行"}], now=NOW)
    mm.write(beliefs=[{"category": "goals", "label": "计划明年四月去日本东京旅行"}], now=NOW)
    labels = {row["label"] for row in mm.memory.state(as_of=NOW.timestamp())}
    assert labels == {"计划明年三月去日本东京旅行", "计划明年四月去日本东京旅行"}


def test_write_does_not_merge_one_character_proper_noun():
    mm = make([])
    mm.write(beliefs=[{"category": "explicit_facts", "label": "项目代号是 Astra"}], now=NOW)
    mm.write(beliefs=[{"category": "explicit_facts", "label": "项目代号是 Atlas"}], now=NOW)
    labels = {row["label"] for row in mm.memory.state(as_of=NOW.timestamp())}
    assert labels == {"项目代号是 Astra", "项目代号是 Atlas"}


def test_write_does_not_merge_different_amounts():
    mm = make([])
    mm.write(beliefs=[{"category": "explicit_facts", "label": "预算是一万二千元"}], now=NOW)
    mm.write(beliefs=[{"category": "explicit_facts", "label": "预算是一万五千元"}], now=NOW)
    labels = {row["label"] for row in mm.memory.state(as_of=NOW.timestamp())}
    assert labels == {"预算是一万二千元", "预算是一万五千元"}

def test_remember_stores_episode_and_retrospective_context():
    mm = make([{
        "beliefs": [{"category": "identity", "label": "名字叫小陌"}],
        "intentions": [],
    }])
    result = mm.remember("我叫小陌", now=NOW, force=True, session_id="s1")
    assert result["episode_id"]
    hits = mm.episodes_search("小陌")
    assert hits and hits[0]["user_msg"] == "我叫小陌"
    assert hits[0]["belief_ids"] == result["belief_ids"]

    pack = mm.build_context("上次我们说了什么", now=NOW)
    assert pack.episodes and pack.episodes[0]["user_msg"] == "我叫小陌"


def test_record_turn_then_link_structured_beliefs():
    mm = make([])
    turn = mm.record_turn("每周二晚上去上课", session_id="s1", now=NOW)
    assert turn["episode_id"]
    written = mm.write(
        beliefs=[{"category": "goals", "label": "每周二晚上去上课"}],
        episode_id=turn["episode_id"], now=NOW,
    )
    assert written["episode_id"] == turn["episode_id"]
    episode = mm.episodes_search("上课")[0]
    assert written["belief_ids"][0] in episode["belief_ids"]


def test_label_routine_appears_in_context_probe():
    mm = make([])
    labeled = mm.label_routine(
        "attend class", time="10:00", weekdays=[5],
        anchor_date="2026-10-16", status="candidate", now=NOW,
    )
    assert labeled["weekdays"] == [5]
    pack = mm.build_context("随便", now=NOW)
    assert pack.routine_probes
    assert pack.routine_probes[0]["action"] == "attend class"
    confirmed = mm.routine_feedback(pack.routine_probes[0]["probe_id"], "confirmed", now=NOW)
    assert confirmed["status"] == "confirmed"


def test_sqlite_path_persists_episodes(tmp_path):
    path = str(tmp_path / "merged.db")
    first = MergedMemory(llm=FakeLLM([]), user_id="u", tz=TZ, sqlite_path=path)
    turn = first.record_turn("上次说要搬家", now=NOW)
    assert turn["episode_id"]

    second = MergedMemory(llm=FakeLLM([]), user_id="u", tz=TZ, sqlite_path=path)
    hits = second.episodes_search("搬家")
    assert hits and hits[0]["user_msg"] == "上次说要搬家"

def test_english_temporal_query_opens_episode_context():
    mm = make([])
    mm.record_turn("I met my aunt last week", now=NOW)
    pack = mm.build_context("How many weeks ago did I meet my aunt?", now=NOW)
    assert pack.episodes
    assert "aunt" in pack.episodes[0]["user_msg"]

