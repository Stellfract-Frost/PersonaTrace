"""Tests for the v1 -> v2 migration mapping (no Chroma needed)."""

from datetime import datetime
from zoneinfo import ZoneInfo

from personatrace.memory_v2 import MemoryBackend, MemoryV2
from scripts.migrate_v1 import migrate_data

TZ = "Asia/Shanghai"
Z = ZoneInfo(TZ)
NOW = datetime(2026, 10, 23, 10, 0, tzinfo=Z)


def fixture():
    return {
        "work": {
            "likes": ["喜欢猫"],
            "goals": [
                {"label": "周五要演示", "confidence": 0.8,
                 "updated": "2026-10-22T10:00:00", "reasoning": "客户演示"},
            ],
            "constraints": [
                {"label": "下周三交周报", "confidence": 0.9,
                 "updated": "2026-10-22T10:00:00"},
            ],
            "explicit_facts": ["上周五去了海边"],
        },
        "persona": {
            "identity": [{"label": "名字叫小陌", "confidence": 0.9,
                          "updated": "2026-10-22T10:00:00"}],
            "rhythms": [{"label": "晚间使用 Chrome", "confidence": 0.7,
                         "time_note": "22:00", "updated": "2026-10-22T10:00:00"}],
        },
        "cross": {"cross_insights": ["他和同事关系不错"]},
        "ephemeral": {"observation_log": [{"label": "闲聊"}], "uncertain": []},
    }


def test_dry_run_writes_nothing():
    mem = MemoryV2(backend=MemoryBackend(), user_id="u")
    report = migrate_data(fixture(), mem, user_id="u", apply=False, now=NOW.timestamp())
    assert report["migrated"] >= 5
    assert mem.stats()["beliefs"] == 0


def test_apply_maps_categories_and_due():
    mem = MemoryV2(backend=MemoryBackend(), user_id="u")
    report = migrate_data(fixture(), mem, user_id="u", apply=True, now=NOW.timestamp())
    assert report["migrated"] >= 5

    rows = {row["label"]: row for row in mem.state(as_of=NOW.timestamp())}
    assert "喜欢猫" in rows
    assert rows["名字叫小陌"]["category"] == "identity"
    assert rows["晚间使用 Chrome"]["category"] == "habit"
    assert rows["周五要演示"]["time_kind"] == "due"
    assert rows["下周三交周报"]["time_kind"] == "due"
    assert "上周五去了海边" not in rows or rows["上周五去了海边"]["time_kind"] == "none"

    # Logs must not become beliefs.
    assert "他和同事关系不错" not in rows
    assert "闲聊" not in rows


def test_apply_is_idempotent():
    mem = MemoryV2(backend=MemoryBackend(), user_id="u")
    migrate_data(fixture(), mem, user_id="u", apply=True, now=NOW.timestamp())
    first = mem.stats()["beliefs"]
    migrate_data(fixture(), mem, user_id="u", apply=True, now=NOW.timestamp())
    assert mem.stats()["beliefs"] == first
