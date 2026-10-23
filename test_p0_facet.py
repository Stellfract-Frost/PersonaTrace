"""P0 tests for the conditional time facet."""

from datetime import datetime
from zoneinfo import ZoneInfo

from personatrace.facet import (
    detect_time_intent, has_time_intent, humanize_due, render_time_line,
    time_bucket, time_bucket_lines,
)

TZ = "Asia/Shanghai"
Z = ZoneInfo(TZ)
NOW = datetime(2026, 10, 23, 10, 0, tzinfo=Z)  # Friday


def ts(month: int, day: int, hour: int = 23, minute: int = 59) -> float:
    return datetime(2026, month, day, hour, minute, tzinfo=Z).timestamp()


ROWS = [
    {"id": "future", "label": "周五要演示", "time_kind": "due", "due_at_ts": ts(10, 23), "time_raw": "周五"},
    {"id": "next", "label": "下周三交周报", "time_kind": "due", "due_at_ts": ts(10, 28), "time_raw": "下周三"},
    {"id": "range", "label": "年底前换工作", "time_kind": "range", "due_at_ts": ts(12, 31), "time_raw": "年底前"},
    {"id": "past", "label": "上周五去了海边", "time_kind": "past", "due_at_ts": ts(10, 16), "time_raw": "上周五"},
    {"id": "habit", "label": "每天十点睡", "time_kind": "recurring", "time_raw": "每天"},
    {"id": "legacy", "label": "旧数据截止", "due_at_ts": ts(10, 22), "time_raw": "", "time_kind": "none"},
    {"id": "old", "label": "很久以前", "time_kind": "due", "due_at_ts": ts(9, 1), "time_raw": ""},
]


def test_time_intent_and_habit():
    assert has_time_intent("我那个 deadline 是几号")
    assert has_time_intent("接下来有什么安排")
    assert not has_time_intent("平时几点睡")
    assert not has_time_intent("今天好累")
    assert detect_time_intent("平时几点睡").is_habit


def test_bucket_keeps_due_and_range_not_past_or_habit():
    rows = time_bucket(ROWS, now=NOW, limit=2)
    ids = [row["id"] for row in rows]
    assert ids == ["future", "next"]
    assert "past" not in ids
    assert "habit" not in ids
    assert "old" not in ids


def test_bucket_legacy_due_is_kept_only_if_recent():
    rows = time_bucket(ROWS, now=NOW, limit=10)
    ids = [row["id"] for row in rows]
    assert "legacy" in ids       # no time_kind, but only 3 days old
    assert "old" not in ids      # too old, treated as narrative


def test_humanize_due_absolute_date_first():
    assert humanize_due(ts(10, 23), now=NOW, tz=TZ) == "截止 今天（2026-10-23，周五）"
    assert humanize_due(ts(10, 28), now=NOW, tz=TZ) == "截止 2026-10-28（周三，还有 5 天）"
    assert humanize_due(ts(10, 16), now=NOW, tz=TZ) == "已过期 7 天（2026-10-16，周五）"


def test_humanize_range_keeps_raw():
    text = humanize_due(ts(12, 31), now=NOW, tz=TZ, kind="range", raw="年底前")
    assert text == "截止 年底前（2026-12-31，周四）"


def test_render_time_line():
    line = render_time_line(ROWS[0], now=NOW, tz=TZ)
    assert "周五要演示" in line
    assert "截止 今天" in line


def test_time_bucket_lines():
    lines = time_bucket_lines(ROWS, now=NOW, tz=TZ, limit=2)
    assert len(lines) == 2
    assert all(line.startswith("- ") for line in lines)
