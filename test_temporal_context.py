"""Offline tests for the time-gap / reply-stance hint."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from personatrace.temporal_context import (
    build_temporal_context, classify_gap, format_temporal_hint, time_period,
)

TZ = "Asia/Shanghai"
Z = ZoneInfo(TZ)


def test_gap_buckets():
    assert classify_gap(None) == "unknown"
    assert classify_gap(30) == "same_flow"
    assert classify_gap(600) == "short_break"
    assert classify_gap(2 * 3600) == "resumed"
    assert classify_gap(8 * 3600) == "cold_start"


def test_time_period_boundaries():
    assert time_period(8) == "上午"
    assert time_period(12) == "中午"
    assert time_period(15) == "下午"
    assert time_period(20) == "晚间"
    assert time_period(2) == "深夜"


def test_cold_start_does_not_pretend_continuity():
    now = datetime(2026, 10, 22, 23, 30, tzinfo=Z)
    ctx = build_temporal_context(now, now - timedelta(hours=8))
    assert ctx.gap_label == "cold_start"
    assert "不要假设" in ctx.hint
    assert "深夜" in ctx.hint


def test_same_flow_keeps_continuity():
    now = datetime(2026, 10, 22, 15, 0, tzinfo=Z)
    ctx = build_temporal_context(now, now - timedelta(seconds=30))
    assert ctx.gap_label == "same_flow"
    assert "承接上文" in ctx.hint


def test_no_last_turn_returns_empty_hint():
    now = datetime(2026, 10, 22, 15, 0, tzinfo=Z)
    assert format_temporal_hint(now, None) == ""


def test_naive_datetime_is_treated_as_local():
    now = datetime(2026, 10, 22, 15, 0)
    ctx = build_temporal_context(now, now - timedelta(seconds=60))
    assert ctx.gap_label == "same_flow"
