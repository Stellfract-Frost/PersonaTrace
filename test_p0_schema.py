"""P0 tests for the storage-free schema and typed time."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from personatrace.schema import (
    ALL_CATEGORIES, BeliefIn, FORWARD_CATS, PINNED_CATEGORIES,
    STABILITY_DAYS, coerce_in, due_for_write, normalize_category, parse_due,
    track_for,
)

TZ = "Asia/Shanghai"
Z = ZoneInfo(TZ)
NOW = datetime(2026, 10, 23, 10, 0, tzinfo=Z)  # Friday


def test_17_categories_two_tracks():
    assert len(ALL_CATEGORIES) == 17
    assert len(STABILITY_DAYS) == 17

    # The stdlib kernel must not rely on MergedMemory to patch its table:
    # standalone MemoryV2 users need the same 17 categories.
    from personatrace.memory_v2 import CATEGORY_STABILITY

    assert set(CATEGORY_STABILITY) == set(ALL_CATEGORIES)
    for category in ALL_CATEGORIES:
        assert CATEGORY_STABILITY[category] == STABILITY_DAYS[category]

    assert track_for("constraints") == "work"
    assert track_for("identity") == "persona"
    assert track_for("habit") == "persona"
    assert PINNED_CATEGORIES == ("boundaries", "constraints", "identity")
    assert FORWARD_CATS == ("projects", "goals", "constraints")
    with pytest.raises(ValueError):
        track_for("not_a_category")


def test_category_aliases_and_normalization():
    assert normalize_category("preferences") == "likes"
    assert normalize_category("preference") == "likes"
    assert normalize_category("Fact") == "explicit_facts"
    assert normalize_category("projects") == "projects"
    assert normalize_category("nope") is None


def test_coerce_cleans_confidence_and_track():
    item = coerce_in({"category": "preferences", "label": " 喜欢猫 ", "confidence": 85})
    assert item is not None
    assert item.category == "likes"
    assert item.label == "喜欢猫"
    assert item.confidence == 0.85
    assert item.track == "work"

    item = coerce_in({"category": "traits", "label": "偏内向", "confidence": 1.5})
    assert item is not None
    assert item.confidence == 1.0
    assert item.track == "persona"


def test_coerce_drops_bad_category():
    assert coerce_in({"category": "unknown", "label": "随便"}) is None
    assert coerce_in("喜欢猫") is None
    assert coerce_in("喜欢猫", default_category="likes") is not None


def test_belief_in_rejects_short_label_and_bad_category():
    with pytest.raises(ValidationError):
        BeliefIn(category="likes", label="猫")
    with pytest.raises(ValidationError):
        BeliefIn(category="nope", label="喜欢猫")


def test_belief_in_normalizes_percent_confidence():
    assert BeliefIn(category="likes", label="喜欢猫", confidence=85).confidence == 0.85
    assert BeliefIn(category="likes", label="喜欢猫", confidence=1.5).confidence == 1.0


def test_parse_last_friday_is_past():
    item = parse_due("上周五去了海边", NOW, TZ)
    assert item is not None
    assert item.kind == "past"
    assert datetime.fromtimestamp(item.at, Z).date() == datetime(2026, 10, 16).date()


def test_parse_next_wednesday_and_today():
    item = parse_due("下周三交周报", NOW, TZ)
    assert item is not None and item.kind == "due"
    assert datetime.fromtimestamp(item.at, Z).strftime("%Y-%m-%d %H:%M") == "2026-10-28 23:59"

    item = parse_due("周五要演示", NOW, TZ)
    assert item is not None and item.kind == "due"
    assert datetime.fromtimestamp(item.at, Z).strftime("%Y-%m-%d") == "2026-10-23"


def test_parse_relative_and_month_end():
    item = parse_due("三天后", NOW, TZ)
    assert item is not None
    assert datetime.fromtimestamp(item.at, Z).strftime("%Y-%m-%d") == "2026-10-26"

    item = parse_due("月底", NOW, TZ)
    assert item is not None
    assert datetime.fromtimestamp(item.at, Z).strftime("%Y-%m-%d") == "2026-10-31"


def test_year_end_is_range():
    item = parse_due("年底前换工作", NOW, TZ)
    assert item is not None
    assert item.kind == "range"
    assert datetime.fromtimestamp(item.end, Z).strftime("%Y-%m-%d") == "2026-12-31"
    assert item.due_ts == item.end


def test_iso_date_is_a_language_neutral_due():
    item = parse_due("2026-10-25", NOW, TZ)
    assert item is not None and item.kind == "due"
    assert datetime.fromtimestamp(item.at, Z).strftime("%Y-%m-%d") == "2026-10-25"

    timed = parse_due("2026-10-25 19:00", NOW, TZ)
    assert timed is not None and timed.confidence == 1.0
    assert datetime.fromtimestamp(timed.at, Z).strftime("%H:%M") == "19:00"

    assert parse_due("2026-10-20", NOW, TZ).kind == "past"


def test_recurring_is_typed_but_not_a_deadline():
    daily = parse_due("每天十点睡", NOW, TZ)
    weekly = parse_due("每周三上课", NOW, TZ)
    assert daily is not None and daily.kind == "recurring"
    assert weekly is not None and weekly.kind == "recurring"
    assert daily.due_ts is None
    assert due_for_write(coerce_in({"category": "habit", "label": "每天十点睡"}), NOW, TZ) is None


def test_clock_time_and_past_today():
    item = parse_due("今天下午3点开会", NOW, TZ)
    assert item is not None and item.kind == "due"
    assert datetime.fromtimestamp(item.at, Z).strftime("%H:%M") == "15:00"

    later = datetime(2026, 10, 23, 18, 0, tzinfo=Z)
    item = parse_due("今天下午3点开会", later, TZ)
    assert item is not None and item.kind == "past"


def test_due_for_write_guards_narrative_and_uses_forward_fallback():
    narrative = coerce_in({"category": "explicit_facts", "label": "上周五去了海边"})
    assert due_for_write(narrative, NOW, TZ) is None

    project = coerce_in({"category": "projects", "label": "周五要演示"})
    typed = due_for_write(project, NOW, TZ)
    assert typed is not None and typed.kind == "due"

    explicit = coerce_in({"category": "state", "label": "最近很累", "due": "下周三"})
    typed = due_for_write(explicit, NOW, TZ)
    assert typed is not None
    assert datetime.fromtimestamp(typed.at, Z).strftime("%Y-%m-%d") == "2026-10-28"

def test_english_relative_due_is_supported():
    assert parse_due("every Tuesday", NOW, TZ).kind == "recurring"
    assert parse_due("tomorrow", NOW, TZ).kind == "due"
    assert parse_due("yesterday", NOW, TZ).kind == "past"
    assert parse_due("in 3 days", NOW, TZ).kind == "due"
    assert parse_due("2 weeks ago", NOW, TZ).kind == "past"
    assert parse_due("next Monday", NOW, TZ).kind == "due"
    assert parse_due("last Friday", NOW, TZ).kind == "past"
    assert parse_due("last night", NOW, TZ).kind == "past"
    assert parse_due("end of month", NOW, TZ).kind == "due"

    tomorrow = parse_due("tomorrow at 8pm", NOW, TZ)
    assert tomorrow is not None
    assert datetime.fromtimestamp(tomorrow.at, Z).strftime("%H:%M") == "20:00"


def test_relative_month_end_uses_the_named_month():
    item = parse_due("下个月底前交", NOW, TZ)
    assert item is not None
    assert datetime.fromtimestamp(item.at, Z).strftime("%Y-%m-%d") == "2026-11-30"

    item = parse_due("下下个月底", NOW, TZ)
    assert item is not None
    assert datetime.fromtimestamp(item.at, Z).strftime("%Y-%m-%d") == "2026-12-31"

    item = parse_due("月底", NOW, TZ)
    assert datetime.fromtimestamp(item.at, Z).strftime("%Y-%m-%d") == "2026-10-31"


def test_next_next_week_is_not_next_week():
    item = parse_due("下下周三", NOW, TZ)
    assert item is not None
    assert datetime.fromtimestamp(item.at, Z).strftime("%Y-%m-%d") == "2026-11-04"

    item = parse_due("下周末", NOW, TZ)
    assert item is not None
    assert datetime.fromtimestamp(item.end, Z).strftime("%Y-%m-%d") == "2026-11-01"
    assert item.kind == "due"


def test_iso_datetime_with_offset():
    item = parse_due("2026-10-25T19:00:00+08:00", NOW, TZ)
    assert item is not None
    assert datetime.fromtimestamp(item.at, Z).strftime("%H:%M") == "19:00"

    utc = parse_due("2026-10-25T11:00:00Z", NOW, TZ)
    assert utc is not None
    assert datetime.fromtimestamp(utc.at, Z).strftime("%H:%M") == "19:00"

