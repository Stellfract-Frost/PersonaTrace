"""Offline tests for the prospective-memory ledger.

No LLM, no network, no Chroma. The clock is always passed in.
"""

import threading
from datetime import datetime
from zoneinfo import ZoneInfo

from personatrace.intentions import IntentStore

TZ = "Asia/Shanghai"
Z = ZoneInfo(TZ)


def dt(day: int, hour: int = 0, minute: int = 0, month: int = 10) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=Z)


def make(store: IntentStore, title: str, due: datetime, **kwargs):
    return store.upsert("u", title, due_at=due, granularity="day", now=dt(22, 23, 7), **kwargs)


def test_day_granularity_uses_date_not_midnight():
    store = IntentStore(":memory:", default_tz=TZ)
    make(store, "上课", dt(24))
    text = store.render_agenda("u", now=dt(22, 23, 7))
    assert "后天 上课" in text
    assert "还有 2 天" in text

    text = store.render_agenda("u", now=dt(24, 10))
    assert "今天 上课" in text

    assert store.render_agenda("u", now=dt(27, 10)) == ""  # past grace -> not injected


def test_due_task_is_in_agenda_without_semantic_relevance():
    store = IntentStore(":memory:", default_tz=TZ)
    make(store, "交报告", dt(24), importance=0.9)
    text = store.render_agenda("u", now=dt(23, 10))
    assert "交报告" in text
    assert "需主动" in text  # T-24h bucket


def test_anti_nag_one_proactive_then_compact():
    store = IntentStore(":memory:", default_tz=TZ, proactive_cooldown_s=6 * 3600)
    make(store, "上课", dt(24))
    first = store.render_agenda("u", now=dt(24, 8))
    assert "需主动" in first
    second = store.render_agenda("u", now=dt(24, 8))
    assert "需主动" not in second
    later = store.render_agenda("u", now=dt(24, 12))
    assert "需主动" not in later
    assert "上课" in later  # still present, just not prominent


def test_overdue_grace_then_archive():
    store = IntentStore(":memory:", default_tz=TZ)
    make(store, "还书", dt(24))
    text = store.render_agenda("u", now=dt(25, 10))
    assert "未确认" in text

    assert store.render_agenda("u", now=dt(26, 10)) == ""  # grace is 1 day


def test_complete_and_reschedule():
    store = IntentStore(":memory:", default_tz=TZ)
    item = make(store, "上课", dt(24))
    store.complete(item.id, now=dt(23, 10))
    assert store.render_agenda("u", now=dt(23, 10)) == ""
    assert store.get(item.id).status == "done"

    other = make(store, "交报告", dt(24))
    store.reschedule(other.id, dt(30), now=dt(23, 10))
    assert store.get(other.id).due_at == dt(30).timestamp()
    assert store.get(other.id).status == "active"


def test_same_title_different_date_is_two_intents():
    store = IntentStore(":memory:", default_tz=TZ)
    make(store, "上课", dt(24))
    make(store, "上课", dt(31))
    assert len(store.list("u")) == 2


def test_proposed_is_not_injected_by_default():
    store = IntentStore(":memory:", default_tz=TZ)
    make(store, "可能要去", dt(24), needs_confirmation=True)
    assert store.render_agenda("u", now=dt(23, 10)) == ""
    assert len(store.list("u", statuses=("proposed",))) == 1


def test_weekly_recurrence_and_exception():
    store = IntentStore(":memory:", default_tz=TZ)
    first = dt(28)  # Wednesday
    item = store.upsert(
        "u", "上课", due_at=first, granularity="day",
        rrule="FREQ=WEEKLY;BYDAY=WE", now=dt(22, 23, 7),
    )
    assert "上课" in store.render_agenda("u", now=dt(28, 8))

    store.complete(item.id, occurrence_ts=first.timestamp(), now=dt(28, 10))
    assert store.render_agenda("u", now=dt(28, 10)) == ""  # this occurrence is done

    # next week's occurrence still exists
    assert "上课" in store.render_agenda("u", now=dt(3, 10, month=11))


def test_event_log_keeps_history():
    store = IntentStore(":memory:", default_tz=TZ)
    item = make(store, "上课", dt(24))
    store.reschedule(item.id, dt(25), now=dt(23))
    store.complete(item.id, now=dt(24))
    rows = store.conn.execute(
        "SELECT kind FROM intent_events WHERE intention_id=? ORDER BY id", (item.id,)
    ).fetchall()
    assert [row["kind"] for row in rows] == ["created", "rescheduled", "done"]


def test_touch_turn_roundtrip():
    store = IntentStore(":memory:", default_tz=TZ)
    store.touch_turn("u", dt(22, 23, 7))
    assert store.last_turn_at("u") == dt(22, 23, 7).timestamp()


def test_concurrent_upserts_are_serialized():
    store = IntentStore(":memory:", default_tz=TZ)
    errors = []

    def worker():
        try:
            store.upsert("u", "并发任务", due_at=dt(24), granularity="day", now=dt(22, 23, 7))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert len(store.list("u")) == 1
