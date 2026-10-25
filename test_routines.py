"""Weekly routine tests: recurrence, term/exceptions, probe state machine."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from personatrace.routines import RoutineStore

TZ = "Asia/Shanghai"
Z = ZoneInfo(TZ)
TUESDAY = datetime(2026, 10, 27, 19, 5, tzinfo=Z)


def make_store():
    return RoutineStore(":memory:", default_tz=TZ)


def test_weekly_occurrence_and_interval():
    store = make_store()
    routine = store.label(
        "u", "attend class", time="19:00", weekdays=[2],
        anchor_date="2026-10-20",
    )
    assert routine is not None
    assert store.next_occurrence(routine, TUESDAY).date().isoformat() == "2026-11-03"
    assert store.next_occurrence(routine, TUESDAY.replace(hour=18, minute=55)).date().isoformat() == "2026-10-27"

    biweekly = store.label(
        "u", "lab", time="19:00", weekdays=[2], interval=2,
        anchor_date="2026-10-20",
    )
    assert biweekly is not None
    # Week of 10/20 is the anchor; 10/27 is the off week; 11/03 is on.
    assert store.next_occurrence(biweekly, TUESDAY).date().isoformat() == "2026-11-03"


def test_probe_window_and_weekly_weekday():
    store = make_store()
    store.label("u", "attend class", time="19:00", weekdays=[2], anchor_date="2026-10-20")

    assert len(store.probe("u", now=TUESDAY)) == 1
    assert store.probe("u", now=TUESDAY.replace(hour=20, minute=0)) == []
    assert store.probe("u", now=TUESDAY.replace(day=28)) == []  # Wednesday


def test_probe_questions_follow_action_language():
    store = make_store()
    store.label("u", "上课", time="19:00", weekdays=[2], anchor_date="2026-10-20")
    probe = store.probe("u", now=TUESDAY)[0]
    assert any("上课" in question for question in probe["suggested_questions"])
    assert all(question.startswith(("你", "今天")) for question in probe["suggested_questions"])


def test_confirm_defaults_and_second_confirm_trusts():
    store = make_store()
    store.label("u", "attend class", time="19:00", weekdays=[2], anchor_date="2026-10-20")
    probe = store.probe("u", now=TUESDAY)[0]
    first = store.feedback(probe["probe_id"], "confirmed", now=TUESDAY)
    assert first.status == "confirmed"
    assert store.probe("u", now=TUESDAY + timedelta(days=7)) == []

    # A stale confirmation can ask again; a second yes becomes trusted.
    later = TUESDAY + timedelta(days=35)  # a Tuesday, beyond the stale window
    stale = store.probe("u", now=later)
    assert stale
    second = store.feedback(stale[0]["probe_id"], "confirmed", now=later)
    assert second.status == "trusted"


def test_skipped_today_does_not_deny_routine():
    store = make_store()
    store.label("u", "attend class", time="19:00", weekdays=[2], anchor_date="2026-10-20")
    probe = store.probe("u", now=TUESDAY)[0]
    current = store.feedback(probe["probe_id"], "skipped_today", now=TUESDAY)
    assert current.status == "candidate"
    assert "2026-10-27" in current.exdates
    assert store.probe("u", now=TUESDAY) == []
    # Next week is still a valid occurrence.
    assert store.next_occurrence(current, TUESDAY + timedelta(days=1)).date().isoformat() == "2026-11-03"


def test_denied_pauses_and_two_silences_dormant():
    store = make_store()
    store.label("u", "gym", time="08:00", weekdays=[2], anchor_date="2026-10-20")
    probe = store.probe("u", now=TUESDAY.replace(hour=8, minute=0))[0]
    denied = store.feedback(probe["probe_id"], "denied", now=TUESDAY.replace(hour=8, minute=0))
    assert denied.status == "paused"

    store.label("u", "read", time="06:00", weekdays=[2], anchor_date="2026-10-20")
    t = TUESDAY.replace(hour=6, minute=0)
    p1 = store.probe("u", now=t)[0]
    store.feedback(p1["probe_id"], "unanswered", now=t)
    t2 = t + timedelta(days=7)
    p2 = store.probe("u", now=t2)[0]
    after = store.feedback(p2["probe_id"], "unanswered", now=t2)
    assert after.status == "dormant"


def test_valid_until_and_exdate():
    store = make_store()
    routine = store.label(
        "u", "seminar", time="19:00", weekdays=[2],
        anchor_date="2026-10-20", valid_until="2026-10-27",
    )
    assert routine is not None
    assert store.probe("u", now=TUESDAY)

    store.label(
        "u", "holiday class", time="19:00", weekdays=[2],
        anchor_date="2026-10-20", exdates=["2026-10-27"],
    )
    assert not any(p["action"] == "holiday class" for p in store.probe("u", now=TUESDAY))

def test_single_observation_is_only_candidate():
    store = make_store()
    routine = store.observe(
        "u", "attend class", occurrence_date="2026-10-27", time="19:00",
    )
    assert routine is not None
    assert routine.status == "candidate"
    probe = store.probe("u", now=TUESDAY)[0]
    confirmed = store.feedback(probe["probe_id"], "confirmed", now=TUESDAY)
    assert confirmed.status == "confirmed"


def test_silence_does_not_erase_confirmed():
    store = make_store()
    store.label(
        "u", "attend class", time="19:00", weekdays=[2],
        anchor_date="2026-10-20", status="confirmed",
    )
    # 35 days later the stale check is allowed; two silences must not demote it.
    later = TUESDAY + timedelta(days=35)
    p1 = store.probe("u", now=later)[0]
    store.feedback(p1["probe_id"], "unanswered", now=later)
    p2 = store.probe("u", now=later + timedelta(days=7))[0]
    after = store.feedback(p2["probe_id"], "unanswered", now=later + timedelta(days=7))
    assert after.status == "confirmed"

