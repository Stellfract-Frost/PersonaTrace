"""Offline tests for intention extraction and date resolution."""

from datetime import datetime
from zoneinfo import ZoneInfo

from personatrace.intent_extractor import IntentExtractor, resolve_when

TZ = "Asia/Shanghai"
Z = ZoneInfo(TZ)
NOW = datetime(2026, 10, 22, 23, 7, tzinfo=Z)


def ex() -> IntentExtractor:
    return IntentExtractor(tz=TZ)


def test_hou_tian_class():
    item = ex().extract("后天要上课", now=NOW)[0]
    assert item["title"] == "上课"
    assert item["due_at"] == datetime(2026, 10, 24, tzinfo=Z).timestamp()
    assert item["granularity"] == "day"
    assert item["due_kind"] == "start"


def test_next_week_deadline():
    item = ex().extract("下周三交报告", now=NOW)[0]
    assert item["title"] == "交报告"
    assert item["due_at"] == datetime(2026, 10, 28, tzinfo=Z).timestamp()
    assert item["due_kind"] == "end"
    assert item["lead_s"] == 7 * 86400


def test_past_and_smalltalk_are_not_intentions():
    assert ex().extract("昨天上过课", now=NOW) == []
    assert ex().extract("今天好累", now=NOW) == []


def test_negative_is_not_a_new_intention():
    assert ex().extract("这周三不上课", now=NOW) == []
    update = ex().extract_update("这周三不上课", now=NOW)
    assert update["kind"] == "cancel"


def test_weekly_recurrence():
    item = ex().extract("每周三上课", now=NOW)[0]
    assert item["rrule"] == "FREQ=WEEKLY;BYDAY=WE"
    assert item["due_at"] == datetime(2026, 10, 28, tzinfo=Z).timestamp()


def test_clock_time_is_not_invented():
    item = ex().extract("后天要上课", now=NOW)[0]
    assert item["granularity"] == "day"
    item = ex().extract("后天晚上八点开会", now=NOW)[0]
    assert item["granularity"] == "datetime"
    assert item["due_at"] == datetime(2026, 10, 24, 20, 0, tzinfo=Z).timestamp()


def test_explicit_clock():
    item = ex().extract("明天下午3点开会", now=NOW)[0]
    assert item["due_at"] == datetime(2026, 10, 23, 15, 0, tzinfo=Z).timestamp()


def test_n_days_later():
    item = ex().extract("3天后交报告", now=NOW)[0]
    assert item["due_at"] == datetime(2026, 10, 25, tzinfo=Z).timestamp()


def test_reschedule_detection():
    update = ex().extract_update("上课改到周五", now=NOW)
    assert update["kind"] == "reschedule"
    assert update["due_at"] == datetime(2026, 10, 23, tzinfo=Z).timestamp()


def test_complete_detection():
    update = ex().extract_update("上完了", now=NOW)
    assert update["kind"] == "complete"


class FakeLLM:
    def __init__(self, payload):
        self.payloads = payload if isinstance(payload, list) else [payload]
        self.calls = 0

    def chat_json(self, messages, max_tokens=1200):
        self.calls += 1
        return self.payloads[min(self.calls - 1, len(self.payloads) - 1)]


def test_llm_json_is_normalized():
    llm = FakeLLM({"intentions": [{
        "action": "交报告",
        "detail": "客户周五要看",
        "when_text": "后天",
        "when_local": "2026-10-24",
        "granularity": "day",
        "due_kind": "end",
        "importance": 4,
        "confidence": 0.8,
        "needs_confirmation": False,
    }]})
    extractor = IntentExtractor(llm=llm, tz=TZ)
    item = extractor.extract("后天要交报告", now=NOW)[0]
    assert item["title"] == "交报告"
    assert item["due_at"] == datetime(2026, 10, 24, tzinfo=Z).timestamp()
    assert item["importance"] == 0.8
    assert item["confidence"] == 0.8


def test_llm_not_called_for_smalltalk():
    llm = FakeLLM({"intentions": []})
    extractor = IntentExtractor(llm=llm, tz=TZ)
    assert extractor.extract("今天好累", now=NOW) == []
    assert llm.calls == 0


def test_llm_retry_on_pydantic_error():
    llm = FakeLLM([
        {"intentions": [{"action": "交报告", "importance": "not-a-number"}]},
        {"intentions": [{
            "action": "交报告", "when_text": "后天", "granularity": "day",
            "importance": 4, "confidence": 0.8,
        }]},
    ])
    extractor = IntentExtractor(llm=llm, tz=TZ)
    item = extractor.extract("后天要交报告", now=NOW)[0]
    assert llm.calls == 2
    assert item["title"] == "交报告"
    assert item["importance"] == 0.8


def test_ambiguous_period_stays_a_period():
    item = ex().extract("下个月交房租", now=NOW)[0]
    assert item["granularity"] == "day"
    assert item["due_at"] == datetime(2026, 11, 1, tzinfo=Z).timestamp()


def test_resolve_when_returns_none_for_no_date():
    due, granularity, rrule = resolve_when("把这件事做好", NOW, TZ)
    assert due is None
    assert granularity == "day"
    assert rrule is None
