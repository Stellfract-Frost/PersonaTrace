"""Extract future intentions ("后天要上课") from one conversation turn.

Two modes:
- With an LLM: a small JSON-only prompt. The model extracts the action and the
  surface time expression; **code resolves the date**. LLMs are bad at
  "后天" across midnight and time zones, so they are not asked to do arithmetic.
- Without an LLM (or when the call fails): a conservative rule extractor for
  common Chinese expressions. It is a fallback, not a general date parser.

The output is a list of dicts accepted by ``intentions.IntentStore.upsert``.
"""

from __future__ import annotations

import calendar
import logging
import re
from datetime import date, datetime, time as dt_time, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

try:  # package import and top-level test import both work
    from .structured import StructuredOutputError, call_structured
except ImportError:  # pragma: no cover - depends on how the caller imports it
    from structured import StructuredOutputError, call_structured

try:  # optional; the rule fallback covers the common cases without it
    import dateparser
except Exception:  # pragma: no cover - depends on the environment
    dateparser = None

logger = logging.getLogger(__name__)

DEFAULT_TZ = "Asia/Shanghai"

_WEEKDAY_INDEX = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_WEEKDAY_CODE = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")

TIME_CUE = re.compile(
    r"今天|今日|明天|明日|后天|大后天|"
    r"下(?:周|星期|礼拜)|这(?:周|星期|礼拜)|本(?:周|星期|礼拜)|"
    r"(?:周|星期|礼拜)[一二三四五六日天]|"
    r"\d+\s*个?\s*(?:天|周|星期|月)\s*[后後]|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*[日号]|"
    r"月底|下个月|今晚|明早|明晚|"
    r"(?:上午|下午|晚上|中午|凌晨|傍晚)"
)

_NEGATIVE = re.compile(r"不(?:上|去|参加|考|用|要)|取消|算了|不去了")
_CANCEL = re.compile(r"取消|不去了|不上了|不上课|不用了|不参加|不考了|不去了")
_COMPLETE = re.compile(r"上完了|上完|交了|提交了|做完了|完成了|搞定了|弄完了|办完了")
_RESCHEDULE = re.compile(r"(?:改到|改在|推迟到|延到|挪到|改期到)\s*(.+)")
_END_WORDS = ("交", "提交", "截止", "报名", "付款", "还款", "缴费", "申请", "交表", "交报告")
_START_WORDS = ("上课", "开会", "面试", "考试", "出发", "见面", "约", "复诊", "就诊")
_INTENT_WORDS = re.compile(
    r"要|得|必须|需要|答应|约|安排|记得|提醒|别忘|帮我记|"
    r"交|提交|截止|报名|上课|开会|面试|考试|出发|见面|复诊|"
    r"还款|缴费|付款|申请|做|买|去|看|参加|准备|计划|打算|见"
)


class IntentDraft(BaseModel):
    """Runtime contract for one LLM-extracted intention."""

    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=2, max_length=120)
    detail: str = Field(default="", max_length=500)
    when_text: str = Field(default="", max_length=120)
    when_local: Optional[str] = None
    granularity: Literal["day", "datetime"] = "day"
    due_kind: Literal["start", "end", "day"] = "day"
    kind: Literal["event", "task", "deadline", "recurring"] = "task"
    importance: float = Field(default=0.6, ge=0.0, le=5.0)
    confidence: float = Field(default=0.6, ge=0.0, le=5.0)
    rrule: Optional[str] = Field(default=None, max_length=200)
    needs_confirmation: bool = False


class IntentBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intentions: List[IntentDraft] = Field(default_factory=list)


def _zone(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz or DEFAULT_TZ)
    except Exception:
        return ZoneInfo(DEFAULT_TZ)


def _aware(now: Optional[datetime], tz: str) -> datetime:
    now = now or datetime.now(_zone(tz))
    if now.tzinfo is None:
        return now.replace(tzinfo=_zone(tz))
    return now.astimezone(_zone(tz))


def _next_weekday(day: date, target: int, *, include_today: bool = True) -> date:
    delta = (target - day.weekday()) % 7
    if delta == 0 and not include_today:
        delta = 7
    return day + timedelta(days=delta)


def _add_months(day: date, months: int) -> date:
    month = day.month - 1 + months
    year = day.year + month // 12
    month = month % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


_CN_DIGITS = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _cn_int(token: str) -> Optional[int]:
    token = (token or "").strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    if token in _CN_DIGITS:
        return _CN_DIGITS[token]
    if token.startswith("十") and len(token) == 2 and token[1] in _CN_DIGITS:
        return 10 + _CN_DIGITS[token[1]]
    if token.endswith("十") and len(token) == 2 and token[0] in _CN_DIGITS:
        return _CN_DIGITS[token[0]] * 10
    if "十" in token and len(token) == 3:
        tens, ones = token.split("十")
        if tens in _CN_DIGITS and ones in _CN_DIGITS:
            return _CN_DIGITS[tens] * 10 + _CN_DIGITS[ones]
    return None


def _parse_clock(text: str, day: date, tz: str) -> Optional[datetime]:
    match = re.search(r"([一二三四五六七八九十两\d]{1,2})\s*[:：]\s*(\d{1,2})", text)
    if not match:
        match = re.search(r"([一二三四五六七八九十两\d]{1,2})\s*点\s*([一二三四五六七八九十两\d]{0,2})?\s*(半)?", text)
    if not match:
        return None
    hour = _cn_int(match.group(1))
    minute = _cn_int(match.group(2) or "") if match.group(2) else None
    if minute is None:
        minute = 30 if "半" in match.group(0) else 0
    if hour is None:
        return None
    if hour > 23 or minute > 59:
        return None
    if re.search(r"下午|晚上|傍晚|晚间", text) and hour < 12:
        hour += 12
    elif re.search(r"中午", text) and hour < 11:
        hour += 12
    elif re.search(r"凌晨", text) and hour == 12:
        hour = 0
    return datetime.combine(day, dt_time(hour, minute), tzinfo=_zone(tz))


def _has_time(text: str) -> bool:
    return bool(re.search(
        r"[一二三四五六七八九十两\d]{1,2}\s*[:：]\s*\d{1,2}|"
        r"[一二三四五六七八九十两\d]{1,2}\s*点",
        text,
    ))


def detect_recurrence(text: str, now: datetime, tz: str) -> Tuple[Optional[str], Optional[datetime]]:
    """Return (rrule, first_occurrence) for simple Chinese repeats."""
    match = re.search(r"每(?:周|星期|礼拜)([一二三四五六日天])", text)
    if match:
        target = _WEEKDAY_INDEX[match.group(1)]
        first = _next_weekday(now.date(), target, include_today=True)
        clock = _parse_clock(text, first, tz)
        if clock:
            return f"FREQ=WEEKLY;BYDAY={_WEEKDAY_CODE[target]}", clock
        return f"FREQ=WEEKLY;BYDAY={_WEEKDAY_CODE[target]}", datetime.combine(
            first, dt_time(0, 0), tzinfo=_zone(tz)
        )
    if "每天" in text:
        first = now.date()
        clock = _parse_clock(text, first, tz)
        if clock:
            return "FREQ=DAILY", clock
        return "FREQ=DAILY", datetime.combine(first, dt_time(0, 0), tzinfo=_zone(tz))
    return None, None


def resolve_when(text: str, now: Optional[datetime] = None, tz: str = DEFAULT_TZ) -> Tuple[Optional[datetime], str, Optional[str]]:
    """Resolve a Chinese time expression to (local datetime, granularity, rrule).

    The fallback is deliberately limited to common expressions. ``dateparser``
    is used only when the deterministic patterns fail.
    """
    tz = tz or DEFAULT_TZ
    now = _aware(now, tz)
    text = (text or "").strip()
    if not text:
        return None, "day", None

    rrule, first = detect_recurrence(text, now, tz)
    if first is not None:
        return first, ("datetime" if _has_time(text) else "day"), rrule

    match = re.search(r"大后天", text)
    if match:
        day = now.date() + timedelta(days=3)
        clock = _parse_clock(text, day, tz)
        return (clock or datetime.combine(day, dt_time(0, 0), tzinfo=_zone(tz)),
                "datetime" if clock else "day", None)
    match = re.search(r"后天", text)
    if match:
        day = now.date() + timedelta(days=2)
        clock = _parse_clock(text, day, tz)
        return (clock or datetime.combine(day, dt_time(0, 0), tzinfo=_zone(tz)),
                "datetime" if clock else "day", None)
    match = re.search(r"明天|明日", text)
    if match:
        day = now.date() + timedelta(days=1)
        clock = _parse_clock(text, day, tz)
        return (clock or datetime.combine(day, dt_time(0, 0), tzinfo=_zone(tz)),
                "datetime" if clock else "day", None)
    match = re.search(r"今天|今日", text)
    if match:
        clock = _parse_clock(text, now.date(), tz)
        return (clock or datetime.combine(now.date(), dt_time(0, 0), tzinfo=_zone(tz)),
                "datetime" if clock else "day", None)

    match = re.search(r"今晚", text)
    if match:
        return datetime.combine(now.date(), dt_time(20, 0), tzinfo=_zone(tz)), "datetime", None
    match = re.search(r"明早", text)
    if match:
        return datetime.combine(now.date() + timedelta(days=1), dt_time(8, 0), tzinfo=_zone(tz)), "datetime", None
    match = re.search(r"明晚", text)
    if match:
        return datetime.combine(now.date() + timedelta(days=1), dt_time(20, 0), tzinfo=_zone(tz)), "datetime", None

    match = re.search(r"下(?:周|星期|礼拜)([一二三四五六日天])", text)
    if match:
        target = _WEEKDAY_INDEX[match.group(1)]
        monday = now.date() - timedelta(days=now.weekday()) + timedelta(days=7)
        day = monday + timedelta(days=target)
        clock = _parse_clock(text, day, tz)
        return (clock or datetime.combine(day, dt_time(0, 0), tzinfo=_zone(tz)),
                "datetime" if clock else "day", None)
    match = re.search(r"(?:这|本)(?:周|星期|礼拜)([一二三四五六日天])", text)
    if match:
        target = _WEEKDAY_INDEX[match.group(1)]
        monday = now.date() - timedelta(days=now.weekday())
        day = monday + timedelta(days=target)
        if day < now.date():
            day += timedelta(days=7)
        clock = _parse_clock(text, day, tz)
        return (clock or datetime.combine(day, dt_time(0, 0), tzinfo=_zone(tz)),
                "datetime" if clock else "day", None)

    match = re.search(r"(\d+)\s*天\s*[后後]", text)
    if match:
        day = now.date() + timedelta(days=int(match.group(1)))
        clock = _parse_clock(text, day, tz)
        return (clock or datetime.combine(day, dt_time(0, 0), tzinfo=_zone(tz)),
                "datetime" if clock else "day", None)
    match = re.search(r"(\d+)\s*(?:周|星期)\s*[后後]", text)
    if match:
        day = now.date() + timedelta(days=7 * int(match.group(1)))
        clock = _parse_clock(text, day, tz)
        return (clock or datetime.combine(day, dt_time(0, 0), tzinfo=_zone(tz)),
                "datetime" if clock else "day", None)
    match = re.search(r"(\d+)\s*个?月\s*[后後]", text)
    if match:
        day = _add_months(now.date(), int(match.group(1)))
        clock = _parse_clock(text, day, tz)
        return (clock or datetime.combine(day, dt_time(0, 0), tzinfo=_zone(tz)),
                "datetime" if clock else "day", None)

    match = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]", text)
    if match:
        month, day_num = int(match.group(1)), int(match.group(2))
        year = now.year
        try:
            day = date(year, month, day_num)
        except ValueError:
            return None, "day", None
        if day < now.date():
            try:
                day = date(year + 1, month, day_num)
            except ValueError:
                return None, "day", None
        clock = _parse_clock(text, day, tz)
        return (clock or datetime.combine(day, dt_time(0, 0), tzinfo=_zone(tz)),
                "datetime" if clock else "day", None)

    if "月底" in text:
        last = calendar.monthrange(now.year, now.month)[1]
        day = date(now.year, now.month, last)
        if day < now.date():
            month = now.month + 1
            year = now.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            day = date(year, month, calendar.monthrange(year, month)[1])
        clock = _parse_clock(text, day, tz)
        return (clock or datetime.combine(day, dt_time(0, 0), tzinfo=_zone(tz)),
                "datetime" if clock else "day", None)
    if "下个月" in text:
        month = now.month + 1
        year = now.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        return datetime.combine(date(year, month, 1), dt_time(0, 0), tzinfo=_zone(tz)), "day", None

    match = re.search(r"(?:周|星期|礼拜)([一二三四五六日天])", text)
    if match:
        target = _WEEKDAY_INDEX[match.group(1)]
        day = _next_weekday(now.date(), target, include_today=True)
        clock = _parse_clock(text, day, tz)
        return (clock or datetime.combine(day, dt_time(0, 0), tzinfo=_zone(tz)),
                "datetime" if clock else "day", None)

    if "下周" in text:
        monday = now.date() - timedelta(days=now.weekday()) + timedelta(days=7)
        return datetime.combine(monday, dt_time(0, 0), tzinfo=_zone(tz)), "day", None

    if dateparser is not None:
        try:
            parsed = dateparser.parse(
                text,
                languages=["zh"],
                settings={
                    "RELATIVE_BASE": now.replace(tzinfo=None),
                    "TIMEZONE": tz,
                    "RETURN_AS_TIMEZONE_AWARE": False,
                },
            )
            if parsed:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=_zone(tz))
                return parsed, ("datetime" if _has_time(text) else "day"), None
        except Exception:
            logger.debug("dateparser failed for %r", text, exc_info=True)
    return None, "day", None


def _infer_due_kind(text: str) -> str:
    if any(word in text for word in _END_WORDS):
        return "end"
    if any(word in text for word in _START_WORDS):
        return "start"
    return "day"


def _lead_seconds(text: str, due_kind: str) -> int:
    if any(word in text for word in ("考试", "面试", "答辩", "机票", "签约", "手术")):
        return 14 * 86400
    if due_kind == "end":
        return 7 * 86400
    return 2 * 86400


def _grace_seconds(text: str, due_kind: str) -> int:
    if due_kind == "end" or any(word in text for word in ("考试", "面试", "答辩", "机票")):
        return 2 * 86400
    return 86400


def _infer_importance(text: str) -> float:
    high = ("必须", "一定", "截止", "考试", "面试", "机票", "违约", "答应",
            "报名", "还款", "手术", "复诊", "签约", "答辩", "别人")
    if any(word in text for word in high):
        return 0.85
    if any(word in text for word in ("要", "得", "交", "提交", "上课", "开会", "约", "安排", "提醒")):
        return 0.70
    return 0.50


def _strip_time(text: str) -> str:
    patterns = [
        r"大后天|后天|明天|明日|今天|今日|今晚|明早|明晚",
        r"下(?:周|星期|礼拜)[一二三四五六日天]",
        r"这(?:周|星期|礼拜)[一二三四五六日天]",
        r"本(?:周|星期|礼拜)[一二三四五六日天]",
        r"(?:周|星期|礼拜)[一二三四五六日天]",
        r"\d+\s*个?\s*(?:天|周|星期|月)\s*[后後]",
        r"\d{1,2}\s*月\s*\d{1,2}\s*[日号]",
        r"月底|下个月",
        r"(?:上午|下午|晚上|中午|凌晨|傍晚)",
        r"[一二三四五六七八九十两\d]{1,2}\s*[:：]\s*\d{1,2}",
        r"[一二三四五六七八九十两\d]{1,2}\s*点\s*[一二三四五六七八九十两\d]{0,2}\s*(?:分|半)?",
        r"截止|之前|以前",
    ]
    for pattern in patterns:
        text = re.sub(pattern, " ", text)
    return re.sub(r"\s+", " ", text).strip(" ，,。；;：:！!？?")


def _title_from_clause(clause: str) -> str:
    title = _strip_time(clause)
    title = re.sub(
        r"^(?:我|你|他|她|它|要|得|必须|需要|记得|提醒我|帮我记|别忘了|"
        r"安排|约了|计划|准备|打算|然后|所以|但是|不过|就|才)+",
        "", title,
    )
    title = re.sub(r"(?:了|吧|啊|呀|呢|哦)$", "", title).strip()
    return title or clause.strip()[:24]


def _clauses(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"[，,。；;！!？?\n]", text or "") if part.strip()]


def _first_clause_with_cue(text: str) -> str:
    for clause in _clauses(text):
        if TIME_CUE.search(clause):
            return clause
    return (text or "").strip()


def _scale_importance(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number > 1.0:
        number = number / 5.0
    return max(0.0, min(1.0, number))


class IntentExtractor:
    """Turn one user message into zero or more intention dicts."""

    def __init__(self, config: Any = None, llm: Any = None, tz: str = DEFAULT_TZ) -> None:
        self.config = config
        self.llm = llm
        self.tz = tz or getattr(config, "tz", DEFAULT_TZ) or DEFAULT_TZ

    def has_cue(self, text: str) -> bool:
        return bool(TIME_CUE.search(text or ""))

    def looks_like_intent(self, text: str) -> bool:
        """A time expression alone is not enough: "今天好累" is not a task."""
        return self.has_cue(text) and bool(_INTENT_WORDS.search(text or ""))

    def extract(
        self,
        user_msg: str,
        ai_reply: str = "",
        history_context: str = "",
        *,
        now: Optional[datetime] = None,
        user_id: str = "default",
    ) -> List[Dict[str, Any]]:
        text = (user_msg or "").strip()
        if not text or not self.looks_like_intent(text):
            return []
        now = _aware(now, self.tz)
        if self.llm is None:
            return self.rule_extract(text, now=now)
        prompt = self._build_prompt(text, ai_reply, history_context, now)
        try:
            batch = call_structured(
                self.llm, prompt, IntentBatch, retries=2, max_tokens=1200
            )
        except StructuredOutputError:
            logger.debug("intent extraction failed validation, using rules", exc_info=True)
            return self.rule_extract(text, now=now)
        except Exception:
            logger.debug("intent extraction LLM failed, using rules", exc_info=True)
            return self.rule_extract(text, now=now)
        out: List[Dict[str, Any]] = []
        for draft in batch.intentions:
            item = self._normalize(draft.model_dump(), now)
            if item:
                out.append(item)
        return out

    def rule_extract(self, text: str, *, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        now = _aware(now, self.tz)
        if not self.looks_like_intent(text):
            return []
        clause = _first_clause_with_cue(text)
        if not clause or _NEGATIVE.search(clause):
            return []
        due, granularity, rrule = resolve_when(clause, now, self.tz)
        if due is None:
            return []
        title = _title_from_clause(clause)
        if len(title) < 2:
            return []
        due_kind = _infer_due_kind(clause)
        return [{
            "title": title,
            "detail": "",
            "due_at": due.timestamp(),
            "due_kind": due_kind,
            "granularity": granularity,
            "kind": "recurring" if rrule else ("event" if due_kind == "start" else "task"),
            "importance": _infer_importance(clause),
            "confidence": 0.55,
            "lead_s": _lead_seconds(clause, due_kind),
            "grace_s": _grace_seconds(clause, due_kind),
            "rrule": rrule,
            "needs_confirmation": False,
            "source_text": text[:500],
        }]

    def extract_update(
        self,
        user_msg: str,
        *,
        now: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """Detect a completion / cancellation / reschedule, without an LLM."""
        now = _aware(now, self.tz)
        for clause in _clauses(user_msg):
            if _CANCEL.search(clause):
                return {"kind": "cancel", "title_hint": clause}
            if _COMPLETE.search(clause):
                return {"kind": "complete", "title_hint": clause}
            match = _RESCHEDULE.search(clause)
            if match:
                due, granularity, _ = resolve_when(match.group(1), now, self.tz)
                if due is not None:
                    return {
                        "kind": "reschedule",
                        "title_hint": clause,
                        "due_at": due.timestamp(),
                        "granularity": granularity,
                    }
        return None

    def _build_prompt(
        self,
        user_msg: str,
        ai_reply: str,
        history_context: str,
        now: datetime,
    ) -> str:
        history = (history_context or "").strip()[:1200]
        reply = (ai_reply or "").strip()[:400]
        return f"""You extract future intentions from one conversation turn.
Current local time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')} (timezone {self.tz})

Only extract an intention when ALL are true:
1. It is in the future (not "yesterday", not a past event).
2. It belongs to the user (not a third party, not the assistant's suggestion).
3. It is actionable: a task, event, deadline, or promise.
4. It is not a pure habit ("I usually study on Wednesday") and not a
   hypothetical ("if I had class...").

Rules:
- Do NOT calculate the date. Return the user's surface expression in
  `when_text`; the caller will resolve it.
- If the user gave a clear calendar date, you may also return `when_local`
  as ISO-8601 local time (without timezone).
- If the exact time is unknown, set `granularity` to "day"; do not invent 09:00.
- `importance` is 0..1 and means consequence if missed, not how often it was
  mentioned.
- If the task is ambiguous or has no clear time, set `needs_confirmation` true.

Conversation:
{history}
User: {user_msg}
Assistant: {reply}

Output JSON only:
{{
  "intentions": [
    {{
      "action": "short action in the user's language",
      "detail": "why / context, one short sentence",
      "when_text": "surface time expression",
      "when_local": "2026-10-24T09:00:00 or null",
      "granularity": "day|datetime",
      "due_kind": "start|end|day",
      "kind": "event|task|deadline|recurring",
      "importance": 0.0,
      "confidence": 0.0,
      "rrule": null,
      "needs_confirmation": false
    }}
  ]
}}"""

    def _normalize(self, raw: Dict[str, Any], now: datetime) -> Optional[Dict[str, Any]]:
        title = str(raw.get("action") or raw.get("title") or "").strip()
        if len(title) < 2:
            return None
        when_local = raw.get("when_local")
        due: Optional[datetime] = None
        granularity = str(raw.get("granularity") or "day")
        rrule = raw.get("rrule") or raw.get("recurrence")
        if when_local:
            try:
                due = datetime.fromisoformat(str(when_local))
                if due.tzinfo is None:
                    due = due.replace(tzinfo=_zone(self.tz))
                due = due.astimezone(_zone(self.tz))
                granularity = "datetime" if ("T" in str(when_local) or ":" in str(when_local)) else "day"
            except (TypeError, ValueError):
                due = None
        if due is None:
            due, resolved_granularity, resolved_rrule = resolve_when(
                str(raw.get("when_text") or raw.get("when") or ""), now, self.tz
            )
            granularity = str(raw.get("granularity") or resolved_granularity)
            rrule = rrule or resolved_rrule
        if due is None:
            return None
        if isinstance(rrule, str) and ("每周" in rrule or "每天" in rrule):
            rrule, first = detect_recurrence(rrule, now, self.tz)
            if first is not None:
                due = first
        due_kind = str(raw.get("due_kind") or _infer_due_kind(title))
        if granularity == "day":
            due = due.replace(hour=0, minute=0, second=0, microsecond=0)
        return {
            "title": title,
            "detail": str(raw.get("detail") or "")[:120],
            "due_at": due.timestamp(),
            "due_kind": due_kind,
            "granularity": granularity,
            "kind": str(raw.get("kind") or ("recurring" if rrule else "task")),
            "importance": _scale_importance(raw.get("importance"), _infer_importance(title)),
            "confidence": _scale_importance(raw.get("confidence"), 0.65),
            "lead_s": _lead_seconds(f"{title} {raw.get('detail') or ''}", due_kind),
            "grace_s": _grace_seconds(f"{title} {raw.get('detail') or ''}", due_kind),
            "rrule": rrule if isinstance(rrule, str) else None,
            "needs_confirmation": bool(raw.get("needs_confirmation", False)),
            "source_text": "",
        }
