"""v1 × v2 merged data contract.

This module is deliberately storage-free. It answers four questions:

1. Which 17 categories survive the merge, and which track do they belong to?
2. What does a valid incoming belief look like (Pydantic)?
3. How do we clean model output before it reaches the store?
4. How do we turn Chinese time expressions into a typed time slot?

The answer to (4) is the missing piece from the original v2 kernel: a single
``due_at_ts`` float cannot distinguish a future deadline from a past event or
a recurring habit.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from typing import Any, Dict, Literal, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_TZ = "Asia/Shanghai"

# ---------------------------------------------------------------------------
# 17 categories = v2's 9 + 8 promoted from v1.
# Logs (observation_log / uncertain / low_confidence / timeline / analysis_log
# / corrections) are intentionally not beliefs: they must not enter retention
# or pinned logic.
# ---------------------------------------------------------------------------

WORK_CATEGORIES = (
    "explicit_facts", "expertise", "projects", "goals",
    "likes", "dislikes", "constraints",
)
PERSONA_CATEGORIES = (
    "identity", "traits", "interests", "style_notes",
    "behavior_patterns", "decision_style", "interpersonal", "boundaries",
    "habit", "state",
)
ALL_CATEGORIES = WORK_CATEGORIES + PERSONA_CATEGORIES

CATEGORY_TRACK: Dict[str, str] = {
    **{category: "work" for category in WORK_CATEGORIES},
    **{category: "persona" for category in PERSONA_CATEGORIES},
}

# Initial stability in days. The merged layer updates MemoryV2's
# CATEGORY_STABILITY from this table at runtime; the kernel itself stays
# unchanged.
STABILITY_DAYS: Dict[str, float] = {
    "explicit_facts": 520.0,
    "identity": 300.0,
    "constraints": 180.0,
    "expertise": 240.0,
    "likes": 90.0,
    "dislikes": 90.0,
    "projects": 60.0,
    "goals": 45.0,
    "state": 10.0,
    "traits": 300.0,
    "interests": 120.0,
    "style_notes": 180.0,
    "behavior_patterns": 120.0,
    "decision_style": 240.0,
    "interpersonal": 180.0,
    "boundaries": 365.0,
    "habit": 90.0,
}

PINNED_CATEGORIES = ("boundaries", "constraints", "identity")
FORWARD_CATS = ("projects", "goals", "constraints")

# Common model spellings. Unknown categories are dropped rather than guessed:
# a wrong category poisons the id, stability and pinned behaviour at once.
CATEGORY_ALIASES: Dict[str, str] = {
    "preference": "likes",
    "preferences": "likes",
    "like": "likes",
    "dislike": "dislikes",
    "dislikes": "dislikes",
    "fact": "explicit_facts",
    "facts": "explicit_facts",
    "explicit_fact": "explicit_facts",
    "constraint": "constraints",
    "project": "projects",
    "goal": "goals",
    "expertise": "expertise",
    "skill": "expertise",
    "skills": "expertise",
    "trait": "traits",
    "interest": "interests",
    "style": "style_notes",
    "style_note": "style_notes",
    "behavior": "behavior_patterns",
    "behaviour": "behavior_patterns",
    "behavior_pattern": "behavior_patterns",
    "decision": "decision_style",
    "relationship": "interpersonal",
    "boundary": "boundaries",
    "habit": "habit",
    "routine": "habit",
    "mood": "state",
    "status": "state",
}


def normalize_category(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if raw in CATEGORY_TRACK:
        return raw
    if raw in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[raw]
    # "preferences" → "preference" → "likes"; also tolerate plural "s".
    singular = raw[:-1] if raw.endswith("s") else raw
    if singular in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[singular]
    if singular in CATEGORY_TRACK:
        return singular
    return None


def track_for(category: str) -> str:
    try:
        return CATEGORY_TRACK[category]
    except KeyError as exc:
        raise ValueError(f"unknown category: {category!r}") from exc


# ---------------------------------------------------------------------------
# Typed time
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TypedTime:
    """A time slot with a kind, not just a timestamp.

    kind:
      due       future deadline/event
      past      explicit past event ("上周五去了海边")
      recurring habit ("每天十点睡") — typed as recurring, never a deadline
      range     fuzzy window ("年底前换工作")
    """

    kind: str
    raw: str = ""
    at: Optional[float] = None
    start: Optional[float] = None
    end: Optional[float] = None
    confidence: float = 1.0

    @property
    def due_ts(self) -> Optional[float]:
        return self.at if self.at is not None else self.end


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


def _end_of_day(day: date, tz: str) -> float:
    return datetime.combine(day, dt_time(23, 59, 59), tzinfo=_zone(tz)).timestamp()


def _cn_int(token: str) -> Optional[int]:
    token = (token or "").strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if token in digits:
        return digits[token]
    if token.startswith("十") and len(token) == 2 and token[1] in digits:
        return 10 + digits[token[1]]
    if token.endswith("十") and len(token) == 2 and token[0] in digits:
        return digits[token[0]] * 10
    if "十" in token and len(token) == 3:
        tens, ones = token.split("十")
        if tens in digits and ones in digits:
            return digits[tens] * 10 + digits[ones]
    return None


def _parse_clock(text: str, day: date, tz: str) -> Optional[datetime]:
    match = re.search(r"([一二三四五六七八九十两\d]{1,2})\s*[:：]\s*(\d{1,2})", text)
    if not match:
        match = re.search(
            r"([一二三四五六七八九十两\d]{1,2})\s*点\s*"
            r"([一二三四五六七八九十两\d]{0,2})?\s*(半)?",
            text,
        )
    if not match:
        return None
    hour = _cn_int(match.group(1))
    minute = _cn_int(match.group(2) or "") if match.group(2) else None
    if minute is None:
        minute = 30 if "半" in match.group(0) else 0
    if hour is None or hour > 23 or minute > 59:
        return None
    if re.search(r"下午|晚上|傍晚|晚间", text) and hour < 12:
        hour += 12
    elif re.search(r"中午", text) and hour < 11:
        hour += 12
    elif re.search(r"凌晨", text) and hour == 12:
        hour = 0
    return datetime.combine(day, dt_time(hour, minute), tzinfo=_zone(tz))


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


_WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}


def _typed(day: date, clock: Optional[datetime], now: datetime, raw: str,
           tz: Optional[str] = None, *, force_past: bool = False) -> TypedTime:
    tz = tz or getattr(now.tzinfo, "key", None) or DEFAULT_TZ
    if clock is not None:
        at = clock.timestamp()
    else:
        at = _end_of_day(day, tz)
    past = force_past or at < now.timestamp()
    return TypedTime(
        kind="past" if past else "due",
        raw=raw,
        at=at,
        confidence=1.0 if clock is not None else 0.9,
    )


def _at(moment: datetime, raw: str, now: datetime) -> TypedTime:
    """A time slot with sub-day precision ("in 2 hours", "两小时后").

    ``_typed`` always starts from a date and an optional clock, which cannot
    express an offset of a few hours. This keeps the same kind contract.
    """
    at = moment.timestamp()
    return TypedTime(
        kind="past" if at < now.timestamp() else "due",
        raw=raw, at=at, confidence=1.0,
    )


_EN_WEEKDAYS = {    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}
_EN_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_EN_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _en_number(text: str) -> Optional[int]:
    text = (text or "").strip().lower()
    if text.isdigit():
        return int(text)
    return _EN_NUMBERS.get(text)


def _en_clock(text: str, day: date, tz: str) -> Optional[datetime]:
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text, re.IGNORECASE)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return None
    if match.group(3).lower() == "pm" and hour < 12:
        hour += 12
    if match.group(3).lower() == "am" and hour == 12:
        hour = 0
    return datetime.combine(day, dt_time(hour, minute), tzinfo=_zone(tz))


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _parse_english_due(raw: str, now: datetime, tz: str) -> Optional[TypedTime]:
    text = re.sub(r"\s+", " ", raw.lower()).strip(" .,!?")
    if not text:
        return None

    # Recurring expressions are routines, not deadlines.
    if re.search(r"\b(?:every|each)\s+(?:day|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text):
        return None

    def make(day: date, force_past: bool = False) -> TypedTime:
        clock = _en_clock(text, day, tz)
        return _typed(day, clock, now, raw, tz, force_past=force_past)

    if "day after tomorrow" in text:
        return make(now.date() + timedelta(days=2))
    if "day before yesterday" in text:
        return make(now.date() - timedelta(days=2), force_past=True)
    for word, offset in (("today", 0), ("tomorrow", 1), ("yesterday", -1)):
        if re.search(rf"\b{word}\b", text):
            return make(now.date() + timedelta(days=offset), force_past=offset < 0)
    if re.search(r"\btonight\b", text):
        return make(now.date())
    if re.search(r"\blast night\b", text):
        return make(now.date() - timedelta(days=1), force_past=True)

    match = re.search(r"\b(?:in|after)\s+([a-z]+|\d+)\s+(day|week|month|year)s?\b", text)
    if match:
        count = _en_number(match.group(1))
        if count is None:
            return None
        unit = match.group(2)
        if unit == "day":
            return make(now.date() + timedelta(days=count))
        if unit == "week":
            return make(now.date() + timedelta(days=7 * count))
        if unit == "month":
            return make(_add_months(now.date(), count))
        return make(_add_months(now.date(), 12 * count))

    match = re.search(r"\b([a-z]+|\d+)\s+(day|week|month|year)s?\s+(ago|from now|later)\b", text)
    if match:
        count = _en_number(match.group(1))
        if count is None:
            return None
        unit = match.group(2)
        back = match.group(3) == "ago"
        if unit == "day":
            day = now.date() - timedelta(days=count) if back else now.date() + timedelta(days=count)
        elif unit == "week":
            day = now.date() - timedelta(days=7 * count) if back else now.date() + timedelta(days=7 * count)
        elif unit == "month":
            day = _add_months(now.date(), -count if back else count)
        else:
            day = _add_months(now.date(), -12 * count if back else 12 * count)
        return make(day, force_past=back)

    # Sub-day offsets. "in 2 hours" and "2 hours later" are both common, and a
    # missed parse drops the whole intention, so both spellings are handled.
    if "half an hour" in text or "half hour" in text:
        return _at(now + timedelta(minutes=30), raw, now)
    match = re.search(r"\b(?:in|after)\s+([a-z]+|\d+)\s+(hour|minute)s?\b", text)
    if match:
        count = _en_number(match.group(1))
        if count is None:
            return None
        delta = timedelta(hours=count) if match.group(2) == "hour" else timedelta(minutes=count)
        return _at(now + delta, raw, now)
    match = re.search(r"\b([a-z]+|\d+)\s+(hour|minute)s?\s+(?:later|from now)\b", text)
    if match:
        count = _en_number(match.group(1))
        if count is None:
            return None
        delta = timedelta(hours=count) if match.group(2) == "hour" else timedelta(minutes=count)
        return _at(now + delta, raw, now)

    match = re.search(r"\b(next|last|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b", text)
    if match:
        prefix, name = match.group(1), match.group(2)
        target = _EN_WEEKDAYS[name]
        if prefix == "next":
            day = _next_weekday(now.date(), target, include_today=False)
        elif prefix == "last":
            day = now.date() - timedelta(days=((now.weekday() - target - 1) % 7) + 1)
        else:
            monday = now.date() - timedelta(days=now.weekday())
            day = monday + timedelta(days=target)
        return make(day, force_past=prefix == "last")

    match = re.search(r"\b(?:on|by|before|after)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b", text)
    if match:
        return make(_next_weekday(now.date(), _EN_WEEKDAYS[match.group(1)], include_today=True))

    match = re.fullmatch(r"(?:on\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)", text)
    if match:
        return make(_next_weekday(now.date(), _EN_WEEKDAYS[match.group(1)], include_today=True))

    match = re.search(r"\b(next|last|this)\s+weekend\b", text)
    if match:
        prefix = match.group(1)
        monday = now.date() - timedelta(days=now.weekday())
        shift = {"next": 7, "last": -7, "this": 0}[prefix]
        return make(monday + timedelta(days=6 + shift), force_past=prefix == "last")

    match = re.search(r"\b(next|last|this)\s+(week|month|year)\b", text)
    if match:
        prefix, unit = match.group(1), match.group(2)
        if unit == "week":
            monday = now.date() - timedelta(days=now.weekday())
            shift = {"next": 7, "last": -7, "this": 0}[prefix]
            day = monday + timedelta(days=6 + shift)
        elif unit == "month":
            shift = {"next": 1, "last": -1, "this": 0}[prefix]
            month_index = now.month - 1 + shift
            year = now.year + month_index // 12
            month = month_index % 12 + 1
            day = _month_end(year, month)
        else:
            year = now.year + {"next": 1, "last": -1, "this": 0}[prefix]
            day = date(year, 12, 31)
        return make(day, force_past=prefix == "last")

    if re.search(r"\bend of (?:the )?(?:month|year)\b", text):
        if "year" in text:
            return make(date(now.year, 12, 31))
        return make(_month_end(now.year, now.month))

    match = re.search(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b",
        text,
    )
    if match:
        return make(_next_weekday(now.date(), _EN_WEEKDAYS[match.group(1)], include_today=True))

    match = re.fullmatch(
        r"([a-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?", text, re.IGNORECASE,
    )
    if match and match.group(1).lower() in _EN_MONTHS:
        month = _EN_MONTHS[match.group(1).lower()]
        year = int(match.group(3) or now.year)
        try:
            return make(date(year, month, int(match.group(2))))
        except ValueError:
            return None
    return None


_ISO_DUE = re.compile(
    r"(\d{4})-(\d{1,2})-(\d{1,2})"
    r"(?:[T ](\d{1,2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?"
    r"(Z|[+-]\d{2}:?\d{2})?)?"
)


def _parse_iso_due(raw: str, now: datetime, tz: str) -> Optional[TypedTime]:
    """Parse an ISO date/datetime before the natural-language parsers.

    The old order let English month/weekday rules run first. That is harmless
    for a bare ``2026-10-25`` but turns strings such as
    ``2026-10-25T19:00:00+08:00`` into ``None``. ISO is the machine format, so
    it gets a strict full-string match and the correct offset.
    """
    match = _ISO_DUE.fullmatch(raw)
    if not match:
        return None
    try:
        day = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None

    clock: Optional[datetime] = None
    if match.group(4) is not None:
        try:
            clock = datetime(
                day.year, day.month, day.day,
                int(match.group(4)), int(match.group(5)),
                int(match.group(6) or 0),
                tzinfo=_zone(tz),
            )
        except ValueError:
            return None
        offset = match.group(7)
        if offset:
            # A supplied offset wins; then convert to the caller's zone so the
            # local calendar date and the clock are consistent.
            parsed = datetime.fromisoformat(
                raw.replace("Z", "+00:00")
            )
            clock = parsed.astimezone(_zone(tz))
            day = clock.date()
    return _typed(day, clock, now, raw, tz)


def parse_due(text: str, now: Optional[datetime] = None, tz: str = DEFAULT_TZ) -> Optional[TypedTime]:
    """Parse common Chinese and English time expressions into a typed slot.

    Returns ``None`` when there is no date. Habit expressions
    (``每天十点睡``) return ``kind="recurring"`` with no due timestamp; explicit
    past expressions return ``kind="past"``. Callers that write deadlines
    refuse both kinds.
    """
    tz = tz or DEFAULT_TZ
    now = _aware(now, tz)
    raw = (text or "").strip()
    if not raw:
        return None

    # Recurring habits are a typed category of their own, not deadlines.
    # Returning the type makes `recurring` reachable in the public contract;
    # due_for_write and IntentStore still refuse to turn it into a deadline.
    if re.search(r"每(?:天|日|周|星期|礼拜|月|年)", raw):
        return TypedTime(kind="recurring", raw=raw, confidence=0.8)
    if re.search(
        r"\b(?:every|each)\s+(?:day|week|month|year|morning|night|monday|tuesday|"
        r"wednesday|thursday|friday|saturday|sunday)\b",
        raw, re.IGNORECASE,
    ):
        return TypedTime(kind="recurring", raw=raw, confidence=0.8)

    iso_due = _parse_iso_due(raw, now, tz)
    if iso_due is not None:
        return iso_due

    # Explicit Chinese year: "2029年3月1日". The month/day parser below
    # deliberately rolls a past date into the future, which must never be
    # allowed to discard an explicitly written year.
    year_match = re.search(
        r"(?<!\d)(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*(?:日|号)?",
        raw,
    )
    if year_match:
        try:
            day = date(int(year_match.group(1)), int(year_match.group(2)),
                       int(year_match.group(3)))
        except ValueError:
            return None
        clock = _parse_clock(raw, day, tz)
        return _typed(day, clock, now, raw, tz)

    english = _parse_english_due(raw, now, tz)
    if english is not None:
        return english

    # Weekends are ranges, not a single day. Check longest prefix first:
    # "下下周末" contains the substring "下周末".
    for token, weeks in (("下下周末", 2), ("下周末", 1), ("这周末", 0), ("本周末", 0)):
        if token in raw:
            monday = now.date() - timedelta(days=now.weekday()) + timedelta(days=7 * weeks)
            start = datetime.combine(monday + timedelta(days=5), dt_time(0, 0), tzinfo=_zone(tz))
            end = datetime.combine(monday + timedelta(days=6), dt_time(23, 59, 59), tzinfo=_zone(tz))
            return TypedTime(
                kind="past" if end.timestamp() < now.timestamp() else "due",
                raw=raw, at=end.timestamp(),
                start=start.timestamp(), end=end.timestamp(),
                confidence=0.8,
            )

    # "下下周三" must not be read as "下周三" (the substring starts later).
    match = re.search(r"下下周([一二三四五六日天])", raw)
    if match:
        monday = now.date() - timedelta(days=now.weekday()) + timedelta(days=14)
        day = monday + timedelta(days=_WEEKDAY[match.group(1)])
        clock = _parse_clock(raw, day, tz)
        return _typed(day, clock, now, raw, tz)

    match = re.search(r"上周([一二三四五六日天])", raw)
    if match:
        monday = now.date() - timedelta(days=now.weekday()) - timedelta(days=7)
        day = monday + timedelta(days=_WEEKDAY[match.group(1)])
        clock = _parse_clock(raw, day, tz)
        return _typed(day, clock, now, raw, tz, force_past=True)

    match = re.search(r"下周([一二三四五六日天])", raw)
    if match:
        monday = now.date() - timedelta(days=now.weekday()) + timedelta(days=7)
        day = monday + timedelta(days=_WEEKDAY[match.group(1)])
        clock = _parse_clock(raw, day, tz)
        return _typed(day, clock, now, raw)

    match = re.search(r"(?:这|本)(?:周|星期|礼拜)([一二三四五六日天])", raw)
    if match:
        monday = now.date() - timedelta(days=now.weekday())
        day = monday + timedelta(days=_WEEKDAY[match.group(1)])
        clock = _parse_clock(raw, day, tz)
        if day < now.date():
            return _typed(day, clock, now, raw, force_past=True)
        return _typed(day, clock, now, raw)

    match = re.search(r"(?:周|星期|礼拜)([一二三四五六日天])", raw)
    if match:
        day = _next_weekday(now.date(), _WEEKDAY[match.group(1)], include_today=True)
        clock = _parse_clock(raw, day, tz)
        return _typed(day, clock, now, raw)

    for word, offset in (("大后天", 3), ("后天", 2), ("明天", 1), ("明日", 1), ("今天", 0), ("今日", 0)):
        if word in raw:
            day = now.date() + timedelta(days=offset)
            clock = _parse_clock(raw, day, tz)
            return _typed(day, clock, now, raw)

    if "今晚" in raw:
        clock = datetime.combine(now.date(), dt_time(23, 59, 59), tzinfo=_zone(tz))
        return _typed(now.date(), clock, now, raw)
    if "明早" in raw:
        clock = datetime.combine(now.date() + timedelta(days=1), dt_time(8, 0), tzinfo=_zone(tz))
        return _typed(now.date() + timedelta(days=1), clock, now, raw)
    if "明晚" in raw:
        clock = datetime.combine(now.date() + timedelta(days=1), dt_time(23, 59, 59), tzinfo=_zone(tz))
        return _typed(now.date() + timedelta(days=1), clock, now, raw)

    # Sub-day offsets come before the day/week/month rules so "两小时后" is not
    # swallowed by a coarser match. "半小时" has no numeric token.
    match = re.search(r"([一二三四五六七八九十两半\d]+)\s*个?\s*(?:小时|钟头)\s*[后後]", raw)
    if match:
        token = match.group(1)
        if token == "半":
            hours = 0.5
        else:
            count = _cn_int(token)
            if count is None:
                return None
            hours = float(count)
        return _at(now + timedelta(hours=hours), raw, now)

    match = re.search(r"([一二三四五六七八九十两\d]+)\s*天\s*[后後]", raw)
    if match:
        count = _cn_int(match.group(1))
        if count is None:
            return None
        day = now.date() + timedelta(days=count)
        clock = _parse_clock(raw, day, tz)
        return _typed(day, clock, now, raw)
    match = re.search(r"([一二三四五六七八九十两\d]+)\s*(?:周|星期)\s*[后後]", raw)
    if match:
        count = _cn_int(match.group(1))
        if count is None:
            return None
        day = now.date() + timedelta(days=7 * count)
        clock = _parse_clock(raw, day, tz)
        return _typed(day, clock, now, raw)
    match = re.search(r"([一二三四五六七八九十两\d]+)\s*个?月\s*[后後]", raw)
    if match:
        count = _cn_int(match.group(1))
        if count is None:
            return None
        day = _add_months(now.date(), count)
        clock = _parse_clock(raw, day, tz)
        return _typed(day, clock, now, raw)

    match = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]", raw)
    if match:
        month, day_num = int(match.group(1)), int(match.group(2))
        try:
            day = date(now.year, month, day_num)
        except ValueError:
            return None
        if day < now.date():
            try:
                day = date(now.year + 1, month, day_num)
            except ValueError:
                return None
        clock = _parse_clock(raw, day, tz)
        return _typed(day, clock, now, raw)

    # Month-end must resolve the month named by the expression. The old code
    # checked the bare substring "月底" first, so "下个月底" silently became
    # *this* month's end.
    for prefix, offset in (
        ("下下个月", 2), ("下个月", 1), ("下下月", 2), ("下月", 1),
        ("这个月", 0), ("本月", 0),
    ):
        if prefix + "底" in raw or prefix + "末" in raw:
            month_index = now.month - 1 + offset
            year = now.year + month_index // 12
            month = month_index % 12 + 1
            day = date(year, month, calendar.monthrange(year, month)[1])
            clock = _parse_clock(raw, day, tz)
            return _typed(day, clock, now, raw, tz)

    if "月底" in raw:
        day = date(now.year, now.month, calendar.monthrange(now.year, now.month)[1])
        clock = _parse_clock(raw, day, tz)
        return _typed(day, clock, now, raw, tz)

    if "年底前" in raw or "年底" in raw:
        end = datetime.combine(date(now.year, 12, 31), dt_time(23, 59, 59), tzinfo=_zone(tz)).timestamp()
        if end < now.timestamp() - 86400:
            end = datetime.combine(date(now.year + 1, 12, 31), dt_time(23, 59, 59), tzinfo=_zone(tz)).timestamp()
        return TypedTime(kind="range", raw=raw, at=end, start=now.timestamp(), end=end, confidence=0.8)

    if "下个月" in raw:
        month = now.month + 1
        year = now.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        start = datetime.combine(date(year, month, 1), dt_time(0, 0), tzinfo=_zone(tz)).timestamp()
        end = datetime.combine(
            date(year, month, calendar.monthrange(year, month)[1]),
            dt_time(23, 59, 59), tzinfo=_zone(tz),
        ).timestamp()
        return TypedTime(kind="range", raw=raw, at=end, start=start, end=end, confidence=0.6)

    if "下下周" in raw:
        monday = now.date() - timedelta(days=now.weekday()) + timedelta(days=14)
        return _typed(monday, None, now, raw)

    if "下周" in raw:
        monday = now.date() - timedelta(days=now.weekday()) + timedelta(days=7)
        return _typed(monday, None, now, raw)

    return None


# ---------------------------------------------------------------------------
# Incoming belief contract
# ---------------------------------------------------------------------------

class BeliefIn(BaseModel):
    """One belief as it is allowed to enter the merged store."""

    model_config = ConfigDict(extra="forbid")

    category: str
    label: str = Field(min_length=2, max_length=80)
    confidence: float = Field(default=0.6, ge=0.0, le=100.0)
    track: Optional[Literal["work", "persona"]] = None
    reasoning: str = Field(default="", max_length=2000)
    evidence_text: str = Field(default="", max_length=4000)
    due: Optional[str] = Field(default=None, max_length=80)

    @field_validator("category")
    @classmethod
    def _valid_category(cls, value: str) -> str:
        normalized = normalize_category(value)
        if normalized is None:
            raise ValueError(f"unknown category: {value!r}")
        return normalized

    @field_validator("label")
    @classmethod
    def _label(cls, value: str) -> str:
        value = (value or "").strip()
        if len(value) < 2:
            raise ValueError("label must be at least 2 characters")
        return value

    @field_validator("confidence")
    @classmethod
    def _confidence(cls, value: float) -> float:
        # Accept both 0..1 and percent/1..5 scales, but store 0..1.
        number = float(value)
        if number > 1.0:
            if number <= 5.0:
                number = 1.0
            elif number <= 100.0:
                number /= 100.0
            else:
                number = 1.0
        return max(0.0, min(1.0, number))

    @field_validator("due")
    @classmethod
    def _due(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    @model_validator(mode="after")
    def _infer_track(self) -> "BeliefIn":
        self.track = CATEGORY_TRACK[self.category]  # type: ignore[assignment]
        return self


class ExtractionBatch(BaseModel):
    """LLM output shape for one write turn."""

    model_config = ConfigDict(extra="forbid")
    beliefs: list[BeliefIn] = Field(default_factory=list)


def coerce_in(raw: Any, *, default_category: Optional[str] = None) -> Optional[BeliefIn]:
    """Clean one model-produced item; return None when it cannot be trusted.

    Deliberately strict about category: a wrong category changes the primary
    key, stability curve and pinned behaviour. Dropping one item is cheaper
    than poisoning all three.
    """
    if isinstance(raw, BeliefIn):
        return raw
    if isinstance(raw, str):
        category = normalize_category(default_category)
        if category is None:
            return None
        try:
            return BeliefIn(category=category, label=raw)
        except Exception:
            return None
    if not isinstance(raw, dict):
        return None

    category = normalize_category(
        raw.get("category") or raw.get("type") or raw.get("dim") or default_category
    )
    if category is None:
        return None

    label = raw.get("label") or raw.get("text") or raw.get("content") or raw.get("fact")
    if isinstance(label, dict):
        label = label.get("label") or label.get("text")
    if label is None:
        return None

    confidence = raw.get("confidence", 0.6)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.6
    if confidence > 1.0:
        # 85 → 0.85; 1.5 is a malformed 0..1 score, so clamp it, not divide it.
        if confidence <= 5.0:
            confidence = 1.0
        elif confidence <= 100.0:
            confidence = confidence / 100.0
        else:
            confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))

    due = raw.get("due") or raw.get("deadline") or raw.get("due_text") or raw.get("time")
    if due is not None:
        due = str(due).strip()

    try:
        return BeliefIn(
            category=category,
            label=str(label),
            confidence=confidence,
            reasoning=str(raw.get("reasoning") or raw.get("reason") or "")[:2000],
            evidence_text=str(raw.get("evidence_text") or raw.get("evidence") or "")[:4000],
            due=due or None,
        )
    except Exception:
        return None


def due_for_write(
    belief: BeliefIn,
    now: Optional[datetime] = None,
    tz: str = DEFAULT_TZ,
) -> Optional[TypedTime]:
    """Resolve the time slot that may be written for a belief.

    - explicit ``due`` is trusted regardless of category;
    - otherwise only FORWARD_CATS labels are parsed;
    - past expressions are refused (a narrative must not become a deadline).
    """
    now_dt = _aware(now, tz)
    source = belief.due
    if not source and belief.category in FORWARD_CATS:
        source = belief.label
    if not source:
        return None
    typed = parse_due(source, now_dt, tz)
    if typed is None:
        return None
    if typed.kind in ("past", "recurring"):
        return None
    return typed
