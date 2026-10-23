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
      recurring habit ("每天十点睡") — currently parse_due returns None for it
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


def parse_due(text: str, now: Optional[datetime] = None, tz: str = DEFAULT_TZ) -> Optional[TypedTime]:
    """Parse common Chinese time expressions into a typed slot.

    Returns ``None`` when there is no date, or when the expression is a habit
    (``每天十点睡``). Explicit past expressions return ``kind="past"`` so the
    caller can refuse to write them as deadlines.
    """
    tz = tz or DEFAULT_TZ
    now = _aware(now, tz)
    raw = (text or "").strip()
    if not raw:
        return None

    # Recurring habits are not deadlines. TypedTime has the field, but
    # parse_due intentionally returns None so they never enter the due bucket.
    if re.search(r"每(?:天|日|周|星期|礼拜|月)", raw):
        return None

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

    if "月底" in raw:
        day = date(now.year, now.month, calendar.monthrange(now.year, now.month)[1])
        clock = _parse_clock(raw, day, tz)
        return _typed(day, clock, now, raw)

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
