"""Query-side time facet: decide whether a query is asking about time, then
retrieve a small high-precision bucket.

This is not a second retriever. It is a conditional bypass:

    time intent?  → time bucket (≤2) ∪ vector results
    habit?        → no time bucket (the answer is a rhythm, not a deadline)
    otherwise     → no time bucket

The negative habit list is the important part. "平时几点睡" must not pull
"周五要演示" to the top just because both contain date-ish words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

DEFAULT_TZ = "Asia/Shanghai"

TIME_INTENT = re.compile(
    r"几号|哪天|什么时候|截止|deadline|ddl|日程|安排|到期|之前|之后"
    # English temporal expressions. Without these the facet never opens on an
    # English query, so "how many days between X and Y" gets no time bucket.
    r"|how many (?:day|week|month|year)s?"
    r"|when did|what date|which day|how long"
    r"|\bago\b|last (?:week|month|year|night)"
    r"|between .+ and |more recently|most recent"
    r"|\b(?:today|tomorrow|yesterday|tonight)\b"
    r"|\b(?:this|next|last)\s+(?:week|month|year)\b"
    r"|\b(?:on|by|before|after)\s+(?:mon|tue|wed|thu|fri|sat|sun)",
    re.IGNORECASE,
)
TIME_HABIT = re.compile(
    r"平时|每天|一般|习惯|通常|总是"
    r"|\bevery\s+(?:day|week|month|year|morning|night)\b"
    r"|\beveryday\b|\beach\s+day\b|\bdaily\b"
    r"|\busually\b|\bnormally\b|\balways\b|\btypically\b"
    r"|\btend\s+to\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TimeIntent:
    is_time: bool
    is_habit: bool
    matched: str = ""


def detect_time_intent(query: str) -> TimeIntent:
    query = query or ""
    habit = TIME_HABIT.search(query)
    if habit:
        return TimeIntent(is_time=False, is_habit=True, matched=habit.group(0))
    match = TIME_INTENT.search(query)
    if match:
        return TimeIntent(is_time=True, is_habit=False, matched=match.group(0))
    return TimeIntent(is_time=False, is_habit=False)


def has_time_intent(query: str) -> bool:
    return detect_time_intent(query).is_time


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


def _epoch(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return None


def _row_due(row: Dict[str, Any]) -> Optional[float]:
    return _epoch(row.get("due_at_ts") if "due_at_ts" in row else row.get("due_ts"))


def time_bucket(
    rows: Iterable[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    tz: str = DEFAULT_TZ,
    limit: int = 2,
) -> List[Dict[str, Any]]:
    """Return the high-precision time bucket.

    Only future/current ``due`` and ``range`` rows are considered. A row with
    no ``time_kind`` but a due timestamp is treated as legacy ``due`` data.
    Rows older than one day are excluded: they are narrative, not a deadline.
    """
    now_dt = _aware(now, tz)
    now_ts = now_dt.timestamp()
    candidates = []
    for row in rows:
        kind = row.get("time_kind")
        due = _row_due(row)
        if due is None:
            continue
        if kind in ("past", "recurring"):
            continue
        # Old v2 rows have a due timestamp but no typed kind; treat them as due.
        if kind in (None, "", "none"):
            kind = "due"
        if kind not in ("due", "range"):
            continue
        if due < now_ts - 86400:
            continue
        candidates.append((due, row))
    candidates.sort(key=lambda item: (item[0] < now_ts, abs(item[0] - now_ts), str(item[1].get("label", ""))))
    return [row for _, row in candidates[: max(0, int(limit))]]


def humanize_due(
    due: Any,
    *,
    now: Optional[datetime] = None,
    tz: str = DEFAULT_TZ,
    kind: str = "due",
    raw: str = "",
    lang: str = "zh",
) -> str:
    """Absolute date first, plus a relative label and a weekday.

    The absolute date is what the model can answer with; "还有 0.1 天" is not.
    """
    due_ts = _epoch(due)
    if due_ts is None:
        return ""
    now_dt = _aware(now, tz)
    local = datetime.fromtimestamp(due_ts, _zone(tz))
    now_local = now_dt
    delta_days = (local.date() - now_local.date()).days
    date_text = local.strftime("%Y-%m-%d")
    if (lang or "zh").lower().startswith("en"):
        weekday_en = local.strftime("%A")
        if kind == "range" and raw:
            return f"by {raw} ({date_text}, {weekday_en})"
        if delta_days == 0:
            return f"due today ({date_text}, {weekday_en})"
        if delta_days == 1:
            return f"due tomorrow ({date_text}, {weekday_en})"
        if delta_days == 2:
            return f"due in 2 days ({date_text}, {weekday_en})"
        if delta_days < 0:
            return f"overdue {abs(delta_days)} days ({date_text}, {weekday_en})"
        return f"due {date_text} ({weekday_en}, in {delta_days} days)"

    weekday = "周" + "一二三四五六日"[local.weekday()]
    if kind == "range" and raw:
        return f"截止 {raw}（{date_text}，{weekday}）"
    if delta_days == 0:
        return f"截止 今天（{date_text}，{weekday}）"
    if delta_days == 1:
        return f"截止 明天（{date_text}，{weekday}）"
    if delta_days == 2:
        return f"截止 后天（{date_text}，{weekday}）"
    if delta_days < 0:
        return f"已过期 {abs(delta_days)} 天（{date_text}，{weekday}）"
    return f"截止 {date_text}（{weekday}，还有 {delta_days} 天）"


def render_time_line(
    row: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
    tz: str = DEFAULT_TZ,
    lang: str = "zh",
) -> str:
    due = _row_due(row)
    label = str(row.get("label", "")).strip()
    if due is None or not label:
        return ""
    human = humanize_due(
        due, now=now, tz=tz,
        kind=str(row.get("time_kind") or "due"),
        raw=str(row.get("time_raw") or ""),
        lang=lang,
    )
    return f"- {label} — {human}" if human else f"- {label}"


def time_bucket_lines(
    rows: Iterable[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    tz: str = DEFAULT_TZ,
    limit: int = 2,
    lang: str = "zh",
) -> List[str]:
    return [
        line for row in time_bucket(rows, now=now, tz=tz, limit=limit)
        if (line := render_time_line(row, now=now, tz=tz, lang=lang))
    ]
