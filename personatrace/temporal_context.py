"""Small temporal context for reply stance.

The point is not to infer emotions from a clock. It is to stop the model from
pretending a conversation is continuous when it is not: a reply after eight
hours should not start with "as we were saying". This is the same idea as the
initial skills_bot ``_analyze_rhythm``, but with explicit, testable buckets.

The host decides how to use the hint; this module only describes the gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

DEFAULT_TZ = "Asia/Shanghai"


def _zone(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz or DEFAULT_TZ)
    except Exception:
        return ZoneInfo(DEFAULT_TZ)


def _aware(value: Any, tz: str) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), _zone(tz))
    if value.tzinfo is None:
        return value.replace(tzinfo=_zone(tz))
    return value.astimezone(_zone(tz))


def time_period(hour: int) -> str:
    if 5 <= hour < 11:
        return "上午"
    if 11 <= hour < 13:
        return "中午"
    if 13 <= hour < 18:
        return "下午"
    if 18 <= hour < 23:
        return "晚间"
    return "深夜"


def classify_gap(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 120:
        return "same_flow"
    if seconds < 1800:
        return "short_break"
    if seconds < 6 * 3600:
        return "resumed"
    return "cold_start"


_GAP_STANCE = {
    "unknown": "没有上一轮时间；不要假设存在未说完的话题。",
    "same_flow": "承接上文，不要重新开场。",
    "short_break": "可以承接，但先确认话题是否还在原来的方向上。",
    "resumed": "不要假设连续性；先接当前这句，必要时一句轻确认。",
    "cold_start": "不要假设连续性；当作新的会话；除非用户主动提起，不要翻旧话题。",
}


@dataclass(frozen=True)
class TemporalContext:
    gap_seconds: Optional[float]
    gap_label: str
    time_period: str
    stance: str
    hint: str


def build_temporal_context(
    now: Optional[datetime] = None,
    last_turn_at: Optional[datetime] = None,
    *,
    tz: str = DEFAULT_TZ,
) -> TemporalContext:
    now = _aware(now, tz) or datetime.now(_zone(tz))
    last = _aware(last_turn_at, tz)
    gap = max(0.0, (now - last).total_seconds()) if last else None
    label = classify_gap(gap)
    period = time_period(now.hour)
    stance = _GAP_STANCE[label]

    if gap is None:
        gap_text = "首轮"
    elif gap < 60:
        gap_text = f"距上一轮 {int(gap)} 秒"
    elif gap < 3600:
        gap_text = f"距上一轮 {int(gap // 60)} 分钟"
    else:
        hours = int(gap // 3600)
        minutes = int((gap % 3600) // 60)
        gap_text = f"距上一轮 {hours} 小时" + (f" {minutes} 分" if minutes else "")

    late_note = "；深夜回复短一些，降低刺激" if period == "深夜" else ""
    hint = f"【对话节奏】{gap_text}，现在是{period}。{stance.rstrip('。')}{late_note}。"
    return TemporalContext(
        gap_seconds=gap, gap_label=label, time_period=period,
        stance=stance, hint=hint,
    )


def format_temporal_hint(
    now: Optional[datetime] = None,
    last_turn_at: Optional[datetime] = None,
    *,
    tz: str = DEFAULT_TZ,
) -> str:
    """Return the one-line hint, or "" when there is no previous turn."""
    if last_turn_at is None:
        return ""
    return build_temporal_context(now, last_turn_at, tz=tz).hint
