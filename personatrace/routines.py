"""Weekly routine memory and in-turn probe state.

A routine is not a belief and not a one-off intention. It is a recurring local
schedule: action + weekday(s) + local time + interval + optional term/validity.

The engine never sends messages. It returns a ``Probe`` candidate when the host
is already handling a user turn; the host decides whether to ask. Confirmation,
denial, today-only skip and silence are explicit feedback events.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence
from zoneinfo import ZoneInfo

DEFAULT_TZ = "Asia/Shanghai"
ALLOWED_STATUSES = frozenset(
    {"candidate", "confirmed", "trusted", "paused", "dormant", "ended"}
)
ALLOWED_FEEDBACK = frozenset(
    {"confirmed", "denied", "skipped_today", "unanswered", "ended"}
)
PROBE_WINDOW_MIN = 30
PAUSE_DAYS = 14
DORMANT_DAYS = 30
STALE_CONFIRM_DAYS = 28


def _zone(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz or DEFAULT_TZ)
    except Exception:
        return ZoneInfo(DEFAULT_TZ)


def _as_datetime(value: Any, tz: str) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, dtime.min)
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), _zone(tz))
    elif value:
        try:
            dt = datetime.fromisoformat(str(value))
        except ValueError:
            dt = datetime.now(_zone(tz))
    else:
        dt = datetime.now(_zone(tz))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_zone(tz))
    return dt.astimezone(_zone(tz))


def _as_date(value: Any, tz: str) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.astimezone(_zone(tz)).date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), _zone(tz)).date()
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def _parse_time(value: Any, default: str = "09:00") -> dtime:
    text = str(value or default).strip()
    try:
        hour, minute = text.split(":")[:2]
        return dtime(hour=int(hour), minute=int(minute))
    except (TypeError, ValueError):
        hour, minute = default.split(":")
        return dtime(hour=int(hour), minute=int(minute))


def _norm_weekdays(values: Iterable[Any]) -> tuple[int, ...]:
    out = []
    for value in values or ():
        try:
            day = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= day <= 7 and day not in out:
            out.append(day)
    return tuple(sorted(out))


def _week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())


@dataclass
class Routine:
    id: str
    user_id: str
    action: str
    local_time: str = "09:00"
    weekdays: tuple[int, ...] = ()
    interval: int = 1
    anchor_date: str = ""
    timezone: str = DEFAULT_TZ
    valid_from: str = ""
    valid_until: str = ""
    exdates: tuple[str, ...] = ()
    rdates: tuple[str, ...] = ()
    status: str = "candidate"
    confidence: float = 0.55
    observations: int = 0
    confirms: int = 0
    denies: int = 0
    unanswered: int = 0
    last_observed_at: float = 0.0
    last_asked_at: float = 0.0
    last_asked_date: str = ""
    last_confirmed_at: float = 0.0
    paused_until: float = 0.0
    dormant_until: float = 0.0
    evidence_episode_ids: List[str] = field(default_factory=list)
    origin: str = "manual"
    allow_probe: bool = True
    sensitivity: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "local_time": self.local_time,
            "weekdays": list(self.weekdays),
            "interval": self.interval,
            "anchor_date": self.anchor_date,
            "timezone": self.timezone,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "exdates": list(self.exdates),
            "rdates": list(self.rdates),
            "status": self.status,
            "confidence": round(self.confidence, 3),
            "observations": self.observations,
            "confirms": self.confirms,
            "denies": self.denies,
            "unanswered": self.unanswered,
            "last_observed_at": self.last_observed_at,
            "last_asked_at": self.last_asked_at,
            "last_asked_date": self.last_asked_date,
            "last_confirmed_at": self.last_confirmed_at,
            "evidence_episode_ids": list(self.evidence_episode_ids),
            "origin": self.origin,
            "allow_probe": self.allow_probe,
            "sensitivity": self.sensitivity,
            "note": self.note,
        }


class RoutineStore:
    """SQLite-backed weekly routine and probe state machine."""

    def __init__(self, path: str = ":memory:", default_tz: str = DEFAULT_TZ) -> None:
        self.path = str(path)
        self.tz = default_tz or DEFAULT_TZ
        if self.path != ":memory:":
            import os
            parent = os.path.dirname(os.path.abspath(self.path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10.0)
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS routines (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    local_time TEXT NOT NULL,
                    weekdays TEXT NOT NULL,
                    interval INTEGER NOT NULL DEFAULT 1,
                    anchor_date TEXT NOT NULL,
                    timezone TEXT NOT NULL,
                    valid_from TEXT NOT NULL DEFAULT '',
                    valid_until TEXT NOT NULL DEFAULT '',
                    exdates TEXT NOT NULL DEFAULT '[]',
                    rdates TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'candidate',
                    confidence REAL NOT NULL DEFAULT 0.55,
                    observations INTEGER NOT NULL DEFAULT 0,
                    confirms INTEGER NOT NULL DEFAULT 0,
                    denies INTEGER NOT NULL DEFAULT 0,
                    unanswered INTEGER NOT NULL DEFAULT 0,
                    last_observed_at REAL NOT NULL DEFAULT 0,
                    last_asked_at REAL NOT NULL DEFAULT 0,
                    last_asked_date TEXT NOT NULL DEFAULT '',
                    last_confirmed_at REAL NOT NULL DEFAULT 0,
                    paused_until REAL NOT NULL DEFAULT 0,
                    dormant_until REAL NOT NULL DEFAULT 0,
                    evidence_episode_ids TEXT NOT NULL DEFAULT '[]',
                    origin TEXT NOT NULL DEFAULT 'manual',
                    allow_probe INTEGER NOT NULL DEFAULT 1,
                    sensitivity TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS i_routines_user
                    ON routines(user_id, status);
                """
            )
            self.conn.commit()
        except Exception:
            self.conn.close()
            raise

    @staticmethod
    def _routine_id(user_id: str, action: str, local_time: str, weekdays: Sequence[int], interval: int) -> str:
        seed = f"{user_id}:{action.strip().lower()}:{local_time}:{','.join(str(x) for x in weekdays)}:{interval}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Routine:
        return Routine(
            id=row["id"], user_id=row["user_id"], action=row["action"],
            local_time=row["local_time"],
            weekdays=tuple(int(x) for x in json.loads(row["weekdays"] or "[]")),
            interval=int(row["interval"]), anchor_date=row["anchor_date"],
            timezone=row["timezone"], valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            exdates=tuple(json.loads(row["exdates"] or "[]")),
            rdates=tuple(json.loads(row["rdates"] or "[]")),
            status=row["status"], confidence=float(row["confidence"]),
            observations=int(row["observations"]), confirms=int(row["confirms"]),
            denies=int(row["denies"]), unanswered=int(row["unanswered"]),
            last_observed_at=float(row["last_observed_at"]),
            last_asked_at=float(row["last_asked_at"]),
            last_asked_date=row["last_asked_date"],
            last_confirmed_at=float(row["last_confirmed_at"]),
            paused_until=float(row["paused_until"]),
            dormant_until=float(row["dormant_until"]),
            evidence_episode_ids=json.loads(row["evidence_episode_ids"] or "[]"),
            origin=row["origin"], allow_probe=bool(row["allow_probe"]),
            sensitivity=row["sensitivity"], note=row["note"],
        )

    def _save(self, routine: Routine) -> Routine:
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO routines
                (id, user_id, action, local_time, weekdays, interval, anchor_date,
                 timezone, valid_from, valid_until, exdates, rdates, status,
                 confidence, observations, confirms, denies, unanswered,
                 last_observed_at, last_asked_at, last_asked_date,
                 last_confirmed_at, paused_until, dormant_until,
                 evidence_episode_ids, origin, allow_probe, sensitivity, note,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    routine.id, routine.user_id, routine.action, routine.local_time,
                    json.dumps(list(routine.weekdays)), routine.interval,
                    routine.anchor_date, routine.timezone, routine.valid_from,
                    routine.valid_until, json.dumps(list(routine.exdates)),
                    json.dumps(list(routine.rdates)), routine.status,
                    routine.confidence, routine.observations, routine.confirms,
                    routine.denies, routine.unanswered, routine.last_observed_at,
                    routine.last_asked_at, routine.last_asked_date,
                    routine.last_confirmed_at, routine.paused_until,
                    routine.dormant_until,
                    json.dumps(list(routine.evidence_episode_ids), ensure_ascii=False),
                    routine.origin, int(routine.allow_probe), routine.sensitivity,
                    routine.note, datetime.now(_zone(self.tz)).timestamp(),
                    datetime.now(_zone(self.tz)).timestamp(),
                ),
            )
            self.conn.commit()
        return routine

    def label(
        self,
        user_id: str,
        action: str,
        *,
        time: str = "09:00",
        weekdays: Iterable[int] = (),
        interval: int = 1,
        anchor_date: Any = None,
        timezone: str = DEFAULT_TZ,
        valid_from: Any = None,
        valid_until: Any = None,
        exdates: Iterable[Any] = (),
        rdates: Iterable[Any] = (),
        status: str = "candidate",
        confidence: float = 0.55,
        episode_id: Optional[str] = None,
        allow_probe: bool = True,
        sensitivity: str = "",
        note: str = "",
    ) -> Optional[Routine]:
        user_id = (user_id or "").strip()
        action = (action or "").strip()
        if not user_id or not action:
            return None
        day = _as_date(anchor_date, timezone) or datetime.now(_zone(timezone)).date()
        local_time = _parse_time(time).strftime("%H:%M")
        days = _norm_weekdays(weekdays or (day.isoweekday(),))
        if not days:
            return None
        interval = 2 if int(interval) == 2 else 1
        status = status if status in ALLOWED_STATUSES else "candidate"
        rid = self._routine_id(user_id, action, local_time, days, interval)
        existing = self.get(rid)
        now = datetime.now(_zone(timezone)).timestamp()
        if existing is None:
            routine = Routine(
                id=rid, user_id=user_id, action=action, local_time=local_time,
                weekdays=days, interval=interval, anchor_date=day.isoformat(),
                timezone=timezone,
                valid_from=_as_date(valid_from, timezone).isoformat() if _as_date(valid_from, timezone) else "",
                valid_until=_as_date(valid_until, timezone).isoformat() if _as_date(valid_until, timezone) else "",
                exdates=tuple(sorted({d.isoformat() for d in (_as_date(x, timezone) for x in exdates) if d})),
                rdates=tuple(sorted({d.isoformat() for d in (_as_date(x, timezone) for x in rdates) if d})),
                status=status, confidence=max(0.0, min(1.0, float(confidence))),
                observations=0,
                last_confirmed_at=now if status in {"confirmed", "trusted"} else 0.0,
                evidence_episode_ids=[episode_id] if episode_id else [],
                origin="manual", allow_probe=allow_probe,
                sensitivity=sensitivity, note=note,
            )
            return self._save(routine)
        existing.weekdays = days
        existing.interval = interval
        existing.anchor_date = day.isoformat()
        existing.timezone = timezone
        existing.valid_from = _as_date(valid_from, timezone).isoformat() if _as_date(valid_from, timezone) else existing.valid_from
        existing.valid_until = _as_date(valid_until, timezone).isoformat() if _as_date(valid_until, timezone) else existing.valid_until
        existing.exdates = tuple(sorted(set(existing.exdates) | {
            d.isoformat() for d in (_as_date(x, timezone) for x in exdates) if d
        }))
        existing.rdates = tuple(sorted(set(existing.rdates) | {
            d.isoformat() for d in (_as_date(x, timezone) for x in rdates) if d
        }))
        existing.status = status
        existing.confidence = max(0.0, min(1.0, float(confidence)))
        existing.allow_probe = allow_probe
        existing.sensitivity = sensitivity or existing.sensitivity
        existing.note = note or existing.note
        if episode_id and episode_id not in existing.evidence_episode_ids:
            existing.evidence_episode_ids.append(episode_id)
        if status in {"confirmed", "trusted"}:
            existing.last_confirmed_at = now
        return self._save(existing)

    def observe(
        self,
        user_id: str,
        action: str,
        *,
        occurred_at: Any = None,
        occurrence_date: Any = None,
        time: Optional[str] = None,
        weekdays: Optional[Iterable[int]] = None,
        interval: int = 1,
        timezone: str = DEFAULT_TZ,
        episode_id: Optional[str] = None,
        confidence: float = 0.55,
    ) -> Optional[Routine]:
        """Record one observation. A single observation is only a candidate."""
        dt = _as_datetime(occurred_at, timezone)
        day = _as_date(occurrence_date, timezone) or dt.date()
        local_time = _parse_time(time or dt.strftime("%H:%M")).strftime("%H:%M")
        days = _norm_weekdays(weekdays or (day.isoweekday(),))
        if not days:
            return None
        rid = self._routine_id(user_id, action, local_time, days, interval)
        routine = self.get(rid)
        now = datetime.now(_zone(timezone)).timestamp()
        if routine is None:
            routine = Routine(
                id=rid, user_id=user_id, action=action, local_time=local_time,
                weekdays=days, interval=2 if int(interval) == 2 else 1,
                anchor_date=day.isoformat(), timezone=timezone,
                status="candidate", confidence=max(0.0, min(0.9, float(confidence))),
                observations=1, last_observed_at=dt.timestamp(),
                evidence_episode_ids=[episode_id] if episode_id else [],
                origin="observed",
            )
            return self._save(routine)
        routine.observations += 1
        routine.last_observed_at = dt.timestamp()
        if routine.status in {"paused", "dormant"} and now >= max(routine.paused_until, routine.dormant_until):
            routine.status = "candidate"
        if routine.status == "candidate":
            routine.confidence = min(0.9, routine.confidence + 0.08)
        if episode_id and episode_id not in routine.evidence_episode_ids:
            routine.evidence_episode_ids.append(episode_id)
        return self._save(routine)

    def get(self, routine_id: str) -> Optional[Routine]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM routines WHERE id = ?", (routine_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def list(self, user_id: str, statuses: Optional[Iterable[str]] = None) -> List[Routine]:
        sql = "SELECT * FROM routines WHERE user_id = ?"
        params: List[Any] = [user_id]
        allowed = tuple(statuses or ())
        if allowed:
            sql += " AND status IN (" + ",".join("?" for _ in allowed) + ")"
            params.extend(allowed)
        sql += " ORDER BY local_time, action"
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [self._from_row(row) for row in rows]

    def _matches(self, routine: Routine, day: date) -> bool:
        if day.isoformat() in routine.exdates:
            return False
        if day.isoformat() in routine.rdates:
            return True
        if day.isoweekday() not in routine.weekdays:
            return False
        if routine.valid_from and day.isoformat() < routine.valid_from:
            return False
        if routine.valid_until and day.isoformat() > routine.valid_until:
            return False
        if routine.interval == 2:
            anchor = _as_date(routine.anchor_date, routine.timezone)
            if anchor is None:
                return False
            weeks = (_week_start(day) - _week_start(anchor)).days // 7
            if weeks % 2 != 0:
                return False
        return True

    def next_occurrence(self, routine: Routine, after: Any = None) -> Optional[datetime]:
        now = _as_datetime(after, routine.timezone)
        start = now.date()
        for offset in range(0, 370):
            day = start + timedelta(days=offset)
            if not self._matches(routine, day):
                continue
            local = _parse_time(routine.local_time)
            candidate = datetime.combine(day, local, tzinfo=_zone(routine.timezone))
            if candidate >= now:
                return candidate
        return None

    def _occurrence_key(self, routine: Routine, day: date) -> str:
        return f"{routine.id}@{day.isoformat()}"

    def _in_window(self, routine: Routine, now: datetime, window_min: int) -> Optional[date]:
        day = now.date()
        if not self._matches(routine, day):
            return None
        start = datetime.combine(day, _parse_time(routine.local_time), tzinfo=_zone(routine.timezone))
        delta_min = (now - start).total_seconds() / 60.0
        return day if abs(delta_min) <= window_min else None

    def probe(
        self,
        user_id: str,
        *,
        now: Any = None,
        window_min: int = PROBE_WINDOW_MIN,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return in-turn probe candidates; never sends or marks asked."""
        now_dt = _as_datetime(now, self.tz)
        out = []
        for routine in self.list(user_id):
            if not routine.allow_probe or routine.status in {"paused", "ended", "trusted", "dormant"}:
                continue
            day = self._in_window(routine, now_dt, window_min)
            if day is None:
                continue
            key = self._occurrence_key(routine, day)
            if routine.last_asked_date == day.isoformat():
                continue
            if routine.status == "confirmed":
                stale = (
                    routine.last_confirmed_at <= 0
                    or now_dt.timestamp() - routine.last_confirmed_at > STALE_CONFIRM_DAYS * 86400
                )
                if not stale:
                    continue
                style = "check"
            elif routine.confidence < 0.65:
                style = "open"
            else:
                style = "direct"
            out.append({
                "probe_id": key,
                "routine_id": routine.id,
                "action": routine.action,
                "window": routine.local_time,
                "status": routine.status,
                "confidence": round(routine.confidence, 3),
                "ask_style": style,
                "suggested_questions": self._questions(routine, style),
                "reason": "same weekday and local time window",
            })
        return out[:max(1, int(limit))]

    @staticmethod
    def _questions(routine: Routine, style: str) -> List[str]:
        action = routine.action
        chinese = bool(re.search(r"[\u4e00-\u9fff]", action))
        if chinese:
            if style == "open":
                return ["你现在在做什么？", f"你是不是要去{action}？"]
            if style == "check":
                return [f"今天还有{action}吗？"]
            return [f"你是不是要去{action}？", f"今天还有{action}吗？"]
        if style == "open":
            return [
                "What are you up to right now?",
                f"Are you about to {action}?",
            ]
        if style == "check":
            return [f"Is today still a {action} day?"]
        return [
            f"Are you about to {action}?",
            f"Is today still a {action} day?",
        ]

    def feedback(
        self,
        probe_id: str,
        result: str,
        *,
        now: Any = None,
        note: str = "",
    ) -> Optional[Routine]:
        if result not in ALLOWED_FEEDBACK or "@" not in probe_id:
            return None
        routine_id, day_text = probe_id.split("@", 1)
        routine = self.get(routine_id)
        if routine is None:
            return None
        now_dt = _as_datetime(now, routine.timezone)
        routine.last_asked_at = now_dt.timestamp()
        routine.last_asked_date = day_text
        if result == "confirmed":
            routine.confirms += 1
            routine.unanswered = 0
            routine.status = "trusted" if routine.confirms >= 2 else "confirmed"
            routine.confidence = min(0.99, routine.confidence + 0.15)
            routine.last_confirmed_at = now_dt.timestamp()
            routine.paused_until = 0.0
        elif result == "denied":
            routine.denies += 1
            routine.status = "paused"
            routine.confidence = max(0.05, routine.confidence * 0.5)
            routine.paused_until = now_dt.timestamp() + PAUSE_DAYS * 86400
        elif result == "skipped_today":
            routine.exdates = tuple(sorted({*routine.exdates, day_text}))
            routine.confidence = max(0.05, routine.confidence - 0.03)
        elif result == "unanswered":
            routine.unanswered += 1
            routine.confidence = max(0.05, routine.confidence - 0.02)
            # Silence never promotes a candidate and never erases a confirmed
            # routine. Only an unconfirmed candidate can go dormant.
            if routine.status not in {"confirmed", "trusted"} and routine.unanswered >= 2:
                routine.status = "dormant"
                routine.dormant_until = now_dt.timestamp() + DORMANT_DAYS * 86400
        elif result == "ended":
            routine.status = "ended"
        if note:
            routine.note = note
        return self._save(routine)

    def purge(self, user_id: Optional[str] = None) -> int:
        sql = "DELETE FROM routines"
        params: List[Any] = []
        if user_id is not None:
            sql += " WHERE user_id = ?"
            params.append(user_id)
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
        return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self.conn.close()
