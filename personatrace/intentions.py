"""Prospective memory: a small, typed intention ledger.

Why this is not a Belief:
- A belief is a proposition ("likes Rust") and can decay. An intention is a
  commitment with a lifecycle (active -> due -> overdue -> done/cancelled).
- A belief is found by relevance. An intention is found by time. Putting them
  in one top-k means a due task loses to a topical fact.

Storage is SQLite (stdlib). It gives range queries, transactions and one
exportable file without a server or a cron job. ``render_agenda()`` is called
when the prompt is assembled; no background scheduler is required.
"""

from __future__ import annotations

import difflib
import json
import math
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

try:  # recurrence is optional; without it a weekly series keeps its first due date
    from dateutil.rrule import rrulestr
except Exception:  # pragma: no cover - depends on the environment
    rrulestr = None

DEFAULT_TZ = "Asia/Shanghai"
DAY = 86400.0
DEFAULT_LEAD_S = 2 * 86400
DEFAULT_GRACE_S = 86400
PROACTIVE_BUCKETS = {"T-24h", "T-2h", "due", "overdue"}
OPEN_STATUSES = ("proposed", "active", "due", "overdue")
CLOSED_STATUSES = ("done", "cancelled", "expired", "superseded")


def _zone(name: Optional[str]) -> ZoneInfo:
    try:
        return ZoneInfo(name or DEFAULT_TZ)
    except Exception:
        return ZoneInfo(DEFAULT_TZ)


def _aware(value: Optional[datetime], tz: str) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=_zone(tz))
    return value.astimezone(_zone(tz))


def _to_epoch(value: Any, tz: str) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return _aware(value, tz).timestamp()
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_zone(tz))
        return parsed.timestamp()
    return None


def _local(epoch: Optional[float], tz: str) -> Optional[datetime]:
    if epoch is None:
        return None
    return datetime.fromtimestamp(float(epoch), _zone(tz))


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))


def _weekday_cn(dt: datetime) -> str:
    return "一二三四五六日"[dt.weekday()]


def _date_label(dt: datetime, now: datetime) -> str:
    delta = (dt.date() - now.date()).days
    if delta == 0:
        return "今天"
    if delta == 1:
        return "明天"
    if delta == 2:
        return "后天"
    if delta == -1:
        return "昨天"
    if delta == -2:
        return "前天"
    return f"{dt.month:02d}-{dt.day:02d} 周{_weekday_cn(dt)}"


def _relative(due: float, now: float) -> str:
    delta = due - now
    if delta >= 0:
        if delta < 3600:
            return f"还有 {max(1, int(delta // 60))} 分钟"
        if delta < DAY:
            return f"还有 {max(1, int(delta // 3600))} 小时"
        return f"还有 {max(1, int(delta // DAY))} 天"
    delta = -delta
    if delta < 3600:
        return f"已超期 {max(1, int(delta // 60))} 分钟"
    if delta < DAY:
        return f"已超期 {max(1, int(delta // 3600))} 小时"
    return f"已超期 {max(1, int(delta // DAY))} 天"


@dataclass
class Intent:
    id: str
    user_id: str
    kind: str
    title: str
    detail: str
    status: str
    due_at: Optional[float]
    due_kind: str
    tz: str
    granularity: str
    lead_s: int
    grace_s: int
    importance: float
    confidence: float
    pinned: bool
    rrule: str
    series_id: Optional[str]
    exdates: List[str]
    source_turn: str
    source_text: str
    superseded_by: Optional[str]
    created_at: float
    updated_at: float
    completed_at: Optional[float]
    last_bucket: str
    last_surfaced_at: Optional[float]
    surface_count: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Intent":
        try:
            exdates = json.loads(row["exdates"] or "[]")
        except (TypeError, ValueError):
            exdates = []
        return cls(
            id=row["id"], user_id=row["user_id"], kind=row["kind"],
            title=row["title"], detail=row["detail"], status=row["status"],
            due_at=row["due_at"], due_kind=row["due_kind"], tz=row["tz"],
            granularity=row["granularity"], lead_s=int(row["lead_s"]),
            grace_s=int(row["grace_s"]), importance=float(row["importance"]),
            confidence=float(row["confidence"]), pinned=bool(row["pinned"]),
            rrule=row["rrule"] or "", series_id=row["series_id"],
            exdates=exdates, source_turn=row["source_turn"],
            source_text=row["source_text"], superseded_by=row["superseded_by"],
            created_at=float(row["created_at"]), updated_at=float(row["updated_at"]),
            completed_at=row["completed_at"], last_bucket=row["last_bucket"] or "",
            last_surfaced_at=row["last_surfaced_at"],
            surface_count=int(row["surface_count"] or 0),
        )


def _locked(method):
    """Serialize read-modify-write methods on the store's RLock."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class IntentStore:
    """SQLite-backed intention ledger.

    One user is the default, but every row carries ``user_id`` so it stays
    additive. The database can be a file or ``:memory:`` for tests.
    """

    def __init__(
        self,
        path: str = ":memory:",
        *,
        default_tz: str = DEFAULT_TZ,
        proactive_cooldown_s: int = 6 * 3600,
    ) -> None:
        self.path = path
        self.default_tz = default_tz
        self.proactive_cooldown_s = max(0, int(proactive_cooldown_s))
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, timeout=5.0, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout = 5000")
        if path != ":memory:":
            try:
                self.conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.DatabaseError:
                pass
        self._init_schema()

    # ---------------------------------------------------------------- schema
    def _init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS intentions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'task',
                    title TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    due_at REAL,
                    due_kind TEXT NOT NULL DEFAULT 'day',
                    tz TEXT NOT NULL DEFAULT 'Asia/Shanghai',
                    granularity TEXT NOT NULL DEFAULT 'day',
                    lead_s INTEGER NOT NULL DEFAULT 172800,
                    grace_s INTEGER NOT NULL DEFAULT 86400,
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.6,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    rrule TEXT,
                    series_id TEXT,
                    exdates TEXT NOT NULL DEFAULT '[]',
                    source_turn TEXT NOT NULL DEFAULT '',
                    source_text TEXT NOT NULL DEFAULT '',
                    superseded_by TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    last_bucket TEXT NOT NULL DEFAULT '',
                    last_surfaced_at REAL,
                    surface_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_intent_due
                    ON intentions(user_id, status, due_at);
                CREATE TABLE IF NOT EXISTS intent_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intention_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    ts REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agenda_meta (
                    user_id TEXT PRIMARY KEY,
                    last_proactive_at REAL
                );
                CREATE TABLE IF NOT EXISTS user_meta (
                    user_id TEXT PRIMARY KEY,
                    last_turn_at REAL
                );
                """
            )
            self.conn.commit()

    # ---------------------------------------------------------------- writes
    @_locked
    def upsert(
        self,
        user_id: str,
        title: str,
        *,
        due_at: Any = None,
        due_kind: str = "day",
        tz: Optional[str] = None,
        granularity: str = "day",
        kind: str = "task",
        detail: str = "",
        importance: float = 0.5,
        confidence: float = 0.6,
        lead_s: Optional[int] = None,
        grace_s: Optional[int] = None,
        pinned: bool = False,
        rrule: Optional[str] = None,
        series_id: Optional[str] = None,
        source_turn: str = "",
        source_text: str = "",
        status: Optional[str] = None,
        needs_confirmation: bool = False,
        now: Any = None,
    ) -> Intent:
        title = (title or "").strip()
        if not title:
            raise ValueError("intent title is empty")
        tz = tz or self.default_tz
        granularity = granularity if granularity in ("day", "datetime") else "day"
        due_kind = due_kind if due_kind in ("start", "end", "day") else "day"
        due = _to_epoch(due_at, tz)
        if due is not None and granularity == "day":
            # An all-day event is compared by date, not by 00:00 precision.
            local = _local(due, tz)
            due = local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        now_ts = _to_epoch(now, tz) if now is not None else time.time()
        now_ts = float(now_ts if now_ts is not None else time.time())
        local_date = _local(due, tz).strftime("%Y-%m-%d") if due else "none"
        key = f"{user_id}:{title.lower()}:{local_date}:{rrule or ''}"
        iid = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))
        existing = self._row(iid)
        if existing:
            old_due = existing["due_at"]
            new_status = status or existing["status"]
            if (
                new_status in ("done", "cancelled")
                and due is not None
                and due > now_ts
                and existing["status"] in ("done", "cancelled")
            ):
                new_status = "active"
            self.conn.execute(
                """
                UPDATE intentions SET
                    kind=?, title=?, detail=?, status=?, due_at=?, due_kind=?,
                    tz=?, granularity=?, lead_s=?, grace_s=?, importance=?,
                    confidence=?, pinned=?, rrule=?, series_id=?, source_turn=?,
                    source_text=?, updated_at=?, completed_at=?
                WHERE id=?
                """,
                (
                    kind, title, detail, new_status, due, due_kind, tz,
                    granularity,
                    int(lead_s) if lead_s is not None else int(existing["lead_s"]),
                    int(grace_s) if grace_s is not None else int(existing["grace_s"]),
                    _clamp(importance, default=0.5), _clamp(confidence, default=0.6),
                    int(bool(pinned)), rrule or None, series_id or existing["series_id"],
                    source_turn or existing["source_turn"],
                    source_text or existing["source_text"], now_ts,
                    existing["completed_at"] if new_status == existing["status"] else None,
                    iid,
                ),
            )
            self._log(iid, "updated", json.dumps({
                "old_due": old_due, "new_due": due, "status": new_status,
            }, ensure_ascii=False), now_ts)
        else:
            new_status = status or ("proposed" if needs_confirmation else "active")
            series = series_id or (iid if rrule else None)
            self.conn.execute(
                """
                INSERT INTO intentions (
                    id, user_id, kind, title, detail, status, due_at, due_kind, tz,
                    granularity, lead_s, grace_s, importance, confidence, pinned,
                    rrule, series_id, exdates, source_turn, source_text,
                    superseded_by, created_at, updated_at, completed_at,
                    last_bucket, last_surfaced_at, surface_count
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    iid, user_id, kind, title, detail, new_status, due, due_kind, tz,
                    granularity,
                    int(lead_s) if lead_s is not None else DEFAULT_LEAD_S,
                    int(grace_s) if grace_s is not None else DEFAULT_GRACE_S,
                    _clamp(importance, default=0.5), _clamp(confidence, default=0.6),
                    int(bool(pinned)), rrule or None, series, "[]", source_turn,
                    source_text, None, now_ts, now_ts, None, "", None, 0,
                ),
            )
            self._log(iid, "created", json.dumps({
                "title": title, "due_at": due, "kind": kind,
            }, ensure_ascii=False), now_ts)
        self.conn.commit()
        return self.get(iid)  # type: ignore[return-value]

    @_locked
    def reschedule(self, intent_id: str, due_at: Any, *, now: Any = None) -> Optional[Intent]:
        row = self._row(intent_id)
        if row is None:
            return None
        tz = row["tz"]
        due = _to_epoch(due_at, tz)
        if due is None:
            return None
        now_ts = _to_epoch(now, tz) if now is not None else time.time()
        now_ts = float(now_ts if now_ts is not None else time.time())
        if row["granularity"] == "day":
            due = _local(due, tz).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        self.conn.execute(
            "UPDATE intentions SET due_at=?, status='active', completed_at=NULL, "
            "last_bucket='', updated_at=? WHERE id=?",
            (due, now_ts, intent_id),
        )
        self._log(intent_id, "rescheduled", json.dumps({
            "old_due": row["due_at"], "new_due": due,
        }, ensure_ascii=False), now_ts)
        self.conn.commit()
        return self.get(intent_id)

    @_locked
    def complete(self, intent_id: str, occurrence_ts: Any = None, *, now: Any = None) -> Optional[Intent]:
        row = self._row(intent_id)
        if row is None:
            return None
        now_ts = _to_epoch(now, row["tz"]) if now is not None else time.time()
        now_ts = float(now_ts if now_ts is not None else time.time())
        if row["rrule"]:
            occurrence = _to_epoch(occurrence_ts, row["tz"]) if occurrence_ts is not None else None
            if occurrence is None:
                next_ones = self._occurrences(row, now_ts, 7)
                occurrence = next_ones[0] if next_ones else row["due_at"]
            if occurrence is not None:
                self.add_exdate(intent_id, occurrence, now=now_ts)
                self._log(intent_id, "occurrence_done", json.dumps({
                    "occurrence": occurrence,
                }, ensure_ascii=False), now_ts)
                self.conn.commit()
                return self.get(intent_id)
        self.conn.execute(
            "UPDATE intentions SET status='done', completed_at=?, updated_at=?, "
            "last_bucket='' WHERE id=?",
            (now_ts, now_ts, intent_id),
        )
        self._log(intent_id, "done", "", now_ts)
        self.conn.commit()
        return self.get(intent_id)

    @_locked
    def confirm(self, intent_id: str, *, now: Any = None) -> Optional[Intent]:
        row = self._row(intent_id)
        if row is None:
            return None
        now_ts = _to_epoch(now, row["tz"]) if now is not None else time.time()
        now_ts = float(now_ts if now_ts is not None else time.time())
        self.conn.execute(
            "UPDATE intentions SET status='active', updated_at=? WHERE id=?",
            (now_ts, intent_id),
        )
        self._log(intent_id, "confirmed", "", now_ts)
        self.conn.commit()
        return self.get(intent_id)

    @_locked
    def cancel(self, intent_id: str, *, now: Any = None) -> Optional[Intent]:
        row = self._row(intent_id)
        if row is None:
            return None
        now_ts = _to_epoch(now, row["tz"]) if now is not None else time.time()
        now_ts = float(now_ts if now_ts is not None else time.time())
        self.conn.execute(
            "UPDATE intentions SET status='cancelled', completed_at=?, updated_at=?, "
            "last_bucket='' WHERE id=?",
            (now_ts, now_ts, intent_id),
        )
        self._log(intent_id, "cancelled", "", now_ts)
        self.conn.commit()
        return self.get(intent_id)

    @_locked
    def add_exdate(self, intent_id: str, occurrence_ts: Any, *, now: Any = None) -> Optional[Intent]:
        row = self._row(intent_id)
        if row is None:
            return None
        tz = row["tz"]
        occurrence = _to_epoch(occurrence_ts, tz)
        if occurrence is None:
            return None
        local = _local(occurrence, tz).replace(tzinfo=None).isoformat(timespec="minutes")
        try:
            dates = json.loads(row["exdates"] or "[]")
        except (TypeError, ValueError):
            dates = []
        if local not in dates:
            dates.append(local)
        now_ts = _to_epoch(now, tz) if now is not None else time.time()
        now_ts = float(now_ts if now_ts is not None else time.time())
        self.conn.execute(
            "UPDATE intentions SET exdates=?, updated_at=? WHERE id=?",
            (json.dumps(dates, ensure_ascii=False), now_ts, intent_id),
        )
        self.conn.commit()
        return self.get(intent_id)

    @_locked
    def apply_update(self, user_id: str, update: Dict[str, Any], *, now: Any = None) -> Optional[Intent]:
        kind = update.get("kind")
        hint = str(update.get("title_hint") or "")
        target = self.find_open_by_title(user_id, hint)
        if target is None:
            return None
        if kind == "complete":
            return self.complete(target.id, update.get("occurrence_ts"), now=now)
        if kind == "cancel":
            return self.cancel(target.id, now=now)
        if kind == "reschedule" and update.get("due_at") is not None:
            return self.reschedule(target.id, update["due_at"], now=now)
        return None

    def find_open_by_title(self, user_id: str, hint: str) -> Optional[Intent]:
        hint = (hint or "").strip()
        if not hint:
            return None
        rows = self.list(user_id, statuses=OPEN_STATUSES)
        if not rows:
            return None
        best: Optional[Tuple[float, Intent]] = None
        for row in rows:
            ratio = difflib.SequenceMatcher(None, hint, row.title).ratio()
            if row.title and (row.title in hint or hint in row.title):
                ratio += 0.2
            if best is None or ratio > best[0]:
                best = (ratio, row)
        if best and best[0] >= 0.34:
            return best[1]
        return None

    # ---------------------------------------------------------------- reads
    def get(self, intent_id: str) -> Optional[Intent]:
        row = self._row(intent_id)
        return Intent.from_row(row) if row else None

    def list(
        self,
        user_id: str,
        *,
        statuses: Optional[Iterable[str]] = None,
        sweep: bool = False,
    ) -> List[Intent]:
        if sweep:
            # Keep the stored status honest before reading it back, so callers
            # that filter on "active" do not get rows that are weeks past due.
            self.sweep(user_id)
        allowed = tuple(statuses) if statuses else None
        if allowed:
            placeholders = ",".join("?" for _ in allowed)
            sql = (
                f"SELECT * FROM intentions WHERE user_id=? AND status IN ({placeholders}) "
                "ORDER BY (due_at IS NULL), due_at ASC, created_at DESC"
            )
            rows = self.conn.execute(sql, (user_id, *allowed)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM intentions WHERE user_id=? "
                "ORDER BY (due_at IS NULL), due_at ASC, created_at DESC",
                (user_id,),
            ).fetchall()
        return [Intent.from_row(row) for row in rows]

    def sweep(self, user_id: str, *, now: Any = None) -> int:
        """Close intentions whose grace window has already passed.

        ``_dynamic_status`` has always computed ``expired`` for display, but
        nothing wrote it back to the row, so ``list(statuses=("active",))`` kept
        returning a commitment that was weeks overdue and any count built on the
        stored status was inflated. Recurring series are skipped: their due date
        moves per occurrence and the series must not be closed as a whole.
        """
        now_ts = _to_epoch(now, self.default_tz) if now is not None else time.time()
        now_ts = float(now_ts if now_ts is not None else time.time())
        closed = 0
        with self._lock:
            for row in self.list(user_id, statuses=OPEN_STATUSES):
                if row.status == "proposed" or row.due_at is None:
                    continue
                if (row.rrule or "").strip():
                    continue
                if self._dynamic_status(row, row.due_at, now_ts) != "expired":
                    continue
                cur = self.conn.execute(
                    "UPDATE intentions SET status='expired', updated_at=? "
                    "WHERE id=? AND status=?",
                    (now_ts, row.id, row.status),
                )
                closed += max(0, cur.rowcount)
            if closed:
                self.conn.commit()
        return closed

    def touch_turn(self, user_id: str, ts: Any = None) -> None:
        value = _to_epoch(ts, self.default_tz) if ts is not None else time.time()
        value = float(value if value is not None else time.time())
        with self._lock:
            self.conn.execute(
                "INSERT INTO user_meta(user_id, last_turn_at) VALUES(?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET last_turn_at=excluded.last_turn_at",
                (user_id, value),
            )
            self.conn.commit()

    def last_turn_at(self, user_id: str) -> Optional[float]:
        row = self.conn.execute(
            "SELECT last_turn_at FROM user_meta WHERE user_id=?", (user_id,)
        ).fetchone()
        return float(row["last_turn_at"]) if row and row["last_turn_at"] else None

    # ---------------------------------------------------------------- agenda
    def select_agenda(
        self,
        user_id: str,
        *,
        now: Any = None,
        max_lines: int = 6,
        horizon_days: int = 7,
        include_proposed: bool = False,
    ) -> Dict[str, Any]:
        now_ts = _to_epoch(now, self.default_tz) if now is not None else time.time()
        now_ts = float(now_ts if now_ts is not None else time.time())
        self.sweep(user_id, now=now_ts)
        items: List[Dict[str, Any]] = []
        for row in self.list(user_id, statuses=OPEN_STATUSES):
            if row.status == "proposed" and not include_proposed:
                continue
            for due in self._due_candidates(row, now_ts, horizon_days):
                view = self._view(row, due, now_ts)
                bucket = self._bucket(view, now_ts)
                if bucket == "far" and not view["pinned"]:
                    continue
                if bucket == "expired":
                    continue
                view["bucket"] = bucket
                view["score"] = self._score(view, now_ts)
                view["must_show"] = self._must_show(view, now_ts)
                items.append(view)
        items.sort(key=lambda item: (
            not item["must_show"], -item["score"], item["due_at"] or 0.0,
        ))
        limit = max(1, int(max_lines))
        return {
            "now": now_ts,
            "items": items[:limit],
            "hidden": max(0, len(items) - limit),
        }

    def render_agenda(
        self,
        user_id: str,
        *,
        now: Any = None,
        max_lines: int = 6,
        horizon_days: int = 7,
        include_proposed: bool = False,
        commit: bool = True,
    ) -> str:
        selected = self.select_agenda(
            user_id, now=now, max_lines=max_lines,
            horizon_days=horizon_days, include_proposed=include_proposed,
        )
        items = selected["items"]
        if not items:
            return ""
        now_ts = selected["now"]

        proactive_id: Optional[str] = None
        meta = self.conn.execute(
            "SELECT last_proactive_at FROM agenda_meta WHERE user_id=?", (user_id,)
        ).fetchone()
        last_global = float(meta["last_proactive_at"]) if meta and meta["last_proactive_at"] else 0.0
        cooldown_ok = (now_ts - last_global) >= self.proactive_cooldown_s
        for item in items:
            if item["bucket"] not in PROACTIVE_BUCKETS:
                continue
            if item["last_bucket"] == item["bucket"]:
                continue
            if not cooldown_ok and item["bucket"] != "overdue":
                continue
            proactive_id = item["id"]
            if commit:
                self._mark_surfaced(item["id"], item["bucket"], now_ts)
                self.conn.execute(
                    "INSERT INTO agenda_meta(user_id, last_proactive_at) VALUES(?,?) "
                    "ON CONFLICT(user_id) DO UPDATE SET last_proactive_at=excluded.last_proactive_at",
                    (user_id, now_ts),
                )
                self.conn.commit()
            break

        lines = ["【时间敏感｜内部】"]
        for item in items:
            lines.append(self._line(item, now_ts, proactive=(item["id"] == proactive_id)))
        if selected["hidden"]:
            lines.append(f"- 另有 {selected['hidden']} 项在 {horizon_days} 天内，未展开")
        lines.append("（只有带主动标记的一条可以主动开口；其余是背景，用户不问就不要提。）")
        return "\n".join(lines)

    # ---------------------------------------------------------------- internals
    def _row(self, intent_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM intentions WHERE id=?", (intent_id,)
        ).fetchone()

    def _log(self, intent_id: str, kind: str, detail: str, ts: float) -> None:
        self.conn.execute(
            "INSERT INTO intent_events(intention_id, kind, detail, ts) VALUES(?,?,?,?)",
            (intent_id, kind, detail, ts),
        )

    def _mark_surfaced(self, intent_id: str, bucket: str, now_ts: float) -> None:
        self.conn.execute(
            "UPDATE intentions SET last_bucket=?, last_surfaced_at=?, "
            "surface_count=surface_count+1 WHERE id=?",
            (bucket, now_ts, intent_id),
        )

    def _due_candidates(self, row: Intent, now_ts: float, horizon_days: int) -> List[float]:
        if row.due_at is None:
            return []
        if not row.rrule:
            return [row.due_at]
        return self._occurrences(row, now_ts, horizon_days)

    def _occurrences(self, row: Intent, now_ts: float, horizon_days: int) -> List[float]:
        if rrulestr is None or not row.rrule:
            return [row.due_at] if row.due_at is not None else []
        tz = _zone(row.tz)
        lookback = max(1.0, float(row.grace_s) / DAY) + 1.0
        start = datetime.fromtimestamp(now_ts, tz).replace(tzinfo=None) - timedelta(days=lookback)
        end = datetime.fromtimestamp(now_ts, tz).replace(tzinfo=None) + timedelta(days=horizon_days)
        try:
            rule = rrulestr(row.rrule, dtstart=datetime.fromtimestamp(row.due_at, tz).replace(tzinfo=None))
            occurrences = rule.between(start, end, inc=True)
        except Exception:
            return [row.due_at] if row.due_at is not None else []
        excluded = set(row.exdates or [])
        out: List[float] = []
        for occurrence in occurrences:
            key = occurrence.isoformat(timespec="minutes")
            if key in excluded:
                continue
            out.append(occurrence.replace(tzinfo=tz).timestamp())
        if not out:
            return []
        # Show the occurrence nearest to now; a passed occurrence within grace is
        # more useful than next week's copy.
        out.sort()
        for due in out:
            if due >= now_ts - row.grace_s:
                return [due]
        return [out[-1]]

    def _view(self, row: Intent, due: float, now_ts: float) -> Dict[str, Any]:
        local = _local(due, row.tz)
        now_local = _local(now_ts, row.tz)
        status = self._dynamic_status(row, due, now_ts)
        return {
            "id": row.id, "user_id": row.user_id, "kind": row.kind,
            "title": row.title, "detail": row.detail, "status": status,
            "stored_status": row.status, "due_at": due,
            "due_local": local.isoformat(timespec="minutes") if local else None,
            "due_kind": row.due_kind, "tz": row.tz, "granularity": row.granularity,
            "lead_s": row.lead_s, "grace_s": row.grace_s,
            "importance": row.importance, "confidence": row.confidence,
            "pinned": row.pinned, "rrule": row.rrule, "series_id": row.series_id,
            "occurrence_ts": due if row.rrule else None,
            "source_turn": row.source_turn, "source_text": row.source_text,
            "last_bucket": row.last_bucket, "surface_count": row.surface_count,
            "day_delta": (local.date() - now_local.date()).days if local else None,
        }

    def _dynamic_status(self, row: Intent, due: float, now_ts: float) -> str:
        if row.status in CLOSED_STATUSES:
            return row.status
        if row.status == "proposed":
            return "proposed"
        if row.granularity == "day":
            delta = (_local(due, row.tz).date() - _local(now_ts, row.tz).date()).days
            grace_days = max(0, int(row.grace_s // DAY))
            if delta < 0 and -delta <= grace_days:
                return "overdue"
            if delta < -grace_days:
                return "expired"
            return row.status
        if due < now_ts:
            return "overdue" if now_ts - due <= row.grace_s else "expired"
        return row.status

    def _score(self, view: Dict[str, Any], now_ts: float) -> float:
        due = view.get("due_at")
        if due is None:
            urgency = 0.0
        elif view["granularity"] == "day":
            delta = int(view["day_delta"] or 0)
            lead_days = max(1e-6, view["lead_s"] / DAY)
            grace_days = max(0.0, view["grace_s"] / DAY)
            if delta > lead_days:
                urgency = 0.0
            elif delta >= 0:
                urgency = (1.0 - delta / lead_days) ** 2
            elif -delta <= grace_days:
                urgency = 1.0
            else:
                urgency = math.exp(-(-delta - grace_days) / 2.0)
        else:
            lead = due - now_ts
            lead_s = max(1.0, float(view["lead_s"]))
            if lead > lead_s:
                urgency = 0.0
            elif lead >= 0:
                urgency = (1.0 - lead / lead_s) ** 2
            elif -lead <= view["grace_s"]:
                urgency = 1.0
            else:
                urgency = math.exp(-(-lead - view["grace_s"]) / max(1.0, 2 * DAY))
        return round(
            0.55 * float(view["importance"])
            + 0.35 * urgency
            + 0.10 * float(view["confidence"]),
            4,
        )

    def _must_show(self, view: Dict[str, Any], now_ts: float) -> bool:
        if view.get("pinned"):
            return True
        due = view.get("due_at")
        if due is None:
            return False
        if view["granularity"] == "day":
            delta = int(view["day_delta"] or 0)
            return delta <= 1 or (delta < 0 and -delta <= max(1, int(view["grace_s"] // DAY)))
        lead = due - now_ts
        return lead <= DAY or (lead < 0 and -lead <= view["grace_s"])

    def _bucket(self, view: Dict[str, Any], now_ts: float) -> str:
        due = view.get("due_at")
        if due is None:
            return "far"
        if view["granularity"] == "day":
            delta = int(view["day_delta"] or 0)
            lead_days = max(1, int(view["lead_s"] // DAY))
            grace_days = max(0, int(view["grace_s"] // DAY))
            if delta > lead_days:
                return "far"
            if delta > 1:
                return "lead"
            if delta == 1:
                return "T-24h"
            if delta == 0:
                return "due"
            if -delta <= grace_days:
                return "overdue"
            return "expired"
        lead = due - now_ts
        if lead > view["lead_s"]:
            return "far"
        if lead > DAY:
            return "lead"
        if lead > 2 * 3600:
            return "T-24h"
        if lead > 0:
            return "T-2h"
        if -lead <= view["grace_s"]:
            return "overdue"
        return "expired"

    def _line(self, item: Dict[str, Any], now_ts: float, *, proactive: bool) -> str:
        local = _local(item["due_at"], item["tz"])
        now_local = _local(now_ts, item["tz"])
        day = _date_label(local, now_local) if local else ""
        if item["granularity"] == "datetime":
            when = f"{day} {local.strftime('%H:%M')}".strip()
            rel = _relative(item["due_at"], now_ts)
        else:
            when = day
            rel = ""
        detail = (item.get("detail") or "").strip()
        if detail and detail not in item["title"] and len(detail) <= 40:
            title = f"{item['title']} — {detail}"
        else:
            title = item["title"]
        line = f"- {when} {title}".strip()
        if rel:
            line += f"（{rel}"
        elif item["granularity"] == "day":
            delta = int(item.get("day_delta") or 0)
            if delta == 0:
                line += "（到期"
            elif delta > 0:
                line += f"（还有 {delta} 天"
            else:
                line += f"（已超期 {-delta} 天"
        else:
            line += "（"
        if item["status"] == "overdue":
            line += "，未确认"
        elif item["status"] == "proposed":
            line += "，待确认"
        line += "）"
        if proactive:
            line += " [需主动]"
        return line

    def stats(self) -> Dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM intentions GROUP BY status"
        ).fetchall()
        return {row["status"]: int(row["n"]) for row in rows}

    def close(self) -> None:
        with self._lock:
            self.conn.close()
