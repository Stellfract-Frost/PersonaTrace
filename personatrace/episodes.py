"""Episode store: raw conversation turns with provenance.

Beliefs are conclusions; episodes are the turns they came from. The same store
also keeps episode links to beliefs and intentions so a retrospective query can
show both "what was said" and "what we concluded".

SQLite is the source of truth. A vector index may be added later as a candidate
generator; this module deliberately starts with lexical + time filtering.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

DEFAULT_TZ = "Asia/Shanghai"
DEFAULT_EPISODE_TTL_DAYS = 90
ALLOWED_SOURCES = frozenset({"direct", "bridge", "console"})

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[a-zA-Z0-9_]+")
RETROSPECTIVE_MARKERS = (
    "上次", "之前", "那天", "上周", "昨天", "还记得", "聊过", "说过",
    "以前", "当时", "last time", "previously", "remember when",
)


def has_retrospective(query: str) -> bool:
    text = (query or "").lower()
    return any(marker in text for marker in RETROSPECTIVE_MARKERS)


def _zone(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz or DEFAULT_TZ)
    except Exception:
        return ZoneInfo(DEFAULT_TZ)


def _epoch(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_zone(DEFAULT_TZ))
        return value.timestamp()
    if isinstance(value, date):
        return datetime.combine(value, dtime.min, tzinfo=_zone(DEFAULT_TZ)).timestamp()
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_zone(DEFAULT_TZ))
        return parsed.timestamp()
    except (TypeError, ValueError):
        return None


def _tokens(text: str) -> List[str]:
    text = (text or "").lower()
    words = _WORD.findall(text)
    # Character bigrams are the useful unit for Chinese; single characters add
    # recall for very short labels.
    cjk_chars = "".join(_CJK.findall(text))
    grams = [cjk_chars[i:i + 2] for i in range(max(0, len(cjk_chars) - 1))]
    return words + grams + list(cjk_chars)


@dataclass
class Episode:
    id: str
    user_id: str
    session_id: str
    ts: float
    user_msg: str
    ai_reply: str
    source: str = "direct"
    belief_ids: List[str] = field(default_factory=list)
    intention_ids: List[str] = field(default_factory=list)

    def to_dict(self, include_reply: bool = True) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ts": self.ts,
            "time": datetime.fromtimestamp(self.ts, _zone(DEFAULT_TZ)).isoformat(),
            "user_msg": self.user_msg,
            "ai_reply": self.ai_reply if include_reply else "",
            "source": self.source,
            "belief_ids": list(self.belief_ids),
            "intention_ids": list(self.intention_ids),
        }


class EpisodeStore:
    """Append-only turn log with lexical + time-window retrieval."""

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
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL DEFAULT '',
                    ts REAL NOT NULL,
                    user_msg TEXT NOT NULL,
                    ai_reply TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'direct',
                    belief_ids TEXT NOT NULL DEFAULT '[]',
                    intention_ids TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS i_episodes_user_ts
                    ON episodes(user_id, ts);
                """
            )
            self.conn.commit()
        except Exception:
            self.conn.close()
            raise

    def add(
        self,
        user_id: str,
        user_msg: str,
        ai_reply: str = "",
        *,
        session_id: str = "",
        source: str = "direct",
        ts: Optional[float] = None,
        belief_ids: Iterable[str] = (),
        intention_ids: Iterable[str] = (),
    ) -> Optional[Episode]:
        user_id = (user_id or "").strip()
        user_msg = (user_msg or "").strip()
        ai_reply = (ai_reply or "").strip()
        if not user_id or not user_msg:
            return None
        source = (source or "direct").strip()
        if source not in ALLOWED_SOURCES:
            return None
        ts = float(ts if ts is not None else datetime.now(_zone(self.tz)).timestamp())
        session_id = (session_id or "").strip()
        seed = f"{user_id}:{session_id}:{ts:.3f}:{user_msg}"
        episode = Episode(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, seed)),
            user_id=user_id,
            session_id=session_id,
            ts=ts,
            user_msg=user_msg,
            ai_reply=ai_reply,
            source=source,
            belief_ids=list(dict.fromkeys(str(x) for x in belief_ids if x)),
            intention_ids=list(dict.fromkeys(str(x) for x in intention_ids if x)),
        )
        with self._lock:
            self.conn.execute(
                """INSERT OR IGNORE INTO episodes
                   (id, user_id, session_id, ts, user_msg, ai_reply, source,
                    belief_ids, intention_ids)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    episode.id, episode.user_id, episode.session_id, episode.ts,
                    episode.user_msg, episode.ai_reply, episode.source,
                    json.dumps(episode.belief_ids, ensure_ascii=False),
                    json.dumps(episode.intention_ids, ensure_ascii=False),
                ),
            )
            self.conn.commit()
        return self.get(episode.id) or episode

    def link(
        self,
        episode_id: str,
        *,
        belief_ids: Iterable[str] = (),
        intention_ids: Iterable[str] = (),
    ) -> Optional[Episode]:
        with self._lock:
            row = self.conn.execute(
                "SELECT belief_ids, intention_ids FROM episodes WHERE id = ?",
                (episode_id,),
            ).fetchone()
            if row is None:
                return None
            beliefs = json.loads(row["belief_ids"] or "[]")
            intentions = json.loads(row["intention_ids"] or "[]")
            beliefs.extend(str(x) for x in belief_ids if x)
            intentions.extend(str(x) for x in intention_ids if x)
            beliefs = list(dict.fromkeys(beliefs))
            intentions = list(dict.fromkeys(intentions))
            self.conn.execute(
                "UPDATE episodes SET belief_ids = ?, intention_ids = ? WHERE id = ?",
                (json.dumps(beliefs, ensure_ascii=False),
                 json.dumps(intentions, ensure_ascii=False), episode_id),
            )
            self.conn.commit()
        return self.get(episode_id)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Episode:
        return Episode(
            id=row["id"], user_id=row["user_id"], session_id=row["session_id"],
            ts=float(row["ts"]), user_msg=row["user_msg"], ai_reply=row["ai_reply"],
            source=row["source"],
            belief_ids=json.loads(row["belief_ids"] or "[]"),
            intention_ids=json.loads(row["intention_ids"] or "[]"),
        )

    def get(self, episode_id: str) -> Optional[Episode]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def recent(self, user_id: str, limit: int = 5) -> List[Episode]:
        with self._lock:
            rows = self.conn.execute(
                """SELECT * FROM episodes WHERE user_id = ?
                   ORDER BY ts DESC LIMIT ?""",
                (user_id, max(1, int(limit))),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def search(
        self,
        user_id: str,
        query: str = "",
        *,
        after: Any = None,
        before: Any = None,
        limit: int = 3,
        include_reply: bool = True,
    ) -> List[Dict[str, Any]]:
        """Lexical + recency retrieval inside an optional time window."""
        after_ts = _epoch(after)
        before_ts = _epoch(before)
        sql = "SELECT * FROM episodes WHERE user_id = ?"
        params: List[Any] = [user_id]
        if after_ts is not None:
            sql += " AND ts >= ?"
            params.append(after_ts)
        if before_ts is not None:
            sql += " AND ts <= ?"
            params.append(before_ts)
        sql += " ORDER BY ts DESC LIMIT 500"
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        episodes = [self._from_row(row) for row in rows]
        if not episodes:
            return []
        q = (query or "").strip()
        if not q:
            return [e.to_dict(include_reply) for e in episodes[:max(1, int(limit))]]

        q_tokens = _tokens(q)
        q_set = set(q_tokens)
        if not q_set:
            return [e.to_dict(include_reply) for e in episodes[:max(1, int(limit))]]
        docs = [(e, _tokens(e.user_msg + " " + e.ai_reply)) for e in episodes]
        df = {tok: 0 for tok in q_set}
        for _ep, toks in docs:
            tok_set = set(toks)
            for tok in q_set & tok_set:
                df[tok] += 1
        total = len(docs)
        scored = []
        for ep, toks in docs:
            tok_set = set(toks)
            lexical = 0.0
            for tok in q_set & tok_set:
                lexical += math.log(1.0 + total / max(1, df[tok]))
            if q in ep.user_msg:
                lexical += 2.0
            if lexical <= 0:
                continue
            age_days = max(0.0, (datetime.now(_zone(self.tz)).timestamp() - ep.ts) / 86400.0)
            score = lexical + 0.25 / (1.0 + age_days / 30.0)
            scored.append((score, ep))
        scored.sort(key=lambda item: (-item[0], -item[1].ts))
        return [
            {**ep.to_dict(include_reply), "score": round(score, 4)}
            for score, ep in scored[:max(1, int(limit))]
        ]

    def delete(self, episode_id: str) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
            self.conn.commit()
        return cur.rowcount > 0

    def purge(self, user_id: Optional[str] = None, before: Any = None) -> int:
        sql = "DELETE FROM episodes"
        clauses, params = [], []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        before_ts = _epoch(before)
        if before_ts is not None:
            clauses.append("ts < ?")
            params.append(before_ts)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
        return cur.rowcount

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            count = self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        return {"episodes": int(count), "path": self.path}

    def close(self) -> None:
        with self._lock:
            self.conn.close()
