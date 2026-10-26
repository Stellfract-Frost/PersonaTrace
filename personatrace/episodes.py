"""Episode store: raw conversation turns with provenance.

Beliefs are conclusions; episodes are the turns they came from. The same store
also keeps episode links to beliefs and intentions so a retrospective query can
show both "what was said" and "what we concluded".

SQLite is the source of truth. A vector index may be added later as a candidate
generator; this module deliberately starts with lexical + time filtering.
"""

from __future__ import annotations

import json
import logging
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
MAX_EPISODE_CANDIDATES = 5000
# Reciprocal rank fusion constant, matching the usual 60. Kept for callers that
# still ask for rank fusion; the store's own merge uses scaled weights below.
RRF_K = 60
# Relative weight of each retrieval arm when the two are blended. The vector arm
# leads because episode text is a whole exchange; lexical only rescues exact
# names and numbers the embedder blurs.
VECTOR_WEIGHT = 1.0
LEXICAL_WEIGHT = 0.5
ALLOWED_SOURCES = frozenset({"direct", "bridge", "console"})
logger = logging.getLogger(__name__)

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[a-zA-Z0-9_]+")
RETROSPECTIVE_MARKERS = (
    "上次", "之前", "那天", "上周", "昨天", "还记得", "聊过", "说过",
    "以前", "当时",
)

# English must use phrases/word boundaries. Substring matching made
# "Chicago" match "ago", "order a pizza" match "order", and "my first task"
# match "first", which replayed transcripts for ordinary queries.
_RETROSPECTIVE_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"\b(?:last|previous|prior)\s+(?:time|week|month|year|weekend|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(?:remember|recall)\b",
    r"\b(?:back then|used to|earlier|previously)\b",
    r"\b(?:yesterday|overnight|ago)\b",
    r"\bhow many\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\b",
    r"\bwhen did\b",
    r"\bwhat (?:date|day|year)\b",
    r"\bwhich (?:day|event|month|year)\b",
    r"\bhappened first\b",
    r"\border of\b",
    r"\bin order\b",
    r"\bmore recently\b|\bmost recent\b",
    r"\bpast (?:week|month|weekend|few|two|three)\b",
    r"\bwhat was the date\b",
    r"\bhow old\b",
    r"\bwhat did\b.*\bbefore\b",
))


def has_retrospective(query: str) -> bool:
    text = (query or "").lower()
    if any(marker in text for marker in RETROSPECTIVE_MARKERS):
        return True
    return any(pattern.search(text) for pattern in _RETROSPECTIVE_PATTERNS)


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


def _like_escape(text: str) -> str:
    return text.replace("~", "~~").replace("%", "~%").replace("_", "~_")


def _episode_doc(episode: Episode, limit: int = 1500) -> str:
    """Text handed to a reranker. Cross-encoders cap at 512 tokens anyway."""
    text = f"{episode.user_msg}\n{episode.ai_reply}".strip()
    return text[:limit] or " "


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

    def to_dict(self, include_reply: bool = True, tz: Optional[str] = None) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ts": self.ts,
            "time": datetime.fromtimestamp(self.ts, _zone(tz or DEFAULT_TZ)).isoformat(),
            "user_msg": self.user_msg,
            "ai_reply": self.ai_reply if include_reply else "",
            "source": self.source,
            "belief_ids": list(self.belief_ids),
            "intention_ids": list(self.intention_ids),
        }


class EpisodeStore:
    """Append-only turn log with lexical + time-window retrieval."""

    def __init__(
        self,
        path: str = ":memory:",
        default_tz: str = DEFAULT_TZ,
        *,
        vector_index: Any = None,
    ) -> None:
        self.path = str(path)
        self.tz = default_tz or DEFAULT_TZ
        self.vector_index = vector_index
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
        if self.vector_index is not None:
            try:
                self.vector_index.upsert([episode])
            except Exception:
                logger.warning("episode vector upsert failed", exc_info=True)
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

    def get_many(self, episode_ids: Iterable[str]) -> Dict[str, Episode]:
        ids = [str(x) for x in episode_ids if x]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM episodes WHERE id IN ({placeholders})", ids,
            ).fetchall()
        return {row["id"]: self._from_row(row) for row in rows}

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
        reranker: Any = None,
        rerank_top: int = 30,
    ) -> List[Dict[str, Any]]:
        """Lexical + recency retrieval inside an optional time window.

        ``reranker`` is an optional cross-encoder-like object exposing
        ``predict([(query, doc), ...]) -> scores``. When given, the head of the
        fused list is re-scored and the tail keeps the fused order, so recall at
        larger k is unchanged. Passing ``None`` (the default) leaves behaviour
        exactly as it was.
        """
        after_ts = _epoch(after)
        before_ts = _epoch(before)
        q = (query or "").strip()
        base = "SELECT * FROM episodes WHERE user_id = ?"
        base_params: List[Any] = [user_id]
        if after_ts is not None:
            base += " AND ts >= ?"
            base_params.append(after_ts)
        if before_ts is not None:
            base += " AND ts <= ?"
            base_params.append(before_ts)

        if not q:
            sql = base + " ORDER BY ts DESC LIMIT 500"
            with self._lock:
                rows = self.conn.execute(sql, base_params).fetchall()
            episodes = [self._from_row(row) for row in rows]
            return [e.to_dict(include_reply, self.tz) for e in episodes[:max(1, int(limit))]]

        # Do not rank only the newest 500 rows: with >500 episodes the oldest
        # ones become permanently unsearchable, and a vector id outside that
        # window is dropped. Build a lexical candidate set with SQL LIKE
        # (bounded) and fetch vector hits by id independently.
        q_tokens = list(dict.fromkeys(tok for tok in _tokens(q) if tok))[:24]
        candidates: List[Episode] = []
        if q_tokens:
            clauses = []
            candidate_params = list(base_params)
            for token in q_tokens:
                escaped = _like_escape(token)
                clauses.append("(user_msg LIKE ? ESCAPE '~' OR ai_reply LIKE ? ESCAPE '~')")
                candidate_params.extend([f"%{escaped}%", f"%{escaped}%"])
            sql = base + " AND (" + " OR ".join(clauses) + ")"
            sql += f" ORDER BY ts DESC LIMIT {MAX_EPISODE_CANDIDATES}"
            with self._lock:
                rows = self.conn.execute(sql, candidate_params).fetchall()
            candidates = [self._from_row(row) for row in rows]

        vector_hits: List[tuple[float, Episode]] = []
        if self.vector_index is not None:
            try:
                hits = self.vector_index.query(
                    q, user_id=user_id, limit=max(10, int(limit) * 3),
                    after=after_ts, before=before_ts,
                )
                ids = [str(hit.get("id")) for hit in hits if hit.get("id")]
                fetched = self.get_many(ids)
                for hit in hits:
                    episode = fetched.get(str(hit.get("id")))
                    if episode is None:
                        continue
                    if episode.user_id != user_id:
                        continue
                    if after_ts is not None and episode.ts < after_ts:
                        continue
                    if before_ts is not None and episode.ts > before_ts:
                        continue
                    vector_hits.append((float(hit.get("score", 0.0)), episode))
            except Exception:
                logger.warning("episode vector query failed; using lexical fallback", exc_info=True)

        q_set = set(_tokens(q))
        lexical_hits: List[tuple[float, Episode]] = []
        if q_set and candidates:
            docs = [(e, _tokens(e.user_msg + " " + e.ai_reply)) for e in candidates]
            df = {tok: 0 for tok in q_set}
            for _ep, toks in docs:
                tok_set = set(toks)
                for tok in q_set & tok_set:
                    df[tok] += 1
            total = len(docs)
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
                lexical_hits.append((lexical + 0.25 / (1.0 + age_days / 30.0), ep))
            lexical_hits.sort(key=lambda item: (-item[0], -item[1].ts))

        # Merge the two arms on one scale instead of concatenating: appending
        # lexical hits after every vector hit meant lexical contributed nothing
        # once the vector arm filled the limit, so "hybrid" was really
        # vector-only. Pure rank fusion (RRF) was measured separately and lost
        # 15 points of recall@3 here, because the caller aggregates episode
        # scores per session with max() and rank-only scores destroy the
        # magnitude information that aggregation relies on. Min-max scaling per
        # arm keeps both the scale and the contribution.
        by_id: Dict[str, Episode] = {}
        blended: Dict[str, float] = {}
        for weight, hits in ((VECTOR_WEIGHT, vector_hits), (LEXICAL_WEIGHT, lexical_hits)):
            if not hits:
                continue
            top = max(score for score, _ in hits) or 1.0
            for score, episode in hits:
                by_id.setdefault(episode.id, episode)
                blended[episode.id] = blended.get(episode.id, 0.0) + (
                    weight * max(0.0, score) / top
                )
        ordered = sorted(blended.items(), key=lambda item: (-item[1], item[0]))
        reranked = False
        if reranker is not None and q and ordered:
            head = ordered[:max(1, int(rerank_top))]
            try:
                pairs = [(q, _episode_doc(by_id[eid])) for eid, _ in head]
                scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
                by_score = sorted(
                    ((eid, float(score)) for (eid, _), score in zip(head, scores)),
                    key=lambda kv: (-kv[1], kv[0]),
                )
                ordered = by_score + ordered[len(head):]
                reranked = True
            except Exception:
                logger.warning("episode rerank failed; keeping fused order", exc_info=True)
        rows = []
        for rank, (episode_id, score) in enumerate(ordered[:max(1, int(limit))], 1):
            # A cross-encoder returns logits on a different scale from the fused
            # fused score (roughly -11..+11 vs 0..2). Mixing the two and letting a
            # caller re-sort by score silently inverted the head and the tail.
            # Emit a rank-monotone score instead so any downstream max/order
            # aggregation sees one consistent scale.
            value = 1.0 / rank if reranked else score
            rows.append({
                **by_id[episode_id].to_dict(include_reply, self.tz),
                "score": round(value, 6),
            })
        return rows

    def delete(self, episode_id: str) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
            self.conn.commit()
        if cur.rowcount and self.vector_index is not None:
            try:
                self.vector_index.delete([episode_id])
            except Exception:
                logger.warning("episode vector delete failed", exc_info=True)
        return cur.rowcount > 0

    def prune(
        self,
        user_id: Optional[str] = None,
        *,
        max_age_days: float = DEFAULT_EPISODE_TTL_DAYS,
        now: Any = None,
    ) -> int:
        """Delete episodes older than the configured retention window.

        There is no background scheduler; the host calls this explicitly.
        """
        if max_age_days is None or float(max_age_days) <= 0:
            return 0
        now_ts = _epoch(now) if now is not None else datetime.now(_zone(self.tz)).timestamp()
        if now_ts is None:
            now_ts = datetime.now(_zone(self.tz)).timestamp()
        cutoff = now_ts - float(max_age_days) * 86400.0
        return self.purge(user_id=user_id, before=cutoff)

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
            rows = self.conn.execute(
                "SELECT id FROM episodes" + (" WHERE " + " AND ".join(clauses) if clauses else ""),
                params,
            ).fetchall()
            ids = [row["id"] for row in rows]
            cur = self.conn.execute(sql, params)
            self.conn.commit()
        if ids and self.vector_index is not None:
            try:
                self.vector_index.delete(ids)
            except Exception:
                logger.warning("episode vector purge failed", exc_info=True)
        return cur.rowcount

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            count = self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        return {
            "episodes": int(count), "path": self.path,
            "vector_index": type(self.vector_index).__name__ if self.vector_index else None,
        }

    def close(self) -> None:
        with self._lock:
            self.conn.close()
        if self.vector_index is not None:
            try:
                self.vector_index.close()
            except Exception:
                pass
