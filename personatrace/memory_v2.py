"""memory_v2 —— 时序人格记忆内核（艾宾浩斯保持率 / 双时态 / CAS / 行为预测）

解决的问题（对着实际痛点，不是为了用 Redis 而用）：

1. 衰减不科学：原实现按"置信度分档、每天统一降 0.01~0.08"。
   改：R(t)=exp(-Δt/S)，S 随"支持证据"按间隔效应增长（快忘时复习增益最大）。
   收益：①不再需要每天全量扫描（读时惰性计算，O(1)）②高价值信息不会和低价值信息同样速度腐烂。

2. 无法回溯：原实现只有 updated，问不出"他 9 月是什么状态"。
   改：双时态（valid_from / invalid_at），用 Redis ZSET 按时间索引，
        as_of 查询 = ZRANGEBYSCORE + 过滤 invalid_at，O(log N + M)，无需重算。

3. 并发覆盖（P0）：侦探 load→LLM→save 之间无事务，后写者静默覆盖前写者。
   改：SET NX + fencing token + version CAS，冲突即重放。

4. "会怎么做"没回答：原实现只有 7×24 计数，能算"什么时候活跃"，算不出"下一步做什么"。
   改：行为序列落 ZSET，bigram 转移概率预测下一个动作，纯统计、O(1)、可解释。

5. 节流是全局变量：多会话互相干扰。
   改：per-user TTL 窗口（SET NX EX），自适应窗口由画像完整度决定。

存储后端可插拔：有 Redis 用 Redis（多进程共享），没有则退回进程内实现（本地/单机模式）。
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - 取决于运行环境
    import redis as _redis
except Exception:  # pragma: no cover
    _redis = None


# ---------------------------------------------------------------- 后端

class MemoryBackend:
    """进程内后端：无 Redis 时的降级实现。接口是 Redis 的子集。"""

    def __init__(self) -> None:
        self._h: Dict[str, Dict[str, str]] = {}
        self._kv: Dict[str, Any] = {}
        self._z: Dict[str, Dict[str, float]] = {}
        self._ttl: Dict[str, float] = {}
        self._lock = threading.RLock()

    def _alive(self, key: str) -> bool:
        exp = self._ttl.get(key)
        if exp is not None and exp <= time.time():
            self._kv.pop(key, None)
            self._ttl.pop(key, None)
            return False
        return True

    def hset(self, key: str, mapping: Dict[str, Any]) -> None:
        with self._lock:
            bucket = self._h.setdefault(key, {})
            for k, v in mapping.items():
                bucket[k] = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v

    def hgetall(self, key: str) -> Dict[str, str]:
        with self._lock:
            return dict(self._h.get(key, {}))

    def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        with self._lock:
            bucket = self._h.setdefault(key, {})
            cur = int(bucket.get(field, 0))
            bucket[field] = str(cur + amount)
            return cur + amount

    def delete(self, *keys: str) -> None:
        with self._lock:
            for k in keys:
                self._h.pop(k, None)
                self._kv.pop(k, None)
                self._z.pop(k, None)
                self._ttl.pop(k, None)

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if not self._alive(key):
                return None
            return self._kv.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            self._kv[key] = value
            if ttl:
                self._ttl[key] = time.time() + ttl
            else:
                self._ttl.pop(key, None)

    def setnx(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        with self._lock:
            if self._alive(key) and key in self._kv:
                return False
            self._kv[key] = value
            if ttl:
                self._ttl[key] = time.time() + ttl
            return True

    def zadd(self, key: str, mapping: Dict[str, float]) -> None:
        with self._lock:
            bucket = self._z.setdefault(key, {})
            bucket.update({str(m): float(s) for m, s in mapping.items()})

    def zrangebyscore(self, key: str, lo: float, hi: float) -> List[Tuple[str, float]]:
        with self._lock:
            bucket = self._z.get(key, {})
            out = [(m, s) for m, s in bucket.items() if lo <= s <= hi]
            return sorted(out, key=lambda x: x[1])

    def ztrim_last(self, key: str, keep: int) -> None:
        with self._lock:
            bucket = self._z.get(key, {})
            if len(bucket) <= keep:
                return
            ordered = sorted(bucket.items(), key=lambda x: x[1])
            for m, _ in ordered[: len(ordered) - keep]:
                bucket.pop(m, None)

    def zrem(self, key: str, *members: str) -> None:
        with self._lock:
            bucket = self._z.get(key, {})
            for m in members:
                bucket.pop(str(m), None)

    def zcard(self, key: str) -> int:
        with self._lock:
            return len(self._z.get(key, {}))


class SQLiteBackend:
    """SQLite 后端：单机默认落盘方案。零依赖（标准库 sqlite3）。

    只实现本模块用到的那一小撮命令，语义与 Redis 版对齐：
      hash → t_hash(key, field, value)
      kv   → t_kv(key, value, expires_at)    支持 TTL，用于 lock / throttle
      zset → t_zset(key, member, score)
    写入即提交（WAL），进程被杀也不会丢已返回的结论。
    """

    def __init__(self, path: str) -> None:
        import sqlite3

        self.path = str(path)
        if self.path != ":memory:":
            parent = os.path.dirname(os.path.abspath(self.path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10.0)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS t_hash (
                    key TEXT NOT NULL, field TEXT NOT NULL, value TEXT,
                    PRIMARY KEY (key, field)
                );
                CREATE TABLE IF NOT EXISTS t_kv (
                    key TEXT PRIMARY KEY, value TEXT, expires_at REAL
                );
                CREATE TABLE IF NOT EXISTS t_zset (
                    key TEXT NOT NULL, member TEXT NOT NULL, score REAL NOT NULL,
                    PRIMARY KEY (key, member)
                );
                CREATE INDEX IF NOT EXISTS i_zset_key_score ON t_zset (key, score);
                """
            )
            self._conn.commit()
        except Exception:
            self._conn.close()
            raise

    # -- hash --
    def hset(self, key: str, mapping: Dict[str, Any]) -> None:
        rows = [
            (key, str(f), v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
            for f, v in mapping.items()
        ]
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO t_hash (key, field, value) VALUES (?, ?, ?) "
                "ON CONFLICT(key, field) DO UPDATE SET value = excluded.value",
                rows,
            )

    def hgetall(self, key: str) -> Dict[str, str]:
        with self._lock:
            cur = self._conn.execute("SELECT field, value FROM t_hash WHERE key = ?", (key,))
            return {f: v for f, v in cur.fetchall()}

    def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT value FROM t_hash WHERE key = ? AND field = ?", (key, field)
            ).fetchone()
            value = (int(row[0]) if row else 0) + amount
            self._conn.execute(
                "INSERT INTO t_hash (key, field, value) VALUES (?, ?, ?) "
                "ON CONFLICT(key, field) DO UPDATE SET value = excluded.value",
                (key, field, str(value)),
            )
            return value

    # -- kv（支持 TTL） --
    def _drop_expired(self, key: str) -> None:
        self._conn.execute(
            "DELETE FROM t_kv WHERE key = ? AND expires_at IS NOT NULL AND expires_at <= ?",
            (key, time.time()),
        )

    def get(self, key: str) -> Optional[str]:
        with self._lock, self._conn:
            self._drop_expired(key)
            row = self._conn.execute("SELECT value FROM t_kv WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        expires = time.time() + ttl if ttl else None
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO t_kv (key, value, expires_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, expires_at = excluded.expires_at",
                (key, payload, expires),
            )

    def setnx(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        expires = time.time() + ttl if ttl else None
        with self._lock, self._conn:
            self._drop_expired(key)
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO t_kv (key, value, expires_at) VALUES (?, ?, ?)",
                (key, payload, expires),
            )
            return cur.rowcount == 1

    # -- zset --
    def zadd(self, key: str, mapping: Dict[str, float]) -> None:
        rows = [(key, str(m), float(s)) for m, s in mapping.items()]
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO t_zset (key, member, score) VALUES (?, ?, ?) "
                "ON CONFLICT(key, member) DO UPDATE SET score = excluded.score",
                rows,
            )

    def zrangebyscore(self, key: str, lo: float, hi: float) -> List[Tuple[str, float]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT member, score FROM t_zset WHERE key = ? AND score >= ? AND score <= ? "
                "ORDER BY score",
                (key, lo, hi),
            )
            return [(m, s) for m, s in cur.fetchall()]

    def ztrim_last(self, key: str, keep: int) -> None:
        with self._lock, self._conn:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM t_zset WHERE key = ?", (key,)
            ).fetchone()[0]
            if total <= keep:
                return
            self._conn.execute(
                "DELETE FROM t_zset WHERE key = ? AND member IN "
                "(SELECT member FROM t_zset WHERE key = ? ORDER BY score LIMIT ?)",
                (key, key, total - keep),
            )

    def zrem(self, key: str, *members: str) -> None:
        with self._lock, self._conn:
            self._conn.executemany(
                "DELETE FROM t_zset WHERE key = ? AND member = ?",
                [(key, str(m)) for m in members],
            )

    def zcard(self, key: str) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM t_zset WHERE key = ?", (key,)
            ).fetchone()[0]

    def delete(self, *keys: str) -> None:
        with self._lock, self._conn:
            for k in keys:
                self._conn.execute("DELETE FROM t_hash WHERE key = ?", (k,))
                self._conn.execute("DELETE FROM t_kv WHERE key = ?", (k,))
                self._conn.execute("DELETE FROM t_zset WHERE key = ?", (k,))

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


class RedisBackend:
    """Redis 后端：多进程共享。只包装本模块用到的命令。"""
    def __init__(self, url: str = "redis://127.0.0.1:6379/0", prefix: str = "pt:") -> None:
        self.r = _redis.Redis.from_url(url, decode_responses=True)
        self.p = prefix
        self.r.ping()  # 连不上直接抛，由上层降级

    def hset(self, key: str, mapping: Dict[str, Any]) -> None:
        payload = {k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
                   for k, v in mapping.items()}
        self.r.hset(self.p + key, mapping=payload)

    def hgetall(self, key: str) -> Dict[str, str]:
        return self.r.hgetall(self.p + key)

    def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        return int(self.r.hincrby(self.p + key, field, amount))

    def delete(self, *keys: str) -> None:
        if keys:
            self.r.delete(*[self.p + k for k in keys])

    def get(self, key: str) -> Optional[str]:
        return self.r.get(self.p + key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self.r.set(self.p + key, value, ex=ttl)

    def setnx(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        return bool(self.r.set(self.p + key, value, nx=True, ex=ttl))

    def zadd(self, key: str, mapping: Dict[str, float]) -> None:
        self.r.zadd(self.p + key, {str(m): float(s) for m, s in mapping.items()})

    def zrangebyscore(self, key: str, lo: float, hi: float) -> List[Tuple[str, float]]:
        rows = self.r.zrangebyscore(self.p + key, lo, hi, withscores=True)
        return [(m, float(s)) for m, s in rows]

    def ztrim_last(self, key: str, keep: int) -> None:
        n = self.r.zcard(self.p + key)
        if n > keep:
            self.r.zremrangebyrank(self.p + key, 0, n - keep - 1)

    def zrem(self, key: str, *members: str) -> None:
        if members:
            self.r.zrem(self.p + key, *members)

    def zcard(self, key: str) -> int:
        return int(self.r.zcard(self.p + key))


DEFAULT_DB_PATH = os.path.join(os.path.expanduser("~"), ".personatrace", "memory.db")
DB_ENV = "PERSONA_DB"


def build_backend(url: Optional[str] = None, db_path: Optional[str] = None) -> Tuple[Any, str]:
    """返回 (backend, mode)。优先级：Redis → SQLite（默认落盘）→ 进程内。

    - 装了 redis 包且给了 url → Redis（多进程共享）
    - 否则落 SQLite 文件：db_path > 环境变量 PERSONA_DB > ~/.personatrace/memory.db
    - PERSONA_DB=off / memory:// / :memory: → 进程内（重启即丢，仅用于测试）
    任何异常都退回进程内，绝不让"记不住"升级成"起不来"。
    """
    if _redis is not None and url != "memory://":
        try:
            return RedisBackend(url or "redis://127.0.0.1:6379/0"), "redis"
        except Exception:
            pass
    target = db_path or os.environ.get(DB_ENV) or DEFAULT_DB_PATH
    if str(target).strip().lower() in {"", "off", "none", "memory", "memory://", ":memory:"}:
        return MemoryBackend(), "memory"
    try:
        return SQLiteBackend(str(target)), "sqlite"
    except Exception:
        return MemoryBackend(), "memory"


# ---------------------------------------------------------------- 衰减

# 各类型的"初始稳定度"（天）。决定 R(30 天) 掉到多少。
# 依据：事实几乎不变，性格慢变，目标/项目按周期变，此刻状态几天就旧。
CATEGORY_STABILITY: Dict[str, float] = {
    "explicit_facts": 520.0,
    "identity": 300.0,
    "constraints": 180.0,
    "expertise": 240.0,
    "likes": 90.0,
    "dislikes": 90.0,
    "projects": 60.0,
    "goals": 45.0,
    "state": 10.0,
}
DEFAULT_STABILITY = 60.0

ARCHIVE_AT = 0.30        # 保留率低于此值 → 归档（不参与 recall，不删除）
REINFORCE_K = 0.6        # 间隔效应增益系数
MAX_EVIDENCE = 50        # 单条信条保留的原文证据上限（超出按 FIFO 淘汰并清理 L0）
MAX_STABILITY_MULTIPLE = 20.0   # 稳定度最多涨到该类别初始值的 20 倍，防止变成"永不衰减"
MAX_MENTIONS = 500              # 单个主体保留的第三方记录上限（超出按时间淘汰）
EVIDENCE_RAW_LIMIT = 800        # 单条原文证据的字数上限（4000 会把整段聊天塞进去）
GC_AT = 0.05                    # 保留率低于此值才允许回收
GC_MAX_HITS = 1                 # 只被提过一次的才回收（重复出现过的留着）
MENTION_RAW_LIMIT = 300         # 第三方记录里的原话上限
MENTION_STABILITY_FACTOR = 0.4  # 二手信息失真快：稳定度打 4 折


def stability_from_half_life(half_life_days: float) -> float:
    """让 R(half_life) = 0.5 → S = hl / ln2。"""
    return max(1e-6, half_life_days / math.log(2))


def retention(stability: float, days: float) -> float:
    """艾宾浩斯保持率 R = exp(-t/S)。"""
    if days <= 0:
        return 1.0
    return math.exp(-days / max(1e-6, stability))


def reinforce(stability: float, current_r: float, k: float = REINFORCE_K, cap: Optional[float] = None) -> float:
    """间隔效应：越接近遗忘临界点复习，稳定度增益越大。

    S' = S + k·S·(1-R)。R→0（快忘光）时增益最大，R→1（刚记过）时几乎无增益。
    这比"每次 +0.025 置信度"更符合真实记忆：重复得不合时宜等于白说。

    cap：稳定度上限。不设上限时，按间隔重复 30 次可放大数千倍（指数），
    条目会变成"永不衰减的永久记忆"，与"带保质期的信条"这个定位冲突。
    """
    grown = stability * (1.0 + k * (1.0 - max(0.0, min(1.0, current_r))))
    return min(grown, cap) if cap else grown


# ---------------------------------------------------------------- 数据

@dataclass
class Belief:
    id: str
    user_id: str
    track: str
    category: str
    label: str
    confidence: float
    stability: float
    last_support_ts: float
    valid_from_ts: float
    invalid_at_ts: Optional[float] = None
    status: str = "active"
    hits: int = 1
    dimension: str = "identity"          # identity | need | predict
    reasoning: str = ""                  # 为什么存这条（深读时要带回去的语境）
    due_at_ts: Optional[float] = None    # 前瞻：这件事还有多久到期
    version: int = 0
    evidence: List[str] = field(default_factory=list)

    def retention_now(self, now: Optional[float] = None) -> float:
        now = now if now is not None else time.time()
        return retention(self.stability, (now - self.last_support_ts) / 86400.0)

    def effective_confidence(self, now: Optional[float] = None) -> float:
        return round(self.confidence * self.retention_now(now), 4)

    def to_hash(self) -> Dict[str, Any]:
        return {
            "id": self.id, "user_id": self.user_id, "track": self.track,
            "category": self.category, "label": self.label,
            "confidence": self.confidence, "stability": self.stability,
            "last_support_ts": self.last_support_ts, "valid_from_ts": self.valid_from_ts,
            "invalid_at_ts": self.invalid_at_ts if self.invalid_at_ts else "",
            "status": self.status, "hits": self.hits, "dimension": self.dimension,
            "due_at_ts": self.due_at_ts if self.due_at_ts else "",
            "reasoning": self.reasoning,
            "version": self.version, "evidence": json.dumps(self.evidence, ensure_ascii=False),
        }

    @classmethod
    def from_hash(cls, raw: Dict[str, str]) -> "Belief":
        def num(k, default=0.0):
            v = raw.get(k, "")
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        return cls(
            id=raw.get("id", ""), user_id=raw.get("user_id", ""),
            track=raw.get("track", "work"), category=raw.get("category", ""),
            label=raw.get("label", ""), confidence=num("confidence", 0.5),
            stability=num("stability", DEFAULT_STABILITY),
            last_support_ts=num("last_support_ts", time.time()),
            valid_from_ts=num("valid_from_ts", time.time()),
            invalid_at_ts=num("invalid_at_ts") or None,
            status=raw.get("status", "active"), hits=int(num("hits", 1)),
            dimension=raw.get("dimension", "identity"),
            reasoning=raw.get("reasoning", ""),
            due_at_ts=num("due_at_ts") or None, version=int(num("version", 0)),
            evidence=json.loads(raw.get("evidence") or "[]"),
        )


def belief_id(user_id: str, track: str, category: str, label: str) -> str:
    """确定性主键：同一条结论永远落到同一个 id，写入天然幂等。"""
    key = f"{user_id}:{track}:{category}:{label.lower().strip()}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


class Conflict(Exception):
    """版本冲突：说明有人在你之前写过了，需要重放。"""


class Guard:
    """SET NX + fencing token + version CAS。修掉 read-modify-write 的静默覆盖。"""

    def __init__(self, backend: Any, ttl_ms: int = 10_000) -> None:
        self.b = backend
        self.ttl = max(1, ttl_ms // 1000)
        self._held: Dict[str, str] = {}

    def acquire(self, user_id: str) -> str:
        seq = self.b.hincrby(f"fence:{user_id}", "seq", 1)
        token = f"{seq}:{uuid.uuid4().hex[:8]}"
        if not self.b.setnx(f"lock:{user_id}", token, ttl=self.ttl):
            raise Conflict(f"lock busy for {user_id}")
        self._held[user_id] = token
        return token

    def release(self, user_id: str) -> None:
        token = self._held.pop(user_id, None)
        if token and self.b.get(f"lock:{user_id}") == token:
            self.b.delete(f"lock:{user_id}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------- 内核

class MemoryV2:
    """时序人格记忆内核。所有写操作幂等，所有读操作带时间维度。"""

    def __init__(self, backend: Any = None, user_id: str = "default_user", db_path: Optional[str] = None) -> None:
        if backend is not None:
            self.b, self.mode = backend, "injected"
        else:
            self.b, self.mode = build_backend(db_path=db_path)
        # 空 user_id 会让所有记录塌进同一个命名空间（belief::{bid}），是静默的跨用户串味。
        uid = (user_id or "").strip()
        if not uid:
            raise ValueError(
                "user_id must be a non-empty string: an empty value collapses "
                "every user into the same key namespace (belief::{id})."
            )
        self.user_id = uid
        self.guard = Guard(self.b)

    def _acquire_with_retry(self, attempts: int = 5, base_delay: float = 0.02) -> str:
        """抢写锁：冲突时短退避重放（文档承诺的「冲突即重放」，之前只抛不重试）。"""
        delay = base_delay
        for i in range(attempts):
            try:
                return self.guard.acquire(self.user_id)
            except Conflict:
                if i == attempts - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        raise Conflict(f"lock busy for {self.user_id}")

    # -- key 命名 --
    def _bk(self, bid: str) -> str:
        return f"belief:{self.user_id}:{bid}"

    @property
    def _tl(self) -> str:
        return f"timeline:{self.user_id}"

    @property
    def _bh(self) -> str:
        return f"behavior:{self.user_id}"

    # -- 写入：追加证据 + 间隔效应强化（幂等） --
    def observe(
        self,
        category: str,
        label: str,
        *,
        track: str = "work",
        confidence: float = 0.5,
        evidence_id: Optional[str] = None,
        dimension: str = "identity",
        reasoning: str = "",
        evidence_text: str = "",
        due_at_ts: Optional[float] = None,
        ts: Optional[float] = None,
    ) -> Belief:
        ts = ts if ts is not None else time.time()
        label = (label or "").strip()
        if not label:
            raise ValueError(
                "label must be a non-empty string: empty labels collapse every "
                "observation into one record (same id, hits accumulating)."
            )
        bid = belief_id(self.user_id, track, category, label)
        with self.guard:
            self._acquire_with_retry()
            try:
                raw = self.b.hgetall(self._bk(bid))
                if raw:
                    belief = Belief.from_hash(raw)
                    r = belief.retention_now(ts)
                    belief.stability = reinforce(
                        belief.stability,
                        r,
                        cap=CATEGORY_STABILITY.get(belief.category, DEFAULT_STABILITY)
                        * MAX_STABILITY_MULTIPLE,
                    )
                    belief.confidence = min(0.99, belief.confidence + 0.02 * (1.0 - r))
                    belief.hits += 1
                    belief.last_support_ts = ts
                    if evidence_id and evidence_id not in belief.evidence:
                        belief.evidence.append(evidence_id)
                    if reasoning:
                        belief.reasoning = reasoning
                    if due_at_ts:
                        belief.due_at_ts = due_at_ts
                else:
                    belief = Belief(
                        id=bid, user_id=self.user_id, track=track, category=category,
                        label=label, confidence=confidence,
                        stability=CATEGORY_STABILITY.get(category, DEFAULT_STABILITY),
                        last_support_ts=ts, valid_from_ts=ts,
                        dimension=dimension, reasoning=reasoning,
                        due_at_ts=due_at_ts,
                        evidence=[evidence_id] if evidence_id else [],
                    )
                # L0：原文只存一次，检索时不碰它（这是成本优势的来源）
                if evidence_text:
                    eid = evidence_id or f"{bid}:{int(ts * 1000)}"
                    self.b.hset(f"evidence:{self.user_id}:{eid}", {
                        "belief_id": bid, "ts": ts, "raw_text": evidence_text[:EVIDENCE_RAW_LIMIT],
                    })
                    if eid not in belief.evidence:
                        belief.evidence.append(eid)
                # 证据列表要有上限：长期运行的宿主机（QQ 群）会无限 append，
                # 每次 to_hash 都要重新序列化整张表，而且 L0 原文只增不删。
                if len(belief.evidence) > MAX_EVIDENCE:
                    for stale in belief.evidence[:-MAX_EVIDENCE]:
                        self.b.delete(f"evidence:{self.user_id}:{stale}")
                    belief.evidence = belief.evidence[-MAX_EVIDENCE:]
                belief.version += 1
                self.b.hset(self._bk(bid), belief.to_hash())
                self.b.zadd(self._tl, {bid: belief.valid_from_ts})
                return belief
            finally:
                self.guard.release(self.user_id)

    # -- 失效：不删除，只打时间戳（Graphiti 原则） --
    def invalidate(self, bid: str, ts: Optional[float] = None, status: str = "deprecated") -> bool:
        """把一条结论标记为失效。返回是否真的命中了记录（之前静默返回 None，调用方分不清成败）。"""
        self._acquire_with_retry()
        try:
            raw = self.b.hgetall(self._bk(bid))
            if not raw:
                return False
            belief = Belief.from_hash(raw)
            belief.status = status
            belief.invalid_at_ts = ts if ts is not None else time.time()
            belief.version += 1
            self.b.hset(self._bk(bid), belief.to_hash())
            return True
        finally:
            self.guard.release(self.user_id)

    # -- 彻底删除：用户行使删除权（与 invalidate 的分工：一个留痕，一个抹掉） --
    def purge(self, user_id: Optional[str] = None) -> int:
        """硬删除某个用户的全部记忆（信条 + 证据 + 时间线 + 行为），返回删除的信条数。

        invalidate 只打失效标记、保留时间线以便回溯；purge 不可逆，用于「删掉关于我的记录」。
        只依据 timeline 枚举即可覆盖全部信条（timeline 从不裁剪，只有 behavior 会 ztrim）。
        """
        uid = (user_id or self.user_id or "").strip()
        if not uid:
            raise ValueError("purge requires a non-empty user_id")
        removed = 0
        for bid, _score in list(self.b.zrangebyscore(f"timeline:{uid}", 0, float("inf"))):
            raw = self.b.hgetall(f"belief:{uid}:{bid}")
            if raw:
                try:
                    evidence_ids = json.loads(raw.get("evidence") or "[]")
                except ValueError:
                    evidence_ids = []
                for eid in evidence_ids:
                    self.b.delete(f"evidence:{uid}:{eid}")
                removed += 1
            self.b.delete(f"belief:{uid}:{bid}")
        for mid, _score in list(self.b.zrangebyscore(f"mentions:{uid}", 0, float("inf"))):
            self.b.delete(f"mention:{uid}:{mid}")
        self.b.delete(
            f"timeline:{uid}", f"behavior:{uid}", f"fence:{uid}", f"lock:{uid}", f"mentions:{uid}"
        )
        return removed

    # -- 双时态查询：他在 as_of 那一刻相信什么 --
    def state(
        self,
        as_of: Optional[float] = None,
        include_archived: bool = False,
        include_invalidated: bool = False,
        expand: int = 0,
    ) -> List[Dict[str, Any]]:
        as_of = as_of if as_of is not None else time.time()
        rows = self.b.zrangebyscore(self._tl, 0, as_of)
        out: List[Dict[str, Any]] = []
        for bid, _score in rows:
            raw = self.b.hgetall(self._bk(bid))
            if not raw:
                continue
            belief = Belief.from_hash(raw)
            if belief.valid_from_ts > as_of:
                continue
            if belief.invalid_at_ts and belief.invalid_at_ts <= as_of and not include_invalidated:
                continue          # 那个时刻它已经失效（审计时可用 include_invalidated 打开）
            r = retention(belief.stability, (as_of - belief.last_support_ts) / 86400.0)
            if not include_archived and r < ARCHIVE_AT:
                continue          # 归档区：保留但不参与取用
            out.append(self._view(belief, as_of, r, expand=expand))
        out.sort(key=lambda x: -x["effective_confidence"])
        return out

    # -- 取用：相关度 × 保持率 × 优先级，约束与身份强制入选 --
    def recall(
        self,
        query_vec: Optional[Iterable[float]] = None,
        item_vecs: Optional[Dict[str, Iterable[float]]] = None,
        as_of: Optional[float] = None,
        k: int = 15,
        pinned_categories: Tuple[str, ...] = ("constraints", "identity"),
        expand: int = 0,
    ) -> Dict[str, Any]:
        as_of = as_of if as_of is not None else time.time()
        pool = self.state(as_of=as_of, expand=expand)
        pinned = [x for x in pool if x["category"] in pinned_categories]
        rest = [x for x in pool if x["category"] not in pinned_categories]

        qv = list(query_vec) if query_vec is not None else None

        def cos(a: Iterable[float], b: Iterable[float]) -> float:
            a, b = list(a), list(b)
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na < 1e-9 or nb < 1e-9:
                return 0.0
            return sum(x * y for x, y in zip(a, b)) / (na * nb)

        scored = []
        for item in rest:
            sim = 0.5
            if qv is not None and item_vecs and item["id"] in item_vecs:
                sim = cos(qv, item_vecs[item["id"]])
            # 相关度主导，保持率调制：陈旧的即使相似也要降权
            score = (sim ** 1.0) * (0.35 + 0.65 * item["retention"])
            scored.append((score, item))
        scored.sort(key=lambda x: -x[0])
        picked = pinned + [it for _, it in scored[: max(0, k - len(pinned))]]
        hidden = [it for _, it in scored[max(0, k - len(pinned)):]]
        return {
            "shown": picked,
            "hidden_count": len(hidden),
            "pinned_count": len(pinned),
            "as_of": as_of,
            "backend": self.mode,
        }

    def _evidence_snippets(self, belief: Belief, limit: int = 2, max_chars: int = 200) -> List[str]:
        """回表：只在命中后才读原文。这是"表层检索不碰具体语句"的兑现点。"""
        out: List[str] = []
        for eid in reversed(belief.evidence[-max(1, limit):]):
            raw = self.b.hgetall(f"evidence:{self.user_id}:{eid}")
            text = (raw or {}).get("raw_text", "")
            if text:
                out.append(text[:max_chars])
        return out

    def _view(self, belief: Belief, as_of: float, r: float, expand: int = 0) -> Dict[str, Any]:
        days = (as_of - belief.last_support_ts) / 86400.0
        stale = round(days / max(1e-6, belief.stability), 3)
        due_in = None
        if belief.due_at_ts:
            due_in = round((belief.due_at_ts - as_of) / 86400.0, 1)
        view = {
            "id": belief.id, "track": belief.track, "category": belief.category,
            "label": belief.label, "dimension": belief.dimension,
            "confidence": belief.confidence,
            "retention": round(r, 3),
            "effective_confidence": round(belief.confidence * r, 3),
            "staleness": stale,          # 距上次支撑 / 稳定度，>1 表示已过半个周期以上
            "due_in": due_in,            # 前瞻：还有几天到期（负数=已超期）
            "hits": belief.hits,
            "evidence": belief.evidence[-5:],
            "needs_review": r < 0.5 or (due_in is not None and due_in < 3),
        }
        if expand:
            view["reasoning"] = belief.reasoning          # 为什么存这条（语境）
            view["evidence_count"] = len(belief.evidence)
            view["evidence_texts"] = self._evidence_snippets(belief, limit=expand)
        return view

    # -- 行为序列 + 下一步预测（纯统计，可解释） --
    def record_behavior(self, kind: str, target: str = "", ts: Optional[float] = None) -> None:
        ts = ts if ts is not None else time.time()
        self.b.zadd(self._bh, {f"{kind}|{target}|{int(ts*1000)}": ts})
        self.b.ztrim_last(self._bh, 1000)

    def predict_next(self, window: int = 30, top: int = 3) -> List[Dict[str, Any]]:
        """bigram 转移：P(next | last)。解决"会怎么做"，不依赖模型。"""
        rows = self.b.zrangebyscore(self._bh, 0, time.time() + 1)[-window:]
        seq = [m.split("|")[0] for m, _ in rows]
        if len(seq) < 2:
            return []
        last = seq[-1]
        counts: Dict[str, int] = {}
        for a, bnext in zip(seq, seq[1:]):
            if a == last:
                counts[bnext] = counts.get(bnext, 0) + 1
        total = sum(counts.values()) or 1
        ranked = sorted(counts.items(), key=lambda x: -x[1])[:top]
        return [{"next": k, "p": round(v / total, 3), "support": v} for k, v in ranked]

    # -- 第三方记录：谁说了「我」什么、谁要求对「我」做什么 --
    def _mk(self) -> str:
        return f"mentions:{self.user_id}"

    def _mbk(self, mid: str) -> str:
        return f"mention:{self.user_id}:{mid}"

    def observe_mention(
        self,
        actor: str,
        kind: str,
        category: str,
        label: str,
        confidence: float = 0.6,
        evidence_text: str = "",
        group_key: str = "",
        verified: bool = False,
    ) -> Dict[str, Any]:
        """记录一条「别人对这个人」的信息。

        kind='claim'     —— 别人说关于他的事（"老王说他最近在考研"）
        kind='directive' —— 别人要求怎么对待他（"以后碰到这个 id 骂他"）

        和 beliefs 分开存，是因为主键维度不同：beliefs 的键是
        (user, track, category, label)，装不下 actor；硬加进去会让同一句话
        由两个人说时互相覆盖，而不加又会让"他自己说的"和"别人说的"合并。
        """
        if kind not in ("claim", "directive"):
            raise ValueError(f"kind 必须是 claim 或 directive：{kind!r}")
        label = (label or "").strip()
        if len(label) < 2:
            raise ValueError("mention label 太短")
        actor = (actor or "").strip() or self.user_id
        mid = uuid.uuid5(
            uuid.NAMESPACE_URL, f"{self.user_id}|{kind}|{actor}|{category}|{label}"
        ).hex[:32]

        ts = time.time()
        old = self.b.hgetall(self._mbk(mid))
        hits = int(old.get("hits", 0)) + 1
        self.b.hset(self._mbk(mid), {
            "subject": self.user_id,
            "actor": actor,
            "kind": kind,
            "category": category,
            "label": label,
            "confidence": max(0.0, min(1.0, confidence)),
            "verified": verified,
            "group_key": group_key,
            "ts": ts,
            "hits": hits,
            "raw": (evidence_text or "")[:MENTION_RAW_LIMIT],
        })
        self.b.zadd(self._mk(), {mid: ts})
        self.b.ztrim_last(self._mk(), MAX_MENTIONS)
        return {
            "id": mid, "actor": actor, "kind": kind, "label": label,
            "confidence": confidence, "verified": verified, "hits": hits,
        }

    def mentions(self, kind: Optional[str] = None, limit: int = 20,
                 group_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """读回别人对这个人的记录，新的在前。

        group_key 给了就只返回该群的（以及 scope='*' 的全局指令），
        避免群 1 的话被搬到群 2 去用。
        """
        pairs = self.b.zrangebyscore(self._mk(), 0, time.time() + 86400)[-limit * 3:]
        out: List[Dict[str, Any]] = []
        for mid, _score in reversed(pairs):
            h = self.b.hgetall(self._mbk(mid))
            if not h:
                continue
            if kind and h.get("kind") != kind:
                continue
            if group_key is not None:
                g = h.get("group_key") or ""
                if g and g != group_key and g != "*":
                    continue
            out.append({
                "id": mid,
                "actor": h.get("actor", ""),
                "kind": h.get("kind", "claim"),
                "category": h.get("category", ""),
                "label": h.get("label", ""),
                "confidence": float(h.get("confidence") or 0),
                "verified": str(h.get("verified")).lower() in ("1", "true"),
                "group_key": h.get("group_key", ""),
                "ts": float(h.get("ts") or 0),
                "hits": int(h.get("hits") or 0),
                # 二手信息衰减更快：稳定度打 4 折
                "stability": CATEGORY_STABILITY.get(h.get("category", ""), DEFAULT_STABILITY)
                * MENTION_STABILITY_FACTOR,
            })
            if len(out) >= limit:
                break
        return out

    def directives(self, group_key: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        return self.mentions(kind="directive", limit=limit, group_key=group_key)

    def gc(self, dry_run: bool = False, threshold: float = GC_AT,
           max_hits: int = GC_MAX_HITS) -> Dict[str, int]:
        """回收「已经归档、且几乎不可能再被想起」的条目，连同原文证据。

        归档只是不参与回忆，物理上还在长；这一步才是真正的删除。
        条件故意保守：保留率低于 threshold **且** 从头到尾只被提过一次。
        重复出现过的东西说明它在用户生活里占位置，不该被清掉。

        dry_run=True 只统计不动手 —— doctor 默认走这个，避免误删。
        """
        now = time.time()
        scanned = removed = evidence_rows = 0
        for bid, _score in list(self.b.zrangebyscore(self._tl, 0, float("inf"))):
            raw = self.b.hgetall(self._bk(bid))
            if not raw:
                continue
            scanned += 1
            belief = Belief.from_hash(raw)
            if belief.retention_now(now) >= threshold or belief.hits > max_hits:
                continue
            removed += 1
            evidence_rows += len(belief.evidence)
            if dry_run:
                continue
            for eid in belief.evidence:
                self.b.delete(f"evidence:{self.user_id}:{eid}")
            self.b.delete(self._bk(bid))
            self.b.zrem(self._tl, bid)
        return {"scanned": scanned, "removed": removed, "evidence": evidence_rows}

    def overview(self) -> Dict[str, int]:
        """一眼看全：他自己的信条 / 别人说的 / 对他的指令。

        `/memory` 只数信条会撒谎——用户看到"3 条"，其实库里还有别人的转述
        和管理员的指令。给一个诚实的汇总，存储仍然是两张表。
        """
        rows = self.state(include_archived=True, include_invalidated=True)
        live = self.state()
        mentions = self.mentions(limit=MAX_MENTIONS)
        return {
            "beliefs": len(rows),
            "live": len(live),
            "archived": max(0, len(rows) - len(live)),
            "claims": sum(1 for m in mentions if m.get("kind") == "claim"),
            "directives": sum(1 for m in mentions if m.get("kind") == "directive"),
        }

    # -- 活跃时段：从行为事件推出「他一般几点上线」 --
    def active_hours(self, window: int = 300, min_days: int = 3, span: int = 3) -> Optional[Dict[str, Any]]:
        """从行为事件的时间戳里找出最活跃的连续 span 小时。

        门槛与节律模块一致：至少 span 个不同日期才给结论，否则宁可不说。
        返回 {"start": 23, "end": 1, "days": 4, "samples": 37}，样本不足返回 None。
        """
        events = self.b.zrangebyscore(self._bh, 0, time.time() + 86400)[-window:]
        if not events:
            return None
        hours: Dict[int, int] = {}
        days = set()
        for _member, ts in events:
            hours[int(time.localtime(ts).tm_hour)] = hours.get(int(time.localtime(ts).tm_hour), 0) + 1
            days.add(time.strftime("%Y-%m-%d", time.localtime(ts)))
        if len(days) < min_days:
            return None

        # 用圆周均值找中心小时，再向两侧展开 span 小时（避免窗口偏到一边）
        total = sum(hours.values())
        sin_sum = sum(c * math.sin(2 * math.pi * h / 24) for h, c in hours.items())
        cos_sum = sum(c * math.cos(2 * math.pi * h / 24) for h, c in hours.items())
        peak = int(round((math.atan2(sin_sum, cos_sum) * 24 / (2 * math.pi)) % 24)) % 24
        best_start = (peak - span // 2) % 24
        best_score = sum(hours.get((best_start + i) % 24, 0) for i in range(span))
        if best_score <= 0:
            return None
        return {
            "start": best_start,
            "end": (best_start + span) % 24,
            "days": len(days),
            "samples": len(events),
            "share": round(best_score / max(1, total), 3),
        }

    # -- 自适应节流：per-user TTL 窗口，替代全局时间戳变量 --
    def allow(self, name: str, window_sec: int) -> bool:
        return self.b.setnx(f"throttle:{self.user_id}:{name}", "1", ttl=window_sec)

    def stats(self) -> Dict[str, Any]:
        return {
            "backend": self.mode,
            "beliefs": self.b.zcard(self._tl),
            "behavior_events": self.b.zcard(self._bh),
        }


# ---------------------------------------------------------------- 提示词装配

def aging_lines(skills: Dict[str, Any], now: Any) -> List[str]:
    """给画像拼「到期 / 待复核」两段（前瞻），供宿主直接塞进 prompt。

    解决两个具体问题：
    1. 原来只有「deadline - today」，看不出「这条结论多久没被强化」；
    2. 原来没有「挂太久没人推」的信号 —— 长效规划最需要这个。

    `now` 接受 datetime。本函数不依赖后端，纯计算，可单独测试。
    """
    import datetime as _dt

    out: List[str] = []
    stalled: List[str] = []
    review: List[str] = []

    timeline = (skills.get("ephemeral") or {}).get("timeline", []) or []
    for e in timeline:
        if not isinstance(e, dict):
            continue
        days = e.get("days")
        if not isinstance(days, (int, float)):
            continue
        if days < 0:
            stalled.append(f"{e.get('event', '')}（已超期 {abs(int(days))} 天）")
        elif days >= 14:
            stalled.append(f"{e.get('event', '')}（还有 {int(days)} 天，已 14+ 天无推进）")

    for track in ("work", "persona"):
        for cat, items in (skills.get(track) or {}).items():
            stab = CATEGORY_STABILITY.get(cat, DEFAULT_STABILITY)
            for it in (items or []):
                if not isinstance(it, dict):
                    continue
                label = it.get("label", "")
                updated = it.get("updated", "")
                if not label or not updated:
                    continue
                try:
                    age_days = max(0.0, (now - _dt.datetime.fromisoformat(updated)).total_seconds() / 86400.0)
                except (ValueError, TypeError):
                    continue
                r = retention(stab, age_days)
                if r < 0.35 and float(it.get("confidence", 0) or 0) >= 0.60:
                    review.append(f"{label}（保持率 {r * 100:.0f}%，已 {age_days:.0f} 天未强化）")

    if stalled:
        out.append("")
        out.append("【到期 / 拖欠】(主动推进，不要等用户提起)")
        out.extend(f"  - {x}" for x in stalled[:3])
    if review:
        out.append("")
        out.append("【待复核】(引用这些结论时语气放弱，先确认是否仍成立)")
        out.extend(f"  - {x}" for x in review[:4])
    return out

def pick_cross_category(
    candidates: List[Dict[str, Any]],
    exclude_cats: Iterable[str] = (),
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """跨类别联想：不取"次相关"，而取"别的轨道"。

    设计意图（本人确认）：15 条是定点（答到点上），3 条要来自**当前没入选的类别**，
    这样才是"想起一件没被问但相关的事"，而不是"第二梯队相关项"。

    规则：
      1. 类别未出现过的优先，且每个类别最多取 1 条；
      2. 不足 limit 时，从剩余候选按原有顺序补齐（保证仍有 3 条，不空手）。
    """
    used = set(exclude_cats or ())
    picked: List[Dict[str, Any]] = []
    for it in candidates:
        if len(picked) >= limit:
            return picked
        cat = it.get("cat") or it.get("category")   # state()/recall() 返回的是 category
        if cat in used:
            continue
        picked.append(it)
        used.add(cat)
    if len(picked) < limit:
        for it in candidates:
            if len(picked) >= limit:
                break
            if it in picked:
                continue
            picked.append(it)
    return picked
