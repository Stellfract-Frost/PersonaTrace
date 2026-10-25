"""memory_v2 —— 时序人格记忆内核（艾宾浩斯保持率 / 双时态 / CAS / 行为预测）

解决的问题（对着实际痛点，不是为了用 Redis 而用）：

1. 衰减不科学：原实现按"置信度分档、每天统一降 0.01~0.08"。
   改：R(t)=exp(-Δt/S)，S 随"支持证据"按间隔效应增长（快忘时复习增益最大）。
   收益：①不再需要每天全量扫描（读时惰性计算，O(1)）②高价值信息不会和低价值信息同样速度腐烂。

2. 无法回溯：原实现只有 updated，问不出"他 9 月是什么状态"。
   改：双时态（valid_from / invalid_at + recorded_at），每次写追加 version 快照；
        as_of 查询取"那一刻已知的版本"再套有效窗口，后写的纠正不会改写过去。

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
import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)
_LAST_BACKEND_WARNING: Optional[str] = None


def last_backend_warning() -> Optional[str]:
    """Return why the last durable backend fell back to memory, if it did."""
    return _LAST_BACKEND_WARNING

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

    # -- compare-and-swap / compare-and-delete --
    def cas_hash(self, key: str, expected_version: int, mapping: Dict[str, Any]) -> bool:
        """Atomically replace a hash only when its version field still matches.

        ``expected_version=0`` means "the record must not exist yet". This is
        the primitive behind the kernel's optimistic CAS: a guard lock can
        expire, but a stale writer must never overwrite a newer record.
        """
        with self._lock:
            bucket = self._h.get(key)
            current = 0
            if bucket is not None:
                try:
                    current = int(bucket.get("version", 0))
                except (TypeError, ValueError):
                    current = 0
            if current != int(expected_version or 0):
                return False
            target = self._h.setdefault(key, {})
            for field_name, value in mapping.items():
                target[str(field_name)] = (
                    value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                )
            return True

    def compare_delete(self, key: str, expected: str) -> bool:
        """Delete a key only when its value is still the caller's token."""
        with self._lock:
            if key in self._kv and self._kv.get(key) == expected:
                self._kv.pop(key, None)
                self._ttl.pop(key, None)
                return True
            return False


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

    # -- compare-and-swap / compare-and-delete --
    def cas_hash(self, key: str, expected_version: int, mapping: Dict[str, Any]) -> bool:
        rows = [
            (key, str(f), v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
            for f, v in mapping.items()
        ]
        expected = int(expected_version or 0)
        with self._lock:
            try:
                # BEGIN IMMEDIATE makes the read-check-write sequence atomic
                # across processes, not just inside this Python process.
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT value FROM t_hash WHERE key = ? AND field = 'version'",
                    (key,),
                ).fetchone()
                current = 0
                if row is not None:
                    try:
                        current = int(row[0])
                    except (TypeError, ValueError):
                        current = 0
                if current != expected:
                    self._conn.rollback()
                    return False
                self._conn.executemany(
                    "INSERT INTO t_hash (key, field, value) VALUES (?, ?, ?) "
                    "ON CONFLICT(key, field) DO UPDATE SET value = excluded.value",
                    rows,
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def compare_delete(self, key: str, expected: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM t_kv WHERE key = ? AND value = ?", (key, expected)
            )
            return cur.rowcount == 1

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

    # -- compare-and-swap / compare-and-delete --
    _CAS_SCRIPT = """
    local current = redis.call('HGET', KEYS[1], 'version')
    if not current then current = '0' end
    if current ~= ARGV[1] then return 0 end
    for i = 2, #ARGV, 2 do
        redis.call('HSET', KEYS[1], ARGV[i], ARGV[i + 1])
    end
    return 1
    """

    def cas_hash(self, key: str, expected_version: int, mapping: Dict[str, Any]) -> bool:
        args: List[str] = [str(int(expected_version or 0))]
        for field_name, value in mapping.items():
            args.append(str(field_name))
            args.append(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
        return bool(self.r.eval(self._CAS_SCRIPT, 1, self.p + key, *args))

    def compare_delete(self, key: str, expected: str) -> bool:
        script = (
            "if redis.call('GET', KEYS[1]) == ARGV[1] then "
            "return redis.call('DEL', KEYS[1]) else return 0 end"
        )
        return bool(self.r.eval(script, 1, self.p + key, expected))


DEFAULT_DB_PATH = os.path.join(os.path.expanduser("~"), ".personatrace", "memory.db")
DB_ENV = "PERSONA_DB"


def build_backend(url: Optional[str] = None, db_path: Optional[str] = None) -> Tuple[Any, str]:
    """返回 (backend, mode)。优先级：Redis → SQLite（默认落盘）→ 进程内。

    - 装了 redis 包且给了 url → Redis（多进程共享）
    - 否则落 SQLite 文件：db_path > 环境变量 PERSONA_DB > ~/.personatrace/memory.db
    - PERSONA_DB=off / memory:// / :memory: → 进程内（重启即丢，仅用于测试）
    任何异常都退回进程内，绝不让"记不住"升级成"起不来"。
    """
    global _LAST_BACKEND_WARNING
    _LAST_BACKEND_WARNING = None
    # Only an explicit Redis URL opts into Redis. Merely installing the redis
    # package must not make a caller who asked for SQLite connect to localhost.
    if _redis is not None and url and url != "memory://":
        try:
            return RedisBackend(url), "redis"
        except Exception as exc:
            logger.warning("Redis backend unavailable (%s); trying SQLite", exc)
    target = db_path or os.environ.get(DB_ENV) or DEFAULT_DB_PATH
    if str(target).strip().lower() in {"", "off", "none", "memory", "memory://", ":memory:"}:
        return MemoryBackend(), "memory"
    try:
        return SQLiteBackend(str(target)), "sqlite"
    except Exception as exc:
        # Fail-open is still the right policy: a broken memory must not stop
        # the host's conversation. Fail-silent is not: a memory product that
        # quietly becomes in-process loses everything on restart. Make the
        # degradation visible in logs and in `personatrace-doctor`.
        detail = f"SQLite backend unavailable at {target!r}: {exc}"
        _LAST_BACKEND_WARNING = detail
        logger.warning(
            "%s; falling back to in-process memory (data will not survive restart)",
            detail,
        )
        return MemoryBackend(), "memory"


# ---------------------------------------------------------------- 衰减

# 各类型的"初始稳定度"（天）。决定 R(30 天) 掉到多少。
# 依据：事实几乎不变，性格慢变，目标/项目按周期变，此刻状态几天就旧。
CATEGORY_STABILITY: Dict[str, float] = {
    "explicit_facts": 520.0,
    "identity": 300.0,
    "traits": 300.0,
    "constraints": 180.0,
    "boundaries": 365.0,
    "expertise": 240.0,
    "decision_style": 240.0,
    "likes": 90.0,
    "dislikes": 90.0,
    "interests": 120.0,
    "style_notes": 180.0,
    "behavior_patterns": 120.0,
    "interpersonal": 180.0,
    "projects": 60.0,
    "goals": 45.0,
    "habit": 90.0,
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
DEFAULT_PINNED_CATEGORIES = ("boundaries", "constraints", "identity")
MAX_BELIEF_VERSIONS = 200       # per-belief transaction-time history budget
CAS_RETRIES = 8                 # optimistic replay attempts on a version conflict
_UNSET = object()               # distinguish "not supplied" from "explicitly clear"
# datetime.timestamp() rounds to microseconds; a write can land a fraction of
# a microsecond after a same-process read on Windows/Python 3.10. Treat this
# as clock granularity, not as future knowledge.
RECORDED_AT_EPSILON = 0.001


def stability_from_half_life(half_life_days: float) -> float:
    """让 R(half_life) = 0.5 → S = hl / ln2。"""
    return max(1e-6, half_life_days / math.log(2))


def retention(stability: float, days: float) -> float:
    """艾宾浩斯保持率 R = exp(-t/S)。"""
    if days <= 0:
        return 1.0
    return math.exp(-days / max(1e-6, stability))


def _bigrams(text: Any) -> set:
    flat = "".join(str(text or "").lower().split())
    if not flat:
        return set()
    if len(flat) == 1:
        return {flat}
    return {flat[i:i + 2] for i in range(len(flat) - 1)}


def _lexical_scores(query: str, rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Candidate-set IDF over character bigrams.

    This is the kernel's own lexical channel. It deliberately returns 0.5 for
    no overlap (neutral), not 0: two unrelated strings must not look like
    negative evidence. It is duplicated, in a smaller form, from ``recall`` so
    the kernel keeps its stdlib-only import surface.
    """
    rows = list(rows)
    query_grams = _bigrams(query)
    if not query_grams:
        return {str(row.get("id")): 0.5 for row in rows}

    docs = {str(row.get("id")): _bigrams(row.get("label", "")) for row in rows}
    df: Dict[str, int] = {}
    for grams in docs.values():
        for gram in grams:
            df[gram] = df.get(gram, 0) + 1
    total = max(1, len(docs))
    idf = {}
    for gram, count in df.items():
        idf[gram] = math.log((total + 1.0) / (count + 1.0)) + 1.0
    query_weight = sum(idf.get(gram, 1.0) for gram in query_grams)
    if query_weight <= 0:
        return {str(row.get("id")): 0.5 for row in rows}

    q = str(query or "").lower()
    out: Dict[str, float] = {}
    for row in rows:
        key = str(row.get("id"))
        label = str(row.get("label", "")).lower()
        if label and (label in q or q in label):
            out[key] = 1.0
            continue
        shared = sum(idf.get(gram, 1.0) for gram in query_grams if gram in docs[key])
        overlap = shared / query_weight
        out[key] = 0.5 if overlap <= 0 else 0.5 + 0.5 * min(1.0, overlap)
    return out


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
    # Transaction time: when this version was written to the store. Valid time
    # is valid_from_ts/invalid_at_ts; the pair is what makes queries real
    # snapshots instead of filters over the latest row.
    recorded_at_ts: float = 0.0
    invalid_at_ts: Optional[float] = None
    status: str = "active"
    hits: int = 1
    dimension: str = "identity"          # identity | need | predict
    reasoning: str = ""                  # 为什么存这条（深读时要带回去的语境）
    due_at_ts: Optional[float] = None    # 前瞻：这件事还有多久到期
    time_kind: str = "none"              # none | due | past | range | recurring
    time_raw: str = ""                   # 原话，回填错了能查、能回滚
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
            "recorded_at_ts": self.recorded_at_ts or self.valid_from_ts,
            "invalid_at_ts": self.invalid_at_ts if self.invalid_at_ts else "",
            "status": self.status, "hits": self.hits, "dimension": self.dimension,
            "due_at_ts": self.due_at_ts if self.due_at_ts else "",
            "time_kind": self.time_kind or "none",
            "time_raw": self.time_raw,
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
            recorded_at_ts=num("recorded_at_ts", 0.0) or num("valid_from_ts", time.time()),
            invalid_at_ts=num("invalid_at_ts") or None,
            status=raw.get("status", "active"), hits=int(num("hits", 1)),
            dimension=raw.get("dimension", "identity"),
            reasoning=raw.get("reasoning", ""),
            due_at_ts=num("due_at_ts") or None,
            time_kind=raw.get("time_kind", "none") or "none",
            time_raw=raw.get("time_raw", ""),
            version=int(num("version", 0)),
            evidence=json.loads(raw.get("evidence") or "[]"),
        )


def belief_id(user_id: str, track: str, category: str, label: str) -> str:
    """确定性主键：同一条结论永远落到同一个 id，写入天然幂等。"""
    key = f"{user_id}:{track}:{category}:{label.lower().strip()}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def evidence_key(user_id: str, bid: str, eid: str) -> str:
    """L0 key scoped by belief so one episode cannot overwrite another belief's row."""
    return f"evidence:{user_id}:{bid}:{eid}"


def legacy_evidence_key(user_id: str, eid: str) -> str:
    """Legacy key; kept readable for existing databases."""
    return f"evidence:{user_id}:{eid}"


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
        if token:
            # Compare-and-delete: never release a lock that a later writer has
            # already acquired after our TTL expired.
            self.b.compare_delete(f"lock:{user_id}", token)

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

    def _hk(self, bid: str) -> str:
        """Transaction-time index for one belief: version -> recorded_at."""
        return f"history:{self.user_id}:{bid}"

    def _hvk(self, bid: str, version: Any) -> str:
        return f"history:{self.user_id}:{bid}:{int(version)}"

    @property
    def _tl(self) -> str:
        return f"timeline:{self.user_id}"

    @property
    def _bh(self) -> str:
        return f"behavior:{self.user_id}"

    def _delete_evidence(self, uid: str, bid: str, eid: str) -> None:
        self.b.delete(evidence_key(uid, bid, eid))
        legacy = self.b.hgetall(legacy_evidence_key(uid, eid))
        if legacy.get("belief_id") == bid:
            self.b.delete(legacy_evidence_key(uid, eid))

    # -- transaction-time history -------------------------------------------
    def _trim_history(self, bid: str) -> None:
        total = self.b.zcard(self._hk(bid))
        if total <= MAX_BELIEF_VERSIONS:
            return
        pairs = self.b.zrangebyscore(self._hk(bid), 0, float("inf"))
        for member, _score in pairs[: total - MAX_BELIEF_VERSIONS]:
            self.b.delete(self._hvk(bid, member))
            self.b.zrem(self._hk(bid), member)

    def _record_version(self, belief: Belief) -> None:
        self.b.hset(self._hvk(belief.id, belief.version), belief.to_hash())
        self.b.zadd(self._hk(belief.id), {str(belief.version): belief.recorded_at_ts})
        self._trim_history(belief.id)

    def _backfill_base(self, bid: str, raw: Dict[str, str]) -> None:
        """Give pre-history records one base version before they are updated.

        Older databases only have the current row. Without a base version an
        ``as_of`` query before the first new write would see the new value,
        which is exactly the mutable-filter bug this history removes.
        """
        if not raw or self.b.zcard(self._hk(bid)) > 0:
            return
        old = Belief.from_hash(raw)
        old.recorded_at_ts = old.valid_from_ts
        self._record_version(old)

    def _known_belief(self, bid: str, known_as_of: float) -> Optional[Belief]:
        pairs = self.b.zrangebyscore(self._hk(bid), 0, known_as_of)
        if not pairs:
            return None
        # Version order is transaction order. recorded_at can be backdated by
        # an importer, so version is the tie-breaker that must win.
        member = max(
            pairs,
            key=lambda pair: (
                int(pair[0]) if str(pair[0]).lstrip("-").isdigit() else 0,
                pair[1],
            ),
        )[0]
        raw = self.b.hgetall(self._hvk(bid, member))
        return Belief.from_hash(raw) if raw else None

    def _belief_at(self, bid: str, known_as_of: float) -> Optional[Belief]:
        """Return the row as it was known at ``known_as_of``.

        If the record predates history, the current row is used only when it
        was written at or before the query time. A current row written later
        is not knowledge the system had at that time.
        """
        raw = self.b.hgetall(self._bk(bid))
        current = Belief.from_hash(raw) if raw else None
        if self.b.zcard(self._hk(bid)) > 0:
            seen = self._known_belief(bid, known_as_of)
            # The current row is written just before its history snapshot. If
            # a reader lands in that tiny window (or history was trimmed), use
            # the current row only when it is already known at this time.
            if (
                current is not None
                and current.recorded_at_ts <= known_as_of
                and (seen is None or current.version > seen.version)
            ):
                return current
            return seen
        if current is None:
            return None
        if current.recorded_at_ts and current.recorded_at_ts > known_as_of:
            return None
        return current

    def _valid_belief_at(self, bid: str, known_as_of: float, as_of: float) -> Optional[Belief]:
        """Valid-time view with current knowledge, across revival intervals.

        A revival starts a new validity window at ``valid_from``. Asking
        "what do we now know about T" must still return the earlier window if
        T predates the revival. If the newest known window starts after T, walk
        the transaction history back to the newest window that contains T.
        """
        pairs = self.b.zrangebyscore(self._hk(bid), 0, known_as_of)
        if not pairs:
            return None
        versions: List[tuple[int, Belief]] = []
        for member, _score in pairs:
            raw = self.b.hgetall(self._hvk(bid, member))
            if not raw:
                continue
            try:
                version = int(member)
            except (TypeError, ValueError):
                version = 0
            versions.append((version, Belief.from_hash(raw)))

        # The highest-version window that has started by T is the current
        # knowledge about T. Older windows are not consulted: an invalidation
        # or a later revival supersedes them for the intervening period.
        started = [(version, belief) for version, belief in versions
                   if belief.valid_from_ts <= as_of]
        if not started:
            return None
        _version, chosen = max(started, key=lambda item: item[0])
        if chosen.invalid_at_ts is None or as_of < chosen.invalid_at_ts:
            return chosen
        return None

    # -- 写入：追加证据 + 间隔效应强化（幂等 + CAS） --
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
        due_at_ts: Any = _UNSET,
        time_kind: Any = _UNSET,
        time_raw: Any = _UNSET,
        revive: bool = True,
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
        self._acquire_with_retry()
        try:
            for attempt in range(CAS_RETRIES):
                raw = self.b.hgetall(self._bk(bid))
                old_version = 0
                previous_recorded = 0.0
                timeline_score = ts
                if raw:
                    self._backfill_base(bid, raw)
                    try:
                        old_version = int(raw.get("version") or 0)
                    except (TypeError, ValueError):
                        old_version = 0
                    belief = Belief.from_hash(raw)
                    previous_recorded = belief.recorded_at_ts
                    timeline_score = belief.valid_from_ts
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
                    # A caller that passes the field can clear it; one that
                    # omits it preserves the previous value. This is what
                    # stops an old due date from sticking forever.
                    if due_at_ts is not _UNSET:
                        belief.due_at_ts = due_at_ts or None
                    if time_kind is not _UNSET:
                        belief.time_kind = time_kind or "none"
                    if time_raw is not _UNSET:
                        belief.time_raw = time_raw or ""
                    # Re-observation after a correction is new evidence, so it
                    # reopens the validity window. Do it only after the prior
                    # invalidation took effect; an observation before that
                    # timestamp belongs to the old window.
                    invalid_at = belief.invalid_at_ts
                    if revive and invalid_at is None and belief.status != "active":
                        belief.status = "active"
                    elif revive and invalid_at is not None and ts >= invalid_at:
                        belief.status = "active"
                        belief.valid_from_ts = ts
                        belief.invalid_at_ts = None
                        timeline_score = min(timeline_score, ts)
                else:
                    initial_due = None if due_at_ts is _UNSET else (due_at_ts or None)
                    initial_kind = "none" if time_kind is _UNSET else (time_kind or "none")
                    initial_raw = "" if time_raw is _UNSET else (time_raw or "")
                    belief = Belief(
                        id=bid, user_id=self.user_id, track=track, category=category,
                        label=label, confidence=confidence,
                        stability=CATEGORY_STABILITY.get(category, DEFAULT_STABILITY),
                        last_support_ts=ts, valid_from_ts=ts,
                        recorded_at_ts=ts, dimension=dimension, reasoning=reasoning,
                        due_at_ts=initial_due, time_kind=initial_kind, time_raw=initial_raw,
                        evidence=[evidence_id] if evidence_id else [],
                    )

                # L0：原文只存一次，检索时不碰它（这是成本优势的来源）
                eid = ""
                if evidence_text:
                    eid = evidence_id or f"{bid}:{int(ts * 1000)}:{old_version + 1}"
                    if eid not in belief.evidence:
                        belief.evidence.append(eid)
                # 证据列表要有上限：长期运行的宿主机（QQ 群）会无限 append，
                # 每次 to_hash 都要重新序列化整张表，而且 L0 原文只增不删。
                if len(belief.evidence) > MAX_EVIDENCE:
                    for stale in belief.evidence[:-MAX_EVIDENCE]:
                        self._delete_evidence(self.user_id, bid, stale)
                    belief.evidence = belief.evidence[-MAX_EVIDENCE:]

                belief.version = old_version + 1
                belief.recorded_at_ts = max(ts, previous_recorded)
                if not self.b.cas_hash(self._bk(bid), old_version, belief.to_hash()):
                    # Someone wrote between our read and our CAS (usually a
                    # guard lock that expired). Re-read and replay; never
                    # overwrite their version with our stale copy.
                    time.sleep(min(0.05, 0.002 * (attempt + 1)))
                    continue

                if eid:
                    record = {
                        "belief_id": bid, "ts": ts,
                        "raw_text": evidence_text[:EVIDENCE_RAW_LIMIT],
                    }
                    self.b.hset(evidence_key(self.user_id, bid, eid), record)
                    # Keep the old key populated for readers of older
                    # databases, but never let a second belief overwrite it.
                    legacy = self.b.hgetall(legacy_evidence_key(self.user_id, eid))
                    if not legacy or legacy.get("belief_id") == bid:
                        self.b.hset(legacy_evidence_key(self.user_id, eid), record)
                self._record_version(belief)
                self.b.zadd(self._tl, {bid: timeline_score})
                return belief
            raise Conflict(f"CAS failed after {CAS_RETRIES} attempts for {bid}")
        finally:
            self.guard.release(self.user_id)

    # -- 失效：不删除，只打时间戳（Graphiti 原则） --
    def invalidate(
        self,
        bid: str,
        ts: Optional[float] = None,
        status: str = "deprecated",
        *,
        recorded_at: Optional[float] = None,
    ) -> bool:
        """Invalidate a belief and record a new transaction-time version.

        ``ts`` is valid time (when the fact stopped being true). The write time
        is now unless ``recorded_at`` is supplied, which lets an importer
        reconstruct what the system knew at a historical moment. Repeated
        invalidations never extend an existing validity window.
        """
        now = time.time()
        valid_to = ts if ts is not None else now
        txn_time = recorded_at if recorded_at is not None else now
        self._acquire_with_retry()
        try:
            for attempt in range(CAS_RETRIES):
                raw = self.b.hgetall(self._bk(bid))
                if not raw:
                    return False
                self._backfill_base(bid, raw)
                try:
                    old_version = int(raw.get("version") or 0)
                except (TypeError, ValueError):
                    old_version = 0
                belief = Belief.from_hash(raw)
                previous_recorded = belief.recorded_at_ts
                belief.status = status
                if belief.invalid_at_ts is None:
                    belief.invalid_at_ts = valid_to
                else:
                    # A correction may shorten validity, never silently extend
                    # it: invalidate(t1) followed by invalidate(t2>t1) would
                    # otherwise make the belief alive again between t1 and t2.
                    belief.invalid_at_ts = min(belief.invalid_at_ts, valid_to)
                belief.version = old_version + 1
                belief.recorded_at_ts = max(txn_time, previous_recorded)
                if not self.b.cas_hash(self._bk(bid), old_version, belief.to_hash()):
                    time.sleep(min(0.05, 0.002 * (attempt + 1)))
                    continue
                self._record_version(belief)
                return True
            raise Conflict(f"CAS failed after {CAS_RETRIES} attempts for {bid}")
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
                    self._delete_evidence(uid, bid, eid)
                removed += 1
            for member, _score in list(self.b.zrangebyscore(f"history:{uid}:{bid}", 0, float("inf"))):
                self.b.delete(f"history:{uid}:{bid}:{member}")
            self.b.delete(f"history:{uid}:{bid}", f"belief:{uid}:{bid}")
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
        known_as_of: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return beliefs valid at ``as_of`` as the system knew them then.

        By default ``known_as_of`` equals ``as_of``: this is snapshot
        semantics, so a later correction cannot rewrite the past. Pass
        ``known_as_of=time.time()`` to ask "what do we currently know about
        that date", which is the old valid-time-only behaviour.
        """
        as_of = as_of if as_of is not None else time.time()
        # Read at a tiny tolerance: datetime.timestamp() rounds to microseconds,
        # so a write can appear a fraction of a microsecond after its own read.
        read_as_of = as_of + RECORDED_AT_EPSILON
        known = (known_as_of if known_as_of is not None else as_of) + RECORDED_AT_EPSILON
        rows = self.b.zrangebyscore(self._tl, 0, read_as_of)
        out: List[Dict[str, Any]] = []
        for bid, _score in rows:
            if known_as_of is not None:
                belief = self._valid_belief_at(bid, known, read_as_of)
                if belief is None:
                    belief = self._belief_at(bid, known)
            else:
                belief = self._belief_at(bid, known)
            if belief is None:
                continue
            if belief.valid_from_ts > read_as_of:
                continue
            if belief.invalid_at_ts and belief.invalid_at_ts <= read_as_of and not include_invalidated:
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
        pinned_categories: Tuple[str, ...] = DEFAULT_PINNED_CATEGORIES,
        expand: int = 0,
        *,
        query: Optional[str] = None,
        known_as_of: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Rank beliefs by relevance and retention.

        ``query`` gives the kernel its own lexical channel. Before this, every
        item received the same constant similarity, so the kernel could only
        sort by recency and honestly scored 2% on the 200/50 fixture. Vector
        scoring is still optional; lexical works with the standard library.
        """
        as_of = as_of if as_of is not None else time.time()
        pool = self.state(as_of=as_of, expand=expand, known_as_of=known_as_of)
        pinned = [x for x in pool if x["category"] in pinned_categories]
        rest = [x for x in pool if x["category"] not in pinned_categories]
        pinned.sort(key=lambda row: (
            -float(row.get("effective_confidence", 0.0)),
            -float(row.get("retention", 0.0)),
            str(row.get("id", "")),
        ))
        # Pinned categories are mandatory, not budget-exempt. Half the budget
        # is enough; forty constraints must not consume k=3 and make recall a
        # constant independent of the query.
        pinned_budget = max(1, int(k) // 2) if int(k) > 0 else 0
        pinned = pinned[:pinned_budget]

        qv = list(query_vec) if query_vec is not None else None
        qtext = (query or "").strip().lower()
        lex = _lexical_scores(qtext, rest) if qtext else {}

        def cos(a: Iterable[float], b: Iterable[float]) -> float:
            a, b = list(a), list(b)
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na < 1e-9 or nb < 1e-9:
                return 0.0
            return sum(x * y for x, y in zip(a, b)) / (na * nb)

        scored = []
        for item in rest:
            key = str(item["id"])
            if qtext and item["label"] and (
                item["label"].lower() in qtext or qtext in item["label"].lower()
            ):
                sim = 1.0
            else:
                parts = []
                if qv is not None and item_vecs and key in item_vecs:
                    parts.append(cos(qv, item_vecs[key]))
                if qtext:
                    parts.append(lex.get(key, 0.5))
                sim = sum(parts) / len(parts) if parts else 0.5
            # 相关度主导，保持率调制：陈旧的即使相似也要降权
            score = (0.15 + 0.85 * sim) * (0.35 + 0.65 * item["retention"])
            scored.append((score, item))
        scored.sort(key=lambda x: -x[0])
        picked = pinned + [it for _, it in scored[: max(0, k - len(pinned))]]
        hidden = [it for _, it in scored[max(0, k - len(pinned)):]]
        return {
            "shown": picked,
            # Aliases shared with the outer ranker. ``pinned_count`` was the
            # only spelling before; probes that used rank() expected a list.
            "pinned": pinned,
            "cross": [],
            "hidden": hidden,
            "hidden_count": len(hidden),
            "pinned_count": len(pinned),
            "candidates": len(pool),
            "k": int(k),
            "as_of": as_of,
            "query": query or "",
            "backend": self.mode,
        }

    def _evidence_snippets(self, belief: Belief, limit: int = 2, max_chars: int = 200) -> List[str]:
        """回表：只在命中后才读原文。这是"表层检索不碰具体语句"的兑现点。"""
        out: List[str] = []
        for eid in reversed(belief.evidence[-max(1, limit):]):
            raw = self.b.hgetall(evidence_key(self.user_id, belief.id, eid))
            if not raw:
                # Records written before the per-belief key still resolve.
                raw = self.b.hgetall(legacy_evidence_key(self.user_id, eid))
            text = (raw or {}).get("raw_text", "")
            if text:
                out.append(text[:max_chars])
        return out

    def enrich(self, rows: Iterable[Dict[str, Any]], expand: int = 1) -> List[Dict[str, Any]]:
        """Add L0/context only to rows that survived the ranking budget."""
        if not expand:
            return list(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            raw = self.b.hgetall(self._bk(str(item.get("id", ""))))
            if not raw:
                out.append(item)
                continue
            belief = Belief.from_hash(raw)
            item["reasoning"] = belief.reasoning
            item["evidence_count"] = len(belief.evidence)
            item["evidence_texts"] = self._evidence_snippets(belief, limit=expand)
            out.append(item)
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
            "status": belief.status,
            "valid_from_ts": belief.valid_from_ts,
            "invalid_at_ts": belief.invalid_at_ts,
            "recorded_at_ts": belief.recorded_at_ts,
            "confidence": belief.confidence,
            "retention": round(r, 3),
            "effective_confidence": round(belief.confidence * r, 3),
            "staleness": stale,          # 距上次支撑 / 稳定度，>1 表示已过半个周期以上
            "due_in": due_in,            # 前瞻：还有几天到期（负数=已超期）
            "due_at_ts": belief.due_at_ts,
            "time_kind": belief.time_kind,
            "time_raw": belief.time_raw,
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
        for attempt in range(CAS_RETRIES):
            old = self.b.hgetall(self._mbk(mid))
            try:
                version = int(old.get("version") or 0)
            except (TypeError, ValueError):
                version = 0
            hits = int(old.get("hits", 0)) + 1
            mapping = {
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
                "version": version + 1,
            }
            if self.b.cas_hash(self._mbk(mid), version, mapping):
                self.b.zadd(self._mk(), {mid: ts})
                self.b.ztrim_last(self._mk(), MAX_MENTIONS)
                return {
                    "id": mid, "actor": actor, "kind": kind, "label": label,
                    "confidence": confidence, "verified": verified, "hits": hits,
                }
            time.sleep(min(0.05, 0.002 * (attempt + 1)))
        raise Conflict(f"CAS failed after {CAS_RETRIES} attempts for mention {mid}")

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
                self._delete_evidence(self.user_id, bid, eid)
            for member, _score in list(
                self.b.zrangebyscore(self._hk(bid), 0, float("inf"))
            ):
                self.b.delete(self._hvk(bid, member))
            self.b.delete(self._hk(bid), self._bk(bid))
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

    def close(self) -> None:
        close = getattr(self.b, "close", None)
        if callable(close):
            close()


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
