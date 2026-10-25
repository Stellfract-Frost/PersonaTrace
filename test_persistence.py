"""持久化测试：SQLite 落盘、重启恢复、TTL 锁、并发写入。

跑法：python -m pytest test_persistence.py -q
"""

import os
import threading
import time

import pytest

from personatrace.memory_v2 import (
    MemoryBackend,
    MemoryV2,
    SQLiteBackend,
    build_backend,
)


@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "memory.db")
    backend = SQLiteBackend(path)
    yield path, backend
    backend.close()


def test_hash_kv_zset_basics(db):
    _, b = db
    b.hset("belief:u:1", {"label": "在准备四级", "hits": 1})
    b.hset("belief:u:1", {"hits": 2})
    assert b.hgetall("belief:u:1")["label"] == "在准备四级"
    assert b.hgetall("belief:u:1")["hits"] == "2"
    assert b.hincrby("fence:u", "seq") == 1
    assert b.hincrby("fence:u", "seq") == 2

    b.zadd("timeline:u", {"a": 10, "b": 30})
    assert [m for m, _ in b.zrangebyscore("timeline:u", 0, 20)] == ["a"]
    b.ztrim_last("timeline:u", 1)
    assert b.zcard("timeline:u") == 1

    b.set("lock:u", "tok", ttl=1)
    assert b.get("lock:u") == "tok"
    assert b.setnx("lock:u", "other") is False
    b.delete("lock:u")
    assert b.setnx("lock:u", "other") is True


def test_ttl_release(db):
    _, b = db
    assert b.setnx("throttle:u:extract", "1", ttl=1) is True
    assert b.setnx("throttle:u:extract", "1", ttl=1) is False
    time.sleep(1.1)
    assert b.setnx("throttle:u:extract", "1", ttl=1) is True, "TTL 过期后应能重新拿到"


def test_memory_survives_restart(db):
    """核心诉求：进程重启后记忆还在。"""
    path, backend = db
    m1 = MemoryV2(backend=backend, user_id="qq:1001")
    m1.observe("goals", "交付驻车集 V1.0", track="work", confidence=0.8)
    m1.record_behavior("ask", "部署")
    m1.close = backend.close()
    backend.close()

    # 模拟重启：新连接、新实例
    backend2 = SQLiteBackend(path)
    m2 = MemoryV2(backend=backend2, user_id="qq:1001")
    rows = m2.state()
    assert len(rows) == 1 and rows[0]["label"] == "交付驻车集 V1.0"
    assert m2.stats()["behavior_events"] == 1
    backend2.close()


def test_hits_and_purge_on_sqlite(db):
    path, backend = db
    m = MemoryV2(backend=backend, user_id="qq:1002")
    m.observe("likes", "偏好深色主题", track="work", evidence_text="我一直用深色")
    m.observe("likes", "偏好深色主题", track="work")
    assert m.state()[0]["hits"] == 2

    removed = m.purge()
    assert removed == 1
    assert m.state(include_archived=True, include_invalidated=True) == []
    assert m.stats()["behavior_events"] == 0
    assert not [k for k in m.b._conn.execute("SELECT key FROM t_hash").fetchall() if "qq:1002" in k[0]]


def test_concurrent_writes_are_serialized(db):
    """两个实例（模拟两个并发请求）写同一个人，锁冲突应自动重放而不是丢写。"""
    _, backend = db
    errors = []

    def worker():
        m = MemoryV2(backend=backend, user_id="qq:1003")
        for _ in range(5):
            try:
                m.observe("state", "在改 bug", track="work")
            except Exception as exc:          # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert MemoryV2(backend=backend, user_id="qq:1003").state()[0]["hits"] == 10


def test_sqlite_cas_is_atomic(db):
    _, b = db
    assert b.cas_hash("belief:u:cas", 0, {"version": 1, "label": "a"}) is True
    assert b.cas_hash("belief:u:cas", 0, {"version": 2, "label": "stale"}) is False
    assert b.cas_hash("belief:u:cas", 1, {"version": 2, "label": "b"}) is True
    assert b.hgetall("belief:u:cas")["label"] == "b"

    b.set("lock:u", "owner")
    assert b.compare_delete("lock:u", "someone-else") is False
    assert b.compare_delete("lock:u", "owner") is True
    assert b.get("lock:u") is None


def test_history_survives_sqlite_restart(db):
    path, backend = db
    t0 = time.time()
    m1 = MemoryV2(backend=backend, user_id="u:history")
    m1.observe("goals", "重建历史", track="work", confidence=0.4, ts=t0)
    m1.observe("goals", "重建历史", track="work", confidence=0.9, ts=t0 + 5 * 86400)
    backend.close()

    backend2 = SQLiteBackend(path)
    m2 = MemoryV2(backend=backend2, user_id="u:history")
    past = m2.state(as_of=t0)
    assert len(past) == 1
    assert past[0]["hits"] == 1
    assert past[0]["confidence"] == 0.4
    backend2.close()


def test_purge_removes_transaction_history(db):
    _, backend = db
    m = MemoryV2(backend=backend, user_id="u:purge")
    m.observe("goals", "要删掉的历史", track="work", ts=time.time())
    m.observe("goals", "要删掉的历史", track="work", ts=time.time() + 1)
    assert m.purge() == 1
    rows = backend._conn.execute(
        "SELECT key FROM t_hash WHERE key LIKE 'history:u:purge%'"
    ).fetchall()
    assert rows == []


def test_backend_selection(tmp_path, monkeypatch):
    db_path = str(tmp_path / "x.db")
    backend, mode = build_backend(db_path=db_path)
    assert mode == "sqlite" and os.path.exists(db_path)

    monkeypatch.setenv("PERSONA_DB", "off")
    backend2, mode2 = build_backend()
    assert mode2 == "memory" and isinstance(backend2, MemoryBackend)
