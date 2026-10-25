"""memory_v2 离线测试：不依赖 LLM / Redis / 网络。

跑法：D:/Anaconda3/python.exe -m pytest test_memory_v2.py -q
"""

import time

import pytest

from personatrace.memory_v2 import (
    ARCHIVE_AT, CATEGORY_STABILITY, Conflict, MemoryBackend, MemoryV2,
    belief_id, evidence_key, reinforce, retention, stability_from_half_life,
)

DAY = 86400.0


@pytest.fixture()
def mem():
    return MemoryV2(backend=MemoryBackend(), user_id="u_test")


def test_half_life_math():
    s = stability_from_half_life(10)
    assert abs(retention(s, 10) - 0.5) < 1e-9
    assert retention(s, 0) == 1.0
    assert retention(s, 100) < 0.01


def test_spacing_effect():
    """快忘时复习，增益必须大于刚记完时复习。"""
    s = 10.0
    gain_low_r = reinforce(s, 0.10) - s
    gain_high_r = reinforce(s, 0.95) - s
    assert gain_low_r > gain_high_r * 5


def test_observe_idempotent_and_reinforce(mem):
    t = time.time()
    b1 = mem.observe("likes", "喜欢 Rust", track="work", ts=t)
    b2 = mem.observe("likes", "喜欢 Rust", track="work", ts=t + 5 * DAY)
    assert b1.id == b2.id == belief_id("u_test", "work", "likes", "喜欢 rust")
    assert b2.hits == 2
    assert b2.stability > b1.stability          # 间隔效应生效
    assert mem.stats()["beliefs"] == 1          # 没有产生副本


def test_time_travel_as_of(mem):
    t0 = time.time()
    mem.observe("goals", "做时序记忆", track="work", ts=t0)
    bid = belief_id("u_test", "work", "goals", "做时序记忆")
    mem.invalidate(bid, ts=t0 + 10 * DAY)

    before = mem.state(as_of=t0 + DAY)
    after = mem.state(as_of=t0 + 20 * DAY)
    assert any(x["label"] == "做时序记忆" for x in before)
    assert not any(x["label"] == "做时序记忆" for x in after)


def test_archive_zone(mem):
    t = time.time()
    mem.observe("state", "在改 bug", track="work", ts=t - 400 * DAY)
    assert mem.state() == []                                   # 保留率低于阈值，不参与取用
    kept = mem.state(include_archived=True)
    assert len(kept) == 1                                      # 但没被删除
    assert kept[0]["retention"] < ARCHIVE_AT


def test_recall_pins_and_freshness(mem):
    now = time.time()
    mem.observe("constraints", "不接受远程办公", track="work", ts=now - 200 * DAY)
    mem.observe("identity", "偏内向", track="persona", ts=now - 200 * DAY)
    mem.observe("likes", "喜欢深色主题", track="work", ts=now - 60 * DAY)
    mem.observe("likes", "喜欢浅色主题", track="work", ts=now - 1 * DAY)

    res = mem.recall(as_of=now, k=4)
    cats = [x["category"] for x in res["shown"]]
    assert res["pinned_count"] == 2                 # 约束 + 身份强制入选
    assert "constraints" in cats and "identity" in cats
    labels = [x["label"] for x in res["shown"]]
    assert labels.index("喜欢浅色主题") < labels.index("喜欢深色主题")  # 新鲜的排前面


def test_behavior_predict(mem):
    t = time.time()
    for i, kind in enumerate(["app", "ask", "app", "ask", "app", "ask"]):
        mem.record_behavior(kind, "debug", ts=t - 60 + i)      # 必须是过去时间
    preds = mem.predict_next(window=10)
    assert preds[0]["next"] == "app"
    assert preds[0]["p"] == 1.0


def test_guard_blocks_double_writer(mem):
    mem.guard.acquire("u_test")
    with pytest.raises(Conflict):
        mem.guard.acquire("u_test")
    mem.guard.release("u_test")
    assert mem.guard.acquire("u_test")


def test_throttle_window(mem):
    assert mem.allow("detective", 120) is True
    assert mem.allow("detective", 120) is False


def test_due_in_forward_looking(mem):
    now = time.time()
    mem.observe("goals", "交付 V1.0", track="work", ts=now, due_at_ts=now + 2 * DAY)
    row = mem.state(as_of=now)[0]
    assert 1.9 < row["due_in"] < 2.1
    assert row["needs_review"] is True        # 3 天内到期 → 提示复核


def test_typed_time_roundtrip_and_preserve(mem):
    now = time.time()
    due = now + 2 * DAY
    mem.observe("goals", "周五要演示", track="work", ts=now, due_at_ts=due,
                time_kind="due", time_raw="周五")
    row = mem.state(as_of=now)[0]
    assert row["time_kind"] == "due"
    assert row["time_raw"] == "周五"
    assert abs(row["due_at_ts"] - due) < 1e-6

    mem.observe("goals", "周五要演示", track="work", ts=now + 60)
    row = mem.state(as_of=now + 60)[0]
    assert row["time_kind"] == "due"
    assert row["time_raw"] == "周五"


def test_api_router():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from personatrace.api_v2 import build_router

    m = MemoryV2(backend=MemoryBackend(), user_id="api_user")
    app = FastAPI()
    app.include_router(build_router(m))
    c = TestClient(app)

    r = c.post("/v1/observe", json={"category": "goals", "label": "练习口语", "track": "work"})
    assert r.status_code == 200 and r.json()["hits"] == 1

    r = c.get("/v1/state")
    assert r.status_code == 200 and r.json()["count"] == 1

    r = c.get("/v1/recall", params={"q": "口语"})
    assert r.status_code == 200 and r.json()["shown"][0]["label"] == "练习口语"

    c.post("/v1/behavior", json={"kind": "ask", "target": "debug"})
    c.post("/v1/behavior", json={"kind": "app", "target": "chrome"})
    c.post("/v1/behavior", json={"kind": "ask", "target": "debug"})
    r = c.get("/v1/behavior/predict")
    assert r.status_code == 200 and r.json()["predictions"][0]["next"] == "app"

    r = c.get("/v1/health")
    assert r.json()["backend"] == "injected"

    r = c.post("/v1/observe", json={"category": "goals", "label": "x"})  # 违反 min_length
    assert r.status_code == 422


def test_api_typed_time_contract_and_invalidate_404():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from personatrace.api_v2 import build_router

    m = MemoryV2(backend=MemoryBackend(), user_id="api_typed")
    app = FastAPI()
    app.include_router(build_router(m))
    c = TestClient(app)

    due = time.time() + 2 * DAY
    r = c.post("/v1/observe", json={
        "category": "goals", "label": "有截止的 API 记录",
        "due_at_ts": due, "time_kind": "due", "time_raw": "后天",
    })
    assert r.status_code == 200

    # Omitted fields preserve; explicit null/"none" clears.
    c.post("/v1/observe", json={"category": "goals", "label": "有截止的 API 记录"})
    row = c.get("/v1/state").json()["beliefs"][0]
    assert row["due_at_ts"] == due
    assert row["time_kind"] == "due"

    c.post("/v1/observe", json={
        "category": "goals", "label": "有截止的 API 记录",
        "due_at_ts": None, "time_kind": "none", "time_raw": "",
    })
    row = c.get("/v1/state").json()["beliefs"][0]
    assert row["due_at_ts"] is None and row["time_kind"] == "none"

    assert c.get("/v1/state").json()["as_of"] > 0
    assert c.post("/v1/beliefs/does-not-exist/invalidate").status_code == 404


def test_aging_lines_forward_looking():
    """前瞻两段：超期/拖太久 + 高价值但没强化过的。"""
    import datetime as dt
    from personatrace.memory_v2 import aging_lines

    now = dt.datetime.now()
    skills = {
        "work": {"goals": [
            {"label": "交付 V1.0", "confidence": 0.8,
             "updated": (now - dt.timedelta(days=200)).isoformat()},
            {"label": "低置信杂项", "confidence": 0.3,
             "updated": (now - dt.timedelta(days=200)).isoformat()},
        ]},
        "persona": {},
        "ephemeral": {"timeline": [
            {"event": "联调", "days": -5, "status": "expired"},
            {"event": "改稿", "days": 20, "status": "upcoming"},
            {"event": "交表", "days": 1, "status": "ongoing"},
        ]},
    }
    text = "\n".join(aging_lines(skills, now))
    assert "已超期 5 天" in text                 # 负数 = 超期，能主动提
    assert "还有 20 天" in text                  # 挂太久 = 拖欠
    assert "交付 V1.0" in text and "保持率" in text
    assert "低置信杂项" not in text              # 低置信不进复核（避免噪声）


def test_aging_lines_no_memory_v2_fallback():
    """没装 v2 时宿主必须完全不受影响。"""
    import datetime as dt
    from personatrace.memory_v2 import aging_lines
    assert aging_lines({}, dt.datetime.now()) == []


def test_two_stage_recall_late_materialization(mem):
    """表层只给标签（便宜），expand 才回表读 reasoning + 原文（贵但只在命中后）。"""
    now = time.time()
    mem.observe("goals", "交付驻车集 V1.0", track="work", ts=now,
                reasoning="他说这周必须给客户演示，否则违约",
                evidence_text="这周五要给客户演示，做不完就违约了")
    mem.observe("likes", "喜欢深色主题", track="work", ts=now,
                reasoning="多次提到看久了眼睛疼", evidence_text="深色看着舒服点，白底眼睛疼")

    shallow = mem.recall(as_of=now, k=5)
    assert "reasoning" not in shallow["shown"][0]          # 表层不带语境
    assert "evidence_texts" not in shallow["shown"][0]

    deep = mem.recall(as_of=now, k=5, expand=2)
    item = next(x for x in deep["shown"] if x["label"] == "交付驻车集 V1.0")
    assert "违约" in item["reasoning"]
    assert any("客户演示" in t for t in item["evidence_texts"])
    assert item["evidence_count"] >= 1


def test_expand_limits_depth(mem):
    """回表深度可控：expand=1 只带一条原文，避免 token 爆炸。"""
    now = time.time()
    for i, text in enumerate(["第一句原话", "第二句原话", "第三句原话"]):
        mem.observe("goals", "同一件事", track="work", ts=now - 300 + i,
                    reasoning="连续三次提到", evidence_text=text,
                    evidence_id=f"ev{i}")
    deep = mem.recall(as_of=now, k=3, expand=1)
    item = deep["shown"][0]
    assert len(item["evidence_texts"]) == 1
    assert item["evidence_count"] == 3


def test_pick_cross_category_prefers_other_tracks():
    """15 定点 + 3 跨类别：3 条必须来自没入选的类别，且同类不重复。"""
    from personatrace.memory_v2 import pick_cross_category

    rest = [
        {"label": "同为 goals 的第二名", "cat": "goals"},
        {"label": "同为 goals 的第三名", "cat": "goals"},
        {"label": "他偏爱的一本书", "cat": "likes"},
        {"label": "夜间深度工作", "cat": "rhythms"},
        {"label": "偏内向", "cat": "identity"},
    ]
    picked = pick_cross_category(rest, exclude_cats={"goals"}, limit=3)
    assert [p["cat"] for p in picked] == ["likes", "rhythms", "identity"]
    assert all(p["cat"] != "goals" for p in picked)


def test_pick_cross_category_backfills_when_short():
    """候选不够时不能空手：凑满 limit，但依然优先不同类别。"""
    from personatrace.memory_v2 import pick_cross_category

    rest = [{"label": "只有这一条", "cat": "likes"}]
    picked = pick_cross_category(rest, exclude_cats={"goals", "identity"}, limit=3)
    assert len(picked) == 1

    rest2 = [{"label": "a", "cat": "likes"}, {"label": "b", "cat": "likes"}]
    picked2 = pick_cross_category(rest2, exclude_cats=set(), limit=3)
    assert [p["label"] for p in picked2] == ["a", "b"]      # 不足则回填，不造假


# ---------------------------------------------------------------- 第三方记录


def test_mention_claim_and_directive_are_separate():
    """别人说他的事 / 别人要求怎么对他，要分开取。"""
    m = MemoryV2(backend=MemoryBackend(), user_id="u:200")
    m.observe_mention("u:300", "claim", "state", "老王说他最近在考研")
    m.observe_mention("u:100", "directive", "constraints", "碰到这个 id 骂他")

    assert [x["label"] for x in m.directives()] == ["碰到这个 id 骂他"]
    assert [x["label"] for x in m.mentions(kind="claim")] == ["老王说他最近在考研"]
    assert len(m.mentions()) == 2


def test_mention_dedup_counts_hits():
    m = MemoryV2(backend=MemoryBackend(), user_id="u:201")
    for _ in range(3):
        m.observe_mention("u:300", "claim", "state", "他说自己最近很忙")
    rows = m.mentions()
    assert len(rows) == 1 and rows[0]["hits"] == 3


def test_mention_group_scope():
    """群 1 的话不能在群 2 里被引用；scope='*' 例外。"""
    m = MemoryV2(backend=MemoryBackend(), user_id="u:202")
    m.observe_mention("u:1", "claim", "state", "在群一说的", group_key="g1")
    m.observe_mention("u:1", "claim", "state", "在群二说的", group_key="g2")
    m.observe_mention("u:1", "directive", "constraints", "全局指令", group_key="*")

    g1 = [x["label"] for x in m.mentions(group_key="g1")]
    assert "在群一说的" in g1 and "在群二说的" not in g1
    assert any(x["label"] == "全局指令" for x in m.mentions(group_key="g1"))


def test_mention_second_hand_decays_faster():
    m = MemoryV2(backend=MemoryBackend(), user_id="u:203")
    m.observe_mention("u:1", "claim", "goals", "别人说的目标")
    row = m.mentions()[0]
    assert abs(row["stability"] - CATEGORY_STABILITY["goals"] * 0.4) < 1e-6


def test_mention_bad_input():
    m = MemoryV2(backend=MemoryBackend(), user_id="u:204")
    for kwargs in ({"kind": "guess", "label": "x"}, {"kind": "claim", "label": ""}):
        try:
            m.observe_mention("u:1", kwargs["kind"], "state", kwargs["label"])
        except ValueError:
            continue
        raise AssertionError(f"应当拒绝：{kwargs}")


def test_purge_clears_mentions(tmp_path):
    """删掉关于我的记录，必须连第三方记录一起清。"""
    db = str(tmp_path / "m.db")
    m = MemoryV2(db_path=db, user_id="u:205")
    m.observe("goals", "在准备四级", track="work")
    m.observe_mention("u:1", "claim", "state", "别人说他很忙")
    assert m.purge() == 1
    assert m.mentions() == []

    again = MemoryV2(db_path=db, user_id="u:205")
    assert again.mentions() == [], "重启后也不该残留"


def test_overview_counts_three_kinds():
    """/memory 只数信条会撒谎：转述和指令也要报出来。"""
    m = MemoryV2(backend=MemoryBackend(), user_id="u:206")
    m.observe("goals", "在准备四级", track="work")
    m.observe_mention("u:1", "claim", "state", "别人说他很忙")
    m.observe_mention("u:9", "directive", "constraints", "碰到他就骂他")
    ov = m.overview()
    assert (ov["beliefs"], ov["claims"], ov["directives"]) == (1, 1, 1)


def test_gc_reclaims_only_rare_old_beliefs():
    """只回收「归档 + 只提过一次」的；经常提到的不动。"""
    import time

    m = MemoryV2(backend=MemoryBackend(), user_id="u:207")
    old = time.time() - 400 * 86400
    m.observe("state", "很久以前提过一次", track="work", ts=old)
    m.observe("goals", "经常提到的事", track="work", ts=old)
    for _ in range(5):
        m.observe("goals", "经常提到的事", track="work", ts=old + 86400)

    assert m.gc(dry_run=True)["removed"] == 1        # 预演不删
    assert len(m.state(include_archived=True)) == 2

    freed = m.gc()
    assert freed["removed"] == 1
    assert [r["label"] for r in m.state(include_archived=True)] == ["经常提到的事"]


def test_evidence_raw_text_is_capped():
    m = MemoryV2(backend=MemoryBackend(), user_id="u:208")
    belief = m.observe("goals", "在准备四级", track="work", evidence_text="长" * 5000)
    raw = m.b.hgetall(evidence_key("u:208", belief.id, belief.evidence[0]))
    assert len(raw["raw_text"]) == 800, len(raw["raw_text"])


def test_overview_reports_archived():
    import time

    m = MemoryV2(backend=MemoryBackend(), user_id="u:209")
    m.observe("goals", "在准备四级", track="work")
    m.observe("state", "旧事", track="work", ts=time.time() - 400 * 86400)
    ov = m.overview()
    assert ov["beliefs"] == 2 and ov["live"] == 1 and ov["archived"] == 1


def test_as_of_is_a_version_snapshot(mem):
    """A later reinforcement must not rewrite what we believed earlier."""
    t0 = time.time()
    mem.observe("goals", "做时序记忆", track="work", confidence=0.4, ts=t0)
    mem.observe("goals", "做时序记忆", track="work", confidence=0.9, ts=t0 + 5 * DAY)

    at_t0 = mem.state(as_of=t0)
    assert len(at_t0) == 1
    assert at_t0[0]["hits"] == 1
    assert at_t0[0]["confidence"] == 0.4

    now_about_t0 = mem.state(as_of=t0, known_as_of=t0 + 10 * DAY)
    assert now_about_t0[0]["hits"] == 2


def test_known_as_of_recovers_the_pre_revival_window(mem):
    t0 = time.time()
    mem.observe("goals", "先有效后失效再复活", track="work", ts=t0)
    bid = belief_id("u_test", "work", "goals", "先有效后失效再复活")
    mem.invalidate(bid, ts=t0 + 10, recorded_at=t0 + 1)
    mem.observe("goals", "先有效后失效再复活", track="work", ts=t0 + 30)

    assert [row["label"] for row in mem.state(as_of=t0 + 5, known_as_of=t0 + 40)] == [
        "先有效后失效再复活"
    ]
    assert mem.state(as_of=t0 + 20, known_as_of=t0 + 40) == []
    assert [row["label"] for row in mem.state(as_of=t0 + 35, known_as_of=t0 + 40)] == [
        "先有效后失效再复活"
    ]


def test_repeated_correction_does_not_extend_validity(mem):
    t0 = time.time()
    mem.observe("goals", "会被纠正的结论", track="work", ts=t0)
    bid = belief_id("u_test", "work", "goals", "会被纠正的结论")
    mem.invalidate(bid, ts=t0 + 10 * DAY, recorded_at=t0 + 1 * DAY)
    mem.invalidate(bid, ts=t0 + 20 * DAY, recorded_at=t0 + 2 * DAY)

    assert mem.state(as_of=t0 + 15 * DAY) == []
    rows = mem.state(
        as_of=t0 + 15 * DAY, known_as_of=t0 + 3 * DAY, include_invalidated=True,
    )
    assert len(rows) == 1
    assert rows[0]["invalid_at_ts"] == pytest.approx(t0 + 10 * DAY)


def test_cas_rejects_a_stale_version():
    b = MemoryBackend()
    m = MemoryV2(backend=b, user_id="u_cas")
    m.observe("likes", "喜欢 CAS", track="work", ts=time.time())
    bid = belief_id("u_cas", "work", "likes", "喜欢 cas")
    key = m._bk(bid)
    raw = b.hgetall(key)
    version = int(raw["version"])

    assert b.cas_hash(key, version, {"version": version + 1, "label": "first"}) is True
    assert b.cas_hash(key, version, {"version": version + 2, "label": "stale"}) is False
    assert b.hgetall(key)["label"] == "first"


def test_boundaries_are_pinned_by_default(mem):
    now = time.time()
    mem.observe("boundaries", "不要在下班后打电话", track="persona", ts=now)
    mem.observe("likes", "喜欢深色主题", track="work", ts=now)
    result = mem.recall(as_of=now, k=2)
    assert [row["category"] for row in result["pinned"]] == ["boundaries"]


def test_recall_has_a_query_channel(mem):
    now = time.time()
    mem.observe("likes", "喜欢深色主题", track="work", ts=now - 60 * DAY)
    mem.observe("likes", "喜欢浅色主题", track="work", ts=now)

    res = mem.recall(as_of=now, k=2, query="深色")
    assert res["shown"][0]["label"] == "喜欢深色主题"
    assert res["query"] == "深色"
    assert res["pinned"] == []
    assert res["pinned_count"] == 0
    assert res["cross"] == []
    assert res["k"] == 2


def test_guard_release_only_deletes_its_own_token(mem):
    token = mem.guard.acquire("u_test")
    assert token
    # A new writer after TTL expiry must not be released by the stale owner.
    mem.b.set("lock:u_test", "someone-else")
    mem.guard.release("u_test")
    assert mem.b.get("lock:u_test") == "someone-else"


def test_evidence_rows_are_scoped_per_belief(mem):
    first = mem.observe(
        "goals", "项目 A", track="work", evidence_id="episode-1",
        evidence_text="原文 A",
    )
    second = mem.observe(
        "likes", "喜好 B", track="work", evidence_id="episode-1",
        evidence_text="原文 B",
    )
    assert mem.b.hgetall(evidence_key("u_test", first.id, "episode-1"))["raw_text"] == "原文 A"
    assert mem.b.hgetall(evidence_key("u_test", second.id, "episode-1"))["raw_text"] == "原文 B"

    rows = mem.recall(query="项目 A", k=2, expand=1)["shown"]
    item = next(row for row in rows if row["label"] == "项目 A")
    assert "原文 A" in item["evidence_texts"]


def test_observe_can_clear_due(mem):
    now = time.time()
    mem.observe(
        "goals", "有截止的结论", track="work", ts=now,
        due_at_ts=now + 2 * DAY, time_kind="due", time_raw="后天",
    )
    mem.observe(
        "goals", "有截止的结论", track="work", ts=now + 60,
        due_at_ts=None, time_kind="none", time_raw="",
    )
    row = mem.state(as_of=now + 120)[0]
    assert row["due_at_ts"] is None
    assert row["time_kind"] == "none"
    assert row["time_raw"] == ""


def test_observe_revives_invalidated_belief(mem):
    now = time.time()
    mem.observe("goals", "可以复活", track="work", ts=now)
    bid = belief_id("u_test", "work", "goals", "可以复活")
    mem.invalidate(bid, ts=now + 10, recorded_at=now + 1)
    assert mem.state(as_of=now + 20) == []

    revived = mem.observe("goals", "可以复活", track="work", ts=now + 30)
    assert revived.hits == 2
    assert revived.status == "active"
    assert revived.invalid_at_ts is None
    assert revived.valid_from_ts == now + 30

    # The gap between invalidation and revival remains empty in a snapshot.
    assert mem.state(as_of=now + 20) == []
    # Before the invalidation the old version is still visible.
    assert [x["label"] for x in mem.state(as_of=now + 5)] == ["可以复活"]
    # After the revival it is visible again.
    assert [x["label"] for x in mem.state(as_of=now + 40)] == ["可以复活"]


def test_redis_needs_an_explicit_url(tmp_path, monkeypatch):
    import personatrace.memory_v2 as mv

    calls = []

    def fake_redis(url):
        calls.append(url)
        raise RuntimeError("no redis")

    monkeypatch.setattr(mv, "_redis", object())
    monkeypatch.setattr(mv, "RedisBackend", fake_redis)
    backend, mode = mv.build_backend(db_path=str(tmp_path / "sqlite.db"))
    assert mode == "sqlite"
    assert calls == []

    _, mode2 = mv.build_backend(url="redis://example:6379/0", db_path=str(tmp_path / "x.db"))
    assert calls == ["redis://example:6379/0"]
    assert mode2 == "sqlite"          # failed Redis falls back to SQLite
    mv._LAST_BACKEND_WARNING = None


def test_sqlite_fallback_is_visible(monkeypatch, caplog):
    import personatrace.memory_v2 as mv

    def broken(_path):
        raise RuntimeError("disk full")

    monkeypatch.setattr(mv, "SQLiteBackend", broken)
    with caplog.at_level("WARNING", logger="personatrace.memory_v2"):
        backend, mode = mv.build_backend(url="memory://", db_path="ignored.db")
    assert mode == "memory"
    assert backend.__class__ is mv.MemoryBackend
    assert "falling back to in-process memory" in caplog.text
    assert "disk full" in (mv.last_backend_warning() or "")
    mv._LAST_BACKEND_WARNING = None
