"""memory_v2 离线测试：不依赖 LLM / Redis / 网络。

跑法：D:/Anaconda3/python.exe -m pytest test_memory_v2.py -q
"""

import math
import time

import pytest

from personatrace.memory_v2 import (
    ARCHIVE_AT, CATEGORY_STABILITY, Conflict, MemoryBackend, MemoryV2,
    belief_id, reinforce, retention, stability_from_half_life,
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


def test_api_router():
    fastapi = pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api_v2 import build_router

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
    raw = m.b.hgetall(f"evidence:u:208:{belief.evidence[0]}")
    assert len(raw["raw_text"]) == 800, len(raw["raw_text"])


def test_overview_reports_archived():
    import time

    m = MemoryV2(backend=MemoryBackend(), user_id="u:209")
    m.observe("goals", "在准备四级", track="work")
    m.observe("state", "旧事", track="work", ts=time.time() - 400 * 86400)
    ov = m.overview()
    assert ov["beliefs"] == 2 and ov["live"] == 1 and ov["archived"] == 1


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
    raw = m.b.hgetall(f"evidence:u:208:{belief.evidence[0]}")
    assert len(raw["raw_text"]) == 800, len(raw["raw_text"])


def test_overview_reports_archived():
    import time

    m = MemoryV2(backend=MemoryBackend(), user_id="u:209")
    m.observe("goals", "在准备四级", track="work")
    m.observe("state", "旧事", track="work", ts=time.time() - 400 * 86400)
    ov = m.overview()
    assert ov["beliefs"] == 2 and ov["live"] == 1 and ov["archived"] == 1
