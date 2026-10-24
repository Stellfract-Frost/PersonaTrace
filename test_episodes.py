"""Episode store tests: raw turns, provenance links, retrospective search."""

from datetime import datetime
from zoneinfo import ZoneInfo

from personatrace.episodes import EpisodeStore, has_retrospective

TZ = "Asia/Shanghai"
Z = ZoneInfo(TZ)
NOW = datetime(2026, 10, 27, 19, 5, tzinfo=Z)


def test_add_search_and_link():
    store = EpisodeStore(":memory:", default_tz=TZ)
    first = store.add(
        "u", "上周二晚上去上课", "好的", session_id="s1",
        ts=NOW.timestamp(), source="direct",
    )
    second = store.add(
        "u", "周末去爬山", "", session_id="s1",
        ts=NOW.timestamp() - 86400 * 3,
    )
    assert first and second
    assert store.link(first.id, belief_ids=["b1"], intention_ids=["i1"])

    hits = store.search("u", "上课")
    assert hits
    assert hits[0]["user_msg"] == "上周二晚上去上课"
    assert hits[0]["belief_ids"] == ["b1"]

    ranged = store.search("u", "爬山", after=NOW.timestamp() - 86400 * 4)
    assert [row["user_msg"] for row in ranged] == ["周末去爬山"]
    assert store.search("u", "爬山", before=NOW.timestamp() - 86400 * 4) == []


def test_source_gate_and_empty_turn():
    store = EpisodeStore(":memory:", default_tz=TZ)
    assert store.add("u", "") is None
    assert store.add("u", "hi", source="attention") is None
    assert store.add("u", "hi", source="bridge") is not None
    assert store.stats()["episodes"] == 1


def test_retrospective_markers():
    assert has_retrospective("上次说的那件事")
    assert has_retrospective("What did we discuss last time?")
    assert not has_retrospective("明天有什么安排")
