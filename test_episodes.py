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

class FakeVectorIndex:
    def __init__(self):
        self.rows = {}
        self.deleted = []
        self.fail = False

    def upsert(self, episodes):
        for episode in episodes:
            self.rows[episode.id] = episode

    def query(self, text, *, user_id, limit=10, after=None, before=None):
        if self.fail:
            raise RuntimeError("vector down")
        return [{"id": episode_id, "score": 0.99} for episode_id in list(self.rows)[:limit]]

    def delete(self, episode_ids):
        for episode_id in episode_ids:
            self.rows.pop(episode_id, None)
            self.deleted.append(episode_id)

    def close(self):
        pass


def test_vector_index_is_optional_and_falls_back():
    index = FakeVectorIndex()
    store = EpisodeStore(":memory:", default_tz=TZ, vector_index=index)
    first = store.add("u", "alpha topic", ts=NOW.timestamp())
    second = store.add("u", "beta topic", ts=NOW.timestamp() - 100)
    assert first and second
    assert set(index.rows) == {first.id, second.id}

    # Vector candidate wins even when it does not share a lexical token.
    index.rows = {second.id: second}
    hits = store.search("u", "alpha topic")
    assert hits and hits[0]["id"] == second.id

    index.fail = True
    fallback = store.search("u", "alpha topic")
    assert fallback and fallback[0]["id"] == first.id

    store.delete(first.id)
    assert first.id in index.deleted


def test_chroma_episode_index_if_available(tmp_path):
    import pytest
    pytest.importorskip("chromadb")
    from personatrace.chroma_index import ChromaEpisodeIndex

    index = ChromaEpisodeIndex(
        tmp_path / "chroma",
        embed_fn=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
        collection_name=f"test_episodes_{tmp_path.name}",
    )
    store = EpisodeStore(":memory:", default_tz=TZ, vector_index=index)
    episode = store.add("u", "vector indexed turn", ts=NOW.timestamp())
    assert episode is not None
    hits = index.query(
        "vector", user_id="u", limit=3,
        after=NOW.timestamp() - 1000, before=NOW.timestamp() + 1000,
    )
    assert hits and hits[0]["id"] == episode.id
    index.delete([episode.id])
    assert index.count() == 0

