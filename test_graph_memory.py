"""Tests for stable-ID graph memory, Pydantic edges, and retrying LLMs."""

import copy
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from personatrace.graph_memory import (
    EdgeType, GraphEdgeBatch, GraphEdgeDraft, GraphExtractor, GraphNodeDraft,
    GraphStore, NodeKind,
)
from personatrace.structured import StructuredOutputError, call_structured

TZ = timezone.utc


def node(node_id: str, label: str, kind: NodeKind = NodeKind.belief) -> GraphNodeDraft:
    return GraphNodeDraft(id=node_id, kind=kind, label=label, user_id="u")


def edge(source: str, target: str, kind: EdgeType = EdgeType.similarity, weight: float = 0.8) -> GraphEdgeDraft:
    return GraphEdgeDraft(
        source_id=source, target_id=target, type=kind,
        weight=weight, confidence=0.9, reason="test",
    )


def test_pydantic_rejects_self_loop_and_bad_weight():
    with pytest.raises(ValidationError):
        GraphEdgeDraft(source_id="a", target_id="a", type=EdgeType.related)
    with pytest.raises(ValidationError):
        GraphEdgeDraft(source_id="a", target_id="b", type=EdgeType.related, weight=1.5)


def test_stable_node_id_survives_rename_and_edge_stays():
    store = GraphStore(":memory:")
    store.upsert_node(node("belief:a", "写 Rust"))
    store.upsert_node(node("belief:b", "喜欢编译器"))
    e = store.upsert_edge(edge("belief:a", "belief:b"), user_id="u")

    store.upsert_node(node("belief:a", "写 Rust 和 Go"))
    assert store.node("belief:a")["label"] == "写 Rust 和 Go"
    assert store.edge(e["id"]) is not None
    assert store.neighbors("u", "belief:a")[0]["node_id"] == "belief:b"
    assert store.search_nodes("u", "Rust")[0]["id"] == "belief:a"


def test_upsert_edge_is_idempotent_for_undirected_types():
    store = GraphStore(":memory:")
    store.upsert_node(node("belief:a", "A"))
    store.upsert_node(node("belief:b", "B"))
    store.upsert_edge(edge("belief:a", "belief:b", weight=0.6), user_id="u")
    store.upsert_edge(edge("belief:b", "belief:a", weight=0.8), user_id="u")
    edges = store.edges("u")
    assert len(edges) == 1
    assert edges[0]["weight"] == 0.8


def test_expand_two_hops_and_exclude_contradiction():
    store = GraphStore(":memory:")
    for nid, label in [("belief:a", "A"), ("belief:b", "B"), ("belief:c", "C"), ("belief:d", "D")]:
        store.upsert_node(node(nid, label))
    store.upsert_edge(edge("belief:a", "belief:b"), user_id="u")
    store.upsert_edge(edge("belief:b", "belief:c", EdgeType.related, 0.5), user_id="u")
    store.upsert_edge(edge("belief:a", "belief:d", EdgeType.contradiction, 0.9), user_id="u")

    one_hop = store.expand("u", ["belief:a"], max_hops=1, limit=5)
    assert [x["id"] for x in one_hop] == ["belief:b"]

    two_hops = store.expand("u", ["belief:a"], max_hops=2, limit=5)
    assert "belief:c" in [x["id"] for x in two_hops]
    assert "belief:d" not in [x["id"] for x in two_hops]  # contradiction is not inspiration


def test_invalidate_keeps_history_and_as_of():
    store = GraphStore(":memory:")
    store.upsert_node(node("belief:a", "A"))
    store.upsert_node(node("belief:b", "B"))
    e = store.upsert_edge(
        GraphEdgeDraft(
            source_id="belief:a", target_id="belief:b", type=EdgeType.similarity,
            valid_from=datetime(2026, 10, 20, tzinfo=TZ),
        ),
        user_id="u",
    )
    cutoff = datetime(2026, 10, 22, tzinfo=TZ).timestamp()
    store.invalidate_edge(e["id"], at=cutoff)

    assert store.neighbors("u", "belief:a", as_of=cutoff - 1)
    assert store.neighbors("u", "belief:a", as_of=cutoff + 1) == []
    assert len(store.edges("u", include_invalid=True)) == 1


class FakeLLM:
    def __init__(self, payloads):
        self.payloads = [copy.deepcopy(p) for p in payloads]
        self.calls = 0

    def chat_json(self, messages, max_tokens=1200):
        self.calls += 1
        if not self.payloads:
            return None
        return self.payloads.pop(0)


def test_structured_retry_feeds_validation_error_back():
    llm = FakeLLM([
        {"edges": [{"source_id": "a", "target_id": "a", "type": "related"}]},
        {"edges": [{"source_id": "a", "target_id": "b", "type": "related"}]},
    ])
    batch = call_structured(llm, "build edges", GraphEdgeBatch, retries=1)
    assert llm.calls == 2
    assert batch.edges[0].target_id == "b"


def test_structured_retry_exhausted_raises():
    llm = FakeLLM([{"edges": [{"source_id": "a", "target_id": "a", "type": "related"}]}] * 2)
    with pytest.raises(StructuredOutputError):
        call_structured(llm, "build edges", GraphEdgeBatch, retries=1)


def test_graph_extractor_retries_and_filters_invented_ids():
    llm = FakeLLM([
        {"edges": [{"source_id": "intention:new", "target_id": "intention:new", "type": "related"}]},
        {"edges": [
            {"source_id": "intention:new", "target_id": "belief:a", "type": "related", "weight": 0.7},
            {"source_id": "intention:new", "target_id": "invented:id", "type": "related"},
        ]},
    ])
    extractor = GraphExtractor(llm, retries=1)
    new = node("intention:new", "上课", NodeKind.intention)
    candidates = [node("belief:a", "课程"), node("belief:b", "老师")]
    edges = extractor.infer_edges("u", new, candidates)
    assert llm.calls == 2
    assert len(edges) == 1
    assert edges[0].target_id == "belief:a"


def test_context_lines_from_query():
    store = GraphStore(":memory:")
    store.upsert_node(node("belief:a", "写 Rust"))
    store.upsert_node(node("belief:b", "喜欢编译器"))
    store.upsert_edge(edge("belief:a", "belief:b"), user_id="u")
    lines = store.context_lines("u", query="Rust", limit=3)
    assert any("喜欢编译器" in line for line in lines)
