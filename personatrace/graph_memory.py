"""Graph memory with stable node IDs and Pydantic-validated edges.

The old skills_bot graph used labels as nodes. That works until a label is
renamed: the node disappears and its edges point at text that no longer exists.

This version keeps:
- nodes with stable IDs (``belief:<uuid>`` / ``intention:<uuid>`` / ``topic:...``);
- edges with a type, weight, confidence and validity window;
- label changes update the node, never the edges;
- invalidation instead of deletion.

The graph is an *expander*, not the primary ranker. Call ``expand()`` after a
semantic top-k, not instead of it.
"""

from __future__ import annotations

import difflib
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:  # package import and top-level test import both work
    from .structured import call_structured
except ImportError:  # pragma: no cover - depends on how the caller imports it
    from structured import call_structured


class NodeKind(str, Enum):
    belief = "belief"
    intention = "intention"
    topic = "topic"
    turn = "turn"


class EdgeType(str, Enum):
    similarity = "similarity"
    contradiction = "contradiction"
    supersede = "supersede"
    related = "related"
    mentions = "mentions"


class GraphNodeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    kind: NodeKind
    label: str = Field(min_length=1, max_length=200)
    user_id: str = Field(default="default", min_length=1)
    valid_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "label", "user_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class GraphEdgeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1, max_length=200)
    target_id: str = Field(min_length=1, max_length=200)
    type: EdgeType
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=500)
    source_turn: str = Field(default="", max_length=200)
    valid_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("source_id", "target_id")
    @classmethod
    def _strip_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("node id must not be blank")
        return value

    @model_validator(mode="after")
    def _no_self_loop(self) -> "GraphEdgeDraft":
        if self.source_id == self.target_id:
            raise ValueError("source_id and target_id must differ")
        return self


class GraphEdgeBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edges: List[GraphEdgeDraft] = Field(default_factory=list)


def _epoch(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _undirected(edge_type: EdgeType) -> bool:
    return edge_type in (EdgeType.similarity, EdgeType.contradiction, EdgeType.related)


def _ordered_pair(source: str, target: str, edge_type: EdgeType) -> Tuple[str, str]:
    if _undirected(edge_type) and target < source:
        return target, source
    return source, target


class GraphStore:
    """SQLite graph: nodes, typed edges, validity windows, bounded traversal."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=5.0, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout = 5000")
        if path != ":memory:":
            try:
                self.conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.DatabaseError:
                pass
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                label TEXT NOT NULL,
                valid_from REAL NOT NULL,
                invalid_at REAL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_graph_nodes_user
                ON graph_nodes(user_id, kind, label);
            CREATE TABLE IF NOT EXISTS graph_edges (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0.5,
                confidence REAL NOT NULL DEFAULT 0.5,
                reason TEXT NOT NULL DEFAULT '',
                source_turn TEXT NOT NULL DEFAULT '',
                valid_from REAL NOT NULL,
                invalid_at REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_graph_edges_source
                ON graph_edges(user_id, source_id, type);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_target
                ON graph_edges(user_id, target_id, type);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_edges_active
                ON graph_edges(user_id, source_id, target_id, type)
                WHERE invalid_at IS NULL;
            """
        )
        self.conn.commit()

    # ---------------------------------------------------------------- writes
    def upsert_node(self, node: GraphNodeDraft) -> Dict[str, Any]:
        now = time.time()
        valid_from = _epoch(node.valid_from)
        existing = self.conn.execute(
            "SELECT * FROM graph_nodes WHERE id=?", (node.id,)
        ).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE graph_nodes SET user_id=?, kind=?, label=?, metadata=?, "
                "updated_at=? WHERE id=?",
                (node.user_id, node.kind.value, node.label,
                 json.dumps(node.metadata, ensure_ascii=False), now, node.id),
            )
        else:
            self.conn.execute(
                "INSERT INTO graph_nodes(id, user_id, kind, label, valid_from, "
                "invalid_at, metadata, created_at, updated_at) "
                "VALUES(?,?,?,?,?,NULL,?,?,?)",
                (node.id, node.user_id, node.kind.value, node.label,
                 valid_from, json.dumps(node.metadata, ensure_ascii=False), now, now),
            )
        self.conn.commit()
        return self.node(node.id) or {}

    def upsert_edge(self, edge: GraphEdgeDraft, *, user_id: str = "default") -> Dict[str, Any]:
        source, target = _ordered_pair(edge.source_id, edge.target_id, edge.type)
        if self.node(source) is None or self.node(target) is None:
            raise ValueError(f"unknown graph node: {source!r} -> {target!r}")
        now = time.time()
        existing = self.conn.execute(
            "SELECT * FROM graph_edges WHERE user_id=? AND source_id=? AND "
            "target_id=? AND type=? AND invalid_at IS NULL",
            (user_id, source, target, edge.type.value),
        ).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE graph_edges SET weight=?, confidence=?, reason=?, "
                "source_turn=?, updated_at=? WHERE id=?",
                (
                    max(float(existing["weight"]), float(edge.weight)),
                    max(float(existing["confidence"]), float(edge.confidence)),
                    edge.reason or existing["reason"],
                    edge.source_turn or existing["source_turn"],
                    now, existing["id"],
                ),
            )
            self.conn.commit()
            return self.edge(existing["id"]) or {}
        edge_id = uuid.uuid4().hex
        self.conn.execute(
            "INSERT INTO graph_edges(id, user_id, source_id, target_id, type, "
            "weight, confidence, reason, source_turn, valid_from, invalid_at, "
            "created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,NULL,?,?)",
            (
                edge_id, user_id, source, target,
                edge.type.value, edge.weight, edge.confidence, edge.reason,
                edge.source_turn, _epoch(edge.valid_from), now, now,
            ),
        )
        self.conn.commit()
        return self.edge(edge_id) or {}

    def invalidate_edge(self, edge_id: str, *, at: Optional[float] = None) -> None:
        now = time.time() if at is None else float(at)
        self.conn.execute(
            "UPDATE graph_edges SET invalid_at=?, updated_at=? WHERE id=? AND invalid_at IS NULL",
            (now, now, edge_id),
        )
        self.conn.commit()

    # ---------------------------------------------------------------- reads
    def node(self, node_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM graph_nodes WHERE id=?", (node_id,)
        ).fetchone()
        return self._node_dict(row) if row else None

    def edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM graph_edges WHERE id=?", (edge_id,)
        ).fetchone()
        return self._edge_dict(row) if row else None

    def nodes(self, user_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM graph_nodes WHERE user_id=? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return [self._node_dict(row) for row in rows]

    def edges(self, user_id: str, *, include_invalid: bool = False) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM graph_edges WHERE user_id=?"
        if not include_invalid:
            sql += " AND invalid_at IS NULL"
        rows = self.conn.execute(sql, (user_id,)).fetchall()
        return [self._edge_dict(row) for row in rows]

    def neighbors(
        self,
        user_id: str,
        node_id: str,
        *,
        types: Optional[Iterable[EdgeType]] = None,
        as_of: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        as_of = time.time() if as_of is None else float(as_of)
        allowed = {t.value if isinstance(t, EdgeType) else str(t) for t in types} if types else None
        rows = self.conn.execute(
            "SELECT * FROM graph_edges WHERE user_id=? AND "
            "(source_id=? OR target_id=?)",
            (user_id, node_id, node_id),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            if allowed and row["type"] not in allowed:
                continue
            if row["valid_from"] > as_of:
                continue
            if row["invalid_at"] is not None and row["invalid_at"] <= as_of:
                continue
            other = row["target_id"] if row["source_id"] == node_id else row["source_id"]
            out.append({
                "node_id": other,
                "edge_id": row["id"],
                "type": row["type"],
                "weight": float(row["weight"]),
                "confidence": float(row["confidence"]),
                "direction": "out" if row["source_id"] == node_id else "in",
            })
        return out

    def expand(
        self,
        user_id: str,
        seed_ids: Iterable[str],
        *,
        max_hops: int = 1,
        limit: int = 5,
        exclude_types: Iterable[EdgeType] = (EdgeType.contradiction,),
        as_of: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """BFS from seeds; returns nodes with hops/path score, closest first."""
        as_of = time.time() if as_of is None else float(as_of)
        excluded = {t.value if isinstance(t, EdgeType) else str(t) for t in exclude_types}
        seeds = [s for s in seed_ids if s]
        visited = set(seeds)
        queue: List[Tuple[str, int, float, Tuple[str, ...]]] = [(s, 0, 1.0, (s,)) for s in seeds]
        found: List[Dict[str, Any]] = []
        while queue:
            node_id, hops, score, path = queue.pop(0)
            if hops >= max_hops:
                continue
            for edge in self.neighbors(user_id, node_id, as_of=as_of):
                if edge["type"] in excluded:
                    continue
                neighbor = edge["node_id"]
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                node = self.node(neighbor)
                if node is None or node["user_id"] != user_id:
                    continue
                next_score = round(score * edge["weight"], 4)
                found.append({
                    "id": neighbor, "label": node["label"], "kind": node["kind"],
                    "hops": hops + 1, "edge_type": edge["type"],
                    "score": next_score, "path": list(path + (neighbor,)),
                })
                queue.append((neighbor, hops + 1, next_score, path + (neighbor,)))
        found.sort(key=lambda item: (item["hops"], -item["score"], item["label"]))
        return found[:max(0, int(limit))]

    def search_nodes(self, user_id: str, query: str, *, limit: int = 5) -> List[Dict[str, Any]]:
        query = (query or "").strip().lower()
        if not query:
            return []
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for node in self.nodes(user_id):
            label = node["label"].lower()
            if query in label or label in query:
                score = 1.0
            else:
                score = difflib.SequenceMatcher(None, query, label).ratio()
            if score >= 0.3:
                scored.append((score, node))
        scored.sort(key=lambda item: (-item[0], item[1]["label"]))
        return [node for _, node in scored[:limit]]

    def context_lines(
        self,
        user_id: str,
        *,
        query: Optional[str] = None,
        seed_ids: Optional[Iterable[str]] = None,
        max_hops: int = 1,
        limit: int = 3,
    ) -> List[str]:
        seeds = list(seed_ids or [])
        if not seeds and query:
            seeds = [node["id"] for node in self.search_nodes(user_id, query)]
        if not seeds:
            return []
        rows = self.expand(user_id, seeds, max_hops=max_hops, limit=limit)
        return [
            f"- {row['label']}（{row['edge_type']}，{row['hops']} 跳）"
            for row in rows
        ]

    def stats(self) -> Dict[str, int]:
        nodes = self.conn.execute("SELECT COUNT(*) AS n FROM graph_nodes").fetchone()["n"]
        edges = self.conn.execute("SELECT COUNT(*) AS n FROM graph_edges").fetchone()["n"]
        return {"nodes": int(nodes), "edges": int(edges)}

    def close(self) -> None:
        self.conn.close()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _node_dict(row: sqlite3.Row) -> Dict[str, Any]:
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except (TypeError, ValueError):
            metadata = {}
        return {
            "id": row["id"], "user_id": row["user_id"], "kind": row["kind"],
            "label": row["label"], "valid_from": row["valid_from"],
            "invalid_at": row["invalid_at"], "metadata": metadata,
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    @staticmethod
    def _edge_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"], "user_id": row["user_id"],
            "source_id": row["source_id"], "target_id": row["target_id"],
            "type": row["type"], "weight": row["weight"],
            "confidence": row["confidence"], "reason": row["reason"],
            "source_turn": row["source_turn"], "valid_from": row["valid_from"],
            "invalid_at": row["invalid_at"], "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


class GraphExtractor:
    """Use an LLM to propose edges, with Pydantic validation and retries.

    The model only selects from candidate IDs; it never invents node IDs. If it
    returns an invalid edge, the schema error is sent back and the call retried.
    """

    def __init__(self, llm: Any, *, retries: int = 2) -> None:
        self.llm = llm
        self.retries = retries

    def infer_edges(
        self,
        user_id: str,
        new_node: GraphNodeDraft,
        candidates: Iterable[GraphNodeDraft],
        *,
        source_turn: str = "",
    ) -> List[GraphEdgeDraft]:
        candidates = list(candidates)
        allowed = {new_node.id} | {c.id for c in candidates}
        candidate_text = "\n".join(
            f"- id={c.id!r} kind={c.kind.value!r} label={c.label!r}" for c in candidates
        )
        prompt = f"""You are building a memory graph for one user.

A new node:
- id={new_node.id!r} kind={new_node.kind.value!r} label={new_node.label!r}

Existing candidate nodes:
{candidate_text}

Choose only edges that involve the new node and one candidate. Allowed types:
- similarity: same meaning, one can help retrieve the other
- contradiction: the two cannot both be true now
- supersede: the new node replaces the candidate
- related: a useful association, but not the same thing
- mentions: the new node mentions the candidate

Rules:
- Do not invent IDs. Only use IDs shown above.
- No self-edges.
- weight and confidence are 0..1.
- An empty list is valid when nothing is related.
- Return JSON: {{"edges": [...]}}.
"""
        try:
            batch = call_structured(
                self.llm, prompt, GraphEdgeBatch, retries=self.retries, max_tokens=1200
            )
        except Exception:
            return []
        result: List[GraphEdgeDraft] = []
        for edge in batch.edges:
            if edge.source_id not in allowed or edge.target_id not in allowed:
                continue
            if new_node.id not in (edge.source_id, edge.target_id):
                continue
            if source_turn and not edge.source_turn:
                edge.source_turn = source_turn[:200]
            result.append(edge)
        return result
