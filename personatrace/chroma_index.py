"""Optional Chroma index for raw episodes.

This is an index, not a ledger. SQLite owns the episode rows and every state
transition; Chroma only returns candidate ids. If Chroma is absent, slow, or
misconfigured, EpisodeStore falls back to lexical + time search.

The collection is separate from v1's ``persona_items`` collection: raw turns and
profile labels have different id/metadata semantics and must not be mixed.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

try:  # optional dependency: personatrace[vector]
    import chromadb
except Exception:  # pragma: no cover - depends on install
    chromadb = None

DEFAULT_COLLECTION = "episode_turns"


def _value(item: Any, key: str, default: Any = "") -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _episode_text(item: Any) -> str:
    user = str(_value(item, "user_msg", "")).strip()
    reply = str(_value(item, "ai_reply", "")).strip()
    return f"{user}\n{reply}".strip() or user


class ChromaEpisodeIndex:
    """Small wrapper around a Chroma collection of raw turns."""

    def __init__(
        self,
        path: str = "./persona_data/chroma",
        *,
        embed_fn: Any,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        if chromadb is None:
            raise ImportError(
                "ChromaDB is not installed. Install with: pip install 'personatrace[vector]'"
            )
        if not callable(embed_fn):
            raise ValueError("embed_fn must be callable: list[str] -> list[list[float]]")
        self.path = str(path)
        self.embed_fn = embed_fn
        self.client = (
            chromadb.EphemeralClient()
            if self.path in {":memory:", "memory://"}
            else chromadb.PersistentClient(path=self.path)
        )
        self.collection = self.client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, episodes: Iterable[Any]) -> None:
        rows = list(episodes)
        if not rows:
            return
        docs = [_episode_text(row) for row in rows]
        vectors = self.embed_fn(docs)
        if len(vectors) != len(rows):
            raise ValueError("embedding count does not match episode count")
        ids = [str(_value(row, "id")) for row in rows]
        metadatas = [{
            "user_id": str(_value(row, "user_id")),
            "ts": float(_value(row, "ts", 0.0)),
            "session_id": str(_value(row, "session_id")),
            "source": str(_value(row, "source", "direct")),
        } for row in rows]
        self.collection.upsert(
            ids=ids, documents=docs, embeddings=vectors, metadatas=metadatas,
        )

    def query(
        self,
        text: str,
        *,
        user_id: str,
        limit: int = 10,
        after: Optional[float] = None,
        before: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        text = (text or "").strip()
        if not text or not user_id:
            return []
        total = self.collection.count()
        if total <= 0:
            return []
        where: Dict[str, Any] = {"user_id": user_id}
        if after is not None and before is not None:
            where = {"$and": [
                {"user_id": user_id},
                {"ts": {"$gte": float(after)}},
                {"ts": {"$lte": float(before)}},
            ]}
        elif after is not None:
            where = {"$and": [{"user_id": user_id}, {"ts": {"$gte": float(after)}}]}
        elif before is not None:
            where = {"$and": [{"user_id": user_id}, {"ts": {"$lte": float(before)}}]}
        vector = self.embed_fn([text])[0]
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=min(max(1, int(limit)), total),
            where=where,
            include=["metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        out = []
        for index, episode_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            out.append({
                "id": str(episode_id),
                "score": round(max(0.0, min(1.0, 1.0 - distance)), 4),
            })
        return out

    def delete(self, episode_ids: Iterable[str]) -> None:
        ids = [str(x) for x in episode_ids if x]
        if ids:
            self.collection.delete(ids=ids)

    def delete_user(self, user_id: str) -> None:
        if user_id:
            self.collection.delete(where={"user_id": user_id})

    def count(self) -> int:
        return int(self.collection.count())

    def close(self) -> None:
        # PersistentClient owns an internal client; there is no public close in
        # older Chroma versions. Let process teardown handle it.
        return None
