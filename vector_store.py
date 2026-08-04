"""
ChromaDB vector store module.
Provides persistent storage and similarity search for profile items.
Embedding function is pluggable — inject any embedding model.
"""
import json
import uuid
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
from .config import PersonaConfig
from .models import WORK_CATEGORIES, PERSONA_CATEGORIES, CROSS_CATEGORIES, EPHEMERAL_CATEGORIES
logger = logging.getLogger(__name__)
CHROMA_COLLECTION = "persona_items"
class VectorStore:
    """ChromaDB-backed vector store.
    Provides CRUD operations and similarity search for profile items.
    Embedding function is injected via set_embed_fn — supports any embedding model.
    Attributes:
        config: configuration object
        client: ChromaDB client
        collection: ChromaDB collection
        embed_fn: embedding function (input List[str], output List[List[float]])
    """
    def __init__(self, config: PersonaConfig):
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Run: pip install chromadb"
            )
        self.config = config
        self.client = chromadb.PersistentClient(path=config.vector_store_path)
        self.collection = self.client.get_or_create_collection(CHROMA_COLLECTION)
        self.embed_fn = None  # injected externally
        self._embed_dim = None  # actual embedding dimension, learned from embed_fn output
        self._lock = threading.Lock()
    def set_embed_fn(self, embed_fn):
        """Set the embedding function.
        Args:
            embed_fn: callable, input List[str], output List[List[float]]
        """
        self.embed_fn = embed_fn
    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors. Returns zero vectors on failure."""
        dims = self.config.embed_dims
        if self.embed_fn is None:
            return [[0.0] * dims for _ in texts]
        try:
            vecs = self.embed_fn(texts)
            out = [list(v) for v in vecs]
            if out:
                self._embed_dim = len(out[0])
            return out
        except Exception as e:
            logger.warning("Embedding generation failed: %s", e)
            return [[0.0] * dims for _ in texts]
    @staticmethod
    def _make_id(user_id: str, category: str, label: str, item: Any = None) -> str:
        """Generate a deterministic UUID based on user_id:category:label.
        Items with a ts field (e.g. observation_log) are additionally keyed by ts
        so repeated labels under the same category don't collide.
        """
        key = f"{user_id}:{category}:{label.lower().strip()}"
        if isinstance(item, dict) and item.get("ts"):
            key += f":{item['ts']}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))
    def upsert_items(
        self, user_id: str, category: str, items: List[Any]
    ) -> None:
        """Batch insert or update profile items."""
        if not items:
            return
        labels = []
        valid_items = []
        for item in items:
            label = self._extract_label(item)
            if label:
                labels.append(label)
                valid_items.append(item)
        if not labels:
            return
        if self.embed_fn is None:
            logger.warning("Embedding function not set, skipping save for category %s", category)
            return
        vecs = self._embed(labels)
        ids, documents, metadatas, embeddings = [], [], [], []
        now_ts = datetime.now().isoformat()
        for item, label, vec in zip(valid_items, labels, vecs):
            ids.append(self._make_id(user_id, category, label, item))
            documents.append(label)
            confidence = item.get("confidence", 0.5) if isinstance(item, dict) else 0.5
            meta = {
                "user_id": user_id,
                "category": category,
                "label": label,
                "updated": now_ts,
                "confidence": confidence,
            }
            if isinstance(item, dict):
                meta["item_data_json"] = json.dumps(item, ensure_ascii=False)
                if "cluster_id" in item:
                    meta["cluster_id"] = item["cluster_id"]
            metadatas.append(meta)
            embeddings.append(vec)
        try:
            self.collection.upsert(
                ids=ids, documents=documents,
                metadatas=metadatas, embeddings=embeddings,
            )
        except Exception as e:
            logger.error("ChromaDB upsert failed [%s]: %s", category, e)
    def upsert_meta(self, user_id: str, label: str, data: Any) -> None:
        """Store metadata (zero vector, not searchable)."""
        point_id = self._make_id(user_id, "meta", label)
        meta = {
            "user_id": user_id, "category": "meta", "label": label,
            "updated": datetime.now().isoformat(), "confidence": 0,
            "item_data_json": json.dumps(data, ensure_ascii=False),
        }
        try:
            self.collection.upsert(
                ids=[point_id], documents=[label], metadatas=[meta],
                embeddings=[[0.0] * (self._embed_dim or self.config.embed_dims)],
            )
        except Exception as e:
            logger.error("Metadata upsert failed [%s]: %s", label, e)
    def delete_category(self, user_id: str, category: str) -> None:
        """Delete all entries for a given user + category."""
        try:
            self.collection.delete(
                where={"$and": [{"user_id": user_id}, {"category": category}]}
            )
        except Exception as e:
            logger.error("Delete failed [%s]: %s", category, e)
    def load_all(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load all profile data for a user, rebuilding the nested dict structure."""
        try:
            result = self.collection.get(
                where={"user_id": user_id},
                include=["metadatas", "documents"],
            )
        except Exception as e:
            logger.error("Load failed: %s", e)
            return None
        if not result or not result.get("ids"):
            return None
        work_dict = {cat: [] for cat in WORK_CATEGORIES}
        persona_dict = {cat: [] for cat in PERSONA_CATEGORIES}
        cross_dict = {cat: [] for cat in CROSS_CATEGORIES}
        ephemeral_dict = {cat: [] for cat in EPHEMERAL_CATEGORIES}
        meta_dict = {}
        for meta, doc in zip(result.get("metadatas", []), result.get("documents", [])):
            cat = meta.get("category", "")
            label = meta.get("label", "")
            item_data = None
            if meta.get("item_data_json"):
                try:
                    item_data = json.loads(meta["item_data_json"])
                except json.JSONDecodeError:
                    pass
            if cat == "meta":
                meta_dict[label] = item_data
            elif cat in work_dict:
                work_dict[cat].append(item_data or label)
            elif cat in persona_dict:
                persona_dict[cat].append(item_data or label)
            elif cat in cross_dict:
                cross_dict[cat].append(item_data or label)
            elif cat in ephemeral_dict:
                ephemeral_dict[cat].append(item_data or label)
        return {
            "work": work_dict,
            "persona": persona_dict,
            "cross": cross_dict,
            "ephemeral": ephemeral_dict,
            "graph_edges": meta_dict.get("_graph_edges", []),
            "last_updated": meta_dict.get("_last_updated", datetime.now().isoformat()),
            "version": meta_dict.get("_version", {}).get("version", 0),
            "interaction": meta_dict.get("_interaction", {"prefer_few_questions": False}),
            "pending_advisor": meta_dict.get("_pending_advisor"),
            "profile_completeness": meta_dict.get("_profile_completeness", {"work": 0.0, "persona": 0.0}),
        }
    def search_similar(
        self, user_id: str, query_text: str,
        category: Optional[str] = None, top_k: int = 10,
    ) -> List[Tuple[float, Dict]]:
        """Vector similarity search.
        Returns:
            [(score, payload_dict), ...] sorted by similarity descending
        """
        vecs = self._embed([query_text])
        if not vecs:
            return []
        # chroma rejects a flat multi-key where; multiple filters must be wrapped in $and
        if category:
            where = {"$and": [{"user_id": user_id}, {"category": category}]}
        else:
            where = {"user_id": user_id}
        try:
            results = self.collection.query(
                query_embeddings=[vecs[0]], n_results=top_k, where=where,
                include=["metadatas", "documents", "distances"],
            )
            scores = []
            for i, meta in enumerate(results.get("metadatas", [[]])[0]):
                dist = results.get("distances", [[]])[0][i]
                score = round(1.0 - dist, 4)
                payload = dict(meta)
                if payload.get("item_data_json"):
                    try:
                        payload["item_data"] = json.loads(payload["item_data_json"])
                    except json.JSONDecodeError:
                        pass
                scores.append((score, payload))
            return scores
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []
    def count(self, user_id: str) -> int:
        """Return the total number of profile items for a user."""
        try:
            result = self.collection.get(where={"user_id": user_id}, include=[])
            return len(result.get("ids", []))
        except Exception:
            return -1
    @staticmethod
    def _extract_label(item: Any) -> str:
        """Extract label text from an item."""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            label = item.get("label", "")
            if isinstance(label, dict):
                label = label.get("label", "")
            return str(label)
        return str(item)
