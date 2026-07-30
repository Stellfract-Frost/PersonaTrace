"""
Similarity computation and dedup module.
Provides embedding-based semantic similarity,
with cross-category dedup functionality.
"""
import logging
import threading
from typing import List, Dict, Any, Optional
import numpy as np
from .config import PersonaConfig
logger = logging.getLogger(__name__)
class SimilarityCalculator:
    """Semantic similarity calculator.
    Uses embedding cosine similarity,
    with Jaccard bigram fallback.
    Supports caching to accelerate repeated computations.
    Attributes:
        config: configuration object
        embed_fn: externally injected embedding function
        _cache: similarity score cache
        _embed_cache: embedding vector cache
    """
    def __init__(self, config: PersonaConfig):
        self.config = config
        self.embed_fn = None
        self._cache: Dict[tuple, float] = {}
        self._embed_cache: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
    def set_embed_fn(self, embed_fn):
        """Set the embedding function."""
        self.embed_fn = embed_fn
    @staticmethod
    def extract_label(item: Any) -> str:
        """Safely extract label text from an item."""
        if item is None:
            return ""
        if isinstance(item, dict):
            label = item.get("label", "")
            if isinstance(label, dict):
                label = label.get("label", "")
            return str(label).strip()
        return str(item).strip()
    def compute(self, a: Any, b: Any) -> float:
        """Compute semantic similarity between two items [0, 1].
        Prefers embedding cosine similarity,
        falls back to Jaccard bigram similarity.
        Args:
            a: item A (string or dict)
            b: item B (string or dict)
        Returns:
            similarity score [0, 1]
        """
        a_str = self.extract_label(a).lower()
        b_str = self.extract_label(b).lower()
        if not a_str or not b_str:
            return 0.0
        if a_str == b_str:
            return 1.0
        if a_str in b_str or b_str in a_str:
            return 0.8
        # check cache
        key = (a_str, b_str) if a_str < b_str else (b_str, a_str)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        # try embedding similarity
        sim = self._embed_similarity(a_str, b_str)
        if sim is None:
            sim = self._jaccard_similarity(a_str, b_str)
        # write cache
        with self._lock:
            self._cache[key] = sim
            if len(self._cache) > 2000:
                # evict oldest entry
                del self._cache[next(iter(self._cache))]
        return sim
    def _embed_similarity(self, a: str, b: str) -> Optional[float]:
        """Compute cosine similarity via embedding vectors."""
        if self.embed_fn is None:
            return None
        try:
            va = self._get_embedding(a)
            vb = self._get_embedding(b)
            if va is not None and vb is not None:
                dot = float(np.dot(va, vb))
                denom = float(np.linalg.norm(va)) * float(np.linalg.norm(vb))
                if denom > 1e-9:
                    return round(dot / denom, 4)
        except Exception as e:
            logger.debug("Embedding similarity computation failed: %s", e)
        return None
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text (with cache)."""
        if text in self._embed_cache:
            return self._embed_cache[text]
        if self.embed_fn is None:
            return None
        try:
            vec = self.embed_fn([text])[0]
            arr = np.array(vec, dtype=np.float64)
            if arr.size > 0 and np.linalg.norm(arr) > 1e-9:
                self._embed_cache[text] = arr
                return arr
        except Exception:
            pass
        return None
    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """Jaccard bigram similarity (fallback)."""
        def bigrams(s: str) -> set:
            return {s[i : i + 2] for i in range(len(s) - 1)}
        ba = bigrams(a)
        bb = bigrams(b)
        if not ba or not bb:
            return 0.0
        return len(ba & bb) / len(ba | bb)
    def dedup_list(
        self, items: List[Any], threshold: Optional[float] = None
    ) -> List[Any]:
        """Semantic dedup of a list, keeping higher-confidence items.
        Args:
            items: list of items to dedup
            threshold: dedup threshold, defaults to config.sim_merge_threshold
        Returns:
            deduplicated list
        """
        if not items or len(items) <= 1:
            return list(items)
        threshold = threshold or self.config.sim_merge_threshold
        kept: List[Any] = []
        for item in items:
            label = self.extract_label(item)
            if not label:
                kept.append(item)
                continue
            found = False
            for i, existing in enumerate(kept):
                existing_label = self.extract_label(existing)
                if self.compute(label, existing_label) >= threshold:
                    found = True
                    # keep the higher-confidence one
                    item_conf = item.get("confidence", 0) if isinstance(item, dict) else 0
                    kept_conf = existing.get("confidence", 0) if isinstance(existing, dict) else 0
                    if item_conf > kept_conf:
                        kept[i] = item
                    break
            if not found:
                kept.append(item)
        return kept
    def cross_category_dedup(
        self, persona: Dict[str, List[Any]],
        categories: Optional[List[str]] = None,
    ) -> bool:
        """Cross-category dedup: remove duplicate items across different categories.
        Only removes lower-confidence cross-category duplicates.
        Intra-category dedup is handled by dedup_list.
        Args:
            persona: persona dict
            categories: categories to include in dedup
        Returns:
            whether any items were removed
        """
        if categories is None:
            categories = [
                "identity", "traits", "interests", "style_notes",
                "behavior_patterns", "decision_style", "interpersonal",
            ]
        threshold = self.config.sim_merge_threshold
        all_items: List[tuple] = []  # (label, confidence, category, index)
        for cat in categories:
            for idx, item in enumerate(persona.get(cat, [])):
                label = self.extract_label(item)
                if label:
                    conf = item.get("confidence", 0.5) if isinstance(item, dict) else 0.5
                    all_items.append((label, conf, cat, idx))
        removed = False
        clustered = [False] * len(all_items)
        for i in range(len(all_items)):
            if clustered[i]:
                continue
            group = [i]
            for j in range(i + 1, len(all_items)):
                if clustered[j] or all_items[i][2] == all_items[j][2]:
                    continue
                if self.compute(all_items[i][0], all_items[j][0]) >= threshold:
                    group.append(j)
                    clustered[j] = True
            if len(group) < 2:
                continue
            # sort by confidence descending, keep the highest
            group.sort(key=lambda g: all_items[g][1], reverse=True)
            for g in group[1:]:
                _, _, cat, idx = all_items[g]
                cat_list = persona.get(cat, [])
                if idx < len(cat_list):
                    del cat_list[idx]
                    removed = True
                    # adjust subsequent indices
                    for k in range(len(all_items)):
                        if all_items[k][2] == cat and all_items[k][3] > idx:
                            all_items[k] = (
                                all_items[k][0], all_items[k][1],
                                all_items[k][2], all_items[k][3] - 1,
                            )
        return removed
