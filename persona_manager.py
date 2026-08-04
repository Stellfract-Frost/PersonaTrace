"""
Core profile management module.
Integrates storage, extraction, similarity computation, rhythm engine,
Mem0 memory, and file parsing.
"""
import os
import json
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from .config import PersonaConfig
from .models import PersonaItem, ProfileData, WORK_CATEGORIES, PERSONA_CATEGORIES, CROSS_CATEGORIES, EPHEMERAL_CATEGORIES
from .vector_store import VectorStore
from .similarity import SimilarityCalculator
from .rhythm import RhythmEngine
from .extractor import PersonaExtractor
from .formatter import PersonaFormatter
from .llm_client import LLMClient
from .file_loader import FileLoader
logger = logging.getLogger(__name__)
class PersonaManager:
    def __init__(self, config: Optional[PersonaConfig] = None, llm: Optional[LLMClient] = None, embed_fn=None):
        self.config = config or PersonaConfig.from_env()
        self.llm = llm or LLMClient(self.config)
        self.embed_fn = embed_fn
        self.store = VectorStore(self.config)
        self.similarity = SimilarityCalculator(self.config)
        self.rhythm = RhythmEngine(sim_threshold=self.config.sim_threshold)
        self.extractor = PersonaExtractor(self.config, self.llm)
        self.formatter = PersonaFormatter(self.config, self.similarity)
        self.file_loader = FileLoader()
        if embed_fn:
            self.store.set_embed_fn(embed_fn)
            self.similarity.set_embed_fn(embed_fn)
        self._lock = threading.Lock()
        self._hourly_density: Dict = {}
        self.memory = None
        if self.config.enable_mem0:
            self._init_mem0()
    def _init_mem0(self):
        """Initialize Mem0 conversation memory."""
        try:
            from mem0 import Memory
            cfg = {
                "llm": {"provider": "openai", "config": {
                    "model": self.config.llm_model, "api_key": self.config.llm_api_key,
                    "openai_base_url": self.config.llm_base_url + "/v1"
                }},
                "embedder": {"provider": "huggingface", "config": {"model": self.config.embed_model}},
                "vector_store": {"provider": "qdrant", "config": {
                    "path": self.config.mem0_store_path, "embedding_model_dims": self.config.embed_dims
                }}
            }
            self.memory = Memory.from_config(cfg)
            logger.info("Mem0 conversation memory initialized successfully")
        except Exception as e:
            logger.error("Mem0 initialization failed: %s", e)
            self.memory = None
    # ── Data load & save ──
    def load(self, user_id: str = "default") -> Dict[str, Any]:
        with self._lock:
            data = self.store.load_all(user_id)
        if data is None:
            data = ProfileData.default().__dict__
        self._decay_all(data)
        self.similarity.cross_category_dedup(data.get("persona", {}))
        return data
    def save(self, data: Dict[str, Any], user_id: str = "default") -> None:
        self.rhythm.process(data, similarity_fn=self.similarity.compute, hourly_density=self._hourly_density)
        with self._lock:
            data["version"] = data.get("version", 0) + 1
            data["last_updated"] = datetime.now().isoformat()
            for track, cats in {"work": WORK_CATEGORIES, "persona": PERSONA_CATEGORIES, "cross": CROSS_CATEGORIES, "ephemeral": EPHEMERAL_CATEGORIES}.items():
                for cat in cats:
                    items = data.get(track, {}).get(cat, [])
                    self.store.delete_category(user_id, cat)
                    if items: self.store.upsert_items(user_id, cat, items)
            # Cross-cutting state is stored as non-searchable meta points, not per-category items
            meta = {
                "_version": {"version": data["version"]},
                "_last_updated": data["last_updated"],
                "_graph_edges": data.get("graph_edges", []),
                "_interaction": data.get("interaction", {"prefer_few_questions": False}),
                "_pending_advisor": data.get("pending_advisor"),
                "_profile_completeness": data.get("profile_completeness", {"work": 0.0, "persona": 0.0}),
            }
            for key, value in meta.items():
                self.store.upsert_meta(user_id, key, value)
    # ── Mem0 conversation memory interface ──
    def add_memory(self, messages: List[Dict[str, str]], user_id: str = "default") -> None:
        """Add conversation to Mem0 memory."""
        if self.memory:
            try: self.memory.add(messages, user_id=user_id)
            except Exception as e: logger.error("Mem0 add failed: %s", e)
    def search_memory(self, query: str, user_id: str = "default", limit: int = 5) -> List[str]:
        """Search Mem0 for relevant conversation history."""
        if not self.memory: return []
        try:
            result = self.memory.search(query, filters={"user_id": user_id}, limit=limit)
            return [m.get('memory', '') for m in result.get("results", []) if isinstance(m, dict)]
        except Exception as e:
            logger.error("Mem0 search failed: %s", e)
            return []
    # ── Profile extraction & merging ──
    def ingest_file(self, file_path: str, user_id: str = "default") -> Dict[str, Any]:
        """Extract profile from a multi-format file."""
        try:
            text, parser_name = self.file_loader.extract(file_path)
            if not text.strip(): return {"status": "empty_file"}
            return self.ingest_text(text, source_label=f"File: {os.path.basename(file_path)} ({parser_name})", user_id=user_id)
        except Exception as e:
            return {"status": "error", "message": str(e)}
    def ingest_text(self, text: str, source_label: str = "Text import", user_id: str = "default") -> Dict[str, Any]:
        result = self.extractor.extract_from_text(text, source_label)
        if not result: return {"status": "no_result"}
        data = self.load(user_id)
        changed = self._apply_extraction_result(data, result, source_label)
        if changed:
            self._recompute_completeness(data)
            self.save(data, user_id)
        return {"status": "success" if changed else "no_change"}
    def ingest_conversation(self, user_msg: str, ai_reply: str, history_context: str = "", user_id: str = "default") -> Dict[str, Any]:
        """V2 detective pipeline: extract profile from conversation with three-stage dedup."""
        now = datetime.now()
        result = self.extractor.extract_from_conversation(
            user_msg, ai_reply, history_context, now,
            vector_store=self.store, user_id=user_id
        )
        if not result: return {"status": "no_result"}
        data = self.load(user_id)
        changed = self._apply_extraction_result(data, result, "Conversation extract")
        # Handle V2 special actions: confidence adjustment, contradiction downgrade, label deprecation, corrections
        if "confidence_adjust" in result:
            self._apply_confidence_adjust(data, result["confidence_adjust"])
            changed = True
        if "_v2_deprecate_labels" in result:
            self._apply_deprecate(data, result["_v2_deprecate_labels"])
            changed = True
        if "corrections" in result:
            self._apply_corrections(data, result["corrections"])
            changed = True
        if "contradictions" in result:
            self._apply_contradictions(data, result["contradictions"])
            changed = True
        # Record an observation for rhythm analysis; forces a save so it persists
        if self._record_observation(data, result, user_msg, now):
            changed = True
        if changed:
            self._recompute_completeness(data)
            self.save(data, user_id)
        return {"status": "success" if changed else "no_change"}
    def _apply_extraction_result(self, data: Dict, result: Dict, source_label: str = "") -> bool:
        changed = False
        ts_now = datetime.now().isoformat()
        wu = result.get("work_updates", {})
        for cat in ["expertise", "goals", "projects"]:
            if wu.get(cat) and self._merge_items(data["work"][cat], wu[cat], ts_now): changed = True
        for cat in ["likes", "dislikes", "constraints", "explicit_facts"]:
            items = wu.get(cat, [])
            if items:
                if isinstance(items[0], dict) and self._merge_items(data["work"][cat], items, ts_now): changed = True
                else:
                    norm = [str(i) for i in items if i]
                    if norm: data["work"][cat] = self._dedup_preferences(data["work"].get(cat, []), norm); changed = True
        pu = result.get("persona_updates", {})
        for cat in PERSONA_CATEGORIES:
            if cat in ("corrections", "rhythms"): continue
            if pu.get(cat) and self._merge_items(data["persona"][cat], pu[cat], ts_now): changed = True
        return changed
    def _apply_confidence_adjust(self, data: Dict, adjustments: List[Dict]):
        """Apply confidence adjustments to matching items."""
        for adj in adjustments:
            label = adj.get("label", "")
            new_c = min(1.0, max(0.0, float(adj.get("new_confidence", 0) or 0)))
            if not label or new_c <= 0: continue
            for container in [data["work"], data["persona"]]:
                for cat, items in container.items():
                    for item in items:
                        if self.similarity.compute(self.similarity.extract_label(item), label) >= self.config.sim_threshold:
                            if isinstance(item, dict):
                                item["confidence"] = new_c
                                item["updated"] = datetime.now().isoformat()
    def _apply_deprecate(self, data: Dict, labels: List[str]):
        """Remove items matching deprecated labels."""
        for label in labels:
            for container in [data["work"], data["persona"]]:
                for cat, items in container.items():
                    container[cat] = [
                        item for item in items
                        if self.similarity.compute(self.similarity.extract_label(item), label) < self.config.sim_threshold
                    ]
    def _apply_corrections(self, data: Dict, corrections: List[Dict]) -> None:
        """Apply correction records: relabel matching old items to the corrected label and log the correction.
        If the corrected label already exists, merge the old items into it instead of creating a duplicate.
        """
        now = datetime.now().isoformat()
        containers = (data.get("work", {}), data.get("persona", {}))
        for c in corrections:
            old = c.get("label", "")
            new = c.get("correction", "")
            if not old or not new:
                continue
            # find an item already carrying the corrected label (string match, not semantic)
            new_target = None
            new_l = new.lower()
            for container in containers:
                for cat, items in container.items():
                    if cat in ("corrections", "rhythms"):
                        continue
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        l = self.similarity.extract_label(item).lower()
                        if l == new_l or (len(l) >= 2 and len(new_l) >= 2 and (l in new_l or new_l in l)):
                            new_target = item
                            break
                    if new_target:
                        break
                if new_target:
                    break
            for container in containers:
                for cat, items in container.items():
                    if cat in ("corrections", "rhythms"):
                        continue
                    keep = []
                    for item in items:
                        if isinstance(item, dict) and self.similarity.compute(self.similarity.extract_label(item), old) >= self.config.sim_threshold:
                            if item is new_target:
                                keep.append(item)
                            elif new_target is not None:
                                new_target["hits"] = new_target.get("hits", 0) + 1
                                new_target["confidence"] = min(0.99, max(item.get("confidence", 0.5), new_target.get("confidence", 0.5)) + 0.025)
                                new_target["updated"] = now
                            else:
                                item["label"] = new
                                item["confidence"] = max(0.20, item.get("confidence", 0.5) - 0.15)
                                item["updated"] = now
                                keep.append(item)
                        else:
                            keep.append(item)
                    container[cat] = keep
            data.get("persona", {}).setdefault("corrections", []).append(
                {"label": old, "correction": new, "reason": c.get("reason", ""), "updated": now}
            )
    def _apply_contradictions(self, data: Dict, contradictions: List[Dict]) -> None:
        """Apply contradiction records: downgrade both sides and log the contradiction."""
        now = datetime.now().isoformat()
        log = data.get("cross", {}).setdefault("analysis_log", [])
        for c in contradictions:
            a = c.get("item_a", "")
            b = c.get("item_b", "")
            for lbl in (a, b):
                if not lbl:
                    continue
                for container in (data.get("work", {}), data.get("persona", {})):
                    for cat, items in container.items():
                        for item in items:
                            if isinstance(item, dict) and self.similarity.compute(self.similarity.extract_label(item), lbl) >= self.config.sim_threshold:
                                item["confidence"] = max(0.20, item.get("confidence", 0.5) - 0.15)
                                item["updated"] = now
            log.append({"label": a or b, "item_a": a, "item_b": b, "resolution": c.get("resolution", ""), "ts": now})
            if len(log) > self.config.max_analysis_log:
                del log[:len(log) - self.config.max_analysis_log]
    def _record_observation(self, data: Dict, result: Dict, user_msg: str, now: datetime) -> bool:
        """Record a user-activity observation and hourly density stats for rhythm analysis."""
        label = ""
        wu = result.get("work_updates", {}) or {}
        pu = result.get("persona_updates", {}) or {}
        for cat in ("expertise", "goals", "projects", "likes", "dislikes", "constraints", "explicit_facts"):
            for it in wu.get(cat, []) or []:
                lbl = it.get("label", "") if isinstance(it, dict) else str(it)
                if lbl and len(lbl) >= 2:
                    label = lbl
                    break
            if label:
                break
        if not label:
            for cat in ("traits", "interests", "identity", "style_notes", "decision_style", "interpersonal", "boundaries"):
                for it in pu.get(cat, []) or []:
                    lbl = it.get("label", "") if isinstance(it, dict) else str(it)
                    if lbl and len(lbl) >= 2:
                        label = lbl
                        break
                if label:
                    break
        if not label:
            label = " ".join(user_msg.split())[:40]
        label = label.strip()
        if not label:
            return False
        obs = data.get("ephemeral", {}).setdefault("observation_log", [])
        obs.append({"ts": now.isoformat(), "hour": now.hour, "label": label, "confidence": 0.5})
        # cap growth beyond what the 60-day prune in rhythm.process keeps around
        if len(obs) > 200:
            del obs[:len(obs) - 200]
        key = (now.weekday(), now.hour)
        d = self._hourly_density.setdefault(key, {"count": 0, "total_len": 0, "ask_count": 0})
        d["count"] += 1
        d["total_len"] += len(user_msg)
        d["ask_count"] += 1 if (user_msg.rstrip().endswith("?") or "?" in user_msg) else 0
        return True
    def _merge_items(self, existing: List, new_items: List, ts_now: str, default_conf: float = 0.5) -> bool:
        changed = False
        for raw in new_items:
            item = PersonaItem.from_dict(raw).to_dict() if not isinstance(raw, dict) else raw
            label = self.similarity.extract_label(item)
            if not label: continue
            best_score, best_idx = 0.0, -1
            for i, ex in enumerate(existing):
                score = self.similarity.compute(label, self.similarity.extract_label(ex))
                if score > best_score: best_score, best_idx = score, i
            if best_idx >= 0 and best_score >= self.config.sim_merge_threshold:
                ex = existing[best_idx]
                if not isinstance(ex, dict): ex = {"label": str(ex)}; existing[best_idx] = ex
                ex["hits"] = ex.get("hits", 0) + 1
                ex["confidence"] = min(0.99, max(item.get("confidence", default_conf), ex.get("confidence", 0)) + 0.025)
                ex["updated"] = ts_now
            else:
                existing.append(item)
            changed = True
        return changed
    def _dedup_preferences(self, existing: List, new_items: List) -> List:
        result = list(existing)
        for n in new_items:
            if not n or len(n) < 2: continue
            merged = False
            for idx, ex in enumerate(result):
                ex_label = self.similarity.extract_label(ex)
                if not ex_label: continue
                if n.lower() in ex_label.lower() or ex_label.lower() in n.lower():
                    result[idx] = n if len(n) >= len(ex_label) else ex_label
                    merged = True
                    break
            if not merged: result.append(n)
        return result
    # ── Utility methods ──
    def _decay_all(self, data: Dict) -> None:
        now = datetime.now()
        ts_now = now.isoformat()
        last_up = data.get("last_updated", "")
        # decay only once per day: if the profile was saved today, any due decay is already applied
        if last_up:
            try:
                if datetime.fromisoformat(last_up).date() == now.date(): return
            except: pass
        rates = self.config.decay_rates
        def process(container: List, is_work: bool = False) -> List:
            kept = []
            for item in container:
                if not isinstance(item, dict): kept.append(item); continue
                try: days = max(0, (now - datetime.fromisoformat(item.get("updated", ts_now))).days)
                except: days = 0
                if days < 1: kept.append(item); continue
                conf = item.get("confidence", 0.3)
                rate = (0.03 if conf < 0.40 else 0.01) if is_work else rates.get("mid", 0.03)
                new_conf = round(conf - rate * days, 2)
                if new_conf >= 0.20:
                    item["confidence"] = new_conf
                    item["updated"] = ts_now
                    kept.append(item)
            return kept
        for cat in ["identity", "traits", "interests", "style_notes", "behavior_patterns", "decision_style", "interpersonal"]:
            data["persona"][cat] = process(data["persona"].get(cat, []))
        for cat in ["expertise", "goals", "projects"]:
            data["work"][cat] = process(data["work"].get(cat, []), is_work=True)
    def _recompute_completeness(self, data: Dict) -> None:
        work_non_empty = sum(1 for cat in WORK_CATEGORIES if data.get("work", {}).get(cat))
        persona_cats = [c for c in PERSONA_CATEGORIES if c != "corrections"]
        persona_non_empty = sum(1 for cat in persona_cats if data.get("persona", {}).get(cat))
        data["profile_completeness"] = {
            "work": round(work_non_empty / max(1, len(WORK_CATEGORIES)), 2),
            "persona": round(persona_non_empty / max(1, len(persona_cats)), 2),
        }
    def format_profile(
        self, data: Optional[Dict] = None, query: Optional[str] = None,
        enable_rerank: bool = False, user_id: str = "default"
    ) -> str:
        """Format profile to readable text. If query is given and enable_rerank=True, applies 15+3 semantic reranking."""
        if data is None: data = self.load(user_id)
        return self.formatter.format(data, query=query, enable_rerank=enable_rerank)
