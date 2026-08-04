"""
Profile extractor module.
Implements the Detective V2 three-stage pipeline:
LLM extraction -> vector search for similar history -> LLM dedup judgment (merge/contradict/deprecate).
"""
import re
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from .config import PersonaConfig
from .llm_client import LLMClient
from .models import WORK_CATEGORIES, PERSONA_CATEGORIES
logger = logging.getLogger(__name__)
class PersonaExtractor:
    def __init__(self, config: PersonaConfig, llm: LLMClient):
        self.config = config
        self.llm = llm
    def extract_from_conversation(
        self, user_msg: str, ai_reply: str,
        history_context: str = "", now: Optional[datetime] = None,
        vector_store=None, user_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """V2 detective pipeline extraction."""
        now = now or datetime.now()
        time_period = self._get_time_period(now.hour)
        # Stage 1: extract new labels from the conversation
        stage1_prompt = self._build_extraction_prompt(user_msg, ai_reply, history_context, now, time_period)
        stage1_result = self.llm.chat_json([{"role": "user", "content": stage1_prompt}], max_tokens=16384)
        if not stage1_result: return None
        # Stage 2 & 3: if vector store is available, search history and LLM dedup
        if vector_store:
            similar_map = self._detective_search(stage1_result, vector_store, user_id)
            if similar_map:
                stage3_result = self._llm_dedup(stage1_result, similar_map, now)
                if stage3_result:
                    # merge stage3 judgments into stage1
                    if "confidence_adjust" in stage3_result:
                        stage1_result["confidence_adjust"] = stage3_result["confidence_adjust"]
                    if "contradictions" in stage3_result:
                        stage1_result["contradictions"] = stage3_result["contradictions"]
                    if "corrections" in stage3_result:
                        stage1_result["corrections"] = stage3_result["corrections"]
                    if "_v2_deprecate_labels" in stage3_result:
                        stage1_result["_v2_deprecate_labels"] = stage3_result["_v2_deprecate_labels"]
        return stage1_result
    def _detective_search(self, stage1_result: Dict, vector_store, user_id: str) -> Dict:
        """Stage 2: vector search for similar historical labels."""
        similar_map = {}
        new_labels = []
        wu = stage1_result.get("work_updates", {}) or {}
        for cat in ["likes", "dislikes", "constraints", "explicit_facts", "expertise", "goals", "projects"]:
            for item in wu.get(cat, []):
                lbl = item.get("label", "") if isinstance(item, dict) else str(item)
                if lbl and len(lbl) >= 2: new_labels.append(lbl)
        pu = stage1_result.get("persona_updates", {}) or {}
        for cat in ["identity", "traits", "interests", "style_notes", "decision_style", "interpersonal", "boundaries"]:
            for item in pu.get(cat, []):
                lbl = item.get("label", "") if isinstance(item, dict) else str(item)
                if lbl and len(lbl) >= 2: new_labels.append(lbl)
        for lbl in new_labels:
            try:
                results = vector_store.search_similar(user_id, lbl, top_k=3)
                filtered = [(s, p) for s, p in results if s >= self.config.sim_threshold]
                if filtered: similar_map[lbl] = filtered
            except Exception:
                pass
        return similar_map
    def _llm_dedup(self, stage1_result: Dict, similar_map: Dict, now: datetime) -> Optional[Dict]:
        """Stage 3: LLM judgment on new-vs-existing label relationships."""
        pairs = []
        for new_lbl, matches in similar_map.items():
            old_info = []
            for score, payload in matches:
                old_info.append({
                    "label": payload.get("label", ""),
                    "confidence": payload.get("confidence", 0.5),
                    "score": round(score, 3)
                })
            pairs.append({"new_label": new_lbl, "similar_existing": old_info})
        prompt = f"""You are a dedup judgment expert. Given newly extracted labels and similar existing labels, determine the relationship between them.
## Label Pairs
{json.dumps(pairs, ensure_ascii=False, indent=2)}
## Judgment Rules
1. Semantically identical or nearly identical → action: "merge"
2. Semantically opposite → action: "contradict"
3. New label corrects an old label → action: "correction"
4. Different concepts → action: "skip"
Output JSON:
{{
    "dedup_actions": [{{"new_label":"","old_label":"","action":"merge|contradict|correction|skip","new_confidence":0.0,"reason":""}}],
    "deprecate_labels": ["labels to deprecate"]
}}"""
        try:
            result = self.llm.chat_json([{"role": "user", "content": prompt}], max_tokens=16384)
            if not result: return None
            # Convert format
            confidence_adjust, contradictions, corrections = [], [], []
            for action in result.get("dedup_actions", []):
                act = action.get("action", "")
                if act == "merge":
                    confidence_adjust.append({"label": action["old_label"], "new_confidence": min(0.99, action.get("new_confidence", 0.5) + 0.05), "reason": action.get("reason", "")})
                elif act == "contradict":
                    contradictions.append({"item_a": action["new_label"], "item_b": action["old_label"], "resolution": action.get("reason", "")})
                elif act == "correction":
                    corrections.append({"label": action["old_label"], "correction": action["new_label"], "reason": action.get("reason", "")})
            return {
                "confidence_adjust": confidence_adjust,
                "contradictions": contradictions,
                "corrections": corrections,
                "_v2_deprecate_labels": result.get("deprecate_labels", [])
            }
        except Exception as e:
            logger.error("LLM dedup judgment failed: %s", e)
            return None
    def extract_from_text(self, text: str, source_label: str = "Text import") -> Optional[Dict[str, Any]]:
        """Extract profile from a long-form text."""
        text = text.strip()
        if not text: return {}
        segments = self._split_text(text, chunk_size=12000, overlap=500)
        all_results = []
        for idx, seg in enumerate(segments, 1):
            seg_label = f"{source_label} part {idx}/{len(segments)}"
            try:
                prompt = self._build_batch_prompt(seg, seg_label)
                result = self.llm.chat_json([{"role": "user", "content": prompt}], max_tokens=8192)
                all_results.append(result or {"uncertain": [{"label": f"No info extracted from segment: {seg_label}", "confidence": 0.2}]})
            except Exception as e:
                all_results.append({"uncertain": [{"label": f"Segment analysis failed: {seg_label}", "confidence": 0.2, "hint": str(e)[:300]}]})
        return self._merge_results(all_results)
    def _build_batch_prompt(self, text: str, source_label: str) -> str:
        return f"""You are a user profile analysis assistant. Extract all profile information about the user from the source material.
Source: {source_label}
Content:
{text}
Extraction dimensions: expertise, projects, goals, constraints, explicit_facts, likes, dislikes (Work)
identity, traits, interests, style_notes, decision_style, interpersonal, boundaries (Persona)
Confidence: explicit statement 0.85, implied 0.70, vague 0.50.
Output JSON:
{{
    "work_updates": {{"expertise":[{{"label":"","confidence":0.0,"reasoning":""}}],"goals":[],"projects":[],"likes":[],"dislikes":[],"constraints":[],"explicit_facts":[]}},
    "persona_updates": {{"identity":[],"traits":[],"interests":[],"style_notes":[],"decision_style":[],"interpersonal":[],"boundaries":[]}},
    "uncertain": [{{"label":"","confidence":0.0,"hint":""}}]
}}"""
    def _build_extraction_prompt(self, user_msg: str, ai_reply: str, history_context: str, now: datetime, time_period: str) -> str:
        return f"""You are a user profile extraction assistant. Extract new information about the user from the current conversation turn.
Current time: {now.strftime("%Y-%m-%d %H:%M:%S")} ({time_period})
## Current Turn
{history_context}
User: {user_msg}
Assistant: {ai_reply[:400]}
Output JSON:
{{
    "work_updates": {{"likes":[{{"label":"","evidence":""}}],"dislikes":[],"constraints":[],"explicit_facts":[],"expertise":[{{"label":"","confidence":0.0,"reasoning":""}}],"goals":[],"projects":[]}},
    "persona_updates": {{"identity":[],"traits":[],"interests":[],"style_notes":[],"decision_style":[],"interpersonal":[],"boundaries":[]}},
    "uncertain": [{{"label":"","confidence":0.0,"hint":""}}]
}}"""
    @staticmethod
    def _split_text(text: str, chunk_size: int = 12000, overlap: int = 500) -> List[str]:
        text = text.strip()
        if not text: return []
        chunks, buf = [], ""
        def flush():
            nonlocal buf
            if buf.strip(): chunks.append(buf.strip()); buf = ""
        for part in re.split(r"\n\s*\n|\n", text):
            part = part.strip()
            if not part: continue
            if len(part) > chunk_size:
                flush()
                start = 0
                while start < len(part):
                    end = min(start + chunk_size, len(part))
                    chunks.append(part[start:end])
                    if end >= len(part): break
                    start = max(0, end - overlap)
                continue
            if len(buf) + len(part) + 1 > chunk_size: flush()
            buf = part if not buf else buf + "\n" + part
        flush()
        return chunks
    @staticmethod
    def _merge_results(results: List[Dict]) -> Dict:
        if not results: return {}
        if len(results) == 1: return results[0]
        merged = {}
        for key in ("work_updates", "persona_updates", "uncertain"):
            if key == "uncertain":
                merged[key] = []
                for r in results: merged[key].extend(r.get("uncertain", []))
            else:
                merged[key] = {}
                cats = WORK_CATEGORIES if key == "work_updates" else PERSONA_CATEGORIES
                for cat in cats:
                    all_items = []
                    for r in results: all_items.extend(r.get(key, {}).get(cat, []))
                    merged[key][cat] = all_items
        return merged
    @staticmethod
    def _get_time_period(hour: int) -> str:
        if 5 <= hour < 11: return "morning"
        if 11 <= hour < 13: return "midday"
        if 13 <= hour < 18: return "afternoon"
        if 18 <= hour < 23: return "evening"
        return "late night"
