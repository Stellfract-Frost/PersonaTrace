"""
Profile formatting module.
Supports query-based semantic filtering with 15+3 retrieval reranking.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from .config import PersonaConfig
from .models import WORK_CATEGORIES, PERSONA_CATEGORIES
from .similarity import SimilarityCalculator
logger = logging.getLogger(__name__)
class PersonaFormatter:
    def __init__(self, config: PersonaConfig, similarity: SimilarityCalculator):
        self.config = config
        self.similarity = similarity
    def format(
        self,
        data: Dict[str, Any],
        query: Optional[str] = None,
        max_items_per_category: int = 8,
        enable_rerank: bool = False
    ) -> str:
        """Format profile data as readable text. If query and enable_rerank are given, applies 15+3 semantic retrieval."""
        today = datetime.now()
        weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        lines = [f"Today is {today.strftime('%Y-%m-%d')} ({weekday_labels[today.weekday()]})"]
        work = data.get("work", {})
        persona = data.get("persona", {})
        cross = data.get("cross", {})
        ephemeral = data.get("ephemeral", {})
        # 15+3 semantic rerank
        filter_shown = None
        if query and enable_rerank:
            filter_shown = self._llm_filter_batch(query, work, persona, ephemeral, cross)
        work_lines = self._format_work_track(work, max_items_per_category, filter_shown)
        if work_lines:
            lines.append("\n### Work Profile")
            lines.extend(work_lines)
        persona_lines = self._format_persona_track(persona, max_items_per_category, filter_shown)
        if persona_lines:
            lines.append("\n### Communication Profile")
            lines.extend(persona_lines)
        if not work_lines and not persona_lines:
            lines.append("\nNo known preferences yet (new user, profile being built)")
        return "\n".join(lines)
    def _llm_filter_batch(self, query: str, work: Dict, persona: Dict, ephemeral: Dict, cross: Dict) -> Dict[str, List]:
        """15+3 retrieval: return the top-15 items most relevant to the query."""
        all_items = []
        # collect all items and score them
        for cat in WORK_CATEGORIES:
            for item in work.get(cat, []):
                lbl = self.similarity.extract_label(item)
                if lbl: all_items.append({"cat": cat, "label": lbl, "item": item})
        for cat in PERSONA_CATEGORIES:
            for item in persona.get(cat, []):
                lbl = self.similarity.extract_label(item)
                if lbl: all_items.append({"cat": cat, "label": lbl, "item": item})
        # compute similarity and take top 15
        scored = []
        for item in all_items:
            score = self.similarity.compute(query, item["label"])
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_15 = scored[:15]
        # aggregate results by category
        shown = {}
        for _, item in top_15:
            if item["cat"] not in shown: shown[item["cat"]] = []
            shown[item["cat"]].append(item["item"])
        return shown
    def _format_work_track(self, work: Dict[str, List], max_items: int, filter_shown: Optional[Dict] = None) -> List[str]:
        lines = []
        def get_items(cat: str, limit: int):
            if filter_shown and cat in filter_shown: return filter_shown[cat][:limit]
            return work.get(cat, [])[:limit]
        id_items = get_items("expertise", max_items)
        if id_items:
            lines.append("**Expertise**")
            for item in id_items:
                label = self.similarity.extract_label(item)
                conf = self._get_conf(item)
                lines.append(f"  - {label} ({self._conf_tag(conf)})")
        pg = get_items("projects", 2) + get_items("goals", 2)
        if pg:
            lines.append("**Projects / Goals**")
            for item in pg[:4]:
                label = self.similarity.extract_label(item)
                conf = self._get_conf(item)
                lines.append(f"  - {label} ({self._conf_tag(conf)})")
        for key, label_text in [("likes", "Likes"), ("dislikes", "Dislikes")]:
            items = get_items(key, 8)
            if items:
                labels = [self.similarity.extract_label(i) for i in items]
                lines.append(f"  {label_text}: {', '.join(labels)}")
        constr = get_items("constraints", 4)
        if constr:
            lines.append("**Constraints**")
            for item in constr:
                lines.append(f"  - {self.similarity.extract_label(item)}")
        facts = get_items("explicit_facts", 4)
        if facts:
            lines.append("**Facts**")
            for item in facts:
                lines.append(f"  - {self.similarity.extract_label(item)}")
        return lines
    def _format_persona_track(self, persona: Dict[str, List], max_items: int, filter_shown: Optional[Dict] = None) -> List[str]:
        lines = []
        def get_items(cat: str, limit: int):
            if filter_shown and cat in filter_shown: return filter_shown[cat][:limit]
            return persona.get(cat, [])[:limit]
        l1_items = []
        for cat, max_n in [("identity", max_items), ("traits", 3), ("interests", 3)]:
            for item in get_items(cat, max_n):
                label = self.similarity.extract_label(item)
                conf = self._get_conf(item)
                l1_items.append(f"  - {label} ({self._conf_tag(conf)})")
        if l1_items:
            lines.append("**L1 Identity & Personality**")
            lines.extend(l1_items)
        ds = get_items("decision_style", 3)
        if ds:
            lines.append("**L3 Decision Style**")
            for item in ds: lines.append(f"  - {self.similarity.extract_label(item)}")
        l2_items = []
        for cat, max_n in [("style_notes", 3), ("behavior_patterns", 3)]:
            for item in get_items(cat, max_n):
                label = self.similarity.extract_label(item)
                conf = self._get_conf(item)
                l2_items.append(f"  - {label} ({self._conf_tag(conf)})")
        if l2_items:
            lines.append("**L2 Communication & Behavior Style**")
            lines.extend(l2_items)
        ip = get_items("interpersonal", 3)
        if ip:
            lines.append("**L4 Interpersonal**")
            for item in ip: lines.append(f"  - {self.similarity.extract_label(item)}")
        bd = get_items("boundaries", 3)
        if bd:
            lines.append("**L5 Boundaries & Safety Rules**")
            for item in bd: lines.append(f"  - {self.similarity.extract_label(item)}")
        rhythms = persona.get("rhythms", [])[:5]
        if rhythms:
            lines.append("**Rhythms**")
            for r in rhythms:
                lines.append(f"  - {r.get('label', '')} — {r.get('time_note', '')} (x{r.get('occurrences', 1)})")
        corrections = persona.get("corrections", [])[:3]
        if corrections:
            lines.append("**Corrections**")
            for c in corrections:
                lines.append(f"  - {c.get('label', '')} → {c.get('correction', '')}")
        return lines
    @staticmethod
    def _get_conf(item: Any) -> float:
        return float(item.get("confidence", 0.5)) if isinstance(item, dict) else 0.5
    @staticmethod
    def _conf_tag(c: float) -> str:
        if c >= 0.85: return "confirmed"
        if c >= 0.70: return "likely"
        if c >= 0.55: return "possible"
        if c >= 0.40: return "uncertain"
        return "speculative"
