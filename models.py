"""
PersonaTrace data models.
Defines the dual-track profiling taxonomy:
- Work Track: what the user knows and does
- Persona Track: how the user communicates and thinks
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
# ── Dual-track taxonomy ──
# Work Track: what the user knows/does — knowledge, preferences, and facts
WORK_CATEGORIES: List[str] = [
    "likes",
    "dislikes",
    "constraints",     # hard constraints (time, budget, platform, etc.)
    "explicit_facts",  # explicitly stated facts
    "expertise",       # domains / tech stack
    "projects",        # project experience
    "goals",           # goals (work / personal)
]
# Persona Track: how the user communicates/thinks — 5-layer model
PERSONA_CATEGORIES: List[str] = [
    "identity",          # L1: demographics (gender, age, location, etc.)
    "traits",            # L1: personality traits / values
    "interests",         # L1: interests and hobbies
    "style_notes",       # L2: expression style / speech habits
    "behavior_patterns", # L2: behavioral patterns
    "decision_style",    # L3: decision-making tendencies
    "interpersonal",     # L4: interpersonal preferences
    "boundaries",        # L5: boundaries / no-go zones
    "corrections",       # correction records
    "rhythms",           # temporal rhythms
]
# Cross-track insights
CROSS_CATEGORIES: List[str] = [
    "cross_insights",  # deep cross-track insights
    "analysis_log",    # analysis log
]
# Ephemeral / temporary data
EPHEMERAL_CATEGORIES: List[str] = [
    "observation_log",    # observation log
    "uncertain",          # uncertain items
    "low_confidence",     # low-confidence items
    "timeline",           # timeline
]
@dataclass
class PersonaItem:
    """A single profile entry.
    Attributes:
        label: label text
        confidence: confidence score [0, 1]
        reasoning: rationale for this extraction
        hints: supporting hints
        source: data source
        time_note: time annotation (used for rhythm analysis)
        hits: hit count (incremented on repeated extraction)
        updated: last update timestamp (ISO format)
        cluster_id: cluster ID
    """
    label: str = ""
    confidence: float = 0.5
    reasoning: str = ""
    hints: List[str] = field(default_factory=list)
    source: str = ""
    time_note: str = ""
    hits: int = 0
    updated: str = field(default_factory=lambda: datetime.now().isoformat())
    cluster_id: Optional[int] = None
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "hints": self.hints,
            "source": self.source,
            "time_note": self.time_note,
            "hits": self.hits,
            "updated": self.updated,
            "cluster_id": self.cluster_id,
        }
    @classmethod
    def from_dict(cls, data: Any) -> "PersonaItem":
        """Create a PersonaItem from a dict or string."""
        if isinstance(data, str):
            return cls(label=data.strip(), confidence=0.5)
        if isinstance(data, dict):
            label = data.get("label", "")
            # handle nested dict label field
            if isinstance(label, dict):
                label = label.get("label", "")
            return cls(
                label=str(label).strip(),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=str(data.get("reasoning", "")),
                hints=data.get("hints", []),
                source=data.get("source", ""),
                time_note=data.get("time_note", ""),
                hits=data.get("hits", 0),
                updated=data.get("updated", datetime.now().isoformat()),
                cluster_id=data.get("cluster_id"),
            )
        return cls(label=str(data).strip())
@dataclass
class GraphEdge:
    """Graph edge between two profile items."""
    source: str        # source label (lowercase)
    target: str        # target label (lowercase)
    type: str          # similarity / contradiction / supersede
    weight: float = 1.0
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    updated: str = field(default_factory=lambda: datetime.now().isoformat())
@dataclass
class ProfileData:
    """Complete user profile data structure."""
    work: Dict[str, List[Any]] = field(default_factory=dict)
    persona: Dict[str, List[Any]] = field(default_factory=dict)
    cross: Dict[str, List[Any]] = field(default_factory=dict)
    ephemeral: Dict[str, List[Any]] = field(default_factory=dict)
    graph_edges: List[Dict] = field(default_factory=list)
    interaction: Dict[str, Any] = field(default_factory=lambda: {
        "prefer_few_questions": False,
        "impatience_count": 0,
    })
    pending_advisor: Optional[Dict] = None
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 0
    profile_completeness: Dict[str, float] = field(default_factory=lambda: {
        "work": 0.0,
        "persona": 0.0,
    })
    @classmethod
    def default(cls) -> "ProfileData":
        """Create an empty default profile."""
        return cls(
            work={cat: [] for cat in WORK_CATEGORIES},
            persona={cat: [] for cat in PERSONA_CATEGORIES},
            cross={cat: [] for cat in CROSS_CATEGORIES},
            ephemeral={cat: [] for cat in EPHEMERAL_CATEGORIES},
        )
