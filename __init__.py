"""
PersonaTrace — User profile extraction and management toolkit.
A modular user profiling system featuring:
- Dual-track profile taxonomy (Work + Persona)
- LLM-driven profile extraction
- Semantic dedup and merging
- Confidence decay
- Temporal rhythm detection
- Vector search
Quick start:
    from persona import PersonaManager, PersonaConfig
    config = PersonaConfig(
        llm_api_key="your-api-key",
        llm_base_url="https://api.openai.com/v1",
        llm_model="gpt-4o",
    )
    manager = PersonaManager(config)
    result = manager.ingest_text("I'm a Python developer who enjoys running and reading")
    profile_text = manager.format_profile()
    print(profile_text)
"""
from .config import PersonaConfig
from .models import (
    PersonaItem, ProfileData, GraphEdge,
    WORK_CATEGORIES, PERSONA_CATEGORIES,
    CROSS_CATEGORIES, EPHEMERAL_CATEGORIES,
)
from .vector_store import VectorStore
from .llm_client import LLMClient, extract_json
from .similarity import SimilarityCalculator
from .rhythm import RhythmEngine
from .extractor import PersonaExtractor
from .formatter import PersonaFormatter
from .persona_manager import PersonaManager
__version__ = "1.0.0"
__all__ = [
    "PersonaConfig",
    "PersonaManager",
    "PersonaItem",
    "ProfileData",
    "GraphEdge",
    "VectorStore",
    "LLMClient",
    "SimilarityCalculator",
    "RhythmEngine",
    "PersonaExtractor",
    "PersonaFormatter",
    "extract_json",
    "WORK_CATEGORIES",
    "PERSONA_CATEGORIES",
    "CROSS_CATEGORIES",
    "EPHEMERAL_CATEGORIES",
]
