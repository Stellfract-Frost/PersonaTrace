"""PersonaTrace — merged memory layer.

Primary API:
    from personatrace import MergedMemory

    memory = MergedMemory(
        user_id="alice",
        sqlite_path="./personatrace.db",
    )
    memory.write(
        beliefs=[{"category": "identity", "label": "Name is Mo"}],
        source="direct",
    )
    memory.record_turn("I have class every Tuesday at 19:00", session_id="s1")
    pack = memory.build_context("What is due next week?")

Storage split:
- ``MergedMemory``: facade over beliefs, intentions, episodes and routines.
- ``MemoryV2``: belief kernel (retention, bi-temporal state, evidence).
- ``IntentStore``: short-term commitments.
- ``EpisodeStore``: raw turns and retrospective search.
- ``RoutineStore``: weekly schedules and in-turn probe state.
- ``ChromaEpisodeIndex``: optional semantic episode index.

The runtime v1 pipeline (``PersonaManager`` / extractor / formatter / rhythm)
has been removed. ``VectorStore`` and ``PersonaConfig`` remain only for the
one-time ``scripts/migrate_v1.py`` migration from old Chroma profiles.
"""

from __future__ import annotations

from typing import Any

__version__ = "2.0.1"

# 轻量、无外部依赖的成员（导入本包时不触发 chromadb / requests）
from .config import PersonaConfig

_LAZY = {
    "MemoryV2": ".memory_v2",
    "MemoryBackend": ".memory_v2",
    "SQLiteBackend": ".memory_v2",
    "Belief": ".memory_v2",
    "MergedMemory": ".merged_memory",
    "ContextPack": ".merged_memory",
    "EpisodeStore": ".episodes",
    "Episode": ".episodes",
    "ChromaEpisodeIndex": ".chroma_index",
    "has_retrospective": ".episodes",
    "RoutineStore": ".routines",
    "Routine": ".routines",
    "IntentStore": ".intentions",
    "parse_due": ".schema",
    "rank": ".recall",
    "LLMClient": ".llm_client",
    "extract_json": ".llm_client",
    # Migration-only surface.
    "VectorStore": ".vector_store",
}

__all__ = [
    "MergedMemory",
    "ContextPack",
    "EpisodeStore",
    "Episode",
    "ChromaEpisodeIndex",
    "RoutineStore",
    "Routine",
    "IntentStore",
    "MemoryV2",
    "MemoryBackend",
    "SQLiteBackend",
    "Belief",
    "parse_due",
    "rank",
    # LLM/config helpers, still used by the evaluation scripts.
    "PersonaConfig",
    "LLMClient",
    "extract_json",
    # Migration-only surface.
    "VectorStore",
]


def __getattr__(name: str) -> Any:
    """PEP 562：按需导入重组件，让 `import personatrace` 在没有 chromadb/requests 时也能成功。"""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value  # 缓存，后续访问不再走 __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
