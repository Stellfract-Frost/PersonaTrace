"""
PersonaTrace — User profile extraction and management toolkit.

分两层，按需加载：
- 零依赖内核：`personatrace.memory_v2`（时序信条 / 艾宾浩斯保持率 / 双时态）
- 重组件：VectorStore（chromadb）、PersonaManager（LLM 抽取）等，只有用到才 import

Quick start（零依赖内核）：
    from personatrace.memory_v2 import MemoryV2
    mem = MemoryV2(user_id="qq:12345")
    mem.observe("goals", "交付 V1.0")

Quick start（完整管线，需要 [llm]/[vector] extras）：
    from personatrace import PersonaManager, PersonaConfig
    config = PersonaConfig(llm_api_key="sk-...", llm_model="gpt-4o")
    manager = PersonaManager(config)
    manager.ingest_text("I'm a Python developer who enjoys running and reading")
"""

from __future__ import annotations

from typing import Any

__version__ = "1.5.0"

# 轻量、无外部依赖的成员（导入本包时不触发 chromadb / requests）
from .config import PersonaConfig
from .models import (
    CROSS_CATEGORIES,
    EPHEMERAL_CATEGORIES,
    PERSONA_CATEGORIES,
    WORK_CATEGORIES,
    GraphEdge,
    PersonaItem,
    ProfileData,
)

_LAZY = {
    "MemoryV2": ".memory_v2",
    "MemoryBackend": ".memory_v2",
    "SQLiteBackend": ".memory_v2",
    "Belief": ".memory_v2",
    "MergedMemory": ".merged_memory",
    "ContextPack": ".merged_memory",
    "IntentStore": ".intentions",
    "IntentExtractor": ".intent_extractor",
    "GraphStore": ".graph_memory",
    "parse_due": ".schema",
    "rank": ".recall",
    "LLMClient": ".llm_client",
    "extract_json": ".llm_client",
    "SimilarityCalculator": ".similarity",
    "RhythmEngine": ".rhythm",
    "PersonaExtractor": ".extractor",
    "PersonaFormatter": ".formatter",
    "PersonaManager": ".persona_manager",
    "VectorStore": ".vector_store",
}

__all__ = [
    "PersonaConfig",
    "PersonaItem",
    "ProfileData",
    "GraphEdge",
    "WORK_CATEGORIES",
    "PERSONA_CATEGORIES",
    "CROSS_CATEGORIES",
    "EPHEMERAL_CATEGORIES",
    *_LAZY,
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
