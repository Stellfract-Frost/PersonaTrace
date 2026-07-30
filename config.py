"""
PersonaKit configuration module.
Configure via environment variables or the PersonaConfig dataclass.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
@dataclass
class PersonaConfig:
    """PersonaKit global configuration."""
    # LLM config
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    # Embedding model config
    embed_model: str = "multi-qa-MiniLM-L6-cos-v1"
    embed_dims: int = 384
    # Storage config
    vector_store_path: str = "./persona_data/chroma"
    data_dir: str = "./persona_data"
    # Mem0 conversation memory config
    enable_mem0: bool = False
    mem0_store_path: str = "./persona_data/mem0"
    # Advanced retrieval config (15+3 rerank)
    enable_rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Similarity thresholds
    sim_threshold: float = 0.55
    sim_merge_threshold: float = 0.85
    # Capacity limits
    max_history_entries: int = 10
    max_explicit_facts: int = 30
    max_analysis_log: int = 10
    backup_count: int = 3
    max_graph_edges: int = 500
    # Decay config
    decay_rates: dict = field(default_factory=lambda: {
        "low_conf": 0.08, "mid_low": 0.05, "mid": 0.03, "mid_high": 0.02, "high": 0.01,
    })
    @classmethod
    def from_env(cls) -> "PersonaConfig":
        return cls(
            llm_api_key=os.environ.get("PERSONA_LLM_API_KEY", ""),
            llm_base_url=os.environ.get("PERSONA_LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_model=os.environ.get("PERSONA_LLM_MODEL", "gpt-4o"),
            embed_model=os.environ.get("PERSONA_EMBED_MODEL", "multi-qa-MiniLM-L6-cos-v1"),
            embed_dims=int(os.environ.get("PERSONA_EMBED_DIMS", "384")),
            data_dir=os.environ.get("PERSONA_DATA_DIR", "./persona_data"),
            enable_mem0=os.environ.get("PERSONA_ENABLE_MEM0", "0") == "1",
        )
    @property
    def vector_store_path(self) -> str:
        return os.path.join(self.data_dir, "chroma")
