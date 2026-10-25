"""
PersonaTrace configuration for the LLM helpers and the v1 migration shim.

The merged memory path configures itself through arguments/environment; this
dataclass no longer carries the retired v1 decay/rerank/capacity knobs.
"""
import os
from dataclasses import dataclass


@dataclass
class PersonaConfig:
    """LLM and v1-Chroma-migration configuration."""

    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    embed_dims: int = 384
    data_dir: str = "./persona_data"

    @classmethod
    def from_env(cls) -> "PersonaConfig":
        return cls(
            llm_api_key=os.environ.get("PERSONA_LLM_API_KEY", ""),
            llm_base_url=os.environ.get("PERSONA_LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_model=os.environ.get("PERSONA_LLM_MODEL", "gpt-4o"),
            embed_dims=int(os.environ.get("PERSONA_EMBED_DIMS", "384")),
            data_dir=os.environ.get("PERSONA_DATA_DIR", "./persona_data"),
        )

    @property
    def vector_store_path(self) -> str:
        return os.path.join(self.data_dir, "chroma")
