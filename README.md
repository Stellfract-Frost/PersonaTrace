# PersonaTrace

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Build structured user profiles from conversations and documents. PersonaTrace watches interactions, extracts what it learns into a dual-track profile (Work + Persona), and stores it in a vector database so future conversations can surface the right context.

## Installation

```bash
pip install -r requirements.txt
```

First import of `sentence-transformers` will download the embedding model (~80 MB, cached after).

**Windows users:** ChromaDB depends on `onnxruntime`, which needs the [Visual C++ Redistributable](https://aka.ms/vcredist). If you see a DLL error on import, install that.

## Quick Start

```python
from persona import PersonaConfig, PersonaManager, LLMClient
from sentence_transformers import SentenceTransformer

config = PersonaConfig(
    llm_api_key="sk-...",
    llm_base_url="https://api.openai.com/v1",
    llm_model="gpt-4o",
)

embed_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

llm = LLMClient(config)
mgr = PersonaManager(
    config, llm=llm,
    embed_fn=lambda texts: embed_model.encode(texts).tolist(),
)

# Feed it a document
mgr.ingest_file("resume.pdf", user_id="alice")

# Feed it a conversation turn
mgr.ingest_conversation(
    user_msg="I've been doing a lot of Rust systems programming lately",
    ai_reply="That's impressive — what kind of systems work?",
    user_id="alice",
)

# Ask for the profile
print(mgr.format_profile(user_id="alice"))
```

## How It Works

Profile data is split into two tracks:

| Track | What it captures |
|-------|-----------------|
| **Work** | expertise, projects, goals, likes, dislikes, constraints, facts |
| **Persona** | identity, traits, interests, communication style, decision patterns, boundaries |

Each conversation turn goes through three stages:

1. **Extract** — LLM pulls candidate labels from the exchange
2. **Search** — Query the vector store for semantically similar items already on file
3. **Merge** — LLM compares new items against existing ones: same thing? merge and boost confidence. Contradiction? flag it. Old info corrected? update it. Something new? add it.

This keeps the profile from bloating — repeated information raises confidence instead of creating duplicates.

The key modules you'll interact with: `PersonaManager` (orchestrator), `LLMClient` (OpenAI-compatible API), `PersonaConfig` (all settings). Supporting modules — `vector_store.py` (ChromaDB), `similarity.py` (embeddings + Jaccard fallback), `rhythm.py` (temporal pattern detection), `extractor.py` (the 3-stage pipeline), `formatter.py` (profile → text), `file_loader.py` (PDF/PPTX/DOCX/XLSX parsing) — are used internally.

## Configuration

Everything is driven through `PersonaConfig` or environment variables:

| Variable | Default |
|----------|---------|
| `PERSONA_LLM_API_KEY` | — *(required)* |
| `PERSONA_LLM_BASE_URL` | `https://api.openai.com/v1` |
| `PERSONA_LLM_MODEL` | `gpt-4o` |
| `PERSONA_EMBED_MODEL` | `multi-qa-MiniLM-L6-cos-v1` |
| `PERSONA_EMBED_DIMS` | `384` |
| `PERSONA_DATA_DIR` | `./persona_data` |
| `PERSONA_ENABLE_MEM0` | `0` |

Key thresholds (set on `PersonaConfig`):

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `sim_threshold` | `0.55` | Minimum similarity to consider two items related |
| `sim_merge_threshold` | `0.85` | Similarity above which items merge as duplicates |
| `decay_rates` | tiered (0.01–0.08) | Confidence lost per day, by confidence tier |

## Features

- **Confidence system** — every trait carries a score. Repeated mentions raise it; inactivity decays it over time. The LLM merge stage can also adjust confidence when new evidence comes in.
- **Temporal rhythms** — clusters observations by hour and weekday. A pattern that appears on 3+ different days within a ±1h window gets promoted to a rhythm record.
- **Multi-format ingestion** — `FileLoader` handles `.pdf`, `.pptx`, `.docx`, `.xlsx`, plus plain text, markdown, JSON, CSV, and source files. Drop a resume or slide deck and it'll extract what it can.
- **Semantic reranking** — when formatting a profile for a specific query, pulls the top-15 most relevant items so output is focused on what matters.
- **Cross-category deduplication** — if the same trait shows up in both `traits` and `style_notes`, only the higher-confidence copy survives. No duplicate clutter.
- **Mem0 integration** (optional) — set `PERSONA_ENABLE_MEM0=1` to add [Mem0](https://github.com/mem0ai/mem0) conversation memory alongside the structured profile. Off by default; the core system doesn't need it.

## What It's Good For

- Adding persistent user memory to chatbots and AI assistants
- Extracting structured profiles from interview transcripts or support tickets
- Tracking user goals, expertise, and decision patterns in coaching tools
- Building team skill maps from aggregated work-track profiles

A note on scope: PersonaTrace is a profiling engine — it builds and queries structured user models. It is not a chatbot, not a prompt manager, and not a general-purpose memory layer (see Mem0 for that). It works at its best when you have ongoing conversations with a known user and want the assistant to remember who they are.

## Dependencies

```
chromadb>=0.4.0          # vector storage
requests>=2.28.0         # LLM API calls
numpy>=1.21.0            # similarity math
sentence-transformers>=2.2.0  # embeddings
# optional
mem0>=0.1.0              # conversation memory
python-pptx>=0.6.23      # .pptx support
python-docx>=1.1.0       # .docx support
pypdf>=4.0.0             # .pdf support
openpyxl>=3.1.0          # .xlsx support
```

## License

MIT
