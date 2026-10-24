"""Run Mem0 on the same LongMemEval temporal subset, same answerer, same judge.

This exists to produce an apples-to-apples number: identical questions, identical
answer prompt, identical grading. The only thing that changes is the memory
system underneath.

Two documented deviations from a vanilla Mem0 integration:

* every message is prefixed with its session date (``[2023/01/04]``), because
  LongMemEval's timestamps live outside the message text and a memory system
  that never sees a date cannot answer a temporal question;
* sessions are grouped into batches of five per ``add`` call to keep the number
  of extraction calls finite. Mem0 still runs its full extraction pipeline
  (``infer=True``); only the batching is ours.

Usage:
    python scripts/eval_mem0_lme.py --limit 20 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.dirname(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from personatrace.config import PersonaConfig  # noqa: E402
from personatrace.llm_client import LLMClient  # noqa: E402
from scripts.eval_longmemeval import (  # noqa: E402
    llm_judge, load_subset, parse_stamp,
)

SESSIONS_PER_ADD = 5


def build_memory_store(path: str, dims: int = 384):
    from mem0 import Memory

    key = os.environ.get("PERSONA_LLM_API_KEY", "")
    model = os.environ.get("PERSONA_LLM_MODEL", "deepseek-flash")
    if not key:
        raise SystemExit("PERSONA_LLM_API_KEY is required for Mem0's extraction step")
    config = {
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": model,
                "api_key": key,
                "deepseek_base_url": os.environ.get(
                    "PERSONA_LLM_BASE_URL", "https://api.deepseek.com"
                ).replace("/v1", ""),
                "max_tokens": 4000,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                "embedding_dims": 384,
            },
        },
        # qdrant is Mem0's own default and supports keyword search, so its BM25
        # hybrid path stays enabled. Chroma would silently downgrade to vector
        # only, which would handicap the baseline.
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": f"mem0_{os.path.basename(path.rstrip('/' + chr(92))) or 'lme'}",
                "path": path,
                "on_disk": True,
                # Without this Mem0 creates a 1536-dim collection (its OpenAI
                # default) and every query against a 384-dim local embedder
                # blows up with a shape mismatch.
                "embedding_model_dims": 384,
            },
        },
        "history_db_path": os.path.join(path, "history.db"),
    }
    return Memory.from_config(config)


def messages_with_dates(item: Dict[str, Any]) -> List[List[Dict[str, str]]]:
    """Group sessions into batches, stamping each message with its date."""
    batches: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    count = 0
    for index, session in enumerate(item["haystack_sessions"]):
        stamp = parse_stamp(item["haystack_dates"][index]).strftime("%Y/%m/%d")
        for turn in session:
            role = str(turn.get("role") or "user")
            content = str(turn.get("content") or "").strip()
            if not content:
                continue
            current.append({"role": role, "content": f"[{stamp}] {content}"})
        count += 1
        if count % SESSIONS_PER_ADD == 0 and current:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def ingest(item: Dict[str, Any], memory) -> int:
    qid = item["question_id"]
    calls = 0
    for batch in messages_with_dates(item):
        memory.add(batch, user_id=qid, infer=True)
        calls += 1
    return calls


def answer_from_memories(client, item: Dict[str, Any], memories: List[Dict[str, Any]]) -> str:
    blocks = []
    for index, row in enumerate(memories, 1):
        text = str(row.get("memory") or "").strip()
        if text:
            blocks.append(f"{index}. {text}")
    context = "\n".join(blocks) or "(no memories retrieved)"
    prompt = f"""You are answering a question about a user's past conversations.
Below are facts recalled from memory. Where a fact carries a date, trust it.

Current date: {item['question_date']}

{context}

Question: {item['question']}

Answer in one short sentence. If the question asks for an amount of time,
give the number and the unit. If it asks for an order, list every item in
order. Do not explain your reasoning."""
    for budget in (4000, 8000):
        text = client.chat([{"role": "user", "content": prompt}],
                           max_tokens=budget, temperature=0.0)
        if (text or "").strip():
            return text.strip()
    return ""


def instrument_usage(memory) -> Dict[str, int]:
    """Capture token usage from Mem0's own OpenAI-compatible client.

    Mem0 discards the usage block, so the only way to compare cost is to wrap
    the client it created. Every extraction call goes through this path.
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    lock = threading.Lock()
    try:
        completions = memory.llm.client.chat.completions
        original = completions.create
    except Exception:
        return usage

    def wrapped(*args, **kwargs):
        response = original(*args, **kwargs)
        block = getattr(response, "usage", None)
        if block is not None:
            with lock:
                usage["prompt_tokens"] += int(getattr(block, "prompt_tokens", 0) or 0)
                usage["completion_tokens"] += int(getattr(block, "completion_tokens", 0) or 0)
                usage["calls"] += 1
        return response

    completions.create = wrapped
    return usage


def run_one(item: Dict[str, Any], path: str, client, *, top_k: int = 200,
            seeds: int = 1, reuse: bool = False) -> Dict[str, Any]:
    memory = build_memory_store(path)
    usage = instrument_usage(memory)
    started = time.time()
    calls = 0 if reuse else ingest(item, memory)
    hits = memory.search(
        item["question"], filters={"user_id": item["question_id"]}, top_k=top_k,
    )
    memories = hits.get("results", hits) if isinstance(hits, dict) else hits
    verdicts = []
    for _ in range(max(1, seeds)):
        response = answer_from_memories(client, item, memories)
        verdicts.append(bool(llm_judge(client, item, response)))
    ok = sum(verdicts) * 2 >= len(verdicts)
    return {
        "question_id": item["question_id"],
        "question": item["question"],
        "gold": item["answer"],
        "response": response,
        "correct": bool(ok),
        "seed_rate": round(sum(verdicts) / len(verdicts), 2),
        "add_calls": calls,
        "memories": len(memories),
        "seconds": round(time.time() - started, 1),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "llm_calls": usage["calls"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chroma-root", default=os.path.join(ROOT, "data", "mem0"))
    parser.add_argument("--results", default=os.path.join(ROOT, "data", "mem0_results.json"))
    parser.add_argument("--top-k", type=int, default=200,
                        help="mem0 official default is 200 retrieved memories")
    parser.add_argument("--seeds", type=int, default=1,
                        help="answer+judge repeats; independent harnesses average 5")
    parser.add_argument("--reuse", action="store_true",
                        help="skip ingestion, reuse an existing vector store")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    items = load_subset(args.limit)
    client = LLMClient(PersonaConfig.from_env())
    print(f"[mem0] {len(items)} questions, {args.workers} workers", file=sys.stderr)

    records = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(run_one, item, os.path.join(args.chroma_root, f"q{index}"), client,
                        top_k=args.top_k, seeds=args.seeds, reuse=args.reuse): item
            for index, item in enumerate(items)
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"[mem0] {record['question_id']} "
                  f"{'ok' if record['correct'] else 'miss'} "
                  f"({record['add_calls']} adds, {record['memories']} memories, "
                  f"{record['seconds']}s)", file=sys.stderr)

    hits = sum(1 for row in records if row["correct"])
    prompt_tokens = sum(row["prompt_tokens"] for row in records)
    completion_tokens = sum(row["completion_tokens"] for row in records)
    calls_total = sum(row["llm_calls"] for row in records)
    n = max(1, len(records))
    with open(args.results, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=1)
    print(f"\nmem0  questions={len(records)}  correct={hits}  "
          f"accuracy={hits / max(1, len(records)) * 100:.1f}%")
    print(f"[tokens] prompt={prompt_tokens:,} completion={completion_tokens:,} "
          f"total={prompt_tokens + completion_tokens:,} calls={calls_total} "
          f"(~{(prompt_tokens + completion_tokens) // n:,} tok/question)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
