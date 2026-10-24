"""LongMemEval temporal-reasoning subset, run against PersonaTrace's episode layer.

This is the benchmark harness, not the project's own 200-belief fixture. It
answers one question: given a 50-session haystack and a temporal question, does
the memory surface the sessions that actually hold the answer?

Design notes, because the choices matter for reading the numbers:

* The retrieval unit is the **session**, because LongMemEval labels evidence as
  ``answer_session_ids``. Episodes are stored per exchange (user + reply) with
  ``session_id`` set, then hits are aggregated up to the session.
* Chroma indexes the episodes; ``EpisodeStore`` stays the source of truth. The
  lexical arm runs the same store with no vector index, so both arms are the
  project's own code, not a reimplementation.
* No belief layer here on purpose. Beliefs need an LLM extractor at write time
  and there is no API key offline; the episode path is the shipped semantic
  path and the one that can be measured without a judge.
* Answers are scored twice:
    - ``evidence recall@k``: did the gold sessions appear in the top-k;
    - ``date-arithmetic accuracy``: a no-LLM solver that reads the dates of the
      retrieved sessions and does the subtraction. It only covers numeric
      questions; ordering questions are reported as unsupported, not as wrong.

Usage:
    python scripts/eval_longmemeval.py --limit 8            # smoke
    python scripts/eval_longmemeval.py                      # full subset
    python scripts/eval_longmemeval.py --embedder bge
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.dirname(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from personatrace import MergedMemory  # noqa: E402
from personatrace.chroma_index import ChromaEpisodeIndex  # noqa: E402
from personatrace.episodes import EpisodeStore, has_retrospective  # noqa: E402

TZ = "Asia/Shanghai"
Z = ZoneInfo(TZ)
DATA = os.path.join(ROOT, "data", "longmemeval_s.json")
COLLECTION = "episode_turns"

_DATE_RE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})(?:\s*\([^)]*\))?\s*(\d{1,2}):(\d{2})")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def parse_stamp(text: str) -> datetime:
    match = _DATE_RE.search(str(text or ""))
    if not match:
        return datetime(2023, 1, 1, tzinfo=Z)
    year, month, day, hour, minute = (int(x) for x in match.groups())
    try:
        return datetime(year, month, day, hour, minute, tzinfo=Z)
    except ValueError:
        return datetime(2023, 1, 1, tzinfo=Z)


def load_subset(limit: int = 0) -> List[Dict[str, Any]]:
    with open(DATA, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    subset = [row for row in rows if row.get("question_type") == "temporal-reasoning"]
    return subset[:limit] if limit else subset


def session_exchanges(session: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """One episode per user+assistant exchange.

    Assistant-first turns are kept as their own document with a role prefix:
    dropping them loses content that a temporal question can depend on.
    """
    pairs: List[Tuple[str, str]] = []
    pending: Optional[List[str]] = None
    for turn in session or []:
        role = str(turn.get("role") or "").lower()
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            if pending is not None:
                pairs.append((pending[0], pending[1] if len(pending) > 1 else ""))
            pending = [content]
        elif role == "assistant":
            if pending is None:
                pairs.append((f"Assistant: {content}", ""))
            else:
                pending.append(content)
                pairs.append((pending[0], pending[1]))
                pending = None
        else:
            pairs.append((content, ""))
    if pending is not None:
        pairs.append((pending[0], pending[1] if len(pending) > 1 else ""))
    return pairs


def session_text(session: Sequence[Dict[str, Any]], limit: int = 1600) -> str:
    parts = []
    for turn in session or []:
        content = str(turn.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts)[:limit]


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------

def load_embedder(name: str):
    import torch
    from sentence_transformers import SentenceTransformer

    if name == "bge":
        model_name = "BAAI/bge-small-zh-v1.5"
    else:
        model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    model = SentenceTransformer(model_name, local_files_only=True)
    try:
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    except Exception:
        pass

    def encode(texts: Sequence[str]) -> List[List[float]]:
        return model.encode(
            list(texts), normalize_embeddings=True,
            batch_size=256, show_progress_bar=False,
        ).tolist()

    return encode, model_name


class Corpus:
    """Everything the retrieval arms need, keyed by question."""

    def __init__(self) -> None:
        # The merged facade owns the episode store, so "merged" is not an
        # ablation of MergedMemory — it IS MergedMemory. The other arms reach
        # into the same store to isolate lexical / vector / fusion.
        self.memories: Dict[str, Any] = {}
        self.store: Optional[EpisodeStore] = None
        self.index: Optional[ChromaEpisodeIndex] = None
        self.episode_session: Dict[str, str] = {}
        self.session_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def ingest(self, items: List[Dict[str, Any]], encode, *, reuse_index: bool = False) -> None:
        started = time.time()
        total = 0
        reused = 0
        for item in items:
            qid = item["question_id"]
            meta: Dict[str, Dict[str, Any]] = {}
            episodes: Dict[str, Any] = {}
            # One MergedMemory per question: the benchmark gives every question
            # its own haystack, so each is a separate user. Writes go through
            # the facade (record_turn), not through EpisodeStore directly.
            store = EpisodeStore(":memory:", default_tz=TZ)
            memory = MergedMemory(user_id=qid, episodes=store)
            for index, session in enumerate(item["haystack_sessions"]):
                sid = str(item["haystack_session_ids"][index])
                stamp = parse_stamp(item["haystack_dates"][index])
                pairs = session_exchanges(session)
                meta[sid] = {
                    "ts": stamp.timestamp(),
                    "date": stamp,
                    "text": session_text(session),
                    # Exchange-level chunks: the cross-encoder is also capped at
                    # 512 tokens, so scoring an exchange fits while scoring a
                    # 10k-char session does not.
                    "chunks": [f"{u}\n{a}".strip()[:2400] for u, a in pairs],
                }
                for user_msg, ai_reply in pairs:
                    result = memory.record_turn(
                        user_msg, ai_reply, session_id=sid, now=stamp,
                    )
                    episode_id = result.get("episode_id") or ""
                    if not episode_id:
                        continue
                    self.episode_session[episode_id] = sid
                    # Episode ids are uuid5 over (user, session, minute, text), so
                    # an identical turn repeated inside the same minute collides.
                    # Keep one row per id instead of failing the index upsert.
                    episodes[episode_id] = store.get(episode_id)
            self.session_meta[qid] = meta
            if self.index is not None and episodes:
                # Reuse must be decided per question, not by a global count:
                # otherwise a run with new questions silently indexes nothing.
                already = False
                if reuse_index:
                    probe = next(iter(episodes.values())).id
                    found = self.index.collection.get(ids=[probe], include=[])
                    already = bool(found.get("ids"))
                if already:
                    reused += 1
                else:
                    self.index.upsert(episodes.values())
            # Vectors are enabled only after the batch upsert: EpisodeStore.add
            # would otherwise embed one episode per call.
            store.vector_index = self.index
            self.memories[qid] = memory
            total += len(episodes)
        if reused:
            print(f"[ingest] reused vectors for {reused}/{len(items)} questions", file=sys.stderr)
        print(
            f"[ingest] {len(items)} questions, {total} episodes in {time.time() - started:.1f}s",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# retrieval arms
# ---------------------------------------------------------------------------

def aggregate(hits: Iterable[Tuple[str, float]], episode_session: Dict[str, str]) -> Dict[str, float]:
    """Episode hits -> session scores. Max wins; extra hits only nudge."""
    scores: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for episode_id, score in hits:
        sid = episode_session.get(episode_id)
        if sid is None:
            continue
        scores[sid] = max(scores.get(sid, -1e9), score)
        counts[sid] = counts.get(sid, 0) + 1
    return {
        sid: value + 0.02 * math.log1p(counts[sid] - 1)
        for sid, value in scores.items()
    }


def lexical_hits(corpus: Corpus, qid: str, query: str, limit: int = 60) -> List[Tuple[str, float]]:
    store = corpus.memories[qid].episodes
    saved = store.vector_index
    store.vector_index = None          # isolate the lexical arm
    try:
        rows = store.search(qid, query, limit=limit)
    finally:
        store.vector_index = saved
    return [(row["id"], float(row.get("score", 0.0))) for row in rows]


def merged_hits(corpus: Corpus, qid: str, query: str, limit: int = 60) -> List[Tuple[str, float]]:
    """The merged facade's own retrospective search, used as-is."""
    rows = corpus.memories[qid].episodes_search(query, limit=limit)
    return [(str(row["id"]), float(row.get("score", 0.0))) for row in rows]


def vector_hits(corpus: Corpus, qid: str, query: str, limit: int = 60) -> List[Tuple[str, float]]:
    if corpus.index is None:
        return []
    rows = corpus.index.query(query, user_id=qid, limit=limit)
    return [(row["id"], float(row.get("score", 0.0))) for row in rows]


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


def rank_sessions(
    corpus: Corpus, qid: str, query: str, arm: str,
    reranker=None, depth: int = 60, top_n: int = 20,
) -> List[str]:
    mapping = corpus.session_meta[qid]
    if arm == "merged":
        hits = merged_hits(corpus, qid, query, depth)
        scores = aggregate(hits, corpus.episode_session)
        return _ordered(scores)
    if arm == "lex":
        hits = lexical_hits(corpus, qid, query, depth)
        scores = aggregate(hits, corpus.episode_session)
        return _ordered(scores)
    if arm == "chroma":
        hits = vector_hits(corpus, qid, query, depth)
        scores = aggregate(hits, corpus.episode_session)
        return _ordered(scores)
    if arm in ("hybrid", "hybrid+rerank"):
        lex_scores = aggregate(lexical_hits(corpus, qid, query, depth), corpus.episode_session)
        vec_scores = aggregate(vector_hits(corpus, qid, query, depth), corpus.episode_session)
        fused = rrf([
            _ordered(lex_scores),
            _ordered(vec_scores),
        ])
        order = _ordered(fused)[:top_n]
        if arm == "hybrid" or reranker is None:
            return order
        # Only the head is re-ordered; the tail keeps the fusion order. Reranking
        # all 20 at chunk level cost ~15x the original time for little gain.
        head = rerank_chunks(reranker, query, mapping, order[:10])
        head_set = set(head)
        return head + [sid for sid in order if sid not in head_set]


def rerank_chunks(reranker, query: str, mapping: Dict[str, Dict[str, Any]],
                  order: Sequence[str], max_chunks: int = 6) -> List[str]:
    """Score exchange-level chunks, not whole sessions.

    The cross-encoder truncates at 512 tokens, so a 10k-char session loses 84%
    of itself before scoring; an exchange fits. A session keeps its best chunk.
    """
    pairs: List[Tuple[str, str]] = []
    owners: List[str] = []
    for sid in order:
        for chunk in mapping[sid].get("chunks", [])[:max_chunks]:
            if chunk.strip():
                pairs.append((query, chunk))
                owners.append(sid)
    if not pairs:
        return list(order)
    try:
        scores = reranker.predict(pairs, batch_size=64, show_progress_bar=False)
    except Exception:
        return list(order)
    best: Dict[str, float] = {}
    for sid, score in zip(owners, scores):
        best[sid] = max(best.get(sid, -1e9), float(score))
    for sid in order:
        best.setdefault(sid, -1e9)
    return _ordered(best)


def _ordered(scores: Dict[str, float]) -> List[str]:
    return [key for key, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def evidence_metrics(order: List[str], gold: Sequence[str]) -> Dict[str, float]:
    gold = list(dict.fromkeys(str(x) for x in gold))
    position = {sid: rank for rank, sid in enumerate(order, 1)}
    out: Dict[str, float] = {}
    for k in (3, 5, 10, 20):
        top = set(order[:k])
        out[f"recall@{k}"] = len(set(gold) & top) / len(gold) if gold else 0.0
        out[f"all@{k}"] = 1.0 if set(gold) <= top else 0.0
    first = min((position.get(sid, 10 ** 6) for sid in gold), default=10 ** 6)
    out["mrr"] = 1.0 / first if first < 10 ** 6 else 0.0
    return out


# ---------------------------------------------------------------------------
# no-LLM date arithmetic
# ---------------------------------------------------------------------------

UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


def unit_of(query: str) -> Optional[str]:
    match = re.search(r"how many (day|week|month|year)s?", query, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    match = re.search(r"how (?:long|much time)", query, re.IGNORECASE)
    return "day" if match else None


def arity_of(query: str) -> int:
    text = query.lower()
    if " between " in text or "since" in text or "did it take" in text or "how long" in text:
        return 2
    if re.search(r"\bago\b", text):
        return 1
    return 0


def gold_numbers(answer: Any) -> set:
    if isinstance(answer, int):
        return {answer}
    return {int(number) for number in re.findall(r"\d+", str(answer))}


def solve_date(corpus: Corpus, item: Dict[str, Any], order: List[str]) -> Optional[int]:
    query = item["question"]
    unit = unit_of(query)
    arity = arity_of(query)
    if unit is None or arity == 0 or not order:
        return None
    meta = corpus.session_meta[item["question_id"]]
    try:
        if arity == 1:
            first = meta[order[0]]["date"]
            anchor = parse_stamp(item["question_date"])
            delta = (anchor.date() - first.date()).days
        else:
            if len(order) < 2:
                return None
            left = meta[order[0]]["date"]
            right = meta[order[1]]["date"]
            delta = abs((left.date() - right.date()).days)
    except Exception:
        return None
    scale = UNIT_DAYS[unit]
    if scale == 1:
        return delta
    return int(round(delta / scale))


# ---------------------------------------------------------------------------
# optional LLM answerer + judge (needs PERSONA_LLM_API_KEY)
# ---------------------------------------------------------------------------


USAGE: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
_USAGE_LOCK = threading.Lock()


def _record_usage(usage: Dict[str, Any]) -> None:
    if not usage:
        return
    with _USAGE_LOCK:
        USAGE["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        USAGE["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        USAGE["calls"] += 1


def _chat(client, messages, *, max_tokens: int, temperature: float = 0.0) -> str:
    """client.chat with backoff. Concurrent runs share one API quota."""
    last = None
    for attempt in range(4):
        try:
            text = client.chat(messages, max_tokens=max_tokens, temperature=temperature)
            _record_usage(getattr(client, "last_usage", {}) or {})
            return text
        except Exception as exc:                      # 429 / 5xx / network
            last = exc
            if attempt == 3:
                raise
            time.sleep(2 ** attempt * 2)
    raise last


def build_client():
    from personatrace.config import PersonaConfig
    from personatrace.llm_client import LLMClient

    config = PersonaConfig.from_env()
    if not config.llm_api_key:
        return None, config
    return LLMClient(config), config


def render_sessions(item: Dict[str, Any], order: Sequence[str], k: int,
                    chars: int = 4000) -> str:
    """Session blocks with their dates. The date is the whole point here."""
    by_id = {
        str(sid): (str(date_text), session_text(session, chars))
        for sid, date_text, session in zip(
            item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"],
        )
    }
    blocks = []
    for sid in order[:k]:
        if sid not in by_id:
            continue
        stamp, text = by_id[sid]
        blocks.append(f"### Session (date: {stamp})\n{text}")
    return "\n\n".join(blocks)


def llm_answer(client, item: Dict[str, Any], order: Sequence[str], k: int = 5,
               chars: int = 4000) -> str:
    context = render_sessions(item, order, k, chars)
    prompt = f"""You are answering a question about a user's past conversations.
Each session below is labelled with the date it happened.

Current date: {item['question_date']}

{context}

Question: {item['question']}

Answer in one short sentence. If the question asks for an amount of time,
give the number and the unit. If it asks for an order, list every item in
order. Do not explain your reasoning."""
    # Both hosted models are reasoning models: thinking tokens are billed
    # against max_tokens, so a small budget comes back as empty content. One
    # observed question spent 5.3k chars of reasoning and returned nothing.
    for budget in (4000, 8000):
        text = _chat(client, [{"role": "user", "content": prompt}], max_tokens=budget)
        if (text or "").strip():
            return text.strip()
    # Last resort: less context to reason about, and no room to deliberate.
    short = render_sessions(item, order, min(2, k), chars)
    forced = prompt.replace(context, short) if context else prompt
    forced += "\n\nAnswer immediately with the result only. No explanation."
    text = _chat(client, [{"role": "user", "content": forced}], max_tokens=8000)
    return (text or "").strip()


def llm_judge(client, item: Dict[str, Any], response: str) -> bool:
    """LongMemEval-style yes/no grading against the gold answer."""
    prompt = f"""I will give you a question, the correct answer, and a model's response.
Decide whether the response is correct. A response that contains the correct
answer plus extra harmless detail is correct. A response with a different
number, a different order, or a different date is incorrect.

Question: {item['question']}
Correct answer: {item['answer']}
Model response: {response or '(empty)'}

Reply with exactly one word: yes or no."""
    text = _chat(client, [{"role": "user", "content": prompt}], max_tokens=300) or ""
    match = re.search(r"\b(yes|no)\b", text.strip().lower())
    if match:
        return match.group(1) == "yes"
    return text.strip().lower().startswith("y")



def _answer_one(client, item, order, args, ctx: str = ""):
    if ctx:
        response = answer_from_context(client, item, ctx)
    else:
        response = llm_answer(client, item, order, args.context_k, args.session_chars)
    ok = llm_judge(client, item, response)
    return item, response, ok


def answer_from_context(client, item: Dict[str, Any], context: str) -> str:
    """Answer from MergedMemory.inject(), i.e. the facade's own rendering."""
    prompt = f"""You are answering a question about a user's past conversations.
Current date: {item['question_date']}

Here is what your memory layer returned:

{context or '(nothing)'}

Question: {item['question']}

Answer in one short sentence. If the question asks for an amount of time,
give the number and the unit. If it asks for an order, list every item in
order. Do not explain your reasoning."""
    for budget in (4000, 8000):
        text = _chat(client, [{"role": "user", "content": prompt}], max_tokens=budget)
        if (text or "").strip():
            return text.strip()
    return ""


def _answer_all(client, pending, args, solve, failures, contexts=None) -> None:
    """Answer + judge in parallel; each question is independent."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    contexts = contexts or {}
    done = 0
    records = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_answer_one, client, item, order, args,
                               contexts.get(item["question_id"], ""))
                   for item, order in pending]
        for future in as_completed(futures):
            item, response, ok = future.result()
            done += 1
            solve["attempt"].append(1)
            solve["covered"].append(1.0)
            solve["hit"].append(1.0 if ok else 0.0)
            if args.results:
                records.append({
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "gold": item["answer"],
                    "response": response,
                    "correct": bool(ok),
                })
            if not ok and len(failures) < 25:
                failures.append(
                    f"{item['question_id']}: said={(response or '')[:70]!r} "
                    f"gold={item['answer']!r}"
                )
            if done % 25 == 0:
                print(f"[llm] {done}/{len(pending)}", file=sys.stderr)
    if args.results and records:
        with open(args.results, "w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=1)
        print(f"[results] {len(records)} rows -> {args.results}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--embedder", default="minilm", choices=["minilm", "bge"])
    parser.add_argument("--chroma", default=os.path.join(ROOT, "data", "chroma_lme"))
    parser.add_argument("--no-chroma", action="store_true")
    parser.add_argument("--arms", default="lex,chroma,hybrid,hybrid+rerank")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--reuse-index", action="store_true",
                        help="skip re-embedding when the Chroma collection is already populated")
    parser.add_argument("--answerer", default="solver", choices=["solver", "llm"])
    parser.add_argument("--orders", default="",
                        help="load a dumped retrieval order file and only run answer/judge")
    parser.add_argument("--dump", default="",
                        help="write retrieval orders to this JSON path at the end")
    parser.add_argument("--context-k", type=int, default=10,
                        help="sessions shown to the LLM answerer")
    parser.add_argument("--session-chars", type=int, default=4000,
                        help="characters per session shown to the answerer")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel answer/judge calls")
    parser.add_argument("--answer-from", default="sessions",
                        choices=["sessions", "context"],
                        help="sessions = harness rendering; context = MergedMemory.inject()")
    parser.add_argument("--only", default="", help="comma-separated question ids")
    parser.add_argument("--results", default="", help="write per-question rows to this JSON")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    items = load_subset(args.limit)
    print(f"[data] {len(items)} temporal-reasoning questions", file=sys.stderr)

    if args.orders:
        return run_orders_only(items, args)

    encode, model_name = load_embedder(args.embedder)
    corpus = Corpus()
    if not args.no_chroma:
        corpus.index = ChromaEpisodeIndex(
            path=args.chroma, embed_fn=encode, collection_name=COLLECTION,
        )
        print(f"[chroma] {corpus.index.count()} vectors already indexed", file=sys.stderr)
    corpus.ingest(items, encode, reuse_index=args.reuse_index)

    reranker = None
    if "hybrid+rerank" in args.arms and not args.no_chroma:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True,
        )

    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    totals: Dict[str, Dict[str, List[float]]] = {arm: {} for arm in arms}
    solve: Dict[str, List[float]] = {"hit": [], "attempt": [], "covered": []}
    failures: List[str] = []
    orders_dump: Dict[str, Dict[str, List[str]]] = {}
    gate = [0]

    client = None
    if args.answerer == "llm":
        client, config = build_client()
        if client is None:
            print(
                "[fatal] --answerer llm needs PERSONA_LLM_API_KEY "
                "(and optionally PERSONA_LLM_BASE_URL / PERSONA_LLM_MODEL)",
                file=sys.stderr,
            )
            return 2
        print(f"[llm] model={config.llm_model} base={config.llm_base_url}", file=sys.stderr)

    pending: List[Tuple[Dict[str, Any], List[str]]] = []
    contexts: Dict[str, str] = {}
    for number, item in enumerate(items, 1):
        qid = item["question_id"]
        gold = item["answer_session_ids"]
        per_arm: Dict[str, List[str]] = {}
        for arm in arms:
            order = rank_sessions(corpus, qid, item["question"], arm, reranker=reranker)
            per_arm[arm] = order
            for key, value in evidence_metrics(order, gold).items():
                totals[arm].setdefault(key, []).append(value)
        orders_dump[qid] = per_arm
        if has_retrospective(item["question"]):
            gate[0] += 1
        best = per_arm.get("hybrid+rerank") or per_arm.get("merged") or per_arm.get("hybrid") or per_arm[arms[0]]
        if client is not None:
            if args.answer_from == "context":
                contexts[qid] = corpus.memories[qid].inject(
                    item["question"], k=args.context_k, max_chars=30000,
                )
            pending.append((item, best))
            continue
        guess = solve_date(corpus, item, best)
        wants_number = bool(gold_numbers(item["answer"])) and bool(unit_of(item["question"]))
        if wants_number:
            solve["attempt"].append(1)
            ok = guess is not None and any(
                abs(guess - number) <= 1 if UNIT_DAYS.get(unit_of(item["question"]) or "day", 1) > 1
                else guess == number
                for number in gold_numbers(item["answer"])
            )
            solve["hit"].append(1.0 if ok else 0.0)
            solve["covered"].append(1.0 if guess is not None else 0.0)
            if not ok and len(failures) < 25:
                failures.append(
                    f"{qid}: guess={guess} gold={item['answer']!r} top={best[:3]}"
                )
        else:
            solve["attempt"].append(0)
            solve["hit"].append(0.0)
            solve["covered"].append(0.0)
        if number % 25 == 0:
            print(f"[run] {number}/{len(items)}", file=sys.stderr)

    if client is not None and pending:
        _answer_all(client, pending, args, solve, failures, contexts)

    if args.dump and orders_dump:
        with open(args.dump, "w", encoding="utf-8") as handle:
            json.dump(orders_dump, handle)
        print(f"[dump] retrieval orders -> {args.dump}", file=sys.stderr)

    print(f"\nembedder: {model_name}")
    _report_usage(len(items))
    print(f"questions: {len(items)}  (LongMemEval temporal-reasoning, longmemeval_s)")
    header = f"{'arm':<16}{'recall@3':>10}{'recall@5':>10}{'recall@10':>11}{'recall@20':>11}{'all@10':>9}{'MRR':>8}"
    print(header)
    print("-" * len(header))
    for arm in arms:
        row = totals[arm]
        print(
            f"{arm:<16}"
            f"{_mean(row, 'recall@3') * 100:9.1f}%"
            f"{_mean(row, 'recall@5') * 100:9.1f}%"
            f"{_mean(row, 'recall@10') * 100:10.1f}%"
            f"{_mean(row, 'recall@20') * 100:10.1f}%"
            f"{_mean(row, 'all@10') * 100:8.1f}%"
            f"{_mean(row, 'mrr'):8.3f}"
        )

    numeric = sum(solve["attempt"])
    print(
        f"\n[gate] MergedMemory.build_context only searches episodes when "
        f"has_retrospective(query) is true: {gate[0]}/{len(items)} questions qualify"
    )
    if client is not None:
        print(
            f"\nLLM answers (model={config.llm_model}), judged by the same model: "
            f"{int(sum(solve['hit']))}/{len(items)} correct"
            f"  = {sum(solve['hit']) / max(1, len(items)) * 100:.1f}%"
        )
        print("  note: judged numbers are only comparable to published LongMemEval")
        print("  results when both the answerer and the judge are GPT-4o.")
    else:
        print(
            f"\ndate-arithmetic answers (no LLM): numeric questions {numeric}/{len(items)}, "
            f"solver produced an answer on {int(sum(solve['covered']))}, "
            f"correct {int(sum(solve['hit']))}"
        )
        if items:
            print(
                f"  accuracy over numeric questions: {sum(solve['hit']) / max(1, numeric) * 100:.1f}%"
                f"   over all questions: {sum(solve['hit']) / len(items) * 100:.1f}%"
            )
    if args.verbose and failures:
        print("\nmisses:")
        for line in failures:
            print("  " + line)
    return 0


def run_orders_only(items: List[Dict[str, Any]], args) -> int:
    """Reuse a dumped retrieval order, so only the LLM stage is paid for."""
    with open(args.orders, "r", encoding="utf-8") as handle:
        orders = json.load(handle)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    client, config = build_client()
    if client is None:
        print("[fatal] --orders needs PERSONA_LLM_API_KEY for answering and judging",
              file=sys.stderr)
        return 2
    hits = 0
    used = 0
    records = []
    for item in items:
        if only and item["question_id"] not in only:
            continue
        per_arm = orders.get(item["question_id"]) or {}
        order = per_arm.get("hybrid+rerank") or per_arm.get("hybrid") or []
        if not order:
            continue
        used += 1
        response = llm_answer(client, item, order, args.context_k, args.session_chars)
        ok = llm_judge(client, item, response)
        if ok:
            hits += 1
        records.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "gold": item["answer"],
            "response": response,
            "correct": bool(ok),
            "evidence_top5": [sid for sid in order[:5] if sid in set(item["answer_session_ids"])],
            "evidence_all_in_top5": set(item["answer_session_ids"]) <= set(order[:5]),
        })
        if used % 25 == 0:
            print(f"[llm] {used}/{len(items)}", file=sys.stderr)
    if args.results:
        with open(args.results, "w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=1)
        print(f"[results] {len(records)} rows -> {args.results}", file=sys.stderr)
    print(f"model={config.llm_model}  questions={used}  correct={hits}  "
          f"accuracy={hits / max(1, used) * 100:.1f}%")
    return 0


def _report_usage(n_questions: int) -> None:
    total = USAGE["prompt_tokens"] + USAGE["completion_tokens"]
    print(f"[tokens] prompt={USAGE['prompt_tokens']:,} "
          f"completion={USAGE['completion_tokens']:,} total={total:,} "
          f"calls={USAGE['calls']} "
          f"(~{total // max(1, n_questions):,} tok/question)")


def _mean(rows: Dict[str, List[float]], key: str) -> float:
    values = rows.get(key) or []
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
