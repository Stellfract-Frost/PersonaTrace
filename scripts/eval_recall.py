"""Offline recall evaluation for the merged retrieval layer.

Fixture: 12 beliefs, 10 queries. Metrics: recall@3 and MRR.

The point is not to reproduce the 92.3% number from the design doc; that number
came from a different integration. This script gives PersonaTrace its own
baseline, with the same fixture for every method.

Usage:
    python scripts/eval_recall.py
    python scripts/eval_recall.py --no-embed
    python scripts/eval_recall.py --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Sequence
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from personatrace.facet import has_time_intent, time_bucket  # noqa: E402
from personatrace.memory_v2 import MemoryBackend, MemoryV2  # noqa: E402
from personatrace.recall import rank  # noqa: E402

TZ = "Asia/Shanghai"
Z = ZoneInfo(TZ)
NOW = datetime(2026, 10, 23, 10, 0, tzinfo=Z)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _epoch(day: int, hour: int = 23, minute: int = 59) -> float:
    return datetime(2026, 10, day, hour, minute, tzinfo=Z).timestamp()


BELIEFS = [
    # id, category, label, due_at, time_kind, time_raw
    ("c", "goals", "周五要演示", _epoch(23), "due", "周五"),
    ("d", "constraints", "下周三交周报", _epoch(28), "due", "下周三"),
    ("r", "goals", "年底前换工作", datetime(2026, 12, 31, 23, 59, tzinfo=Z).timestamp(), "range", "年底前"),
    ("tired", "state", "这周比较累", None, "none", ""),
    ("agent", "projects", "在赶 agent 项目", None, "none", ""),
    ("night", "state", "熬夜到两点", None, "none", ""),
    ("name", "identity", "名字叫小陌", None, "none", ""),
    ("peanut", "constraints", "对花生过敏", None, "none", ""),
    ("hz", "explicit_facts", "住在杭州", None, "none", ""),
    ("fullname", "dislikes", "讨厌被叫全名", None, "none", ""),
    ("backend", "expertise", "写后端", None, "none", ""),
    ("short", "style_notes", "说话很短", None, "none", ""),
]

QUERIES = [
    ("我那个 deadline 是几号", {"c", "d"}),
    ("接下来有什么安排", {"c", "d", "r"}),
    ("我平时几点睡", {"night"}),
    ("我住哪儿", {"hz"}),
    ("该怎么称呼他", {"name", "fullname"}),
    ("我对什么过敏", {"peanut"}),
    ("agent 项目怎么样了", {"agent", "c"}),
    ("我做什么方向", {"backend"}),
    ("我最近状态怎么样", {"tired", "night"}),
    ("我年底有什么打算", {"r"}),
]

ID_TO_LABEL = {bid: label for bid, _category, label, _due, _kind, _raw in BELIEFS}
EXPECTED_LABELS = {
    query: {ID_TO_LABEL[bid] for bid in expected}
    for query, expected in QUERIES
}


def build_rows() -> List[Dict]:
    mem = MemoryV2(backend=MemoryBackend(), user_id="eval")
    for bid, category, label, due, kind, raw in BELIEFS:
        mem.observe(
            category, label, track="work" if category not in ("identity", "state", "style_notes") else "persona",
            confidence=0.85, due_at_ts=due, time_kind=kind, time_raw=raw,
            ts=NOW.timestamp(),
        )
        # observe() derives the id; map the fixture id through the label.
    by_label = {row["label"]: row for row in mem.state(as_of=NOW.timestamp(), expand=1)}
    rows = []
    for bid, _category, label, _due, _kind, _raw in BELIEFS:
        row = by_label.get(label)
        if row is None:
            continue
        row = dict(row)
        row["fixture_id"] = bid
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class HashEmbedder:
    """Deterministic fallback; not semantic, only keeps the script runnable."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for i in range(max(1, len(text) - 1)):
                gram = text[i:i + 2]
                index = int(hashlib.sha1(gram.encode("utf-8")).hexdigest()[:8], 16) % self.dim
                vec[index] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def load_embedder(no_embed: bool):
    if no_embed:
        return None, "none"
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)

        def encode(texts):
            return model.encode(list(texts), normalize_embeddings=True).tolist()

        return encode, "bge-small-zh-v1.5"
    except Exception as exc:
        print(f"[warn] embedding model unavailable ({exc}); using hash fallback", file=sys.stderr)
        return HashEmbedder(), "hash-fallback"


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def labels(rows: Iterable[Dict]) -> List[str]:
    return [str(row.get("label", "")) for row in rows]


def v2_native(rows, query, *, pinned: bool):
    # Use the kernel only as a baseline; the merged path does not call this.
    mem = MemoryV2(backend=MemoryBackend(), user_id="eval-native")
    for row in rows:
        mem.observe(
            row["category"], row["label"],
            track=row.get("track", "work"), confidence=row.get("confidence", 0.85),
            due_at_ts=row.get("due_at_ts"), time_kind=row.get("time_kind", "none"),
            time_raw=row.get("time_raw", ""), ts=NOW.timestamp(),
        )
    result = mem.recall(as_of=NOW.timestamp(), k=3,
                        pinned_categories=("constraints", "identity") if pinned else ())
    return labels(result["shown"])


def v1_string(rows, query, embed_fn=None):
    from personatrace.config import PersonaConfig
    from personatrace.similarity import SimilarityCalculator
    sim = SimilarityCalculator(PersonaConfig())
    if embed_fn is not None:
        sim.set_embed_fn(embed_fn)
    scored = [(sim.compute(query, row["label"]), row) for row in rows]
    scored.sort(key=lambda item: (-item[0], item[1]["label"]))
    return labels(row for _, row in scored[:3])


def outer_rank(rows, query, *, embed_fn=None, facet=False):
    query_vec = None
    vectors = None
    if embed_fn is not None:
        query_vec = embed_fn([query])[0]
        vectors = {str(row["id"]): embed_fn([row["label"]])[0] for row in rows}
    result = rank(rows, query, query_vec=query_vec, vectors=vectors, k=3,
                  pinned_categories=())
    shown = list(result["shown"])
    if facet and has_time_intent(query):
        bucket = time_bucket(rows, now=NOW, tz=TZ, limit=2)
        merged = []
        seen = set()
        for row in bucket + shown + list(result["cross"]):
            key = str(row.get("id"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
        shown = merged[:3]
    return labels(shown)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def score_method(rows, embed_fn, method) -> Dict[str, float]:
    recalls = []
    rr = []
    details = []
    for query, expected_ids in QUERIES:
        expected = EXPECTED_LABELS[query]
        got = method(rows, query, embed_fn)
        top = got[:3]
        hit = expected & set(top)
        recalls.append(len(hit) / len(expected))
        reciprocal = 0.0
        for index, label in enumerate(top, 1):
            if label in expected:
                reciprocal = 1.0 / index
                break
        rr.append(reciprocal)
        details.append((query, expected, top, reciprocal))
    return {
        "recall@3": sum(recalls) / len(recalls),
        "mrr": sum(rr) / len(rr),
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = build_rows()
    embed_fn, embed_name = load_embedder(args.no_embed)

    methods = [
        ("v2 native (no pinned)", lambda r, q, e: v2_native(r, q, pinned=False)),
        ("v2 native (pinned)", lambda r, q, e: v2_native(r, q, pinned=True)),
        ("v1 string", lambda r, q, e: v1_string(r, q, None)),
        ("v1 + bge", lambda r, q, e: v1_string(r, q, e)),
        ("outer lex", lambda r, q, e: outer_rank(r, q, embed_fn=None)),
        ("outer + bge", lambda r, q, e: outer_rank(r, q, embed_fn=e)),
        ("outer + bge + facet", lambda r, q, e: outer_rank(r, q, embed_fn=e, facet=True)),
    ]

    print(f"embedder: {embed_name}; beliefs: {len(rows)}; queries: {len(QUERIES)}")
    print(f"{'method':<26} {'recall@3':>9} {'MRR':>7}")
    for name, method in methods:
        result = score_method(rows, embed_fn, method)
        print(f"{name:<26} {result['recall@3'] * 100:8.1f}% {result['mrr']:7.3f}")
        if args.verbose:
            for query, expected, top, reciprocal in result["details"]:
                print(f"  {query} → {top} (expected {sorted(expected)}, rr={reciprocal:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
