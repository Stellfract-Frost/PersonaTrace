"""Offline recall evaluation for the merged retrieval layer.

Fixture: 200 beliefs, 50 queries. Metrics: recall@3 and MRR.

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

try:
    from scripts.eval_data_200 import BELIEFS, LABEL_SET, QUERIES
except ImportError:  # `python scripts/eval_recall.py`
    from eval_data_200 import BELIEFS, LABEL_SET, QUERIES

PERSONA_CATEGORIES = {
    "identity", "traits", "interests", "style_notes", "behavior_patterns",
    "decision_style", "interpersonal", "boundaries", "habit", "state",
}


def build_rows() -> List[Dict]:
    mem = MemoryV2(backend=MemoryBackend(), user_id="eval")
    for category, label, due, kind, raw in BELIEFS:
        mem.observe(
            category, label,
            track="persona" if category in PERSONA_CATEGORIES else "work",
            confidence=0.85, due_at_ts=due, time_kind=kind, time_raw=raw,
            ts=NOW.timestamp(),
        )

    by_label = {row["label"]: row for row in mem.state(as_of=NOW.timestamp(), expand=1)}
    missing = LABEL_SET - set(by_label)
    if missing:
        raise AssertionError(f"fixture rows missing after observe: {sorted(missing)[:5]}")
    rows = []
    for _category, label, _due, _kind, _raw in BELIEFS:
        row = dict(by_label[label])
        rows.append(row)
    if len(rows) != 200:
        raise AssertionError(f"expected 200 rows, got {len(rows)}")
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
    # The kernel now has its own lexical query channel; this method is still
    # the no-outer-layer baseline, not the merged path.
    mem = MemoryV2(backend=MemoryBackend(), user_id="eval-native")
    for row in rows:
        mem.observe(
            row["category"], row["label"],
            track=row.get("track", "work"), confidence=row.get("confidence", 0.85),
            due_at_ts=row.get("due_at_ts"), time_kind=row.get("time_kind", "none"),
            time_raw=row.get("time_raw", ""), ts=NOW.timestamp(),
        )
    result = mem.recall(
        as_of=NOW.timestamp(), k=3, query=query,
        pinned_categories=("constraints", "identity") if pinned else (),
    )
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
    for query, expected_labels in QUERIES:
        expected = expected_labels
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
    if not recalls:
        raise AssertionError("no queries scored")
    return {
        "recall@3": sum(recalls) / len(recalls),
        "mrr": sum(rr) / len(rr),
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--fail-under", type=float, default=None,
        help="CI gate: exit 1 when the gate method's recall@3 is below this fraction (0..1)",
    )
    parser.add_argument(
        "--gate-method", default=None,
        help="method name to gate; default: outer lex without embeddings, else outer + bge + facet",
    )
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
    results: Dict[str, Dict[str, float]] = {}
    for name, method in methods:
        result = score_method(rows, embed_fn, method)
        results[name] = result
        print(f"{name:<26} {result['recall@3'] * 100:8.1f}% {result['mrr']:7.3f}")
        if args.verbose:
            for query, expected, top, reciprocal in result["details"]:
                print(f"  {query} → {top} (expected {sorted(expected)}, rr={reciprocal:.2f})")

    if args.fail_under is not None:
        gate = args.gate_method or (
            "outer lex" if embed_name == "none" else "outer + bge + facet"
        )
        if gate not in results:
            print(f"[gate] unknown gate method: {gate}", file=sys.stderr)
            return 2
        value = results[gate]["recall@3"]
        if value < args.fail_under:
            print(
                f"[gate] FAIL {gate}: recall@3 {value * 100:.1f}% < {args.fail_under * 100:.1f}%",
                file=sys.stderr,
            )
            return 1
        print(
            f"[gate] PASS {gate}: recall@3 {value * 100:.1f}% >= {args.fail_under * 100:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
