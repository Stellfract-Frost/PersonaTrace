"""Outer retrieval for the v1 × v2 merge.

The v2 kernel stores beliefs well but its own ``recall()`` has no query: ``sim``
is a constant 0.5. This module is the missing outer layer. It is a pure
function over candidate rows, so it can be tested and evaluated without a
database.

Scoring (deliberately the same shape as the design doc):

    sim   = label ⊆ query ? 1.0 : 0.5*vector_norm + 0.5*lexical
    score = (0.15 + 0.85*sim) * (0.35 + 0.65*retention)

``vector_norm`` is candidate-set z-score + tanh, so 0.5 means "average", not
"irrelevant". Lexical also returns 0.5 when there is no shared character: a
neutral signal is safer than pretending two unrelated strings are a match.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .schema import PINNED_CATEGORIES
except ImportError:  # top-level test import
    from schema import PINNED_CATEGORIES


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _bigrams(text: str) -> set:
    text = re.sub(r"\s+", "", _text(text))
    if not text:
        return set()
    if len(text) == 1:
        return {text}
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _cosine(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if a is None or b is None:
        return 0.0
    a = list(a)
    b = list(b)
    if not a or not b or len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _vector_norm(cosines: Dict[str, float]) -> Dict[str, float]:
    """Candidate-set z-score + tanh; 0.5 is the neutral point."""
    if not cosines:
        return {}
    values = list(cosines.values())
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std < 1e-9:
        return {key: 0.5 for key in cosines}
    return {key: 0.5 + 0.5 * math.tanh((value - mean) / std) for key, value in cosines.items()}


def _lexical_scores(query: str, rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not query:
        return {_id(row): 0.5 for row in rows}
    query_bigrams = _bigrams(query)
    if not query_bigrams:
        return {_id(row): 0.5 for row in rows}

    docs = {_id(row): _bigrams(row.get("label", "")) for row in rows}
    df = Counter()
    for grams in docs.values():
        df.update(grams)
    total = max(1, len(docs))
    idf = {gram: math.log((total + 1.0) / (count + 1.0)) + 1.0 for gram, count in df.items()}
    query_weight = sum(idf.get(gram, 1.0) for gram in query_bigrams)
    if query_weight <= 0:
        return {_id(row): 0.5 for row in rows}

    q = _text(query)
    out: Dict[str, float] = {}
    for row in rows:
        key = _id(row)
        label = _text(row.get("label", ""))
        if label and (label in q or q in label):
            out[key] = 1.0
            continue
        shared = sum(idf.get(gram, 1.0) for gram in query_bigrams if gram in docs[key])
        overlap = shared / query_weight if query_weight else 0.0
        out[key] = 0.5 if overlap <= 0 else 0.5 + 0.5 * min(1.0, overlap)
    return out


def _id(row: Dict[str, Any]) -> str:
    return str(row.get("id") or row.get("belief_id") or row.get("label") or "")


def _retention(row: Dict[str, Any]) -> float:
    try:
        value = float(row.get("retention", 1.0))
    except (TypeError, ValueError):
        value = 1.0
    return max(0.0, min(1.0, value))


def _confidence(row: Dict[str, Any]) -> float:
    if "effective_confidence" in row:
        try:
            return float(row["effective_confidence"])
        except (TypeError, ValueError):
            pass
    try:
        return float(row.get("confidence", 0.5)) * _retention(row)
    except (TypeError, ValueError):
        return 0.5


def rank(
    rows: Iterable[Dict[str, Any]],
    query: str,
    *,
    query_vec: Optional[Sequence[float]] = None,
    vectors: Optional[Dict[str, Sequence[float]]] = None,
    k: int = 4,
    pinned_categories: Tuple[str, ...] = PINNED_CATEGORIES,
    cross_limit: int = 2,
) -> Dict[str, Any]:
    """Rank candidate rows.

    ``rows`` can be ``MemoryV2.state()`` views or synthetic dicts. The function
    does not mutate them.

    Returns pinned, shown, cross, hidden and counts. ``pinned`` is always kept;
    if it consumes the whole budget, ``shown`` is empty — the caller can then
    rely on the separate time-facet section rather than silently dropping the
    query. This is the same tradeoff the design doc flags in S5.
    """
    rows = list(rows)
    pinned = [row for row in rows if row.get("category") in pinned_categories]
    rest = [row for row in rows if row.get("category") not in pinned_categories]

    pinned.sort(key=lambda row: (-_confidence(row), -_retention(row), _id(row)))

    cosines: Dict[str, float] = {}
    if query_vec is not None and vectors:
        for row in rest:
            key = _id(row)
            if key in vectors:
                cosines[key] = _cosine(query_vec, vectors[key])
    vec_norm = _vector_norm(cosines)
    lex = _lexical_scores(query, rest)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in rest:
        key = _id(row)
        q = _text(query)
        label = _text(row.get("label", ""))
        if q and label and (label in q or q in label):
            sim = 1.0
        else:
            sim = 0.5 * vec_norm.get(key, 0.5) + 0.5 * lex.get(key, 0.5)
        score = (0.15 + 0.85 * sim) * (0.35 + 0.65 * _retention(row))
        scored.append((round(score, 6), row))
    scored.sort(key=lambda item: (-item[0], _id(item[1])))

    budget = max(0, int(k) - len(pinned))
    shown = [row for _, row in scored[:budget]]
    tail = [row for _, row in scored[budget:]]

    used_categories = {row.get("category") for row in pinned}
    used_categories.update(row.get("category") for row in shown)
    cross: List[Dict[str, Any]] = []
    for row in tail:
        category = row.get("category")
        if category in used_categories:
            continue
        cross.append(row)
        used_categories.add(category)
        if len(cross) >= max(0, int(cross_limit)):
            break

    cross_ids = {_id(row) for row in cross}
    hidden = [row for row in tail if _id(row) not in cross_ids]

    return {
        "query": query,
        "pinned": pinned,
        "pinned_count": len(pinned),
        "shown": shown,
        "cross": cross,
        "hidden": hidden,
        "hidden_count": len(hidden),
        "candidates": len(rows),
        "k": int(k),
    }
