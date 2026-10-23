"""One-time v1 (Chroma profile) -> v2 (SQLite/Redis beliefs) migration.

Dry-run by default. It reads Chroma raw through VectorStore.load_all(), not
PersonaManager.load(), because load() applies the old _decay_all and would
double-decay confidence before the migration.

Usage:
    python scripts/migrate_v1.py --db ./persona_data/memory.db
    python scripts/migrate_v1.py --db ./persona_data/memory.db --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, Tuple
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from personatrace.config import PersonaConfig  # noqa: E402
from personatrace.memory_v2 import CATEGORY_STABILITY, MemoryV2, SQLiteBackend  # noqa: E402
from personatrace.schema import (  # noqa: E402
    ALL_CATEGORIES, DEFAULT_TZ, STABILITY_DAYS, coerce_in, due_for_write,
    track_for,
)
from personatrace.vector_store import VectorStore  # noqa: E402

CATEGORY_STABILITY.update(STABILITY_DAYS)

TZ = DEFAULT_TZ
TARGETS = set(ALL_CATEGORIES)

# v1 category -> v2 category. Logs are intentionally skipped: they are not
# conclusions and must not enter retention/pinned logic.
CATEGORY_MAP = {
    "likes": "likes",
    "dislikes": "dislikes",
    "constraints": "constraints",
    "explicit_facts": "explicit_facts",
    "expertise": "expertise",
    "projects": "projects",
    "goals": "goals",
    "identity": "identity",
    "traits": "traits",
    "interests": "interests",
    "style_notes": "style_notes",
    "behavior_patterns": "behavior_patterns",
    "decision_style": "decision_style",
    "interpersonal": "interpersonal",
    "boundaries": "boundaries",
    "rhythms": "habit",
}

SKIP_CATEGORIES = {
    "cross_insights", "analysis_log", "observation_log", "uncertain",
    "low_confidence", "timeline", "corrections",
}


def _label(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        value = item.get("label", "")
        if isinstance(value, dict):
            value = value.get("label", "")
        return str(value).strip()
    return str(item or "").strip()


def _confidence(item: Any) -> float:
    try:
        value = float(item.get("confidence", 0.5)) if isinstance(item, dict) else 0.5
    except (TypeError, ValueError):
        value = 0.5
    if value > 1.0:
        value = value / 100.0 if value <= 100.0 else 1.0
    return max(0.0, min(1.0, value))


def _updated_ts(item: Any, fallback: float, tz: str) -> float:
    if isinstance(item, dict):
        raw = item.get("updated") or item.get("last_seen")
        if raw:
            try:
                parsed = datetime.fromisoformat(str(raw))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=ZoneInfo(tz))
                return parsed.timestamp()
            except (TypeError, ValueError):
                pass
    return fallback


def _items(data: Dict[str, Any]) -> Iterable[Tuple[str, str, Any]]:
    for track in ("work", "persona", "cross", "ephemeral"):
        for category, items in (data.get(track, {}) or {}).items():
            for item in (items or []):
                yield track, category, item


def migrate_data(
    data: Dict[str, Any],
    memory: MemoryV2,
    *,
    user_id: str = "default",
    apply: bool = False,
    now: float | None = None,
) -> Dict[str, Any]:
    """Migrate one profile dict. Pure except for optional memory.observe()."""
    now_ts = float(now if now is not None else datetime.now(ZoneInfo(TZ)).timestamp())
    report: Dict[str, Any] = {
        "seen": 0, "migrated": 0, "skipped": 0,
        "by_category": Counter(), "skipped_categories": Counter(),
    }
    for _track, category, item in _items(data):
        report["seen"] += 1
        if category in SKIP_CATEGORIES:
            report["skipped"] += 1
            report["skipped_categories"][category] += 1
            continue
        target = CATEGORY_MAP.get(category)
        label = _label(item)
        if target not in TARGETS or len(label) < 2:
            report["skipped"] += 1
            report["skipped_categories"][category] += 1
            continue

        confidence = _confidence(item)
        reasoning = ""
        evidence = ""
        due_text = None
        if isinstance(item, dict):
            reasoning = str(item.get("reasoning") or "")[:2000]
            evidence = str(item.get("evidence") or item.get("evidence_text") or "")[:4000]
            due_text = item.get("due")
            if category == "rhythms" and item.get("time_note"):
                reasoning = f"{reasoning}（节律：{item['time_note']}）"[:2000]

        draft = coerce_in({
            "category": target,
            "label": label,
            "confidence": confidence,
            "reasoning": reasoning,
            "evidence_text": evidence,
            "due": due_text,
        })
        if draft is None:
            report["skipped"] += 1
            continue

        typed = due_for_write(draft, datetime.fromtimestamp(now_ts, ZoneInfo(TZ)), TZ)
        # For a forward category with no explicit due, due_for_write already
        # parses the label. A past expression is refused there.
        due_ts = typed.due_ts if typed else None
        time_kind = typed.kind if typed else "none"
        time_raw = typed.raw if typed else (draft.due or "")

        if apply:
            memory.observe(
                draft.category, draft.label,
                track=draft.track or track_for(draft.category),
                confidence=draft.confidence,
                reasoning=draft.reasoning,
                evidence_text=draft.evidence_text,
                due_at_ts=due_ts,
                time_kind=time_kind,
                time_raw=time_raw,
                ts=_updated_ts(item, now_ts, TZ),
            )
        report["migrated"] += 1
        report["by_category"][target] += 1
    report["by_category"] = dict(report["by_category"])
    report["skipped_categories"] = dict(report["skipped_categories"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./persona_data")
    parser.add_argument("--db", default="./persona_data/memory.db")
    parser.add_argument("--user-id", default="default")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = PersonaConfig(data_dir=args.data_dir)
    store = VectorStore(config)
    data = store.load_all(args.user_id)
    if data is None:
        print("no v1 profile found; nothing to migrate")
        return 1

    memory = MemoryV2(backend=SQLiteBackend(args.db), user_id=args.user_id)
    report = migrate_data(data, memory, user_id=args.user_id, apply=args.apply)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"{'APPLIED' if args.apply else 'DRY RUN'} user={args.user_id}")
        print(f"seen={report['seen']} migrated={report['migrated']} skipped={report['skipped']}")
        print(f"by_category={report['by_category']}")
        print(f"skipped_categories={report['skipped_categories']}")
        if not args.apply:
            print("re-run with --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
