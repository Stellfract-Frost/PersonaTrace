# Upgrading to 2.x

2.0 removes the old runtime v1 pipeline. This is a breaking API change; the
merged kernel is the only write/read path.

## Removed imports

These are no longer importable from `personatrace`:

- `PersonaManager`
- `PersonaExtractor`, `PersonaFormatter`
- `GraphStore`, `GraphEdge`
- `IntentExtractor`
- `SimilarityCalculator`, `RhythmEngine`
- `PersonaItem`, `ProfileData`
- `WORK_CATEGORIES`, `PERSONA_CATEGORIES`, `CROSS_CATEGORIES`,
  `EPHEMERAL_CATEGORIES`

Use:

```python
from personatrace import MergedMemory, MemoryV2, IntentStore, EpisodeStore
```

## Old Chroma profiles

The migration shim is still shipped:

```bash
python scripts/migrate_v1.py --db ./personatrace.db          # dry run
python scripts/migrate_v1.py --db ./personatrace.db --apply
```

It uses `VectorStore.load_all()` to read the old `persona_items` collection.
Install the `vector` extra if Chroma is not already present. Run the migration
with the last 1.x release if you need to inspect old v1 behavior first; 2.x
does not expose the v1 runtime.

## Behavior changes worth checking

- `MergedMemory(user_id=...)` without `sqlite_path` now keeps beliefs,
  intentions, episodes and routines in one durable database. `PERSONA_DB=off`
  is the explicit in-memory test mode.
- `state(as_of=T)` is a transaction-time snapshot. Use
  `state(as_of=T, known_as_of=now)` to ask what the current store knows about
  T, including a validity interval that has since been revived.
- `memory_observe` / `observe` preserve typed-time fields when omitted; pass
  `due_at_ts=None`, `time_kind="none"`, `time_raw=""` to clear them.
- Pinned categories are capped at `max(1, k // 2)` so they cannot consume the
  whole retrieval budget.
- `parse_due` returns `kind="recurring"` for habits instead of `None`; callers
  that write deadlines still refuse it.
- SQLite fallback is logged and `personatrace-doctor` exits non-zero for an
  unexpected fallback; `PERSONA_DB=off` remains a normal test mode.

## Dependency changes

`python-dateutil` is required for weekly recurrence, and Windows installs
`tzdata` so `ZoneInfo("Asia/Shanghai")` works without a system IANA database.
