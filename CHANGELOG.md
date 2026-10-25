# Changelog

## 2.0.2

- Fix CI: add `python-dateutil` (weekly recurrence) and Windows `tzdata`
  (ZoneInfo) to runtime dependencies.
- Use one wall-clock source for `MergedMemory` defaults and `MemoryV2.observe`;
  this fixes a Windows/Python 3.10 case where a just-written belief was hidden
  by a few milliseconds of `datetime.now()` vs `time.time()` skew.
- Cap pinned categories at half the top-k budget (`max(1, k // 2)`) instead of
  letting constraints/boundaries consume every slot. Kernel `v2 native
  (pinned)` goes from 2.0% to 52.0% recall@3 on the 200/50 fixture.
- Parse explicit Chinese years (`2029年3月1日`) without dropping the year or
  applying the future rollover; `每年12月25日` is now typed as `recurring`.
- Make the MCP timeout test tolerate cold-runner imports.

## 2.0.1

- Trim the retired v1 configuration surface (`mem0`, rerank, decay, capacity
  fields); keep only LLM, embedding dimensions and the migration `data_dir`.

## 2.0.0

- **Breaking:** remove the runtime v1 pipeline (`PersonaManager`, extractor,
  formatter, file loader, rhythm, graph memory, intent extractor, similarity,
  models, temporal context, utils) and their tests/exports.
- Keep a minimal migration shim: `PersonaConfig`, `VectorStore` and
  `scripts/migrate_v1.py` for old Chroma profiles.
- Version target is Python 3.10+.

## 1.6.5

- MCP worker isolation with an enforceable 8s deadline, batch handling and
  UTF-8 stdio; one user namespace for both tool families.
- Bi-temporal belief history, `as_of` snapshots, `known_as_of`, revival,
  explicit due clearing, CAS and compare-and-delete.
- Visible backend fallback (`last_backend_warning`, doctor exit code); one
  durable store for beliefs/intentions/episodes/routines by default.
- Kernel lexical retrieval, pinned-category unification, episode search beyond
  the recent 500, evidence keys scoped per belief, safer dedup, CI and ruff
  gate.
