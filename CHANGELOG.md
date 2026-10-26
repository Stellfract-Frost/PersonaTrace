# Changelog

## 2.1.1

- `晚上12点` now resolves to midnight at the end of the named day
  (`次日 00:00`), not noon. `中午12点` stays noon; `晚上12点半` keeps the
  half-hour and rolls the date.
- Bare three-letter English weekdays are no longer date cues in free text.
  `备考 SAT` returns `None` instead of next Saturday; explicit forms such as
  `next Sat` / `on Sat` still work. Full weekday names still parse.
- Add regression coverage for both cases.

## 2.1.0

- Episode fusion no longer concatenates the two retrieval arms. Each arm is
  min-max scaled and blended (vector 1.0 / lexical 0.5); the old concatenation
  made the lexical arm unreachable whenever the vector arm filled the limit.
  `merged` recall@3 75.6% → 79.9%, `all@10` 88.0% → 93.2% on the 133-question
  LongMemEval subset. Pure RRF was measured and rejected: it loses the score
  magnitude that session aggregation relies on (recall@3 79.9% → 60.1%).
- Optional cross-encoder reranking inside the store: `EpisodeStore.search(...,
  reranker=, rerank_top=30)`, `MergedMemory(reranker=...)` and
  `MergedMemory.episodes_search(..., reranker=)`. Reranked results report
  rank-monotone scores so downstream aggregation cannot invert the order.
  With reranking: recall@3 85.5%, `all@10` 94.7%, MRR 0.925.
- Intentions now expire for real: `IntentStore.sweep()` writes `expired` back to
  rows whose grace window has passed, `select_agenda()` sweeps before reading,
  and `list(..., sweep=True)` is available. Recurring series are skipped.
- Unparsed or past due dates are no longer dropped silently:
  `_write_intentions` returns `(written, dropped)` and logs each drop,
  `write()`/`remember()` report a real `dropped_intentions` count.
- Time parsing covers hour-level offsets and more English forms: `两小时后`,
  `半小时后`, `in 2 hours`, `2 hours later`, `after 2 hours`, `half an hour`,
  `3 days later`, `after 3 days`.
- `LLMClient.chat()` accepts `thinking="enabled"|"disabled"`; the default
  (`None`) leaves the provider setting untouched. Reasoning models bill thinking
  tokens against `max_tokens`, so an undersized budget returns empty content.
- `VectorStore` (retired v1 layer) is no longer exported from `__all__`; it stays
  importable for `scripts/migrate_v1.py`.
- Evaluation harness: `--session-chars`, `--context-k`, `--assistant-chars`,
  `--workers`, `--no-thinking`, `--date-hint`, `--ids`; per-stage token
  accounting and incremental result checkpoints.
- LongMemEval_s 500-question result: 431/500 = 86.2% LLM self-judge; rule-judge
  lower bound 78.4%; about ¥5.50. The 133 temporal subset is 119/133 = 89.5%.

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
