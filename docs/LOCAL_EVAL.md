# Local regression fixture: 200 beliefs × 50 queries

This fixture is a deterministic regression gate, not a benchmark. It is kept
here so the main README can show the external LongMemEval result first.

## Retrieval arms

```text
method                         recall@3     MRR
v2 native (query, no pinned)      66.0%   0.707
v2 native (pinned)                52.0%   0.290
outer lex                         61.0%   0.637
outer + bge                       74.0%   0.770
outer + bge + time facet          76.0%   0.760
```

`v2 native (query)` is the kernel's own lexical channel with no outer layer.
`pinned` is capped at half the budget, so constraints and boundaries cannot
consume the whole query budget. The time facet adds recall but can lower MRR by
promoting a correct due item above an equally correct semantic item.

## Commands

```bash
python scripts/eval_recall.py
python scripts/eval_recall.py --no-embed
python scripts/eval_recall.py --no-embed --fail-under 0.60 --gate-method "outer lex"
```

The last command is the CI gate. It needs no embedding model, so it is stable
on Windows and Linux runners.

## Files

```text
scripts/eval_data_200.py
scripts/eval_recall.py
test_eval_dataset.py
```

## Why it stays small

The 50 queries are enough to catch a ranking regression and to keep CI fast.
They are not enough to support a claim about general accuracy. Use the external
LongMemEval result in [LONGMEMEVAL.md](LONGMEMEVAL.md) for that claim.
