"""Contract tests for the 200-belief / 50-query evaluation fixture."""

from collections import Counter

from scripts.eval_data_200 import BELIEFS, LABEL_SET, QUERIES, validate


def test_fixture_contract():
    validate()
    assert len(BELIEFS) == 200
    assert len(QUERIES) == 50
    assert len(LABEL_SET) == 200
    assert all(expected <= LABEL_SET for _query, expected in QUERIES)


def test_category_counts_cover_the_contract():
    counts = Counter(category for category, _label, _due, _kind, _raw in BELIEFS)
    assert counts["identity"] == 10
    assert counts["goals"] == 15
    assert counts["constraints"] == 20
    assert sum(counts.values()) == 200


def test_eval_rows_are_materialized():
    from scripts.eval_recall import build_rows

    rows = build_rows()
    assert len(rows) == 200
    assert len({row["label"] for row in rows}) == 200
    assert any(row.get("time_kind") == "due" for row in rows)
