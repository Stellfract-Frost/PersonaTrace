"""P0 tests for the pure outer ranker."""

from personatrace.recall import rank

ROWS = [
    {"id": "pin1", "category": "constraints", "label": "对花生过敏", "retention": 0.9, "effective_confidence": 0.80},
    {"id": "pin2", "category": "identity", "label": "名字叫小陌", "retention": 0.85, "effective_confidence": 0.75},
    {"id": "c", "category": "goals", "label": "周五要演示", "retention": 0.90, "effective_confidence": 0.70},
    {"id": "d", "category": "goals", "label": "下周三交周报", "retention": 0.95, "effective_confidence": 0.80},
    {"id": "e", "category": "state", "label": "这周比较累", "retention": 0.80, "effective_confidence": 0.50},
    {"id": "f", "category": "likes", "label": "喜欢猫", "retention": 0.70, "effective_confidence": 0.60},
    {"id": "g", "category": "projects", "label": "在赶 agent 项目", "retention": 0.85, "effective_confidence": 0.65},
]


def test_pinned_always_kept_and_budget_accounted():
    result = rank(ROWS, "我那个 deadline 是几号", k=4)
    assert [r["id"] for r in result["pinned"]] == ["pin1", "pin2"]
    assert result["pinned_count"] == 2
    assert len(result["shown"]) == 2  # k - pinned
    assert result["hidden_count"] >= 0


def test_pinned_eats_budget_does_not_crash():
    result = rank(ROWS, "住哪儿", k=2)
    assert len(result["pinned"]) == 2
    assert result["shown"] == []
    # tail = 5; cross takes at most 2; hidden = tail - cross.
    assert result["hidden_count"] == 3


def test_vector_channel_puts_deadline_on_top():
    result = rank(
        ROWS,
        "我那个 deadline 是几号",
        k=4,
        query_vec=[1.0, 0.0],
        vectors={
            "c": [1.0, 0.0],     # 周五要演示
            "d": [0.0, 1.0],     # 下周三交周报
            "e": [0.0, 1.0],
            "f": [0.1, 0.9],
            "g": [0.2, 0.8],
        },
    )
    assert result["shown"][0]["id"] == "c"


def test_exact_label_substring_wins():
    result = rank(ROWS, "周五要演示是什么时候", k=4)
    assert result["shown"][0]["id"] == "c"


def test_cross_categories_are_not_duplicates():
    result = rank(ROWS, "随便问问", k=4)
    cross_categories = [row["category"] for row in result["cross"]]
    assert len(cross_categories) == len(set(cross_categories))
    assert len(result["cross"]) <= 2


def test_no_vectors_still_returns_stable_order():
    first = rank(ROWS, "agent 项目", k=4)
    second = rank(ROWS, "agent 项目", k=4)
    assert [r["id"] for r in first["shown"]] == [r["id"] for r in second["shown"]]
    assert first["hidden_count"] == second["hidden_count"]


def test_empty_query_is_neutral():
    result = rank(ROWS, "", k=4)
    assert len(result["shown"]) == 2
    assert result["shown"][0]["id"] in {"c", "d", "e", "f", "g"}
