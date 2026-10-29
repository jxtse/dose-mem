from __future__ import annotations

import pytest

from conftest import item
from dose_mem_eval.packing import (DEFAULT_BUDGETS, RANKING_BYTES, budget_items, cap_utf8, injected_bytes,
                                  parse_budgets, visible, visible_item_id)


def test_utf8_budget_counts_separator_and_never_scores_hidden_fragment() -> None:
    ranked = [item("a", "中文", 0.8), item("b", "abc", 0.7), item("c", "x", 0.6)]
    assert [value["item_id"] for value in budget_items(ranked, 6)] == ["a"]
    # "中文" is 6 bytes, separator 2, "abc" 3 -> 11
    assert [value["item_id"] for value in budget_items(ranked, 10)] == ["a"]
    assert [value["item_id"] for value in budget_items(ranked, 11)] == ["a", "b"]
    assert not budget_items(ranked, 5)
    assert budget_items(ranked, 11, threshold=0.75) == [ranked[0]]


def test_packing_stops_at_first_misfit_and_never_promotes_lower_ranks() -> None:
    ranked = [item("a", "x" * 10), item("b", "y" * 50), item("c", "z")]
    # "c" would fit in the leftover space but must not be promoted past "b".
    assert [value["item_id"] for value in budget_items(ranked, 20)] == ["a"]


def test_threshold_skips_low_scores_but_keeps_rank_order() -> None:
    ranked = [item("a", "aa", 0.9), item("b", "bb", 0.1), item("c", "cc", 0.8)]
    assert [value["item_id"] for value in budget_items(ranked, 100, threshold=0.5)] == ["a", "c"]
    with pytest.raises(ValueError):
        budget_items([{"item_id": "a", "text": "x"}], 10, threshold=0.5)


def test_packing_rejects_duplicates_and_empty_items() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        budget_items([item("a"), item("a")], 100)
    with pytest.raises(ValueError):
        budget_items([{"item_id": "a", "text": ""}], 100)


def test_visible_caps_on_character_boundary_and_rekeys_by_text() -> None:
    long = "中" * 400  # 1200 bytes
    view = visible([item("raw", long)])[0]
    assert len(view["text"].encode()) <= RANKING_BYTES
    assert view["text"].endswith("\u2026")
    assert view["visible_truncated"] is True
    assert view["raw_item_id"] == "raw"
    assert view["item_id"] == visible_item_id(view["text"])
    # Same visible text from two systems has the same identity; duplicates collapse.
    both = visible([item("x", "same text"), item("y", "same text")])
    assert len(both) == 1
    assert cap_utf8("short") == "short"


def test_injected_bytes_and_budget_overrides() -> None:
    assert injected_bytes([item("a", "ab"), item("b", "cd")]) == 6
    assert DEFAULT_BUDGETS == {"assistant": 1200, "working": 600}
    assert parse_budgets(["working=900"]) == {"assistant": 1200, "working": 900}
    with pytest.raises(ValueError):
        parse_budgets(["unknown=5"])
