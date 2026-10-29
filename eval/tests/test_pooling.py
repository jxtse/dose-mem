from __future__ import annotations

from conftest import item
from titrate_eval.packing import visible, visible_item_id
from titrate_eval.pooling import judged_ids, merge_judgments, pool_candidates, unjudged_items


def test_pool_is_union_of_visible_top10_deduplicated_and_anonymous() -> None:
    a = [item(f"a{i}", f"text {i}", 1 - i / 100) for i in range(12)]
    b = [item("b0", "text 0", 0.9), item("b1", "only b", 0.8)]
    pool = pool_candidates({"sys-a": a, "sys-b": b})
    texts = [row["text"] for row in pool]
    assert len(texts) == len(set(texts)) == 11  # a's top-10 + "only b"; "text 0" deduplicated
    assert "text 11" not in texts and "text 10" not in texts
    assert all(set(row) == {"item_id", "text"} for row in pool)
    assert all(row["item_id"] == visible_item_id(row["text"]) for row in pool)


def test_pool_includes_optional_extra_items() -> None:
    pool = pool_candidates({"s": [item("a", "one")]}, extra=[item("x", "cross-project negative")])
    assert [row["text"] for row in pool] == ["one", "cross-project negative"]


def test_unjudged_items_feed_supplementary_judging_and_merge_keeps_canonical_labels() -> None:
    ranked = visible([item("a", "known"), item("b", "new item")])
    known = {visible_item_id("known")}
    fresh = unjudged_items(ranked, known)
    assert [row["text"] for row in fresh] == ["new item"]
    primary = {"probe_id": "p", "status": "valid", "memory_needed": True,
               "items": [{"item_id": visible_item_id("known"), "label": "necessary"}]}
    supplement = {"probe_id": "p", "status": "valid", "memory_needed": False,
                  "items": [{"item_id": visible_item_id("new item"), "label": "harmful"},
                            {"item_id": visible_item_id("known"), "label": "irrelevant"}]}
    merged = merge_judgments(primary, [supplement])
    labels = {row["item_id"]: row["label"] for row in merged["items"]}
    assert labels[visible_item_id("known")] == "necessary"
    assert labels[visible_item_id("new item")] == "harmful"
    assert merged["memory_needed"] is True  # never taken from a supplement
    assert merge_judgments(None, [supplement]) is None
    assert judged_ids(merged) == set(labels)
