"""Candidate pooling.

The judge labels the union of every system's top-10 *visible* items for a
probe (plus optional unscoped BM25 extras, which give the judge negative and
cross-project examples). Items are keyed by a hash of the visible text, so the
same text returned by two systems is judged once and the judge cannot tell
which system produced it.

A pool is frozen once judged. If a later system (or a new view) returns items
that are not in the frozen pool, ``unjudged_items`` finds them and they are
labelled in a *supplementary* pass with the identical rubric and protocol.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .packing import RANKING_BYTES, visible
from .systems.base import TOP_K


def pool_candidates(ranked_by_system: Mapping[str, Sequence[Mapping[str, Any]]], *,
                    extra: Sequence[Mapping[str, Any]] = (), top_k: int = TOP_K,
                    byte_limit: int = RANKING_BYTES) -> list[dict[str, Any]]:
    """Deduplicated ``{item_id, text}`` pool for one probe; no system names or scores are kept."""
    pool: dict[str, dict[str, Any]] = {}
    for system in sorted(ranked_by_system):
        for item in visible(list(ranked_by_system[system])[:top_k], byte_limit):
            pool.setdefault(item["item_id"], {"item_id": item["item_id"], "text": item["text"]})
    for item in visible(list(extra), byte_limit):
        pool.setdefault(item["item_id"], {"item_id": item["item_id"], "text": item["text"]})
    return list(pool.values())


def judged_ids(judgment: Mapping[str, Any] | None) -> set[str]:
    if not judgment or judgment.get("status", "valid") != "valid":
        return set()
    return {str(item["item_id"]) for item in judgment.get("items", [])}


def unjudged_items(ranked: Sequence[Mapping[str, Any]], known_ids: set[str]) -> list[dict[str, Any]]:
    """Visible items of one ranked list that still need a supplementary label."""
    return [{"item_id": item["item_id"], "text": item["text"]} for item in ranked if item["item_id"] not in known_ids]


def merge_judgments(primary: Mapping[str, Any] | None, supplements: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Combine a canonical judgment with supplementary labels for new items.

    ``memory_needed`` and the update chain always come from the canonical
    judgment; supplementary passes only add labels for previously unseen items
    (an item already labelled keeps its canonical label).
    """
    if not primary or primary.get("status", "valid") != "valid":
        return None
    labels = {item["item_id"]: dict(item) for item in primary["items"]}
    for supplement in supplements:
        if not supplement or supplement.get("status", "valid") != "valid":
            continue
        for item in supplement["items"]:
            labels.setdefault(item["item_id"], dict(item))
    return {**primary, "items": list(labels.values())}
