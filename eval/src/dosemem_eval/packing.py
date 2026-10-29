"""Equal-budget packing and the per-item ranking view.

Every system is compared under the same injected byte budget. Two rules keep
this fair:

* ``visible``: each returned item is shown as at most ``RANKING_BYTES`` UTF-8
  bytes (cut on a character boundary). Labels attach to exactly that visible
  text, identified by a content hash, so no system is credited for text the
  agent would never see.
* ``budget_items``: an ordered prefix of whole visible items is packed until
  the next one does not fit. Packing stops there; lower ranks are never
  promoted into the leftover space and nothing is silently truncated.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

RANKING_BYTES = 480
DEFAULT_BUDGETS: dict[str, int] = {"assistant": 1200, "working": 600}
SEPARATOR = "\n\n"
ELLIPSIS = "\u2026"


def visible_item_id(text: str) -> str:
    """Content identity permits pool union without leaking source-system IDs."""
    return "item_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def cap_utf8(text: str, byte_limit: int = RANKING_BYTES) -> str:
    """Return ``text`` or a prefix ending in an ellipsis, at most ``byte_limit`` bytes."""
    encoded = text.encode("utf-8")
    if len(encoded) <= byte_limit:
        return text
    room = byte_limit - len(ELLIPSIS.encode("utf-8"))
    if room <= 0:
        raise ValueError("byte limit too small for a visible view")
    return encoded[:room].decode("utf-8", errors="ignore") + ELLIPSIS


def visible(items: Sequence[Mapping[str, Any]], byte_limit: int = RANKING_BYTES) -> list[dict[str, Any]]:
    """Ranking view: cap each item, re-key it by visible text and drop duplicate views."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        raw = str(item["text"])
        if not raw.strip():
            continue
        text = cap_utf8(raw, byte_limit)
        identity = visible_item_id(text)
        if identity in seen:
            continue
        seen.add(identity)
        result.append({**item, "raw_item_id": item.get("item_id"), "item_id": identity, "text": text,
                       "visible_truncated": text != raw})
    return result


def budget_items(items: Sequence[Mapping[str, Any]], budget: int, *, threshold: float | None = None,
                 separator: str = SEPARATOR) -> list[dict[str, Any]]:
    """Pack an ordered prefix of whole visible items; never credit hidden text.

    A fragment must already be a separately identified/judged candidate. When
    the next item cannot fit, packing stops rather than promoting a lower rank.
    Items scoring below ``threshold`` are skipped (they are not injected).
    """
    if budget < 0:
        raise ValueError("negative byte budget")
    if threshold is not None and not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    selected: list[dict[str, Any]] = []
    used = 0
    seen: set[str] = set()
    for original in items:
        item = dict(original)
        item_id, text = item.get("item_id"), item.get("text")
        if not isinstance(item_id, str) or not item_id or not isinstance(text, str) or not text:
            raise ValueError("items require nonempty item_id and text")
        if item_id in seen:
            raise ValueError("duplicate item ID")
        seen.add(item_id)
        if threshold is not None:
            score = item.get("score")
            if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score):
                raise ValueError("threshold selection requires finite score")
            if score < threshold:
                continue
        cost = len(text.encode("utf-8")) + (len(separator.encode("utf-8")) if selected else 0)
        if used + cost > budget:
            break
        selected.append(item)
        used += cost
    return selected


def injected_bytes(items: Sequence[Mapping[str, Any]], separator: str = SEPARATOR) -> int:
    return len(separator.join(str(item["text"]) for item in items).encode("utf-8"))


def parse_budgets(values: Sequence[str] | None) -> dict[str, int]:
    """Parse ``["assistant=1200", "working=600"]`` overrides onto the defaults."""
    budgets = dict(DEFAULT_BUDGETS)
    for value in values or []:
        group, _, number = value.partition("=")
        if group not in DEFAULT_BUDGETS or not number.isdigit():
            raise ValueError(f"invalid budget {value!r}; expected assistant=N or working=N")
        budgets[group] = int(number)
    return budgets
