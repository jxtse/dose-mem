"""System adapter interface.

A memory system answers ``retrieve(probe, epoch)`` with up to ten ranked
items ``{item_id, text, score, source?}``. ``probe`` is ``Probe.system_view()``:
it never contains hindsight, split, session or judge information. ``epoch``
names the only snapshot the system may use for this probe.

Systems that keep state (an index, a vector store) implement ``prepare(corpus)``
which receives every snapshot in time order before any probe is asked.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..data import Corpus

TOP_K = 10


@runtime_checkable
class MemorySystem(Protocol):
    name: str

    def prepare(self, corpus: Corpus) -> None: ...

    def retrieve(self, probe: dict[str, Any], epoch: str) -> list[dict[str, Any]]: ...


def check_items(items: list[dict[str, Any]], limit: int = TOP_K) -> list[dict[str, Any]]:
    """Normalize adapter output: drop empty texts, require finite scores, keep rank order."""
    import math

    output = []
    for item in items:
        text = str(item.get("text") or "")
        if not text.strip():
            continue
        score = float(item.get("score", 0.0) or 0.0)
        if not math.isfinite(score):
            raise ValueError("nonfinite retrieval score")
        row = {"item_id": str(item["item_id"]), "text": text, "score": score}
        if item.get("source") is not None:
            row["source"] = str(item["source"])
        output.append(row)
    return output[:limit]
