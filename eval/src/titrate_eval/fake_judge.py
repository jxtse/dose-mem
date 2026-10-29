"""Deterministic offline fake judge — for tests and the example ONLY.

It speaks the exact Chat Completions wire format and the span-anchored
protocol, so the whole pipeline (validation, alias restoration, scoring) is
exercised without any network call. Its labels come from a crude token
overlap heuristic and mean nothing; never report results produced with it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .judging import EVIDENCE_BEGIN, EVIDENCE_END

STOP = frozenset("the a an and or of to in on for is are was were be it this that with what which do does did we i you my our".split())


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_-]+", text.lower()) if len(w) > 2 and w not in STOP}


def _first_span(source: Mapping[str, Any]) -> str | None:
    segments = source.get("segments") or []
    return segments[0]["span_id"] if segments else None


def fake_judge_answer(payload: Mapping[str, Any], *, flavor: str = "primary") -> dict[str, Any]:
    """Build a valid judgment JSON for a request payload. ``flavor='audit'`` flips some labels."""
    content = payload["messages"][-1]["content"]
    evidence = json.loads(content[len(EVIDENCE_BEGIN):-len(EVIDENCE_END)])
    query = "".join(segment["text"] for segment in evidence["query"]["segments"])
    hindsight = " ".join(segment["text"] for turn in evidence["hindsight"] for segment in turn["segments"])
    project = "".join(segment["text"] for segment in evidence["project_context"]["segments"])
    need_words = _words(query) | _words(hindsight)
    memory_needed = bool(re.search(r"\b(remember|recall|last time|again|previous|we decided|what was)\b", query.lower()))
    items = []
    for candidate in evidence["candidates"]:
        text = "".join(segment["text"] for segment in candidate["segments"])
        overlap = len(_words(text) & need_words)
        lowered = text.lower()
        if "superseded" in lowered or "deprecated" in lowered or (project and "project:" in lowered and project.lower() not in lowered):
            label = "harmful"
        elif overlap >= 3 and memory_needed:
            label = "necessary"
        elif overlap >= 2:
            label = "helpful"
        else:
            label = "irrelevant"
        if flavor == "audit" and label == "helpful" and overlap == 2:
            label = "irrelevant"
        items.append({"item_id": candidate["item_id"], "label": label, "rationale": f"token overlap {overlap}",
                      "anchors": [{"source": candidate["item_id"], "span_id": _first_span(candidate)}]})
    return {"memory_needed": "yes" if memory_needed else "no", "rationale": "keyword heuristic (fake judge)",
            "anchors": [{"source": "query", "span_id": _first_span(evidence["query"])}],
            "items": items,
            "update_chain": {"established": False, "newer_item_ids": [], "stale_item_ids": [], "anchors": []}}


def fake_transport(flavor: str = "primary"):
    """A drop-in ``Transport`` returning a Chat Completions body."""
    def send(payload: Mapping[str, Any]) -> tuple[int, Any]:
        answer = fake_judge_answer(payload, flavor=flavor)
        return 200, {"id": "fake", "object": "chat.completion", "model": payload.get("model"),
                     "choices": [{"index": 0, "finish_reason": "stop",
                                  "message": {"role": "assistant", "content": json.dumps(answer)}}]}
    return send
