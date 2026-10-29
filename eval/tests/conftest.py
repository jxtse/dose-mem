"""Shared fixtures: tiny synthetic runs, judgments and judge responses."""

from __future__ import annotations

from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tiny"


def item(name: str = "a", text: str = "Use the newer setting.", score: float = 0.7) -> dict:
    return {"item_id": name, "text": text, "score": score}


def probe(name: str = "p1", session: str = "s1") -> dict:
    return {"probe_id": name, "session_id": session, "user_turn": "Use my previous setting.",
            "previous_turn": "A previous task.",
            "hindsight": [{"role": "user", "text": "That setting is old; use the newer setting."}],
            "client_group": "working", "split": "test", "stratum": "memory-needed", "epoch": "e1",
            "project_hint": None}


def run(name: str = "p1", session: str = "s1", **kwargs) -> dict:
    return {**probe(name, session), "system": "dense", "regime": "fixed", "status": "ok",
            "injected_items": [item()], "ranked_items": [item()], "latency_ms": 20.0, **kwargs}


def judgment(name: str = "p1", needed: bool = True, labels: dict | None = None, **kwargs) -> dict:
    return {"probe_id": name, "status": "valid", "memory_needed": needed,
            "items": [{"item_id": key, "label": label}
                      for key, label in (labels if labels is not None else {"a": "necessary"}).items()],
            "update_chain": {"established": False, "newer_item_ids": [], "stale_item_ids": []}, **kwargs}


def response_for(request: dict, label: str = "necessary") -> dict:
    catalogue = request["span_catalogue"]
    return {"memory_needed": "yes", "rationale": "The user explicitly refers to previous settings.",
            "anchors": [{"source": "query", "span_id": next(iter(catalogue["query"]))}],
            "items": [{"item_id": value["item_id"], "label": label, "rationale": "Contains the required setting.",
                       "anchors": [{"source": value["item_id"], "span_id": next(iter(catalogue[value["item_id"]]))}]}
                      for value in request["packet"]["candidates"]],
            "update_chain": {"established": False, "newer_item_ids": [], "stale_item_ids": [], "anchors": []}}


@pytest.fixture
def example_dir() -> Path:
    return EXAMPLE
