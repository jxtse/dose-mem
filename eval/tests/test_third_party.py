"""Third-party adapters, exercised only against mocked SDK objects (no installs, no network)."""

from __future__ import annotations

import builtins

import pytest

from dose_mem_eval.data import Corpus, Document
from dose_mem_eval.systems.mem0 import NEVER_REMOVED, Mem0System, split_text
from dose_mem_eval.systems.openviking import OpenVikingSystem


def doc(identity: str, text: str, project: str = "global") -> Document:
    return Document.from_dict({"id": identity, "path": f"notes/{identity}.md", "project": project, "text": text})


def corpus() -> Corpus:
    return Corpus({"2026-01-05": [doc("a", "alpha v1", "proj"), doc("g", "global note")],
                   "2026-01-12": [doc("a", "alpha v2", "proj"), doc("g", "global note")]},
                  ["2026-01-05", "2026-01-12"])


# ----------------------------------------------------------------------------- OpenViking
class FakeVikingClient:
    def __init__(self):
        self.added = []
        self.searches = []

    def add_resource(self, path, to=None, wait=False, options=None):
        from pathlib import Path

        files = sorted(str(p.relative_to(path)) for p in Path(path).rglob("*.md"))
        self.added.append({"to": to, "wait": wait, "options": options, "files": files})
        return {"status": "accepted", "task_id": "t"}

    def search_context(self, query, limit=10, options=None, **kwargs):
        assert "target_uri" not in kwargs  # mode=context rejects target_uri; scope is a filter
        self.searches.append({"query": query, "limit": limit, "options": options})
        return {"entries": [{"uri": "viking://resources/dose-mem/2026-01-12/proj/notes/a", "text": "alpha v2", "score": 0.8},
                            {"uri": "viking://resources/dose-mem/2026-01-12/global/x", "text": "", "abstract": "abs", "score": 0.5},
                            {"uri": "viking://empty", "text": "", "score": 0.1}],
                "stats": {}}


def test_openviking_ingests_each_epoch_natively_under_its_own_root(tmp_path) -> None:
    client = FakeVikingClient()
    system = OpenVikingSystem(client, workdir=tmp_path)
    system.prepare(corpus())
    assert [call["to"] for call in client.added] == ["viking://resources/dose-mem/2026-01-05",
                                                     "viking://resources/dose-mem/2026-01-12"]
    assert client.added[0]["options"]["processing_mode"] == "semantic_and_vectors"
    assert client.added[0]["files"] == ["global/notes/g.md", "proj/notes/a.md"]


def test_openviking_queries_search_context_with_uri_scope_filter(tmp_path) -> None:
    client = FakeVikingClient()
    system = OpenVikingSystem(client, workdir=tmp_path, max_tokens=900)
    system.prepare(corpus())
    rows = system.retrieve({"user_turn": "what is alpha?\nverbatim", "project_hint": "proj"}, "2026-01-12")
    call = client.searches[-1]
    assert call["query"] == "what is alpha?\nverbatim"
    assert call["options"]["max_tokens"] == 900 and call["options"]["score_threshold"] == 0
    native_filter = call["options"]["filter"]
    assert native_filter["op"] == "or"
    assert [c["conds"] for c in native_filter["conds"]] == [["viking://resources/dose-mem/2026-01-12/proj"],
                                                            ["viking://resources/dose-mem/2026-01-12/global"]]
    assert [row["text"] for row in rows] == ["alpha v2", "abs"]
    assert all(row["item_id"].startswith("openviking:") for row in rows)
    system.retrieve({"user_turn": "q", "project_hint": None}, "2026-01-05")
    assert client.searches[-1]["options"]["filter"] == {"op": "must", "field": "uri", "para": "-d=-1",
                                                        "conds": ["viking://resources/dose-mem/2026-01-05"]}


def test_openviking_sdk_is_optional(monkeypatch) -> None:
    from dose_mem_eval.systems import openviking

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("openviking"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match=r"dose_mem_eval\[openviking\]"):
        openviking.make_client("http://example.invalid")


# ----------------------------------------------------------------------------- mem0
class FakePoint:
    def __init__(self, payload):
        self.payload = payload


class FakeVectorStore:
    def __init__(self):
        self.payloads: dict[str, dict] = {}

    def get(self, vector_id):
        return FakePoint(dict(self.payloads[vector_id]))

    def update(self, vector_id, vector=None, payload=None):
        self.payloads[vector_id] = payload


class FakeMemory:
    """Mimics mem0 Memory.add(infer=True) / search(filters=...) semantics in memory."""

    def __init__(self):
        self.vector_store = FakeVectorStore()
        self.calls = []
        self.counter = 0

    def add(self, messages, *, user_id, infer, metadata):
        self.calls.append({"messages": messages, "user_id": user_id, "infer": infer, "metadata": dict(metadata)})
        self.counter += 1
        memory_id = f"m{self.counter}"
        text = "fact: " + " ".join(m["content"] for m in messages)
        self.vector_store.payloads[memory_id] = {**metadata, "data": text, "user_id": user_id}
        return {"results": [{"id": memory_id, "memory": text, "event": "ADD"}]}

    def search(self, query, *, filters, top_k, threshold, rerank):
        self.last_search = {"query": query, "filters": filters, "top_k": top_k, "threshold": threshold, "rerank": rerank}
        rows = []
        for memory_id, payload in self.vector_store.payloads.items():
            if payload["epoch_idx"] > filters["epoch_idx"]["lte"] or payload["removed_idx"] <= filters["removed_idx"]["gt"]:
                continue
            if "project" in filters and payload["project"] not in filters["project"]["in"]:
                continue
            rows.append({"id": memory_id, "memory": payload["data"], "score": 0.5, "metadata": payload})
        return {"results": rows[:top_k]}


def test_mem0_ingests_in_time_order_with_infer_and_lifetime_metadata() -> None:
    memory = FakeMemory()
    system = Mem0System(memory)
    system.prepare(corpus())
    assert all(call["infer"] is True for call in memory.calls)
    epochs = [call["metadata"]["epoch_idx"] for call in memory.calls]
    assert epochs == sorted(epochs) == [0, 0, 1]  # unchanged "g" is not re-added
    assert memory.calls[0]["metadata"]["removed_idx"] == NEVER_REMOVED
    # alpha v1 was retired at snapshot 1 when its source changed.
    retired = [p for p in memory.vector_store.payloads.values() if p["source_id"] == "a" and p["epoch_idx"] == 0]
    assert retired[0]["removed_idx"] == 1


def test_mem0_query_filters_epoch_window_and_project_scope() -> None:
    memory = FakeMemory()
    system = Mem0System(memory)
    system.prepare(corpus())
    old = system.retrieve({"user_turn": "alpha?", "project_hint": "proj"}, "2026-01-05")
    new = system.retrieve({"user_turn": "alpha?", "project_hint": "proj"}, "2026-01-12")
    assert memory.last_search["filters"]["epoch_idx"] == {"lte": 1}
    assert memory.last_search["filters"]["removed_idx"] == {"gt": 1}
    assert memory.last_search["filters"]["project"] == {"in": ["proj", "global"]}
    assert memory.last_search["rerank"] is False and memory.last_search["query"] == "alpha?"
    assert any("alpha v1" in row["text"] for row in old) and not any("alpha v2" in row["text"] for row in old)
    assert any("alpha v2" in row["text"] for row in new) and not any("alpha v1" in row["text"] for row in new)
    system.retrieve({"user_turn": "x", "project_hint": None}, "2026-01-12")
    assert "project" not in memory.last_search["filters"]


def test_mem0_failed_add_is_retried_once_then_recorded() -> None:
    memory = FakeMemory()
    attempts = []

    def flaky(messages, **kwargs):
        attempts.append(1)
        raise RuntimeError("provider down")

    memory.add = flaky
    system = Mem0System(memory, retry_delay=0)
    system.ingest_epoch(Corpus({"e": [doc("a", "x")]}, ["e"]), "e")
    assert len(attempts) == 2 and system.failures[0]["id"] == "a"


def test_mem0_split_is_lossless() -> None:
    text = "line\n" * 4000
    parts = split_text(text, 1000)
    assert "".join(parts) == text and all(len(p.encode()) <= 1000 for p in parts)
