"""mem0 adapter: native ``add(infer=True)`` extraction with epoch-filtered search.

Why "native" matters: an earlier round of the benchmark this kit came from
stored raw text in mem0 with ``infer=False`` and so measured a plain vector
store, not mem0. That was unfair. Here mem0's own LLM extracts facts from each
document (``Memory.add(messages, infer=True)``) exactly as its integrations do.

Time travel without rebuilding: all snapshots are written into ONE store in
time order. Every memory carries ``epoch_idx`` (first snapshot in which its
source document version is visible) and ``removed_idx`` (first snapshot where
that version changed or disappeared; ``NEVER_REMOVED`` until then). A probe at
snapshot ``k`` searches with ``epoch_idx <= k < removed_idx`` plus the shared
project scope ``project in [hint, "global"]``. Because ingestion is in time
order, extraction at snapshot ``k`` never sees memories from later snapshots.

Requires ``pip install 'dosemem_eval[mem0]'`` and a mem0 config (LLM, embedder,
vector store) of your choice; the kit does not pick providers for you.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..data import GLOBAL_PROJECT, Corpus
from .base import TOP_K, check_items

NEVER_REMOVED = 1_000_000
MAX_MESSAGE_BYTES = 7500


def split_text(text: str, max_bytes: int = MAX_MESSAGE_BYTES) -> list[str]:
    """Split only oversized documents, at line boundaries, losslessly."""
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]
    parts: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        while len(line.encode("utf-8")) > max_bytes:
            head = line.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
            if current:
                parts.append(current)
                current = ""
            parts.append(head)
            line = line[len(head):]
        if len((current + line).encode("utf-8")) > max_bytes:
            parts.append(current)
            current = ""
        current += line
    if current:
        parts.append(current)
    return [part for part in parts if part]


def make_memory(config_path: str | Path) -> Any:
    try:
        from mem0 import Memory
    except ImportError as exc:
        raise RuntimeError("mem0 adapter needs: pip install 'dosemem_eval[mem0]'") from exc
    path = Path(config_path)
    if path.suffix == ".toml":
        import tomllib

        config = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        config = json.loads(path.read_text(encoding="utf-8"))
    return Memory.from_config(config)


class Mem0System:
    name = "mem0"

    def __init__(self, memory: Any, *, user_id: str = "dosemem-user", retry_delay: float = 8.0):
        self.memory = memory
        self.user_id = user_id
        self.retry_delay = retry_delay
        self.epoch_index: dict[str, int] = {}
        self.manifest: dict[str, dict[str, Any]] = {}
        self.failures: list[dict[str, Any]] = []

    def _retire(self, memory_ids: list[str], removed_idx: int) -> None:
        store = self.memory.vector_store
        for memory_id in memory_ids:
            current = store.get(vector_id=memory_id)
            payload = dict(getattr(current, "payload", None) or {})
            payload["removed_idx"] = removed_idx
            store.update(vector_id=memory_id, payload=payload)

    def ingest_epoch(self, corpus: Corpus, epoch: str) -> dict[str, Any]:
        epoch_idx = corpus.index(epoch)
        documents = {doc.id: doc for doc in corpus.documents(epoch)}
        retired = 0
        for doc_id, entry in list(self.manifest.items()):
            current = documents.get(doc_id)
            if current is None or current.sha256 != entry["sha256"]:
                self._retire(entry["memory_ids"], epoch_idx)
                del self.manifest[doc_id]
                retired += 1
        pending = [doc for doc in documents.values() if doc.id not in self.manifest]
        facts = 0
        for attempt in range(2):
            retry = []
            for doc in pending:
                try:
                    result = self.memory.add([{"role": "user", "content": part} for part in split_text(doc.text)],
                                             user_id=self.user_id, infer=True,
                                             metadata={"project": doc.project or GLOBAL_PROJECT, "source_id": doc.id,
                                                       "source_path": doc.path, "source_sha256": doc.sha256,
                                                       "epoch_idx": epoch_idx, "removed_idx": NEVER_REMOVED})
                except Exception as error:  # noqa: BLE001 - one infrastructure retry, then recorded
                    retry.append(doc)
                    if attempt == 1:
                        self.failures.append({"epoch": epoch, "id": doc.id, "error": f"{type(error).__name__}: {error}"[:300]})
                    continue
                added = [item for item in (result or {}).get("results", []) if isinstance(item, dict) and item.get("id")]
                facts += len(added)
                self.manifest[doc.id] = {"sha256": doc.sha256, "memory_ids": [item["id"] for item in added],
                                         "epoch_idx": epoch_idx}
            if not retry:
                break
            pending = retry
            time.sleep(self.retry_delay)
        return {"epoch": epoch, "retired_sources": retired, "facts_added": facts}

    def prepare(self, corpus: Corpus) -> None:
        # In time order: extraction at snapshot k only ever sees memories from <= k.
        for epoch in corpus.order:
            self.epoch_index[epoch] = corpus.index(epoch)
            self.ingest_epoch(corpus, epoch)

    def filters(self, epoch_idx: int, project_hint: str | None) -> dict[str, Any]:
        filters: dict[str, Any] = {"user_id": self.user_id, "epoch_idx": {"lte": epoch_idx},
                                   "removed_idx": {"gt": epoch_idx}}
        if project_hint and project_hint != GLOBAL_PROJECT:
            filters["project"] = {"in": [project_hint, GLOBAL_PROJECT]}
        return filters

    def retrieve(self, probe: dict[str, Any], epoch: str) -> list[dict[str, Any]]:
        filters = self.filters(self.epoch_index[epoch], probe.get("project_hint"))
        result = self.memory.search(probe["user_turn"], filters=filters, top_k=TOP_K, threshold=0, rerank=False)
        items = []
        for row in (result or {}).get("results", []):
            metadata = row.get("metadata") or {}
            items.append({"item_id": "mem0:" + str(row["id"]), "text": row.get("memory", ""),
                          "score": float(row.get("score", 0) or 0), "source": metadata.get("source_path")})
        return check_items(items)
