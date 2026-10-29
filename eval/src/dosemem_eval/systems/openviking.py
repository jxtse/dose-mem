"""OpenViking adapter: native context-assembly recall through the public SDK.

Why "native" matters: an earlier round of the benchmark this kit came from
drove third-party systems through a generic, non-native path (plain
similarity search over raw chunks, our own text rendering). That measured our
glue code, not the product, and was unfair to the systems. Each adapter here
therefore uses the write and read paths the vendor's own agent integrations
use, and only constrains what every system is constrained by: the snapshot a
probe may see and the shared project scope.

For OpenViking that means:

* write: ``add_resource(...)`` of the snapshot's Markdown tree with the native
  ``semantic_and_vectors`` processing (OpenViking builds its own layered
  abstracts and vectors);
* read: ``search_context(query, limit, options={max_tokens, score_threshold: 0,
  filter})``, i.e. ``mode="context"`` recall used for automatic injection. That
  mode rejects ``target_uri``, so the project scope is expressed as the native
  ``uri`` path filter (OR of directory subtrees).

Each epoch is written under its own root ``<root_uri>/<epoch>/<project>/...`` so
a probe can only reach its own snapshot. Requires ``pip install
'dosemem_eval[openviking]'`` and a running OpenViking server you configured
(embedding/VLM providers are OpenViking's own configuration, not ours).
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any

from ..data import GLOBAL_PROJECT, Corpus
from .base import TOP_K, check_items


def _clean(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not cleaned:
        raise ValueError(f"cannot derive a safe path segment from {value!r}")
    return cleaned


def _relative(path: str) -> Path:
    parts = [segment for segment in Path(path).parts if segment not in ("", ".", "..", "/")]
    if not parts:
        raise ValueError(f"document path {path!r} has no usable segments")
    return Path(*parts)


def make_client(url: str, api_key: str | None = None, timeout: float = 900.0, **kwargs: Any) -> Any:
    """Create the vendor's sync HTTP client lazily (optional dependency)."""
    try:
        from openviking_sdk import SyncHTTPClient  # lightweight SDK
    except ImportError:
        try:
            from openviking import SyncHTTPClient  # full package re-exports it
        except ImportError as exc:
            raise RuntimeError("OpenViking adapter needs: pip install 'dosemem_eval[openviking]'") from exc
    client = SyncHTTPClient(url=url, api_key=api_key, timeout=timeout, **kwargs)
    client.initialize()
    return client


class OpenVikingSystem:
    name = "openviking"

    def __init__(self, client: Any, *, root_uri: str = "viking://resources/dosemem", max_tokens: int = 1600,
                 workdir: str | Path | None = None, wait: bool = True):
        self.client = client
        self.root_uri = root_uri.rstrip("/")
        self.max_tokens = max_tokens
        self.workdir = Path(workdir) if workdir else None
        self.wait = wait
        self.projects: dict[str, set[str]] = {}
        self.ingest_results: dict[str, Any] = {}

    def epoch_uri(self, epoch: str) -> str:
        return f"{self.root_uri}/{_clean(epoch)}"

    def prepare(self, corpus: Corpus) -> None:
        for epoch in corpus.order:
            documents = corpus.documents(epoch)
            self.projects[epoch] = {_clean(doc.project) for doc in documents}
            base = Path(tempfile.mkdtemp(prefix="openviking-upload-", dir=self.workdir))
            upload = base / _clean(epoch)
            for doc in documents:
                target = upload / _clean(doc.project) / _relative(doc.path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(doc.text, encoding="utf-8")
            result = self.client.add_resource(str(upload), to=self.epoch_uri(epoch), wait=self.wait,
                                              options={"preserve_structure": True, "create_parent": True,
                                                       "processing_mode": "semantic_and_vectors"})
            if isinstance(result, dict) and result.get("status") in {"error", "failed", "cancelled"}:
                raise RuntimeError(f"openviking ingestion failed for epoch {epoch}: {result}")
            self.ingest_results[epoch] = result

    def scope_filter(self, epoch: str, project_hint: str | None) -> tuple[list[str], dict[str, Any]]:
        available = self.projects.get(epoch)
        requested = _clean(project_hint) if project_hint else None
        wanted = [p for p in dict.fromkeys([requested, GLOBAL_PROJECT]) if p and (available is None or p in available)]
        targets = [f"{self.epoch_uri(epoch)}/{p}" for p in wanted] if requested and requested != GLOBAL_PROJECT \
            else [self.epoch_uri(epoch)]
        if not targets:
            targets = [self.epoch_uri(epoch)]
        conditions = [{"op": "must", "field": "uri", "conds": [t], "para": "-d=-1"} for t in targets]
        return targets, conditions[0] if len(conditions) == 1 else {"op": "or", "conds": conditions}

    def retrieve(self, probe: dict[str, Any], epoch: str) -> list[dict[str, Any]]:
        _, native_filter = self.scope_filter(epoch, probe.get("project_hint"))
        # Query is the verbatim user turn; no session id, so no LLM query rewrite.
        result = self.client.search_context(probe["user_turn"], limit=TOP_K,
                                            options={"max_tokens": self.max_tokens, "score_threshold": 0,
                                                     "filter": native_filter})
        items = []
        for entry in (result or {}).get("entries", []) or []:
            uri = str(entry.get("uri", ""))
            text = entry.get("text") or entry.get("abstract") or ""
            if not text:
                continue
            items.append({"item_id": "openviking:" + hashlib.sha256(uri.encode()).hexdigest()[:24],
                          "text": text, "score": float(entry.get("score", 0) or 0), "source": uri})
        return check_items(items)
