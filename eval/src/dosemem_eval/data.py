"""Probe and snapshot-corpus schemas plus loaders.

Probes (one JSON object per line)::

    {"probe_id": "p-001", "session_id": "s-01", "client_group": "working",
     "split": "dev", "epoch": "2026-01-05", "project_hint": "widget-detector",
     "user_turn": "...", "previous_turn": "..." | null, "hindsight": ["...", ...]}

``hindsight`` holds the turns that followed ``user_turn`` in the real session.
It is shown ONLY to the judge and never to a memory system.

Snapshot corpus: a directory with one ``<epoch>.jsonl`` file per snapshot and
an optional ``epochs.json`` listing epoch ids in time order (otherwise file
names are sorted). Each line is a document::

    {"id": "doc-1", "path": "projects/widget-detector/notes.md",
     "project": "widget-detector" | "global", "text": "...", "sha256": "..."}
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .io import read_json, read_jsonl

CLIENT_GROUPS = ("assistant", "working")
SPLITS = ("dev", "test")
GLOBAL_PROJECT = "global"


class SchemaError(ValueError):
    pass


def _text(value: Any) -> str:
    """Accept a plain string or a ``{"text": ...}`` turn object."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return str(value.get("text") or value.get("content") or "")
    raise SchemaError("turn must be a string or an object with text")


def _turn(value: Any) -> dict[str, str]:
    role = str(value.get("role") or "unknown") if isinstance(value, Mapping) else "unknown"
    return {"role": role, "text": _text(value)}


@dataclass(frozen=True)
class Probe:
    probe_id: str
    session_id: str
    client_group: str
    split: str
    epoch: str
    user_turn: str
    project_hint: str | None = None
    previous_turn: str | None = None
    hindsight: tuple[dict[str, str], ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Probe":
        for key in ("probe_id", "session_id", "client_group", "split", "epoch", "user_turn"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise SchemaError(f"probe field {key!r} must be a non-empty string")
        if row["client_group"] not in CLIENT_GROUPS:
            raise SchemaError(f"client_group must be one of {CLIENT_GROUPS}")
        if row["split"] not in SPLITS:
            raise SchemaError(f"split must be one of {SPLITS}")
        hint = row.get("project_hint")
        if hint is not None and not isinstance(hint, str):
            raise SchemaError("project_hint must be a string or null")
        hindsight = row.get("hindsight") or []
        if not isinstance(hindsight, list):
            raise SchemaError("hindsight must be a list")
        previous = row.get("previous_turn")
        return cls(probe_id=row["probe_id"], session_id=row["session_id"], client_group=row["client_group"],
                   split=row["split"], epoch=row["epoch"], user_turn=row["user_turn"],
                   project_hint=hint or None, previous_turn=_text(previous) or None,
                   hindsight=tuple(_turn(turn) for turn in hindsight if _text(turn)))

    def to_dict(self) -> dict[str, Any]:
        return {"probe_id": self.probe_id, "session_id": self.session_id, "client_group": self.client_group,
                "split": self.split, "epoch": self.epoch, "project_hint": self.project_hint,
                "user_turn": self.user_turn, "previous_turn": self.previous_turn,
                "hindsight": [dict(turn) for turn in self.hindsight]}

    def system_view(self) -> dict[str, Any]:
        """The only fields a memory system may see. Hindsight and split are withheld."""
        return {"probe_id": self.probe_id, "epoch": self.epoch, "project_hint": self.project_hint,
                "user_turn": self.user_turn, "previous_turn": self.previous_turn}


def load_probes(path: str | Path) -> list[Probe]:
    probes = [Probe.from_dict(row) for row in read_jsonl(path)]
    seen: set[str] = set()
    for probe in probes:
        if probe.probe_id in seen:
            raise SchemaError(f"duplicate probe_id {probe.probe_id}")
        seen.add(probe.probe_id)
    sessions: dict[str, str] = {}
    for probe in probes:
        # A session must not straddle dev and test, or tuning leaks into evaluation.
        if sessions.setdefault(probe.session_id, probe.split) != probe.split:
            raise SchemaError(f"session {probe.session_id} appears in both dev and test")
    return probes


@dataclass(frozen=True)
class Document:
    id: str
    path: str
    project: str
    text: str
    sha256: str

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Document":
        for key in ("id", "path", "text"):
            if not isinstance(row.get(key), str):
                raise SchemaError(f"document field {key!r} must be a string")
        digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        declared = row.get("sha256") or digest
        if declared != digest:
            raise SchemaError(f"document {row['id']} sha256 does not match its text")
        return cls(id=row["id"], path=row["path"], project=row.get("project") or GLOBAL_PROJECT,
                   text=row["text"], sha256=digest)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "path": self.path, "project": self.project, "text": self.text, "sha256": self.sha256}


class Corpus:
    """Time-ordered snapshots. A probe may only see the snapshot named by its epoch."""

    def __init__(self, snapshots: Mapping[str, list[Document]], order: list[str]):
        if sorted(order) != sorted(snapshots) or len(set(order)) != len(order):
            raise SchemaError("epoch order must list every snapshot exactly once")
        self.order = list(order)
        self.snapshots = {epoch: list(snapshots[epoch]) for epoch in order}

    @classmethod
    def load(cls, directory: str | Path) -> "Corpus":
        root = Path(directory)
        files = {path.stem: path for path in root.glob("*.jsonl")}
        if not files:
            raise SchemaError(f"no snapshot files in {root}")
        order_file = root / "epochs.json"
        order = list(read_json(order_file)) if order_file.exists() else sorted(files)
        snapshots: dict[str, list[Document]] = {}
        for epoch in order:
            if epoch not in files:
                raise SchemaError(f"epochs.json names missing snapshot {epoch}")
            documents = [Document.from_dict(row) for row in read_jsonl(files[epoch])]
            if len({doc.id for doc in documents}) != len(documents):
                raise SchemaError(f"duplicate document id in snapshot {epoch}")
            snapshots[epoch] = documents
        return cls(snapshots, order)

    def documents(self, epoch: str) -> list[Document]:
        if epoch not in self.snapshots:
            raise SchemaError(f"unknown epoch {epoch}")
        return self.snapshots[epoch]

    def index(self, epoch: str) -> int:
        return self.order.index(epoch)

    def lifetimes(self) -> list[dict[str, Any]]:
        """Distinct document versions with ``epoch_idx`` (first seen) and ``removed_idx``
        (first snapshot where this exact version is gone; ``len(order)`` if never)."""
        versions: dict[tuple[str, str], dict[str, Any]] = {}
        for index, epoch in enumerate(self.order):
            present = {(doc.id, doc.sha256) for doc in self.snapshots[epoch]}
            for doc in self.snapshots[epoch]:
                key = (doc.id, doc.sha256)
                entry = versions.get(key)
                if entry is None or entry["removed_idx"] is not None:
                    if entry is not None:
                        # A version that reappears gets a fresh lifetime.
                        key = (doc.id, doc.sha256 + f"#{index}")
                    versions[key] = {"document": doc, "epoch_idx": index, "removed_idx": None}
            for key, entry in versions.items():
                if entry["removed_idx"] is None and (key[0], key[1].split("#")[0]) not in present:
                    entry["removed_idx"] = index
        result = []
        for entry in versions.values():
            if entry["removed_idx"] is None:
                entry["removed_idx"] = len(self.order)
            result.append(entry)
        return sorted(result, key=lambda row: (row["epoch_idx"], row["document"].id))


def project_scope(project_hint: str | None) -> list[str] | None:
    """Every system receives the same scope: the hinted project plus global notes."""
    if not project_hint or project_hint == GLOBAL_PROJECT:
        return None
    return [project_hint, GLOBAL_PROJECT]
