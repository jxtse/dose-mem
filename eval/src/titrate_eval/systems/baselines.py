"""Independent baselines: heading-aware chunks, stdlib BM25, dense cosine.

Neither baseline knows anything about the memory layer under test. They are
the floor a memory layer must beat under the same byte budget.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Sequence

from ..data import Corpus, Document, project_scope
from .base import TOP_K, check_items
from .embedders import Embedder

CHUNK_BYTES = 512


def tokens(text: str) -> list[str]:
    """Lowercase ASCII words plus CJK unigrams and bigrams."""
    out = re.findall(r"[a-z0-9_]+", text.lower())
    for span in re.findall(r"[\u3400-\u9fff]+", text):
        out.extend(span)
        out.extend(span[i:i + 2] for i in range(len(span) - 1))
    return out


def chunks(documents: Sequence[Document], max_bytes: int = CHUNK_BYTES) -> list[dict[str, Any]]:
    """Keep heading provenance and partition bodies at line/UTF-8 boundaries."""
    if type(max_bytes) is not int or max_bytes < 64:
        raise ValueError("chunk_budget_too_small")
    result: list[dict[str, Any]] = []
    for doc in documents:
        heading = ""
        buf = ""

        def emit(value: str) -> None:
            if not value.strip():
                return
            text = (heading + "\n" + value).strip()
            identity = hashlib.sha256((doc.path + "\0" + text).encode()).hexdigest()[:24]
            result.append({"item_id": "chunk-" + identity, "text": text, "source": doc.path, "project": doc.project})

        for line in doc.text.splitlines():
            if re.match(r"^#{1,6}\s", line):
                emit(buf)
                buf = ""
                heading = line.encode()[:min(160, max_bytes // 3)].decode(errors="ignore")
                continue
            addition = line + "\n"
            if len((heading + "\n" + buf + addition).encode()) <= max_bytes:
                buf += addition
                continue
            emit(buf)
            buf = ""
            while len((heading + "\n" + addition).encode()) > max_bytes:
                room = max(32, max_bytes - len((heading + "\n").encode()))
                piece = addition.encode()[:room].decode(errors="ignore")
                emit(piece)
                addition = addition[len(piece):]
            buf = addition
        emit(buf)
    return list({x["item_id"]: x for x in result}.values())


def _in_scope(item: dict[str, Any], scope: list[str] | None) -> bool:
    return scope is None or item.get("project") in scope


class BM25Index:
    def __init__(self, items: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.items = items
        self.k1 = k1
        self.b = b
        self.counts = [Counter(tokens(x["text"])) for x in items]
        self.lengths = [sum(x.values()) for x in self.counts]
        self.avg = sum(self.lengths) / max(1, len(items))
        self.df: Counter[str] = Counter()
        for count in self.counts:
            self.df.update(count.keys())

    def search(self, query: str, limit: int = TOP_K, scope: list[str] | None = None) -> list[dict[str, Any]]:
        n = len(self.items)
        scored = []
        terms = set(tokens(query))
        for item, count, length in zip(self.items, self.counts, self.lengths):
            if not _in_scope(item, scope):
                continue
            score = 0.0
            for term in terms:
                tf = count[term]
                if tf:
                    idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / max(self.avg, 1)))
            scored.append({**item, "score": score})
        return sorted(scored, key=lambda x: (-x["score"], x["item_id"]))[:limit]


class DenseIndex:
    def __init__(self, items: list[dict[str, Any]], vectors: list[list[float]]):
        if len(items) != len(vectors):
            raise ValueError("vector_count_mismatch")
        if any(not row or len(row) != len(vectors[0]) or any(not math.isfinite(x) for x in row) for row in vectors):
            raise ValueError("invalid_vectors")
        self.items = items
        self.vectors = vectors
        self.norms = [math.sqrt(sum(x * x for x in row)) for row in vectors]

    def search(self, vector: list[float], limit: int = TOP_K, scope: list[str] | None = None) -> list[dict[str, Any]]:
        if not vector or any(not math.isfinite(x) for x in vector) or (self.vectors and len(vector) != len(self.vectors[0])):
            raise ValueError("invalid_query_vector")
        norm = math.sqrt(sum(x * x for x in vector))
        scored = []
        for item, row, rnorm in zip(self.items, self.vectors, self.norms):
            if not _in_scope(item, scope):
                continue
            score = sum(x * y for x, y in zip(row, vector)) / (norm * rnorm) if norm and rnorm else 0.0
            scored.append({**item, "score": score})
        return sorted(scored, key=lambda x: (-x["score"], x["item_id"]))[:limit]


class BM25System:
    """BM25 (k1=1.5, b=0.75) over heading-aware 512-byte chunks of the probe's snapshot."""

    name = "bm25"

    def __init__(self, chunk_bytes: int = CHUNK_BYTES):
        self.chunk_bytes = chunk_bytes
        self.indexes: dict[str, BM25Index] = {}

    def prepare(self, corpus: Corpus) -> None:
        self.indexes = {epoch: BM25Index(chunks(corpus.documents(epoch), self.chunk_bytes)) for epoch in corpus.order}

    def retrieve(self, probe: dict[str, Any], epoch: str, *, scoped: bool = True,
                 limit: int = TOP_K) -> list[dict[str, Any]]:
        scope = project_scope(probe.get("project_hint")) if scoped else None
        return check_items(self.indexes[epoch].search(probe["user_turn"], limit, scope), limit)


class DenseSystem:
    """Cosine similarity over embedded chunks; the embedder is pluggable."""

    name = "dense"

    def __init__(self, embedder: Embedder, chunk_bytes: int = CHUNK_BYTES):
        self.embedder = embedder
        self.chunk_bytes = chunk_bytes
        self.indexes: dict[str, DenseIndex] = {}
        self._cache: dict[str, list[float]] = {}

    def _embed(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if missing:
            for text, vector in zip(missing, self.embedder.embed(missing), strict=True):
                self._cache[text] = vector
        return [self._cache[text] for text in texts]

    def prepare(self, corpus: Corpus) -> None:
        for epoch in corpus.order:
            items = chunks(corpus.documents(epoch), self.chunk_bytes)
            self.indexes[epoch] = DenseIndex(items, self._embed([item["text"] for item in items]))

    def retrieve(self, probe: dict[str, Any], epoch: str) -> list[dict[str, Any]]:
        vector = self._embed([probe["user_turn"]])[0]
        return check_items(self.indexes[epoch].search(vector, TOP_K, project_scope(probe.get("project_hint"))))
