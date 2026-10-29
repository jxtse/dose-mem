"""Embedders for the dense baseline.

``OpenAICompatibleEmbedder`` calls any ``/embeddings`` endpoint that follows
the OpenAI wire format. ``HashEmbedder`` is a deterministic, offline
bag-of-tokens hash projection used by tests and the example; it is NOT a
semantic model and should never be used for real results.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence

from ..providers import Endpoint, post_json, with_one_retry


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbedder:
    name = "openai-compatible"

    def __init__(self, target: Endpoint, batch_size: int = 64, retry_delay: float = 8.0):
        self.target = target
        self.batch_size = batch_size
        self.retry_delay = retry_delay

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            batch = list(texts[offset:offset + self.batch_size])
            payload = {"model": self.target.model, "input": batch, "encoding_format": "float"}
            status, body = with_one_retry(lambda: post_json(self.target, "/embeddings", payload), delay=self.retry_delay)
            if status != 200 or not isinstance(body, dict) or not isinstance(body.get("data"), list):
                raise RuntimeError(f"embedding request failed with HTTP {status}")
            rows = sorted(body["data"], key=lambda row: row["index"])
            if len(rows) != len(batch):
                raise RuntimeError("embedding count mismatch")
            result.extend([float(x) for x in row["embedding"]] for row in rows)
        return result


class HashEmbedder:
    """Deterministic fake embedder: signed feature hashing of tokens (offline, tests only)."""

    name = "hash-fake"

    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        from .baselines import tokens

        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in tokens(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                vector[index] += 1.0 if digest[4] & 1 else -1.0
            if not any(vector):
                vector[0] = 1e-6
            norm = math.sqrt(sum(x * x for x in vector))
            vectors.append([x / norm for x in vector])
        return vectors
