"""Memory-system adapters. Third-party adapters import their SDKs lazily."""

from __future__ import annotations

from typing import Any, Mapping

from .base import TOP_K, MemorySystem, check_items
from .baselines import BM25System, DenseSystem
from .embedders import HashEmbedder, OpenAICompatibleEmbedder

__all__ = ["TOP_K", "MemorySystem", "check_items", "BM25System", "DenseSystem", "HashEmbedder",
           "OpenAICompatibleEmbedder", "build_system", "AVAILABLE"]

AVAILABLE = ("bm25", "dense", "dense-fake", "openviking", "mem0")


def build_system(name: str, config: Mapping[str, Any] | None = None) -> MemorySystem:
    """Construct an adapter by name. ``config`` is the parsed TOML config file."""
    config = config or {}
    if name == "bm25":
        return BM25System()
    if name == "dense":
        from ..providers import endpoint

        return DenseSystem(OpenAICompatibleEmbedder(endpoint("embedding", config)))
    if name == "dense-fake":
        system = DenseSystem(HashEmbedder())
        system.name = "dense-fake"
        return system
    if name == "openviking":
        from .openviking import OpenVikingSystem, make_client

        section = dict(config.get("openviking", {}))
        import os

        url = os.environ.get("DOSE_MEM_OPENVIKING_URL") or section.pop("url", None)
        if not url:
            raise RuntimeError("[openviking] url (or DOSE_MEM_OPENVIKING_URL) is required")
        api_key = os.environ.get("DOSE_MEM_OPENVIKING_API_KEY") or section.pop("api_key", None)
        options = {key: section.pop(key) for key in ("root_uri", "max_tokens") if key in section}
        return OpenVikingSystem(make_client(url, api_key=api_key, **section.get("client", {})), **options)
    if name == "mem0":
        from .mem0 import Mem0System, make_memory

        section = config.get("mem0", {})
        if not section.get("config_path"):
            raise RuntimeError("[mem0] config_path (a mem0 Memory.from_config JSON/TOML file) is required")
        return Mem0System(make_memory(section["config_path"]), user_id=section.get("user_id", "dose-mem-user"))
    raise ValueError(f"unknown system {name!r}; choose from {AVAILABLE}")
