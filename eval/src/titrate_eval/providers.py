"""Provider configuration for OpenAI-compatible endpoints.

Values come from a TOML config file (``--config``) and/or environment
variables; environment variables win. Nothing is hard-coded: there is no
default endpoint, and every network client refuses to run without one.

Example ``titrate.toml``::

    [judge]
    base_url = "https://api.example.com/v1"
    model = "your-judge-model"
    api_key_env = "JUDGE_API_KEY"      # name of the env var holding the key

    [audit_judge]
    base_url = "https://api.other-provider.example/v1"
    model = "a-different-judge-model"
    api_key_env = "AUDIT_API_KEY"

    [embedding]
    base_url = "https://api.example.com/v1"
    model = "your-embedding-model"
    api_key_env = "EMBED_API_KEY"

Environment overrides: ``TITRATE_<SECTION>_BASE_URL``, ``TITRATE_<SECTION>_MODEL``
and ``TITRATE_<SECTION>_API_KEY`` (SECTION = JUDGE, AUDIT_JUDGE, EMBEDDING).
"""

from __future__ import annotations

import json
import os
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 180.0

    def url(self, path: str) -> str:
        return self.base_url.rstrip("/") + "/" + path.lstrip("/")


def load_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def endpoint(section: str, config: Mapping[str, Any] | None = None) -> Endpoint:
    values = dict((config or {}).get(section, {}))
    prefix = "TITRATE_" + section.upper() + "_"
    base_url = os.environ.get(prefix + "BASE_URL") or values.get("base_url")
    model = os.environ.get(prefix + "MODEL") or values.get("model")
    key_env = values.get("api_key_env")
    api_key = os.environ.get(prefix + "API_KEY") or (os.environ.get(key_env) if key_env else None) or values.get("api_key")
    if not base_url or not model:
        raise RuntimeError(f"[{section}] needs base_url and model (config file or {prefix}BASE_URL / {prefix}MODEL)")
    return Endpoint(base_url=str(base_url), model=str(model), api_key=api_key,
                    timeout=float(values.get("timeout", 180.0)))


class TransportError(RuntimeError):
    """Infrastructure failure (connection, 429, 5xx). Retried once by callers."""

    def __init__(self, message: str, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


def post_json(target: Endpoint, path: str, payload: Mapping[str, Any]) -> tuple[int, Any]:
    """POST JSON with the stdlib; returns (status, parsed body). Raises TransportError on infra failure."""
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if target.api_key:
        headers["Authorization"] = "Bearer " + target.api_key
    request = urllib.request.Request(target.url(path), data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=target.timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TransportError(f"{type(exc).__name__}: {exc}") from exc
    try:
        body = json.loads(raw.decode("utf-8")) if raw else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        body = {"unparsed": raw.decode("utf-8", errors="replace")[:2000]}
    if status == 429 or status >= 500:
        raise TransportError(f"http_{status}", status, body)
    return status, body


def with_one_retry(call, *, delay: float = 8.0):
    """Run ``call`` and retry exactly once on TransportError (infrastructure only)."""
    try:
        return call()
    except TransportError:
        time.sleep(delay)
        return call()
