"""Judge transport: OpenAI-compatible Chat Completions, one infrastructure retry, raw responses kept.

Each judged probe produces one JSON record on disk containing the frozen
candidate list, the request hash, every attempt (HTTP status + raw response)
and the validated judgment or ``null``. Records are revalidated from the raw
response when loaded, so a later protocol fix never silently reuses stale
parses. A transport-successful but invalid response is terminal: it is NOT
resampled (resampling until something parses would bias the labels), and the
probe stays *unknown*.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .io import read_json, write_json
from .judging import JudgeValidationError, build_judge_request, extract_response_text, validate_judgment
from .providers import Endpoint, TransportError, post_json

# (payload) -> (http_status, parsed_body); may raise TransportError.
Transport = Callable[[Mapping[str, Any]], tuple[int, Any]]


def http_transport(target: Endpoint) -> Transport:
    def send(payload: Mapping[str, Any]) -> tuple[int, Any]:
        return post_json(target, "/chat/completions", payload)
    return send


def record_path(directory: Path, stage: str, probe_id: str, request_sha256: str = "") -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in probe_id)
    # Supplementary passes can happen several times per probe (one per new system).
    suffix = f"--{request_sha256[:12]}" if stage == "supplementary" else ""
    return directory / stage / f"{safe}{suffix}.json"


def _finish(record: dict[str, Any], outcome: dict[str, Any], request: Mapping[str, Any]) -> None:
    if outcome.get("http_status") != 200:
        return
    try:
        text = extract_response_text(outcome.get("response") or {}, endpoint=request["endpoint"])
        record["judgment"] = validate_judgment(text, request)
    except JudgeValidationError as exc:
        outcome["validation_error"] = str(exc)


def acquire(probe: Mapping[str, Any], items: Sequence[Mapping[str, Any]], *, stage: str, directory: Path,
            transport: Transport, model: str, seed: int = 10, max_attempts: int = 2,
            retry_delay: float = 8.0) -> dict[str, Any]:
    """Judge one probe's candidates and persist the raw record (resumable)."""
    if max_attempts not in (1, 2):
        raise ValueError("max_attempts must be 1 or 2 (one infrastructure retry)")
    items = [{"item_id": str(item["item_id"]), "text": str(item["text"])} for item in items]
    request = build_judge_request(probe, items, stage=stage, seed=seed, model=model)
    path = record_path(directory, stage, str(probe["probe_id"]), request["request_sha256"])
    if path.exists():
        record = read_json(path)
        if record.get("request_sha256") != request["request_sha256"]:
            raise ValueError(f"{path}: stored request differs from the frozen pool; refusing to overwrite")
        if record.get("terminal"):
            return record
    else:
        record = {"probe_id": str(probe["probe_id"]), "stage": stage, "protocol": request["protocol"],
                  "model": model, "request_sha256": request["request_sha256"], "seed": seed,
                  "items": items, "attempts": [], "judgment": None, "terminal": False}
    while len(record["attempts"]) < max_attempts:
        if record["attempts"]:
            time.sleep(retry_delay)
        outcome: dict[str, Any] = {"sent_at": time.time()}
        started = time.monotonic()
        try:
            status, body = transport(request["payload"])
            outcome.update(http_status=status, response=body)
        except TransportError as exc:
            outcome.update(http_status=exc.status, response=exc.body, error=str(exc))
        outcome["wall_ms"] = (time.monotonic() - started) * 1000
        record["attempts"].append(outcome)
        if "error" not in outcome:
            # Transport completed (any non-retryable status): terminal, never resampled.
            _finish(record, outcome, request)
            break
        write_json(path, record)
    record["terminal"] = True
    write_json(path, record)
    return record


def load_judgment(record: Mapping[str, Any], probe: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Revalidate a stored record against its own frozen candidates. Returns (judgment, error)."""
    request = build_judge_request(probe, record["items"], stage=record["stage"], seed=record.get("seed", 10),
                                  model=record.get("model", "judge-model"))
    if request["request_sha256"] != record.get("request_sha256"):
        return None, "request_changed"
    completed = [attempt for attempt in record.get("attempts", []) if attempt.get("http_status") == 200]
    if not completed:
        return None, "no_completed_response"
    try:
        text = extract_response_text(completed[-1].get("response") or {}, endpoint=request["endpoint"])
        return validate_judgment(text, request), None
    except JudgeValidationError as exc:
        return None, str(exc)


def load_stage(directory: Path, stage: str, probes: Mapping[str, Mapping[str, Any]]
               ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    """All revalidated judgments of one stage, as lists per probe (supplementary may have several)."""
    labels: dict[str, list[dict[str, Any]]] = {}
    invalid: dict[str, list[str]] = {}
    folder = directory / stage
    if not folder.exists():
        return labels, invalid
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        probe = probes.get(record["probe_id"])
        if probe is None:
            continue
        judgment, error = load_judgment(record, probe)
        if judgment is not None:
            labels.setdefault(record["probe_id"], []).append(judgment)
        else:
            invalid.setdefault(record["probe_id"], []).append(error or "invalid")
    return labels, invalid
