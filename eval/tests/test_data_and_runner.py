from __future__ import annotations

import json

import pytest

from conftest import item, probe, response_for
from dose_mem_eval.data import Corpus, Probe, SchemaError, load_probes
from dose_mem_eval.judge_runner import acquire, load_judgment
from dose_mem_eval.providers import TransportError, endpoint


def chat(content: str) -> dict:
    return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}


def test_probe_schema_and_system_view_withholds_hindsight(example_dir) -> None:
    probes = load_probes(example_dir / "probes.jsonl")
    assert {p.client_group for p in probes} == {"assistant", "working"}
    view = probes[0].system_view()
    assert "hindsight" not in view and "split" not in view and "session_id" not in view
    with pytest.raises(SchemaError):
        Probe.from_dict({**probes[0].to_dict(), "client_group": "other"})


def test_sessions_cannot_straddle_dev_and_test(tmp_path) -> None:
    rows = [dict(probe("a", "s"), split="dev"), dict(probe("b", "s"), split="test")]
    path = tmp_path / "probes.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SchemaError, match="both dev and test"):
        load_probes(path)


def test_corpus_epochs_are_ordered_and_hash_checked(example_dir, tmp_path) -> None:
    corpus = Corpus.load(example_dir / "snapshots")
    assert corpus.order == ["2026-01-05", "2026-01-12"]
    lifetimes = {(e["document"].id, e["epoch_idx"]): e["removed_idx"] for e in corpus.lifetimes()}
    assert lifetimes[("wd-notes", 0)] == 1 and lifetimes[("wd-notes", 1)] == 2
    bad = tmp_path / "e.jsonl"
    bad.write_text(json.dumps({"id": "x", "path": "x.md", "text": "t", "sha256": "0" * 64}) + "\n")
    with pytest.raises(SchemaError, match="sha256"):
        Corpus.load(tmp_path)


def test_judge_retries_infrastructure_once_and_saves_raw_responses(tmp_path) -> None:
    calls = []

    def transport(payload):
        calls.append(payload)
        if len(calls) == 1:
            raise TransportError("http_503", 503, {"error": "busy"})
        request = __import__("dose_mem_eval.judging", fromlist=["x"]).build_judge_request(probe(), [item()], model="m")
        return 200, chat(json.dumps(response_for(request)))

    record = acquire(probe(), [item()], stage="canonical", directory=tmp_path, transport=transport, model="m",
                     retry_delay=0)
    assert len(calls) == 2 and record["judgment"]["items"][0]["item_id"] == "a"
    assert record["attempts"][0]["http_status"] == 503 and record["attempts"][0]["response"] == {"error": "busy"}
    stored = json.loads((tmp_path / "canonical" / "p1.json").read_text())
    judgment, error = load_judgment(stored, probe())
    assert error is None and judgment["memory_needed"] is True


def test_invalid_judge_output_is_terminal_unknown_and_not_resampled(tmp_path) -> None:
    calls = []

    def transport(payload):
        calls.append(payload)
        return 200, chat("not json")

    record = acquire(probe(), [item()], stage="canonical", directory=tmp_path, transport=transport, model="m",
                     retry_delay=0)
    assert len(calls) == 1 and record["judgment"] is None
    assert record["attempts"][0]["validation_error"] == "invalid_json"
    again = acquire(probe(), [item()], stage="canonical", directory=tmp_path, transport=transport, model="m")
    assert len(calls) == 1 and again["terminal"]
    with pytest.raises(ValueError, match="frozen pool"):
        acquire(probe(), [item(), item("b", "more")], stage="canonical", directory=tmp_path, transport=transport,
                model="m")


def test_endpoint_config_from_env_and_file(monkeypatch) -> None:
    monkeypatch.setenv("JUDGE_KEY_FOR_TEST", "secret")
    config = {"judge": {"base_url": "https://example.invalid/v1", "model": "m", "api_key_env": "JUDGE_KEY_FOR_TEST"}}
    target = endpoint("judge", config)
    assert target.api_key == "secret" and target.url("/chat/completions") == "https://example.invalid/v1/chat/completions"
    monkeypatch.setenv("DOSE_MEM_JUDGE_MODEL", "override")
    assert endpoint("judge", config).model == "override"
    with pytest.raises(RuntimeError, match="base_url and model"):
        endpoint("audit_judge", {})
