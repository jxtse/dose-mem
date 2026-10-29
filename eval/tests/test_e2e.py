"""End-to-end smoke test on the fictional example: bm25 + dense(fake) + fake judge, no network."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

import pytest

from dose_mem_eval.cli import main


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted during an offline test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_full_pipeline_offline(example_dir, tmp_path, no_network, capsys) -> None:
    run = tmp_path / "run"
    common = ["--probes", str(example_dir / "probes.jsonl"), "--run", str(run)]
    assert main(["retrieve", *common, "--snapshots", str(example_dir / "snapshots"),
                 "--system", "bm25", "--system", "dense-fake"]) == 0
    assert main(["pool", *common]) == 0
    assert main(["judge", *common, "--stage", "canonical", "--fake", "--retry-delay", "0"]) == 0
    assert main(["judge", *common, "--stage", "audit", "--fake", "--retry-delay", "0"]) == 0
    assert main(["calibrate", *common]) == 0
    assert main(["score", *common]) == 0
    assert main(["report", *common, "--bootstrap", "200"]) == 0
    capsys.readouterr()

    summary = json.loads((run / "summary.json").read_text())
    systems = {(s["system"], s["regime"]) for s in summary["systems"] if s["client_group"] == "all"}
    assert systems == {("bm25", "fixed"), ("bm25", "calibrated"), ("dense-fake", "fixed"), ("dense-fake", "calibrated")}
    assert summary["judge"]["valid"] == 7 and not summary["judge"]["invalid"]
    assert summary["audit"]["selected_n"] == 2 and summary["audit"]["paired_n"] == 2
    assert "confusion_matrix" in summary["audit"]["item_labels"]
    assert summary["paired_differences"]
    for entry in summary["systems"]:
        metrics = entry["metrics"]
        assert set(metrics) >= {"hit_rate", "useful_precision", "harmful_rate", "need_recall", "quiet_rate",
                                "composite", "items_per_turn", "bytes_per_turn", "error_rate"}
        assert metrics["composite"]["ci_method"] == "session_cluster_percentile_95"
    per_probe = [json.loads(line) for line in (run / "per-probe.jsonl").read_text().splitlines()]
    assert len(per_probe) == 7 * 2 * 2
    # Equal budgets: no system ever exceeds its client-group byte budget.
    budgets = {"assistant": 1200, "working": 600}
    assert all(row["injected_bytes"] <= budgets[row["client_group"]] for row in per_probe)
    # Dev thresholds are frozen; recalibrating with the same inputs is a no-op, not a drift.
    assert main(["calibrate", *common]) == 0
    # Pools are frozen; supplementary judging finds nothing new for the same systems.
    assert main(["judge", *common, "--stage", "supplementary", "--fake"]) == 0
    assert '"probes": 0' in capsys.readouterr().out


def test_cli_entry_point_runs_all_in_a_subprocess(example_dir, tmp_path) -> None:
    env = {**os.environ, "NO_PROXY": "*"}
    result = subprocess.run([sys.executable, "-m", "dose_mem_eval.cli", "all",
                             "--probes", str(example_dir / "probes.jsonl"),
                             "--snapshots", str(example_dir / "snapshots"), "--run", str(tmp_path / "r"),
                             "--system", "bm25", "--system", "dense-fake", "--fake", "--audit",
                             "--retry-delay", "0", "--bootstrap", "100"],
                            capture_output=True, text=True, timeout=120, env=env)
    assert result.returncode == 0, result.stderr
    assert "bm25" in result.stdout and (tmp_path / "r" / "summary.json").exists()
