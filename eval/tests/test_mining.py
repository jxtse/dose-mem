from __future__ import annotations

from datetime import datetime, timezone

from titrate_eval.cli import main
from titrate_eval.data import load_probes
from titrate_eval.mining import epoch_for, mine


def test_epoch_is_latest_snapshot_not_after_the_turn() -> None:
    epochs = ["2026-01-05", "2026-01-12"]
    assert epoch_for(datetime(2026, 1, 4, tzinfo=timezone.utc), epochs) is None
    assert epoch_for(datetime(2026, 1, 6, tzinfo=timezone.utc), epochs) == "2026-01-05"
    assert epoch_for(datetime(2026, 1, 12, tzinfo=timezone.utc), epochs) == "2026-01-12"


def test_mine_builds_probes_with_previous_turn_hindsight_and_session_split() -> None:
    sessions = [{"session_id": "s", "client_group": "working", "turns": [
        {"role": "user", "text": "first question here", "timestamp": "2026-01-06T09:00:00Z"},
        {"role": "assistant", "text": "answer", "timestamp": "2026-01-06T09:00:01Z"},
        {"role": "user", "text": "follow-up question", "timestamp": "2026-01-13T09:00:00Z"}]}]
    probes = mine(sessions, epochs=["2026-01-05", "2026-01-12"], test_from="2026-01-10")
    assert [p["epoch"] for p in probes] == ["2026-01-05", "2026-01-12"]
    assert {p["split"] for p in probes} == {"dev"}  # split follows the session start, not the turn
    assert probes[0]["hindsight"][0]["text"] == "answer" and probes[1]["previous_turn"] == "answer"


def test_mine_cli_on_example(example_dir, tmp_path, capsys) -> None:
    out = tmp_path / "probes.jsonl"
    assert main(["mine", "--sessions", str(example_dir / "sessions.jsonl"), "--snapshots",
                 str(example_dir / "snapshots"), "--test-from", "2026-01-12", "--out", str(out)]) == 0
    probes = load_probes(out)
    assert len(probes) == 3  # the 2-character "ok" turn is skipped
    assert {p.split for p in probes} == {"dev", "test"}
