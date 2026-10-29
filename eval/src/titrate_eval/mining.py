"""Generic probe miner: normalized session logs -> probes JSONL.

Every agent stores sessions differently, so this module deliberately accepts
one simple, neutral input format and leaves log parsing to a small adapter you
write (or ask your coding agent to write) for your own tools::

    {"session_id": "s-01", "client_group": "working",          # one session per line
     "turns": [{"role": "user", "text": "...", "timestamp": "2026-01-06T09:00:00+00:00"},
               {"role": "assistant", "text": "...", "timestamp": "..."}]}

For every user turn it emits a probe with the previous turn, the next
``hindsight`` turns (judge-only), the latest snapshot epoch at or before the
turn time, and a dev/test split decided per *session* by its start time, so no
session straddles the split. Project hints are left to a callable you supply,
because they must be derived from information available at query time only
(e.g. the working directory or an alias named in the turn), never hindsight.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

MAX_HINDSIGHT = 6


def _time(value: str) -> datetime:
    """Parse an ISO date or datetime; naive values are taken as UTC."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def epoch_for(at: datetime, epochs: Sequence[str]) -> str | None:
    """Latest snapshot id (ISO date/datetime) not after ``at``; None if the turn predates all snapshots."""
    chosen = None
    for epoch in sorted(epochs, key=_time):
        if _time(epoch) <= at:
            chosen = epoch
    return chosen


def mine(sessions: Iterable[Mapping[str, Any]], *, epochs: Sequence[str], test_from: str,
         project_hint: Callable[[Mapping[str, Any], Mapping[str, Any]], str | None] | None = None,
         min_chars: int = 8, hindsight_turns: int = MAX_HINDSIGHT) -> list[dict[str, Any]]:
    """Return probe dicts. Sessions starting at/after ``test_from`` are test, earlier ones dev."""
    boundary = _time(test_from)
    probes: list[dict[str, Any]] = []
    for session in sessions:
        turns = [t for t in session.get("turns", []) if str(t.get("text", "")).strip()]
        if not turns:
            continue
        start = _time(turns[0]["timestamp"])
        split = "test" if start >= boundary else "dev"
        for index, turn in enumerate(turns):
            if turn.get("role") != "user" or len(turn["text"].strip()) < min_chars:
                continue
            at = _time(turn["timestamp"])
            epoch = epoch_for(at, epochs)
            if epoch is None:
                continue
            previous = turns[index - 1]["text"] if index else None
            probes.append({
                "probe_id": f"{session['session_id']}-{index:04d}",
                "session_id": str(session["session_id"]),
                "client_group": session.get("client_group", "assistant"),
                "split": split, "epoch": epoch,
                "project_hint": project_hint(session, turn) if project_hint else None,
                "user_turn": turn["text"], "previous_turn": previous,
                "hindsight": [{"role": t.get("role", "unknown"), "text": t["text"]}
                              for t in turns[index + 1:index + 1 + hindsight_turns]],
            })
    return probes
