from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import item, judgment, run
from titrate_eval.scoring import (clustered_ci, composite_score, freeze_threshold, metric_value,
                                  paired_difference, score_run, summarize, tune_threshold)


@pytest.mark.parametrize(("needed", "labels", "expected"), [
    (True, ["necessary"], 2), (True, ["necessary", "irrelevant"], 2),
    (True, ["necessary", "harmful"], 0), (True, ["helpful"], 1),
    (True, ["helpful", "irrelevant"], 0), (True, [], 0),
    (False, [], 2), (False, ["helpful"], 1), (False, ["irrelevant"], 0),
])
def test_symbolic_composite(needed: bool, labels: list[str], expected: int) -> None:
    assert composite_score(needed, labels) == expected


def test_hit_rate_counts_turns_with_any_useful_item() -> None:
    hit = score_run(run(), judgment(labels={"a": "helpful"}))
    miss = score_run(run("p2", "s2"), judgment("p2", labels={"a": "irrelevant"}))
    empty = score_run(run("p3", "s3", injected_items=[]), judgment("p3", needed=False, labels={}))
    assert metric_value([hit, miss, empty], "hit_rate")["value"] == pytest.approx(1 / 3)


def test_system_error_is_not_successful_quiet_and_stays_in_recall_denominator() -> None:
    success = score_run(run(), judgment())
    error = score_run(run("p2", "s2", status="timeout", injected_items=[]), judgment("p2"))
    assert metric_value([success, error], "need_recall")["value"] == 0.5
    quiet_error = score_run(run(status="error", injected_items=[]), judgment(needed=False, labels={}))
    assert quiet_error["composite"] == 0
    assert quiet_error["quiet"] == 0
    assert metric_value([success, error], "error_rate")["value"] == 0.5


def test_missing_judge_is_unknown_with_full_denominator_bounds() -> None:
    success = score_run(run(), judgment())
    unknown = score_run(run("p2", "s2"), None)
    assert unknown["composite"] is None
    mean = metric_value([success, unknown], "composite")
    assert mean["value"] is None
    assert mean["denominator"] == 2
    assert (mean["lower_bound"], mean["upper_bound"]) == (1.0, 2.0)
    recall = metric_value([success, unknown], "need_recall")
    assert recall["value"] is None and recall["unknown_n"] == 1
    assert recall["lower_bound"] == 0.5
    # An injected item missing from the judged pool makes the row unknown, never zero.
    assert score_run(run(), judgment(labels={}))["composite"] is None


def test_precision_is_micro_item_share_and_empty_is_undefined() -> None:
    first = score_run(run(), judgment(labels={"a": "helpful"}))
    second = score_run(run("p2", "s2", injected_items=[item("b"), item("c")]),
                       judgment("p2", labels={"b": "harmful", "c": "irrelevant"}))
    assert metric_value([first, second], "useful_precision")["value"] == pytest.approx(1 / 3)
    assert metric_value([first, second], "harmful_rate")["value"] == pytest.approx(1 / 3)
    empty = score_run(run(injected_items=[]), judgment(needed=False, labels={}))
    assert metric_value([empty], "useful_precision")["value"] is None


def test_quiet_rate_and_items_bytes_per_turn() -> None:
    quiet = score_run(run(injected_items=[]), judgment(needed=False, labels={}))
    noisy = score_run(run("p2", "s2"), judgment("p2", needed=False, labels={"a": "irrelevant"}))
    assert metric_value([quiet, noisy], "quiet_rate")["value"] == 0.5
    assert metric_value([quiet, noisy], "items_per_turn")["value"] == 0.5
    assert metric_value([quiet, noisy], "bytes_per_turn")["value"] == len("Use the newer setting.".encode()) / 2


def test_update_correctness_requires_new_fact_and_no_old_fact() -> None:
    labels = judgment(labels={"a": "necessary", "b": "harmful"}, update_chain={
        "established": True, "newer_item_ids": ["a"], "stale_item_ids": ["b"]})
    current = score_run(run(stratum="update"), labels)
    stale = score_run(run(stratum="update", injected_items=[item("a"), item("b")]), labels)
    absent = score_run(run(stratum="update", injected_items=[]), labels)
    assert [value["update_correct"] for value in [current, stale, absent]] == [1, 0, 0]
    assert score_run(run(stratum="update"), judgment())["update_correct"] is None


def test_blocked_system_is_not_executed_or_quiet_even_with_valid_judge() -> None:
    blocked = run(status="blocked", attempted=False, injected_items=[], latency_ms=None)
    scored = score_run(blocked, judgment(needed=False, labels={}))
    assert scored["attempted"] is False and scored["error"] is None
    assert scored["quiet"] is None and scored["composite"] is None
    assert metric_value([scored], "error_rate")["value"] is None
    assert metric_value([scored], "execution_coverage")["value"] == 0
    summary = summarize([scored], bootstrap_samples=100)
    assert summary["systems"][0]["blocked_n"] == 1
    assert summary["systems"][0]["metrics"]["composite"]["value"] is None


@pytest.mark.parametrize("field,value", [("latency_ms", 0), ("injected_items", [item()])])
def test_blocked_rows_cannot_fabricate_measurements(field: str, value) -> None:
    blocked = run(status="blocked", attempted=False, injected_items=[], latency_ms=None)
    blocked[field] = value
    with pytest.raises(ValueError, match="unexecuted"):
        score_run(blocked, None)


def test_bootstrap_resamples_sessions_including_failures() -> None:
    # A long all-success session and a short error session. The session
    # bootstrap can select only the error session, so its lower CI is zero.
    rows = [score_run(run(f"p{i}", "long"), judgment(f"p{i}")) for i in range(20)]
    rows.append(score_run(run("bad", "short", status="timeout", injected_items=[]), judgment("bad")))
    result = clustered_ci(rows, "composite", bootstrap_samples=1000, seed=10)
    assert result["clusters"] == 2
    assert result["ci_low"] == 0
    assert result["ci_high"] == 2
    assert result["denominator"] == 21


def test_bootstrap_is_seeded_and_needs_two_sessions() -> None:
    rows = [score_run(run(f"p{i}", f"s{i % 4}"), judgment(f"p{i}", labels={"a": "necessary" if i % 3 else "irrelevant"}))
            for i in range(12)]
    first = clustered_ci(rows, "composite", bootstrap_samples=300, seed=7)
    assert first == clustered_ci(rows, "composite", bootstrap_samples=300, seed=7)
    single = clustered_ci(rows[:1], "composite", bootstrap_samples=100)
    assert single["ci_low"] is None and single["ci_reason"] == "fewer_than_two_sessions"


def test_paired_difference_resamples_the_same_sessions_for_both_systems() -> None:
    a = [score_run(run(f"p{i}", f"s{i % 3}"), judgment(f"p{i}")) for i in range(9)]
    b = [score_run(run(f"p{i}", f"s{i % 3}", system="bm25"), judgment(f"p{i}", labels={"a": "irrelevant"}))
         for i in range(9)]
    diff = paired_difference(a, b, "composite", bootstrap_samples=200)
    assert diff["difference"] == 2.0
    # Every resample has A=2 and B=0 per probe, so the paired CI is degenerate at 2.
    assert diff["ci_low"] == diff["ci_high"] == 2.0
    with pytest.raises(ValueError, match="identical probe sets"):
        paired_difference(a, b[:-1], "composite", bootstrap_samples=100)


def test_latency_uses_observed_failed_attempt_elapsed_time() -> None:
    rows = [score_run(run(latency_ms=5), judgment()),
            score_run(run("p2", "s2", status="timeout", latency_ms=1000), judgment("p2"))]
    assert metric_value(rows, "latency_p50_ms")["value"] == 502.5
    assert metric_value(rows, "latency_p95_ms")["value"] == pytest.approx(950.25)


def test_threshold_is_dev_only_single_threshold_and_immutable(tmp_path: Path) -> None:
    needed = run("needed", "s1", split="dev", ranked_items=[item(score=0.8)])
    routine = run("routine", "s2", split="dev", ranked_items=[item(score=0.2)])
    labels = {"needed": judgment("needed"), "routine": judgment("routine", needed=False, labels={"a": "irrelevant"})}
    selected = tune_threshold([needed, routine], labels, system="dense")
    assert selected["threshold"] == 0.8
    assert selected["dev_mean_composite"] == 2
    target = tmp_path / "threshold.json"
    freeze_threshold(target, selected)
    with pytest.raises(FileExistsError):
        freeze_threshold(target, selected)
    with pytest.raises(ValueError, match="dev-only"):
        tune_threshold([run()], {"p1": judgment()}, system="dense")
    with pytest.raises(ValueError, match="missing dev"):
        tune_threshold([needed], {}, system="dense")


def test_threshold_ties_choose_the_higher_threshold() -> None:
    # Both items are irrelevant on a turn that needs no memory: any threshold that
    # injects nothing scores 2; the highest (inject nothing) must win the tie.
    dev = run("d", "s1", split="dev", ranked_items=[item("a", score=0.3), item("b", "other", score=0.6)])
    labels = {"d": judgment("d", needed=False, labels={"a": "irrelevant", "b": "irrelevant"})}
    selected = tune_threshold([dev], labels, system="dense")
    best = [e for e in selected["evaluated"] if e["mean_composite"] == 2]
    assert len(best) == 1 and selected["threshold"] > 0.6
    # Equal composite at 0.3 and 0.6 (both inject only helpful "b") -> the higher wins.
    labels = {"d": judgment("d", needed=False, labels={"a": "helpful", "b": "helpful"})}
    dev2 = run("d", "s1", split="dev", ranked_items=[item("b", "other", score=0.6), item("a", score=0.3)])
    selected = tune_threshold([dev2], labels, system="dense")
    assert selected["threshold"] > 0.6  # quiet beats helpful (2 > 1)
    tie_rows = [e for e in selected["evaluated"] if e["mean_composite"] == 1]
    assert {e["threshold"] for e in tie_rows} == {0.3, 0.6}


def test_test_summary_excludes_dev_and_rejects_duplicate_outcomes() -> None:
    rows = [score_run(run(), judgment()), score_run(run("dev", "dev", split="dev"), judgment("dev"))]
    result = summarize(rows, bootstrap_samples=100)
    assert all(group["n"] == 1 for group in result["systems"])
    scored = score_run(run(), judgment())
    with pytest.raises(ValueError, match="retries are not extra probes"):
        summarize([scored, scored], bootstrap_samples=100)


def test_summary_reports_paired_differences_between_systems() -> None:
    rows = []
    for i in range(6):
        rows.append(score_run(run(f"p{i}", f"s{i % 3}"), judgment(f"p{i}")))
        rows.append(score_run(run(f"p{i}", f"s{i % 3}", system="bm25"), judgment(f"p{i}")))
    summary = summarize(rows, bootstrap_samples=100)
    pairs = [p for p in summary["paired_differences"] if p["client_group"] == "all" and p["metric"] == "composite"]
    assert pairs and pairs[0]["difference"] == 0.0
    json.dumps(summary, allow_nan=False)
