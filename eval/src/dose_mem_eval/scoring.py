"""Symbolic scoring, dev-only threshold tuning, and session-clustered CIs.

No model, vector store or network dependency is imported. An unknown judge
result stays unknown: label-dependent primary estimates become null and the
full-denominator lower/upper bounds are reported instead. Attempted system
errors stay in every denominator and earn zero effectiveness (they are never
counted as a successful quiet turn).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from .judging import LABELS, USEFUL_LABELS
from .packing import DEFAULT_BUDGETS, SEPARATOR, budget_items

OK_STATUSES = frozenset({"ok", "success", "degraded"})
UNEXECUTED_STATUSES = frozenset({"blocked", "not_run", "not_executed", "unavailable", "missing_snapshot"})
UPDATE_STRATA = frozenset({"update", "chain", "update_correction"})
METRICS = ("hit_rate", "useful_precision", "harmful_rate", "need_recall", "quiet_rate", "update_correctness",
           "composite", "items_per_turn", "bytes_per_turn", "latency_p50_ms", "latency_p95_ms", "error_rate",
           "judge_missing_rate", "execution_coverage")
PAIRED_METRICS = ("composite", "hit_rate", "useful_precision", "harmful_rate", "need_recall", "quiet_rate")


def composite_score(memory_needed: bool, labels: Sequence[str], *, error: bool = False) -> int:
    """Per-turn 0/1/2 score.

    * error or any harmful injected item -> 0
    * memory needed: a necessary item -> 2; only helpful items -> 1; else 0
    * memory not needed: nothing injected -> 2; only helpful items -> 1; else 0
    """
    if any(label not in LABELS for label in labels):
        raise ValueError("unknown item label")
    if error or "harmful" in labels:
        return 0
    if memory_needed:
        if "necessary" in labels:
            return 2
        return 1 if labels and all(label == "helpful" for label in labels) else 0
    if not labels:
        return 2
    return 1 if all(label == "helpful" for label in labels) else 0


def is_attempted(status: str, attempted: Any = None) -> bool:
    if attempted is None:
        return status not in UNEXECUTED_STATUSES and not str(status).startswith("blocked_")
    if not isinstance(attempted, bool):
        raise ValueError("attempted must be boolean")
    return attempted


def score_run(run: Mapping[str, Any], judgment: Mapping[str, Any] | None) -> dict[str, Any]:
    """Turn one actual (packed) retrieval outcome into sufficient statistics.

    Query-system errors get zero effectiveness, not successful abstention.
    Missing judge labels give null effectiveness, even on system errors. An
    injected item absent from the judged pool also makes the row unknown; run
    supplementary judging instead of guessing its label.
    """
    items = run.get("injected_items", [])
    if not isinstance(items, list):
        raise ValueError("injected_items must be a list")
    ids = [str(item["item_id"]) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate injected item IDs")
    status = str(run.get("status", "ok"))
    attempted = is_attempted(status, run.get("attempted"))
    if not attempted and items:
        raise ValueError("unexecuted outcome cannot contain injected items")
    error = attempted and status not in OK_STATUSES
    valid = bool(judgment and judgment.get("status", "valid") == "valid")
    if valid and str(judgment.get("probe_id")) != str(run["probe_id"]):  # type: ignore[union-attr]
        raise ValueError("judgment probe mismatch")
    labels_by_id = {item["item_id"]: item["label"] for item in judgment.get("items", [])} if valid else {}  # type: ignore[union-attr]
    if any(label not in LABELS for label in labels_by_id.values()):
        raise ValueError("invalid stored item label")
    if valid and (not isinstance(judgment.get("memory_needed"), bool) or any(x not in labels_by_id for x in ids)):  # type: ignore[union-attr]
        valid = False
    labels = [labels_by_id[item_id] for item_id in ids] if valid else []
    needed = judgment["memory_needed"] if valid else None  # type: ignore[index]
    known = valid and attempted
    group = run.get("client_group")
    if group not in DEFAULT_BUDGETS:
        raise ValueError("run requires client_group 'assistant' or 'working'")
    latency = run.get("latency_ms")
    if latency is not None and (isinstance(latency, bool) or not isinstance(latency, (int, float))
                                or not math.isfinite(latency) or latency < 0):
        raise ValueError("invalid latency")
    if not attempted and latency is not None:
        raise ValueError("unexecuted outcome cannot have a measured latency")
    is_chain = run.get("stratum") in UPDATE_STRATA or bool(run.get("is_update_chain"))
    update = judgment.get("update_chain", {}) if valid else {}  # type: ignore[union-attr]
    update_known = bool(known and update.get("established"))
    update_correct = (bool(set(ids) & set(update.get("newer_item_ids", [])))
                      and not bool(set(ids) & set(update.get("stale_item_ids", []))) and not error) if update_known else None
    useful = sum(label in USEFUL_LABELS for label in labels) if known else None
    result = {key: run.get(key) for key in ("probe_id", "session_id", "system", "regime", "variant", "split", "stratum")}
    result.update({
        "client_group": group, "status": status, "attempted": attempted,
        "error": int(error) if attempted else None,
        "degraded": int(attempted and (status == "degraded" or bool(run.get("degraded")))),
        "judge_valid": valid, "judge_missing": int(not valid), "memory_needed": needed,
        "injected_count": len(ids), "injected_item_ids": ids,
        "injected_bytes": len(SEPARATOR.join(str(item["text"]) for item in items).encode("utf-8")) if attempted else None,
        "useful_count": useful,
        "harmful_count": labels.count("harmful") if known else None,
        "hit": int(bool(useful) and not error) if known else None,
        "need_recalled": int("necessary" in labels and not error) if known and needed else None,
        "quiet": int(not ids and not error) if known and not needed else None,
        "composite": composite_score(needed, labels, error=error) if known else None,  # type: ignore[arg-type]
        "is_update_chain": is_chain, "update_correct": int(update_correct) if update_correct is not None else None,
        "latency_ms": latency})
    return result


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def metric_value(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    """Primary estimates are null if unresolved labels could change them."""
    n = len(rows)
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric}")
    if metric in {"latency_p50_ms", "latency_p95_ms"}:
        values = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
        missing = n - len(values)
        return {"value": _quantile(values, 0.5 if metric == "latency_p50_ms" else 0.95) if not missing else None,
                "denominator": n, "known_n": len(values), "unknown_n": missing}
    if metric in {"composite", "hit_rate", "items_per_turn", "bytes_per_turn", "error_rate",
                  "judge_missing_rate", "execution_coverage"}:
        key = {"hit_rate": "hit", "items_per_turn": "injected_count", "bytes_per_turn": "injected_bytes",
               "error_rate": "error", "judge_missing_rate": "judge_missing",
               "execution_coverage": "attempted"}.get(metric, metric)
        values = [row[key] for row in rows if row.get(key) is not None]
        missing = n - len(values)
        result = {"value": sum(values) / n if n and not missing else None,
                  "numerator": sum(values), "denominator": n, "known_n": len(values), "unknown_n": missing}
        if metric in {"composite", "hit_rate"} and n:
            best = 2 if metric == "composite" else 1
            result.update(lower_bound=sum(values) / n, upper_bound=(sum(values) + best * missing) / n)
        return result
    if metric in {"useful_precision", "harmful_rate"}:
        key = "useful_count" if metric == "useful_precision" else "harmful_count"
        denom = sum(row["injected_count"] for row in rows)
        unknown = sum(row["injected_count"] for row in rows if row.get(key) is None)
        numerator = sum(row[key] for row in rows if row.get(key) is not None)
        unexecuted = sum(not row.get("attempted", True) for row in rows)
        return {"value": numerator / denom if denom and not unknown and not unexecuted else None,
                "numerator": numerator, "denominator": denom, "known_n": denom - unknown, "unknown_n": unknown,
                "unexecuted_probe_n": unexecuted,
                "denominator_scope": "observed_injected_items; unexecuted_item_count_unknown" if unexecuted else "all_injected_items",
                "lower_bound": numerator / denom if denom and not unexecuted else None,
                "upper_bound": (numerator + unknown) / denom if denom and not unexecuted else None}
    if metric == "update_correctness":
        eligible = [row for row in rows if row["is_update_chain"]]
        unknown = sum(row.get("update_correct") is None for row in eligible)
        numerator = sum(row["update_correct"] for row in eligible if row.get("update_correct") is not None)
        denom = len(eligible) - unknown
    else:
        target_needed = metric == "need_recall"
        key = "need_recalled" if target_needed else "quiet"
        eligible = [row for row in rows if row["memory_needed"] is target_needed]
        unknown = sum(row["memory_needed"] is None for row in rows) + sum(row.get(key) is None for row in eligible)
        numerator = sum(row[key] for row in eligible if row.get(key) is not None)
        denom = sum(row.get(key) is not None for row in eligible)
    return {"value": numerator / denom if denom and not unknown else None,
            "numerator": numerator, "denominator": denom + unknown, "known_n": denom, "unknown_n": unknown,
            "lower_bound": numerator / (denom + unknown) if denom + unknown else None,
            "upper_bound": (numerator + unknown) / (denom + unknown) if denom + unknown else None}


def _clusters(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("session_id"):
            raise ValueError("session_id required for clustered bootstrap")
        groups[str(row["session_id"])].append(row)
    return groups


def clustered_ci(rows: Sequence[Mapping[str, Any]], metric: str, *, bootstrap_samples: int = 2000,
                 seed: int = 10) -> dict[str, Any]:
    """Percentile 95% CI resampling whole sessions, keeping every probe and error in a sampled session."""
    if bootstrap_samples < 100:
        raise ValueError("at least 100 bootstrap samples required")
    result = metric_value(rows, metric)
    groups = _clusters(rows)
    result.update(ci_low=None, ci_high=None, clusters=len(groups), bootstrap_samples=bootstrap_samples,
                  ci_method="session_cluster_percentile_95")
    if result["value"] is None or len(groups) < 2:
        result["ci_reason"] = "missing_or_undefined_metric" if result["value"] is None else "fewer_than_two_sessions"
        return result
    clusters = list(groups.values())
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = [row for cluster in rng.choices(clusters, k=len(clusters)) for row in cluster]
        value = metric_value(sampled, metric)["value"]
        if value is not None:
            estimates.append(value)
    result.update(ci_low=_quantile(estimates, 0.025), ci_high=_quantile(estimates, 0.975),
                  bootstrap_defined_samples=len(estimates))
    return result


def paired_difference(rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]], metric: str, *,
                      bootstrap_samples: int = 2000, seed: int = 10) -> dict[str, Any]:
    """metric(A) - metric(B) on the same probes; both sides use the SAME resampled sessions."""
    by_a = {str(row["probe_id"]): row for row in rows_a}
    by_b = {str(row["probe_id"]): row for row in rows_b}
    if by_a.keys() != by_b.keys():
        raise ValueError("paired comparison requires identical probe sets")
    a_value, b_value = metric_value(rows_a, metric)["value"], metric_value(rows_b, metric)["value"]
    sessions: dict[str, list[str]] = defaultdict(list)
    for probe_id, row in sorted(by_a.items()):
        if str(row.get("session_id")) != str(by_b[probe_id].get("session_id")):
            raise ValueError("paired rows disagree on session_id")
        sessions[str(row["session_id"])].append(probe_id)
    result: dict[str, Any] = {"metric": metric, "a_value": a_value, "b_value": b_value,
                              "difference": a_value - b_value if a_value is not None and b_value is not None else None,
                              "ci_low": None, "ci_high": None, "clusters": len(sessions), "n": len(by_a),
                              "bootstrap_samples": bootstrap_samples, "ci_method": "paired_session_cluster_percentile_95"}
    if result["difference"] is None or len(sessions) < 2:
        result["ci_reason"] = "missing_or_undefined_metric" if result["difference"] is None else "fewer_than_two_sessions"
        return result
    clusters = list(sessions.values())
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(bootstrap_samples):
        chosen = [probe_id for cluster in rng.choices(clusters, k=len(clusters)) for probe_id in cluster]
        left = metric_value([by_a[p] for p in chosen], metric)["value"]
        right = metric_value([by_b[p] for p in chosen], metric)["value"]
        if left is not None and right is not None:
            estimates.append(left - right)
    result.update(ci_low=_quantile(estimates, 0.025), ci_high=_quantile(estimates, 0.975),
                  bootstrap_defined_samples=len(estimates))
    return result


def summarize(scored_rows: Sequence[Mapping[str, Any]], *, split: str = "test", bootstrap_samples: int = 2000,
              seed: int = 10, paired_metrics: Sequence[str] = PAIRED_METRICS) -> dict[str, Any]:
    """Per (system, regime, variant, client_group) metrics with CIs, plus paired system differences."""
    selected = [row for row in scored_rows if row.get("split") == split]
    identities = [(str(row["probe_id"]), str(row["system"]), str(row["regime"]), str(row.get("variant") or "default"))
                  for row in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate probe-system-regime-variant outcome; retries are not extra probes")
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        for group in (row["client_group"], "all"):
            groups[str(row["system"]), str(row["regime"]), str(row.get("variant") or "default"), group].append(row)
    output: list[dict[str, Any]] = []
    for (system, regime, variant, group), rows in sorted(groups.items()):
        output.append({"system": system, "regime": regime, "variant": variant, "client_group": group,
                       "n": len(rows), "errors": sum(row["error"] or 0 for row in rows),
                       "attempted_n": sum(row["attempted"] for row in rows),
                       "blocked_n": sum(not row["attempted"] for row in rows),
                       "status_counts": dict(Counter(str(row["status"]) for row in rows)),
                       "degraded": sum(row["degraded"] for row in rows),
                       "judge_missing": sum(row["judge_missing"] for row in rows),
                       "metrics": {metric: clustered_ci(rows, metric, bootstrap_samples=bootstrap_samples, seed=seed)
                                   for metric in METRICS}})
    paired: list[dict[str, Any]] = []
    by_regime: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(dict)
    for (system, regime, variant, group), rows in groups.items():
        by_regime[regime, group][f"{system}/{variant}"] = rows
    for (regime, group), systems in sorted(by_regime.items()):
        for left, right in combinations(sorted(systems), 2):
            for metric in paired_metrics:
                try:
                    diff = paired_difference(systems[left], systems[right], metric,
                                             bootstrap_samples=bootstrap_samples, seed=seed)
                except ValueError as exc:
                    diff = {"metric": metric, "difference": None, "ci_reason": str(exc)}
                paired.append({"regime": regime, "client_group": group, "a": left, "b": right, **diff})
    return {"split": split, "bootstrap_seed": seed, "bootstrap_samples": bootstrap_samples, "systems": output,
            "paired_differences": paired,
            "missing_judgments_policy": "unknown; primary metrics null, full-denominator bounds retained",
            "errors_policy": "attempted failures retained in denominators; effectiveness zero only with valid judgment; unexecuted outcomes unknown"}


def tune_threshold(dev_runs: Sequence[Mapping[str, Any]], judgments: Mapping[str, Mapping[str, Any]], *,
                   system: str, byte_budgets: Mapping[str, int] = DEFAULT_BUDGETS) -> dict[str, Any]:
    """One score threshold per system across client groups, selected only on complete dev.

    Every distinct dev score (plus one just above the maximum, i.e. inject
    nothing) is a candidate. The threshold maximizing mean dev composite wins;
    ties select the highest threshold (the quietest operating point). Test rows
    and missing dev labels are refused.
    """
    if not dev_runs or any(row.get("split") != "dev" for row in dev_runs):
        raise ValueError("threshold tuning requires nonempty dev-only rows")
    if any(row.get("system") != system for row in dev_runs):
        raise ValueError("threshold tuning cannot mix systems")
    if len({row["probe_id"] for row in dev_runs}) != len(dev_runs):
        raise ValueError("duplicate dev probe")
    scores = sorted({float(item["score"]) for row in dev_runs for item in row.get("ranked_items", [])})
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("nonfinite retrieval score")
    candidates = scores + ([math.nextafter(scores[-1], math.inf)] if scores else [0.0])
    evaluated: list[dict[str, Any]] = []
    for threshold in candidates:
        rows: list[dict[str, Any]] = []
        for run in dev_runs:
            chosen = budget_items(run.get("ranked_items", []), byte_budgets[run["client_group"]], threshold=threshold)
            scored = score_run(dict(run, injected_items=chosen), judgments.get(str(run["probe_id"])))
            if not scored["judge_valid"]:
                raise ValueError("cannot tune threshold with missing dev judgments")
            if not scored["attempted"]:
                raise ValueError("cannot tune threshold with unexecuted dev runs")
            rows.append(scored)
        evaluated.append({"threshold": threshold, "mean_composite": sum(row["composite"] for row in rows) / len(rows),
                          "injected_items": sum(row["injected_count"] for row in rows)})
    best = max(evaluated, key=lambda entry: (entry["mean_composite"], entry["threshold"]))
    frozen_input = {"system": system, "dev_runs": list(dev_runs),
                    "judgments": {str(row["probe_id"]): judgments[str(row["probe_id"])] for row in dev_runs},
                    "byte_budgets": dict(byte_budgets)}
    return {"system": system, "threshold": best["threshold"], "dev_mean_composite": best["mean_composite"],
            "dev_n": len(dev_runs), "dev_probe_ids": sorted(str(row["probe_id"]) for row in dev_runs),
            "scope": "one_score_threshold_per_system_all_client_groups", "split": "dev", "frozen": True,
            "tie_break": "highest_threshold", "evaluated": evaluated,
            "input_sha256": hashlib.sha256(json.dumps(frozen_input, ensure_ascii=False, sort_keys=True,
                                                      separators=(",", ":"), allow_nan=False).encode()).hexdigest()}


def freeze_threshold(path: str | Path, selection: Mapping[str, Any]) -> None:
    """Exclusive creation: a frozen threshold can never silently drift."""
    if not selection.get("frozen") or selection.get("split") != "dev":
        raise ValueError("not a dev-frozen selection")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
