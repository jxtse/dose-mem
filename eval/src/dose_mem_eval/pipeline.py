"""End-to-end stages operating on a run directory.

Layout of ``<run>/``::

    retrieval/<system>.jsonl    one record per probe: status, latency, raw + visible ranked items
    pools.json                  frozen judge pool per probe (union of visible top-10s)
    judgments/<stage>/*.json    raw judge records (canonical, audit, supplementary)
    audit-selection.json        fixed audit sample (seeded, drawn from ALL probes)
    thresholds/<system>.json    dev-frozen threshold per system (create-once)
    scored.jsonl                one scored row per probe x system x regime
    summary.json, per-probe.jsonl
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import Corpus, Probe
from .io import read_json, read_jsonl, write_json, write_jsonl
from .judge_runner import Transport, acquire, load_stage
from .judging import audit_agreement, select_audit_probe_ids
from .packing import DEFAULT_BUDGETS, budget_items, visible
from .pooling import judged_ids, merge_judgments, pool_candidates, unjudged_items
from .scoring import freeze_threshold, score_run, summarize, tune_threshold
from .systems import MemorySystem
from .systems.baselines import BM25System

EXTRA_BM25_DEFAULT = 5


def _probe_map(probes: Sequence[Probe]) -> dict[str, dict[str, Any]]:
    return {probe.probe_id: probe.to_dict() for probe in probes}


def retrieval_systems(run: Path) -> list[str]:
    folder = run / "retrieval"
    return sorted(path.stem for path in folder.glob("*.jsonl")) if folder.exists() else []


def load_retrieval(run: Path, system: str) -> dict[str, dict[str, Any]]:
    return {row["probe_id"]: row for row in read_jsonl(run / "retrieval" / f"{system}.jsonl")}


# --------------------------------------------------------------------------- retrieve
def retrieve(probes: Sequence[Probe], corpus: Corpus, system: MemorySystem, run: Path, *,
             extra_unscoped: int = EXTRA_BM25_DEFAULT) -> dict[str, Any]:
    """Prepare ``system`` on all snapshots, then query every probe against its own epoch."""
    for probe in probes:
        corpus.documents(probe.epoch)  # fail fast on an unknown epoch
    started = time.perf_counter()
    system.prepare(corpus)
    prepare_ms = (time.perf_counter() - started) * 1000
    records = []
    for probe in probes:
        view = probe.system_view()
        started = time.perf_counter()
        record: dict[str, Any] = {"probe_id": probe.probe_id, "system": system.name, "epoch": probe.epoch,
                                  "project_hint": probe.project_hint, "attempted": True}
        try:
            items = system.retrieve(view, probe.epoch)
            record.update(status="ok", error=None, items=items, ranked_items=visible(items))
        except Exception as exc:  # noqa: BLE001 - a per-probe system failure is an outcome, not a crash
            record.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500], items=[], ranked_items=[],
                          traceback=traceback.format_exc(limit=3))
        record["latency_ms"] = (time.perf_counter() - started) * 1000
        if isinstance(system, BM25System) and extra_unscoped and record["status"] == "ok":
            # Unscoped extras give the judge cross-project / negative examples for the pool.
            record["extra_items"] = visible(system.retrieve(view, probe.epoch, scoped=False, limit=extra_unscoped))
        records.append(record)
    write_jsonl(run / "retrieval" / f"{system.name}.jsonl", records)
    return {"system": system.name, "probes": len(records), "errors": sum(r["status"] != "ok" for r in records),
            "prepare_ms": round(prepare_ms, 1)}


# --------------------------------------------------------------------------- pool
def build_pools(probes: Sequence[Probe], run: Path, *, extra_bm25: int = EXTRA_BM25_DEFAULT) -> dict[str, Any]:
    systems = retrieval_systems(run)
    if not systems:
        raise RuntimeError("no retrieval outputs; run `dose-mem-eval retrieve` first")
    records = {system: load_retrieval(run, system) for system in systems}
    pools: dict[str, list[dict[str, Any]]] = {}
    for probe in probes:
        ranked = {system: records[system].get(probe.probe_id, {}).get("items", []) for system in systems}
        extra = records.get("bm25", {}).get(probe.probe_id, {}).get("extra_items", [])[:extra_bm25]
        pools[probe.probe_id] = pool_candidates(ranked, extra=extra)
    target = run / "pools.json"
    if target.exists() and read_json(target)["pools"] != pools:
        raise RuntimeError("pools.json is frozen and differs; new items belong in supplementary judging")
    write_json(target, {"systems": systems, "pools": pools})
    return {"probes": len(pools), "systems": systems, "items": sum(len(v) for v in pools.values())}


# --------------------------------------------------------------------------- judge
def judge(probes: Sequence[Probe], run: Path, *, stage: str, transport: Transport, model: str, seed: int = 10,
          audit_fraction: float = 0.2, retry_delay: float = 8.0) -> dict[str, Any]:
    probe_map = _probe_map(probes)
    directory = run / "judgments"
    jobs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    if stage in ("canonical", "audit"):
        pools = read_json(run / "pools.json")["pools"]
        selected = set(probe_map)
        if stage == "audit":
            selection_path = run / "audit-selection.json"
            ids = select_audit_probe_ids(list(probe_map), seed=seed, fraction=audit_fraction)
            if selection_path.exists() and read_json(selection_path)["probe_ids"] != ids:
                raise RuntimeError("audit selection changed; it must stay fixed")
            write_json(selection_path, {"seed": seed, "fraction": audit_fraction, "probe_ids": ids,
                                        "labels_used_for_scores": False})
            selected = set(ids)
        jobs = [(probe_map[pid], pools[pid]) for pid in probe_map if pid in selected and pools.get(pid)]
    elif stage == "supplementary":
        canonical, _ = load_stage(directory, "canonical", probe_map)
        supplements, _ = load_stage(directory, "supplementary", probe_map)
        for pid, probe in probe_map.items():
            primary = (canonical.get(pid) or [None])[0]
            if primary is None:
                continue  # the canonical judgment is missing; this probe is unknown, not supplementable
            known = judged_ids(primary).union(*[judged_ids(s) for s in supplements.get(pid, [])])
            fresh: dict[str, dict[str, Any]] = {}
            for system in retrieval_systems(run):
                ranked = load_retrieval(run, system).get(pid, {}).get("ranked_items", [])
                for item in unjudged_items(ranked, known):
                    fresh.setdefault(item["item_id"], item)
            if fresh:
                jobs.append((probe, list(fresh.values())))
    else:
        raise ValueError("stage must be canonical, audit or supplementary")
    counts = {"stage": stage, "probes": len(jobs), "valid": 0, "invalid": 0}
    for probe, items in jobs:
        record = acquire(probe, items, stage=stage, directory=directory, transport=transport, model=model,
                         seed=seed, retry_delay=retry_delay)
        counts["valid" if record.get("judgment") else "invalid"] += 1
    return counts


def labels_for(run: Path, probes: Sequence[Probe]) -> dict[str, dict[str, Any] | None]:
    """Canonical judgment per probe merged with supplementary labels for later items."""
    probe_map = _probe_map(probes)
    canonical, _ = load_stage(run / "judgments", "canonical", probe_map)
    supplements, _ = load_stage(run / "judgments", "supplementary", probe_map)
    return {pid: merge_judgments((canonical.get(pid) or [None])[0], supplements.get(pid, [])) for pid in probe_map}


# --------------------------------------------------------------------------- calibrate / score
def _run_row(probe: Probe, record: Mapping[str, Any] | None, system: str, *, regime: str, variant: str,
             injected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = {"probe_id": probe.probe_id, "session_id": probe.session_id, "split": probe.split,
            "client_group": probe.client_group, "stratum": None, "system": system, "regime": regime, "variant": variant}
    if record is None:
        return {**base, "attempted": False, "status": "not_run", "latency_ms": None, "injected_items": [],
                "ranked_items": []}
    return {**base, "attempted": record.get("attempted", True), "status": record.get("status", "ok"),
            "latency_ms": record.get("latency_ms"), "injected_items": list(injected),
            "ranked_items": record.get("ranked_items", [])}


def calibrate(probes: Sequence[Probe], run: Path, *, budgets: Mapping[str, int] = DEFAULT_BUDGETS) -> dict[str, Any]:
    labels = labels_for(run, probes)
    dev = [probe for probe in probes if probe.split == "dev"]
    if not dev:
        raise RuntimeError("no dev probes; thresholds can only be tuned on dev")
    output: dict[str, Any] = {}
    for system in retrieval_systems(run):
        records = load_retrieval(run, system)
        rows = [_run_row(p, records.get(p.probe_id), system, regime="calibrated", variant="dev_tuned", injected=[])
                for p in dev]
        selection = tune_threshold(rows, {pid: j for pid, j in labels.items() if j}, system=system, byte_budgets=budgets)
        target = run / "thresholds" / f"{system}.json"
        if target.exists():
            if read_json(target) != selection:
                raise RuntimeError(f"{target} is frozen and the dev selection changed")
        else:
            freeze_threshold(target, selection)
        output[system] = {"threshold": selection["threshold"], "dev_mean_composite": selection["dev_mean_composite"],
                          "dev_n": selection["dev_n"]}
    return output


def score(probes: Sequence[Probe], run: Path, *, budgets: Mapping[str, int] = DEFAULT_BUDGETS) -> dict[str, Any]:
    labels = labels_for(run, probes)
    rows: list[dict[str, Any]] = []
    for system in retrieval_systems(run):
        records = load_retrieval(run, system)
        threshold_path = run / "thresholds" / f"{system}.json"
        threshold = read_json(threshold_path)["threshold"] if threshold_path.exists() else None
        for probe in probes:
            record = records.get(probe.probe_id)
            ranked = record.get("ranked_items", []) if record else []
            budget = budgets[probe.client_group]
            fixed = _run_row(probe, record, system, regime="fixed", variant="default",
                             injected=budget_items(ranked, budget))
            rows.append(score_run(fixed, labels.get(probe.probe_id)))
            if threshold is not None:
                calibrated = _run_row(probe, record, system, regime="calibrated", variant="dev_tuned",
                                      injected=budget_items(ranked, budget, threshold=threshold))
                rows.append(score_run(calibrated, labels.get(probe.probe_id)))
    write_jsonl(run / "scored.jsonl", rows)
    return {"rows": len(rows), "systems": retrieval_systems(run),
            "judge_missing": sum(row["judge_missing"] for row in rows)}


# --------------------------------------------------------------------------- report
def report(probes: Sequence[Probe], run: Path, *, bootstrap_samples: int = 2000, seed: int = 10,
           split: str = "test") -> dict[str, Any]:
    rows = read_jsonl(run / "scored.jsonl")
    summary = summarize(rows, split=split, bootstrap_samples=bootstrap_samples, seed=seed)
    probe_map = _probe_map(probes)
    canonical, invalid = load_stage(run / "judgments", "canonical", probe_map)
    audit_path = run / "audit-selection.json"
    if audit_path.exists():
        audit_labels, _ = load_stage(run / "judgments", "audit", probe_map)
        summary["audit"] = audit_agreement({k: v[0] for k, v in canonical.items()},
                                           {k: v[0] for k, v in audit_labels.items()},
                                           read_json(audit_path)["probe_ids"])
    else:
        summary["audit"] = None
    summary["judge"] = {"valid": len(canonical), "invalid": {k: v for k, v in invalid.items()},
                        "probes": len(probe_map)}
    thresholds = {}
    for path in sorted((run / "thresholds").glob("*.json")) if (run / "thresholds").exists() else []:
        thresholds[path.stem] = read_json(path)["threshold"]
    summary["thresholds"] = thresholds
    write_json(run / "summary.json", summary)
    write_jsonl(run / "per-probe.jsonl", rows)
    return summary


def headline(summary: Mapping[str, Any]) -> list[str]:
    """Compact text table of the 'all' client group for terminal output."""
    lines = [f"{'system':<14}{'regime':<12}{'n':>4} {'hit':>6} {'prec':>6} {'harm':>6} {'recall':>7} {'quiet':>6} "
             f"{'comp':>6} {'bytes':>7} {'err':>5}"]

    def fmt(value: Any, width: int) -> str:
        return f"{value:>{width}.2f}" if isinstance(value, (int, float)) else f"{'-':>{width}}"

    for entry in summary["systems"]:
        if entry["client_group"] != "all":
            continue
        m = entry["metrics"]
        lines.append(f"{entry['system']:<14}{entry['regime']:<12}{entry['n']:>4} {fmt(m['hit_rate']['value'], 6)} "
                     f"{fmt(m['useful_precision']['value'], 6)} {fmt(m['harmful_rate']['value'], 6)} "
                     f"{fmt(m['need_recall']['value'], 7)} {fmt(m['quiet_rate']['value'], 6)} "
                     f"{fmt(m['composite']['value'], 6)} {fmt(m['bytes_per_turn']['value'], 7)} "
                     f"{fmt(m['error_rate']['value'], 5)}")
    return lines
