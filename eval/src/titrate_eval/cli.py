"""``titrate-eval`` command line.

    titrate-eval mine      --sessions S --snapshots DIR --test-from DATE --out P
    titrate-eval retrieve  --probes P --snapshots DIR --run RUN --system bm25 [--system dense ...]
    titrate-eval pool      --probes P --run RUN
    titrate-eval judge     --probes P --run RUN --stage canonical|audit|supplementary [--fake]
    titrate-eval calibrate --probes P --run RUN
    titrate-eval score     --probes P --run RUN
    titrate-eval report    --probes P --run RUN
    titrate-eval all       ... (every stage in order; handy for the offline example)

Provider settings (judge, audit judge, embeddings, OpenViking, mem0) come from
``--config titrate.toml`` and/or ``TITRATE_*`` environment variables; see
``titrate_eval.providers``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import pipeline
from .data import Corpus, load_probes
from .packing import parse_budgets
from .providers import endpoint, load_config


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--probes", required=True, type=Path, help="probes JSONL")
    parser.add_argument("--run", required=True, type=Path, help="run directory (outputs)")
    parser.add_argument("--config", type=Path, default=None, help="TOML provider config")


def _budget(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--budget", action="append", default=[], metavar="GROUP=BYTES",
                        help="override a byte budget, e.g. --budget assistant=1500 (defaults 1200/600)")


def _judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fake", action="store_true",
                        help="offline deterministic fake judge (tests/demo only; labels are meaningless)")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--audit-fraction", type=float, default=0.2)
    parser.add_argument("--retry-delay", type=float, default=8.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="titrate-eval", description="Blind, equal-budget memory-layer evaluation.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("mine", help="turn normalized session logs into probes JSONL")
    p.add_argument("--sessions", required=True, type=Path, help="normalized sessions JSONL (see titrate_eval.mining)")
    p.add_argument("--snapshots", required=True, type=Path, help="snapshot directory (epoch ids)")
    p.add_argument("--test-from", required=True, help="ISO date: sessions starting on/after it are test")
    p.add_argument("--out", required=True, type=Path)

    p = sub.add_parser("retrieve", help="query each system on every probe at its own snapshot")
    _common(p)
    p.add_argument("--snapshots", required=True, type=Path, help="directory of <epoch>.jsonl snapshots")
    p.add_argument("--system", action="append", required=True,
                   help="bm25 | dense | dense-fake | openviking | mem0 (repeatable)")

    p = sub.add_parser("pool", help="freeze the judge pool: union of visible top-10s (+ BM25 extras)")
    _common(p)
    p.add_argument("--extra-bm25", type=int, default=pipeline.EXTRA_BM25_DEFAULT)

    p = sub.add_parser("judge", help="blind LLM judging of the frozen pool")
    _common(p)
    p.add_argument("--stage", choices=["canonical", "audit", "supplementary"], default="canonical")
    _judge_args(p)

    p = sub.add_parser("calibrate", help="freeze one score threshold per system on dev only")
    _common(p)
    _budget(p)

    p = sub.add_parser("score", help="pack under equal byte budgets and score every probe")
    _common(p)
    _budget(p)

    p = sub.add_parser("report", help="write summary.json (bootstrap CIs, paired diffs, audit kappa) + per-probe.jsonl")
    _common(p)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--split", choices=["dev", "test"], default="test")

    p = sub.add_parser("all", help="retrieve, pool, judge (+audit), calibrate, score, report")
    _common(p)
    p.add_argument("--snapshots", required=True, type=Path)
    p.add_argument("--system", action="append", required=True)
    p.add_argument("--audit", action="store_true", help="also run the audit judge on the fixed sample")
    p.add_argument("--bootstrap", type=int, default=2000)
    _judge_args(p)
    _budget(p)
    return parser


def _transport(args: argparse.Namespace, config: dict[str, Any], stage: str):
    from .judge_runner import http_transport

    if args.fake:
        from .fake_judge import fake_transport

        return fake_transport("audit" if stage == "audit" else "primary"), f"fake-{stage}-judge"
    section = "audit_judge" if stage == "audit" else "judge"
    target = endpoint(section, config)
    return http_transport(target), target.model


def _retrieve(args: argparse.Namespace, config: dict[str, Any], probes) -> list[dict[str, Any]]:
    from .systems import build_system

    corpus = Corpus.load(args.snapshots)
    return [pipeline.retrieve(probes, corpus, build_system(name, config), args.run) for name in args.system]


def _judge(args: argparse.Namespace, config: dict[str, Any], probes, stage: str) -> dict[str, Any]:
    transport, model = _transport(args, config, stage)
    return pipeline.judge(probes, args.run, stage=stage, transport=transport, model=model, seed=args.seed,
                          audit_fraction=args.audit_fraction, retry_delay=args.retry_delay)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "mine":
        from .io import read_jsonl, write_jsonl
        from .mining import mine

        mined = mine(read_jsonl(args.sessions), epochs=Corpus.load(args.snapshots).order, test_from=args.test_from)
        write_jsonl(args.out, mined)
        load_probes(args.out)  # validate the schema and the dev/test session rule
        print(json.dumps({"probes": len(mined), "out": str(args.out)}))
        return 0
    config = load_config(args.config)
    probes = load_probes(args.probes)
    args.run.mkdir(parents=True, exist_ok=True)
    budgets = parse_budgets(getattr(args, "budget", None))
    result: Any
    if args.command == "retrieve":
        result = _retrieve(args, config, probes)
    elif args.command == "pool":
        result = pipeline.build_pools(probes, args.run, extra_bm25=args.extra_bm25)
    elif args.command == "judge":
        result = _judge(args, config, probes, args.stage)
    elif args.command == "calibrate":
        result = pipeline.calibrate(probes, args.run, budgets=budgets)
    elif args.command == "score":
        result = pipeline.score(probes, args.run, budgets=budgets)
    elif args.command == "report":
        summary = pipeline.report(probes, args.run, bootstrap_samples=args.bootstrap, seed=args.seed, split=args.split)
        print("\n".join(pipeline.headline(summary)))
        result = {"summary": str(args.run / "summary.json"), "per_probe": str(args.run / "per-probe.jsonl"),
                  "audit_item_kappa": (summary.get("audit") or {}).get("item_labels", {}).get("kappa")}
    else:  # all
        result = {"retrieve": _retrieve(args, config, probes),
                  "pool": pipeline.build_pools(probes, args.run),
                  "judge": _judge(args, config, probes, "canonical")}
        if args.audit:
            result["audit"] = _judge(args, config, probes, "audit")
        result["calibrate"] = pipeline.calibrate(probes, args.run, budgets=budgets)
        result["score"] = pipeline.score(probes, args.run, budgets=budgets)
        summary = pipeline.report(probes, args.run, bootstrap_samples=args.bootstrap, seed=args.seed)
        print("\n".join(pipeline.headline(summary)))
        result["report"] = str(args.run / "summary.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
