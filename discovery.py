#!/usr/bin/env python3
"""Validate discovery reports and rank graph regions by e2e impact.

Usage:
    python discovery.py validate path/to/discovery.json
    python discovery.py rank path/to/discovery.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autokernel.discovery import (
    DiscoveryError,
    load_discovery_report,
    load_profiler_export,
    rank_operators,
    rank_regions,
    write_discovery_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and rank MotionKernel discovery reports"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "rank"):
        cmd = sub.add_parser(name)
        cmd.add_argument("report", type=Path)
        if name == "rank":
            cmd.add_argument(
                "--impact-floor",
                type=float,
                default=0.005,
                help="Minimum optimistic e2e improvement to search (default 0.5%)",
            )
    ingest = sub.add_parser(
        "ingest-profiler",
        help="Convert a FastVideo profiler export into a discovery report",
    )
    ingest.add_argument("profile", type=Path)
    ingest.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "ingest-profiler":
            report = load_profiler_export(args.profile)
            write_discovery_report(report, args.output)
            print("PROFILER_INGEST: PASS")
            print(f"operators: {len(report.operators)}")
            print(f"output: {args.output}")
            return 0
        report = load_discovery_report(args.report)
    except DiscoveryError as exc:
        print(f"DISCOVERY: FAIL\n{exc}", file=sys.stderr)
        return 2

    if args.command == "validate":
        print("DISCOVERY_VALIDATION: PASS")
        print(f"workload_id: {report.workload.get('workload_id')}")
        print(f"operators: {len(report.operators)}")
        print(f"regions: {len(report.regions)}")
        print(f"graph_breaks: {len(report.graph_breaks)}")
        return 0

    if args.command == "rank":
        ranked = rank_regions(
            report.regions,
            total_cuda_time_us=report.total_cuda_time_us,
            impact_floor=args.impact_floor,
        )
        ops = rank_operators(
            report.operators,
            total_cuda_time_us=report.total_cuda_time_us,
        )
        payload = {
            "workload_id": report.workload.get("workload_id"),
            "total_cuda_time_us": report.total_cuda_time_us,
            "operators": [
                {
                    "name": op.name,
                    "op_key": op.op_key,
                    "share_of_e2e": share,
                    "cuda_time_us": op.cuda_time_us,
                    "calls": op.calls,
                }
                for op, share in ops
            ],
            "regions": [item.as_dict() for item in ranked],
            "graph_breaks": [item.as_dict() for item in report.graph_breaks],
        }
        print(json.dumps(payload, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
