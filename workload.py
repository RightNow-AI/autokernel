#!/usr/bin/env python3
"""Validate and drive versioned FastVideo workload manifests.

Usage:
    python workload.py validate workloads/ltx_480p.yaml
    python workload.py show workloads/wan_t2v_1.3b_480p.yaml
    python workload.py run-ab \\
        --fastvideo-checkout /path/to/FastVideo \\
        --workload workloads/wan_t2v_1.3b_480p.yaml \\
        --output workspace/wan_ab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autokernel.workload import WorkloadError, load_workload
from autokernel.workload.launcher import run_ab
from autokernel.workload.result import load_generation_result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and execute FastVideo workload manifests"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="Validate a workload manifest"
    )
    validate.add_argument("workload", type=Path)

    show = subparsers.add_parser(
        "show", help="Print the normalized workload JSON"
    )
    show.add_argument("workload", type=Path)

    run = subparsers.add_parser(
        "run-ab",
        help="Run native and optimized modes via a FastVideo launcher",
    )
    run.add_argument("--fastvideo-checkout", type=Path, required=True)
    run.add_argument("--workload", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model", help="Optional model_id override")
    run.add_argument(
        "--launcher-script",
        type=Path,
        help="Override path to generation_launcher.py",
    )
    run.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore completed stages and re-run everything",
    )
    run.add_argument(
        "--modes",
        default="native,optimized",
        help="Comma-separated modes (default: native,optimized)",
    )

    result = subparsers.add_parser(
        "validate-result",
        help="Validate a generation_launcher result JSON",
    )
    result.add_argument("result", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            workload = load_workload(args.workload)
            print("WORKLOAD_VALIDATION: PASS")
            print(f"workload_id: {workload.workload_id}")
            print(f"model_id: {workload.model.model_id}")
            print(f"task: {workload.task}")
            return 0

        if args.command == "show":
            workload = load_workload(args.workload)
            print(json.dumps(workload.as_dict(), indent=2))
            return 0

        if args.command == "validate-result":
            result = load_generation_result(args.result)
            print("RESULT_VALIDATION: PASS")
            print(f"mode: {result.mode}")
            print(f"status: {result.status}")
            print(f"workload_id: {result.workload_id}")
            return 0

        if args.command == "run-ab":
            payload = run_ab(
                fastvideo_checkout=args.fastvideo_checkout,
                workload=args.workload,
                output_dir=args.output,
                launcher_script=args.launcher_script,
                model_override=args.model,
                modes=tuple(
                    mode.strip()
                    for mode in args.modes.split(",")
                    if mode.strip()
                ),
                resume=not args.no_resume,
            )
            print(json.dumps(payload, indent=2))
            return 0
    except WorkloadError as exc:
        print(f"WORKLOAD: FAIL\n{exc}", file=sys.stderr)
        return 2

    print(f"unknown command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
