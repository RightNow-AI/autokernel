#!/usr/bin/env python3
"""Validate and prepare model optimization campaigns.

Usage:
    python campaign.py validate path/to/campaign.json
    python campaign.py rank path/to/campaign.json
    python campaign.py plan path/to/campaign.json --output workspace/optimization_plan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autokernel.campaign import (
    CampaignError,
    load_campaign,
    prepare_campaign,
    rank_targets,
    write_optimization_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and prepare an AutoKernel optimization campaign"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "rank"):
        command = subparsers.add_parser(name)
        command.add_argument("campaign", type=Path)
    plan = subparsers.add_parser("plan")
    plan.add_argument("campaign", type=Path)
    plan.add_argument(
        "--output",
        type=Path,
        default=Path("workspace/optimization_plan.json"),
    )
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("campaign", type=Path)
    prepare.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workspace"),
    )
    prepare.add_argument(
        "--trust-specs",
        action="store_true",
        help="Allow loading Python spec locators from this campaign",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        campaign = load_campaign(args.campaign)
    except CampaignError as exc:
        print(f"CAMPAIGN_VALIDATION: FAIL\n{exc}", file=sys.stderr)
        return 2

    if args.command == "validate":
        print("CAMPAIGN_VALIDATION: PASS")
        print(f"workload_id: {campaign.workload['workload_id']}")
        print(f"targets: {len(campaign.targets)}")
        return 0

    if args.command == "rank":
        rows = []
        for rank, target in enumerate(rank_targets(campaign), start=1):
            rows.append(
                {
                    "rank": rank,
                    "name": target.name,
                    "operation": target.operation,
                    "impact_pct": target.impact_pct(
                        campaign.total_profiled_device_time_us
                    ),
                    "calls": target.calls,
                    "spec_locator": target.spec_locator,
                }
            )
        print(json.dumps(rows, indent=2))
        return 0

    if args.command == "plan":
        plan = write_optimization_plan(campaign, args.output)
        print("CAMPAIGN_PLAN: PASS")
        print(f"output: {args.output}")
        print(f"targets: {len(plan['kernels_to_optimize'])}")
        return 0

    try:
        receipt = prepare_campaign(
            campaign,
            args.output_dir,
            trust_specs=args.trust_specs,
        )
    except CampaignError as exc:
        print(f"CAMPAIGN_PREPARE: FAIL\n{exc}", file=sys.stderr)
        return 2
    print("CAMPAIGN_PREPARE: PASS")
    print(f"output: {args.output_dir}")
    print(f"targets: {len(receipt['targets'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
