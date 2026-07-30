"""Optimization campaign schema, ranking, and orchestrator bridge."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from autokernel.campaign import (
    CampaignError,
    OptimizationCampaign,
    load_campaign,
    prepare_campaign,
    rank_targets,
    write_optimization_plan,
)


def test_load_and_rank_wan_campaign(fixtures_dir):
    campaign = load_campaign(fixtures_dir / "wan_campaign.json")
    assert campaign.workload["workload_id"] == "wan-t2v-1.3b"
    assert [target.name for target in rank_targets(campaign)] == [
        "wan.self_attn_residual_norm",
        "wan.rope",
    ]
    assert rank_targets(campaign)[0].impact_pct(
        campaign.total_profiled_device_time_us
    ) == pytest.approx(32.0)


def test_campaign_rejects_tensor_values(fixtures_dir):
    payload = json.loads(
        (fixtures_dir / "wan_campaign.json").read_text(encoding="utf-8")
    )
    payload["targets"][0]["observations"][0]["inputs"][0]["values"] = [1.0]
    with pytest.raises(CampaignError, match="unknown field.*values"):
        OptimizationCampaign.from_dict(payload)


def test_campaign_rejects_nonfinite_timing(fixtures_dir):
    payload = json.loads(
        (fixtures_dir / "wan_campaign.json").read_text(encoding="utf-8")
    )
    payload["targets"][0]["total_device_time_us"] = float("inf")
    with pytest.raises(CampaignError, match="finite non-negative"):
        OptimizationCampaign.from_dict(payload)


def test_campaign_observation_counts_must_match_calls(fixtures_dir):
    payload = json.loads(
        (fixtures_dir / "wan_campaign.json").read_text(encoding="utf-8")
    )
    payload["targets"][0]["calls"] = 41
    with pytest.raises(CampaignError, match="counts sum to 40"):
        OptimizationCampaign.from_dict(payload)


def test_campaign_allows_overlapping_target_timings(fixtures_dir):
    payload = json.loads(
        (fixtures_dir / "wan_campaign.json").read_text(encoding="utf-8")
    )
    payload["targets"][1]["total_device_time_us"] = 8_000
    campaign = OptimizationCampaign.from_dict(payload)
    assert len(campaign.targets) == 2


@pytest.mark.parametrize("operation", ["../../escape", "bad-name", "bad.name"])
def test_campaign_rejects_unsafe_operation_identifiers(fixtures_dir, operation):
    payload = json.loads(
        (fixtures_dir / "wan_campaign.json").read_text(encoding="utf-8")
    )
    payload["targets"][0]["operation"] = operation
    with pytest.raises(CampaignError, match="safe identifier"):
        OptimizationCampaign.from_dict(payload)


def test_campaign_rejects_content_hidden_in_attributes(fixtures_dir):
    payload = json.loads(
        (fixtures_dir / "wan_campaign.json").read_text(encoding="utf-8")
    )
    payload["targets"][0]["attributes"]["prompt"] = "do not export me"
    with pytest.raises(CampaignError, match="content or secret fields"):
        OptimizationCampaign.from_dict(payload)


def test_write_plan_bridges_to_existing_orchestrator(fixtures_dir, tmp_path):
    campaign = load_campaign(fixtures_dir / "wan_campaign.json")
    output = tmp_path / "optimization_plan.json"
    plan = write_optimization_plan(campaign, output)
    assert output.is_file()
    assert plan["kernels_to_optimize"][0] == {
        "rank": 1,
        "file": "workspace/kernel_wan_gated_residual_norm_1.py",
        "op_type": "wan_gated_residual_norm",
        "target_name": "wan.self_attn_residual_norm",
        "spec_locator": "models/wan_gated_residual_norm.py:SPEC",
        "pct_total": 32.0,
        "calls": 40,
        "requires_backward": False,
    }


def test_campaign_cli_validate_rank_and_plan(repo_root, fixtures_dir, tmp_path):
    campaign = fixtures_dir / "wan_campaign.json"
    validate = subprocess.run(
        [sys.executable, "campaign.py", "validate", str(campaign)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert validate.returncode == 0
    assert "CAMPAIGN_VALIDATION: PASS" in validate.stdout

    rank = subprocess.run(
        [sys.executable, "campaign.py", "rank", str(campaign)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rank.returncode == 0
    assert json.loads(rank.stdout)[0]["impact_pct"] == 32.0

    output = tmp_path / "plan.json"
    plan = subprocess.run(
        [
            sys.executable,
            "campaign.py",
            "plan",
            str(campaign),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert plan.returncode == 0
    assert "CAMPAIGN_PLAN: PASS" in plan.stdout
    assert output.is_file()


def test_prepare_requires_explicit_trust(fixtures_dir, tmp_path):
    campaign = load_campaign(fixtures_dir / "wan_campaign.json")
    with pytest.raises(CampaignError, match="trust_specs=True"):
        prepare_campaign(campaign, tmp_path)


def test_prepare_materializes_starter_and_receipt(
    fixtures_dir, tmp_path, monkeypatch
):
    payload = json.loads(
        (fixtures_dir / "wan_campaign.json").read_text(encoding="utf-8")
    )
    payload["targets"] = payload["targets"][:1]
    payload["targets"][0][
        "spec_locator"
    ] = "models/wan_gated_residual_norm.py:SPEC"
    campaign = OptimizationCampaign.from_dict(payload, source="test-campaign")
    monkeypatch.chdir(fixtures_dir.parent.parent)
    output = tmp_path / "workspace"
    receipt = prepare_campaign(campaign, output, trust_specs=True)
    candidate = output / "kernel_wan_gated_residual_norm_1.py"
    assert candidate.is_file()
    assert "KERNEL_TYPE = \"wan_gated_residual_norm\"" in candidate.read_text(
        encoding="utf-8"
    )
    assert receipt["status"] == "prepared"
    assert (output / "optimization_plan.json").is_file()
    assert (output / "campaign_receipt.json").is_file()
