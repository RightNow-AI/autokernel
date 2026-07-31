"""Ranking, profiler parse, and CPU FX capture — real shipped entry points."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from autokernel.discovery import (
    DEFAULT_IMPACT_FLOOR,
    GraphRegion,
    OperatorHotspot,
    TensorMeta,
    capture_callable_region,
    capture_module_region,
    load_discovery_report,
    optimistic_e2e_improvement,
    parse_key_averages_rows,
    profiler_export_to_report,
    rank_operators,
    rank_regions,
)


class _GatedResidual(nn.Module):
    """Tiny pure-tensor stand-in for residual * gate + residual patterns."""

    def forward(self, residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return residual + residual * gate


def test_optimistic_e2e_and_impact_floor_on_wan_like_share():
    # 0.5% of e2e with 90% reducible => 0.45% optimistic < 0.5% floor
    share = 0.005
    improvement = optimistic_e2e_improvement(share, reducible_fraction=0.9)
    assert improvement == pytest.approx(0.0045)
    assert improvement < DEFAULT_IMPACT_FLOOR


def test_rank_regions_marks_low_value_and_high_value():
    low = GraphRegion.build(
        name="wan.elementwise",
        operations=["aten::mul", "aten::add", "aten::layer_norm"],
        inputs=[
            TensorMeta("r", (1, 128, 64), (8192, 64, 1), "bfloat16", "cpu"),
            TensorMeta("g", (1, 1, 64), (64, 64, 1), "float32", "cpu"),
        ],
            cuda_time_us=50.0,
            self_cuda_time_us=50.0,
        calls=40,
    )
    high = GraphRegion.build(
        name="hot.epilogue",
        operations=["aten::mul", "aten::add", "aten::silu"],
        inputs=[
            TensorMeta("x", (1, 128, 64), (8192, 64, 1), "float16", "cpu"),
        ],
            cuda_time_us=2500.0,
            self_cuda_time_us=2500.0,
        calls=40,
    )
    ranked = rank_regions(
        [low, high],
        total_cuda_time_us=10_000.0,
        impact_floor=0.005,
        reducible_fraction=0.9,
    )
    assert ranked[0].region.name == "hot.epilogue"
    assert ranked[0].search_worthy is True
    assert ranked[0].estimated_max_e2e_improvement == 0.225  # 25% * 0.9
    assert ranked[1].region.name == "wan.elementwise"
    assert ranked[1].search_worthy is False
    assert any("below_impact_floor" in r for r in ranked[1].rejection_reasons)


def test_parse_key_averages_rows_real_aliases():
    rows = [
        {
            "name": "aten::mm",
            "cuda_time_total": 6000.0,
            "self_cuda_time_total": 5900.0,
            "count": 100,
        },
        {
            "key": "aten::mul",
            "cuda_time_ms": 0.4,
            "self_cuda_time_ms": 0.4,
            "calls": 40,
        },
    ]
    ops = parse_key_averages_rows(rows)
    assert ops[0].op_key == "aten::mm"
    assert ops[0].cuda_time_us == 6000.0
    assert ops[0].calls == 100
    assert ops[1].op_key == "aten::mul"
    assert ops[1].cuda_time_us == 400.0


def test_capture_module_region_cpu_fx():
    module = _GatedResidual()
    residual = torch.randn(2, 8, 16)
    gate = torch.randn(2, 1, 16)
    result = capture_module_region(
        module,
        (residual, gate),
        name="test.gated_residual",
        parent_module="blocks.0",
    )
    assert result.region is not None
    assert result.region.name == "test.gated_residual"
    assert len(result.region.operations) >= 1
    assert result.region.fingerprint
    # Fingerprint stable across re-capture with same parent_module
    again = capture_module_region(
        module,
        (residual, gate),
        name="test.gated_residual",
        parent_module="blocks.0",
    )
    assert again.region is not None
    assert again.region.operations == result.region.operations
    assert again.region.fingerprint == result.region.fingerprint
    # Ranking can consume the region with injected timing
    timed = GraphRegion.build(
        name=result.region.name,
        operations=result.region.operations,
        inputs=result.region.inputs,
        outputs=result.region.outputs,
        parent_module=result.region.parent_module,
        pattern_family=result.region.pattern_family,
        rejection_reasons=result.region.rejection_reasons,
        cuda_time_us=2000.0,
        self_cuda_time_us=2000.0,
        calls=32,
    )
    ranked = rank_regions([timed], total_cuda_time_us=10_000.0)
    assert ranked[0].search_worthy is True


def test_capture_callable_region():
    def pure(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y + x

    x = torch.randn(4, 8)
    y = torch.randn(4, 8)
    result = capture_callable_region(pure, (x, y), name="test.mul_add")
    assert result.region is not None
    joined = " ".join(result.region.operations).lower()
    assert "mul" in joined or "add" in joined


def test_discovery_cli_rank_entry_point(tmp_path):
    from discovery import main as discovery_main

    region = GraphRegion.build(
        name="cli.region",
        operations=["aten::mul", "aten::add"],
        inputs=[TensorMeta("x", (1, 8, 8), (64, 8, 1), "float16", "cpu")],
        cuda_time_us=3000.0,
        calls=10,
    )
    payload = {
        "schema_version": 1,
        "producer": {"name": "test", "version": "0"},
        "workload": {"workload_id": "unit", "model_id": "m"},
        "environment": {
            "hardware_profile_id": "h",
            "software_profile_id": "s",
        },
        "total_cuda_time_us": 10000.0,
        "operators": [
            {
                "name": "aten::mm",
                "op_key": "aten::mm",
                "calls": 1,
                "cuda_time_us": 5000.0,
                "self_cuda_time_us": 5000.0,
            }
        ],
        "regions": [region.as_dict()],
        "graph_breaks": [],
        "unsupported": [],
    }
    path = tmp_path / "d.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert discovery_main(["validate", str(path)]) == 0
    assert discovery_main(["rank", str(path)]) == 0


def test_profiler_export_ingestion_and_cli(tmp_path):
    from discovery import main as discovery_main

    export = {
        "schema_version": 1,
        "producer": {"name": "fastvideo", "version": "1"},
        "workload": {"workload_id": "ltx-unit", "model_id": "ltx"},
        "environment": {"torch": "2.x", "gpu_name": "unit"},
        "total_cuda_time_us": 100.0,
        "rows": [
            {
                "name": "ProfilerStep*",
                "calls": 1,
                "cuda_time_us": 100.0,
                "self_cuda_time_us": 10.0,
                "cpu_time_us": 20.0,
            },
            {
                "name": "aten::mul",
                "calls": 4,
                "cuda_time_us": 90.0,
                "self_cuda_time_us": 90.0,
                "cpu_time_us": 5.0,
                "input_shapes": [[1, 8], [1, 8]],
            },
        ],
    }
    report = profiler_export_to_report(export)
    assert report.total_cuda_time_us == 100.0
    assert [op.op_key for op in report.operators] == [
        "ProfilerStep",
        "aten::mul",
    ]

    source = tmp_path / "profile.json"
    output = tmp_path / "discovery.json"
    source.write_text(json.dumps(export), encoding="utf-8")
    assert (
        discovery_main(
            ["ingest-profiler", str(source), "--output", str(output)]
        )
        == 0
    )
    loaded = load_discovery_report(output)
    assert len(loaded.operators) == 2


def test_operator_ranking_uses_self_cuda_time():
    nested_scope = OperatorHotspot(
        name="attention_scope",
        op_key="attention_scope",
        calls=1,
        cuda_time_us=100.0,
        self_cuda_time_us=1.0,
        cpu_time_us=0.0,
        input_shapes=(),
        parent_module=None,
        source="torch_profiler",
    )
    kernel = OperatorHotspot(
        name="attention_kernel",
        op_key="attention_kernel",
        calls=1,
        cuda_time_us=80.0,
        self_cuda_time_us=80.0,
        cpu_time_us=0.0,
        input_shapes=(),
        parent_module=None,
        source="torch_profiler",
    )

    ranked = rank_operators(
        [nested_scope, kernel],
        total_cuda_time_us=100.0,
    )

    assert ranked[0] == (kernel, pytest.approx(0.8))
    assert ranked[1] == (nested_scope, pytest.approx(0.01))


def test_profiler_export_rejects_nested_secret_metadata():
    export = {
        "schema_version": 1,
        "producer": {"name": "fastvideo", "version": "1"},
        "workload": {"workload_id": "unit", "model_id": "model"},
        "environment": {"runtime": {"token": "do-not-store"}},
        "total_cuda_time_us": 1.0,
        "rows": [
            {
                "name": "aten::add",
                "calls": 1,
                "cuda_time_us": 1.0,
                "self_cuda_time_us": 1.0,
            }
        ],
    }
    with pytest.raises(ValueError, match="forbidden"):
        profiler_export_to_report(export)
