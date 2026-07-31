"""Discovery report schema, fingerprint stability, and region safety."""

from __future__ import annotations

import json

import pytest

from autokernel.discovery import (
    DiscoveryError,
    DiscoveryReport,
    GraphRegion,
    TensorMeta,
    graph_fingerprint,
    is_region_safe,
    load_discovery_report,
    reject_region,
    write_discovery_report,
)


def _tensor(name: str = "x") -> TensorMeta:
    return TensorMeta(
        name=name,
        shape=(1, 128, 64),
        stride=(8192, 64, 1),
        dtype="bfloat16",
        device_type="cuda",
    )


def test_fingerprint_stable_across_equivalent_regions():
    ops = ("aten::mul", "aten::add", "aten::layer_norm")
    inputs = [_tensor("residual").signature_dict(), _tensor("gate").signature_dict()]
    a = graph_fingerprint(operations=ops, input_signatures=inputs)
    b = graph_fingerprint(operations=ops, input_signatures=inputs)
    assert a == b
    assert len(a) == 32

    different = graph_fingerprint(
        operations=("aten::add", "aten::mul"),
        input_signatures=inputs,
    )
    assert different != a


def test_fingerprint_rejects_nonfinite_and_unsupported_constants():
    with pytest.raises(ValueError, match="finite"):
        graph_fingerprint(
            operations=("aten::add",),
            input_signatures=[_tensor().signature_dict()],
            safe_constants={"scale": float("inf")},
        )
    with pytest.raises(ValueError, match="unsupported"):
        graph_fingerprint(
            operations=("aten::add",),
            input_signatures=[_tensor().signature_dict()],
            safe_constants={"opaque": object()},
        )


def test_rejects_in_place_mutation_and_invalid_region_name():
    assert any("in-place mutation" in reason for reason in reject_region(["aten::add_"]))
    with pytest.raises(ValueError, match="invalid graph region name"):
        GraphRegion.build(
            name="invalid name",
            operations=["aten::add"],
            inputs=[_tensor()],
        )


def test_graph_region_build_and_report_roundtrip(tmp_path):
    region = GraphRegion.build(
        name="elementwise.mul_add",
        operations=["aten::mul", "aten::add"],
        inputs=[_tensor("a"), _tensor("b")],
        outputs=[_tensor("out")],
        parent_module="blocks.0",
        cuda_time_us=1200.0,
        self_cuda_time_us=1100.0,
        calls=40,
        pattern_family="elementwise_chain",
        shape_frequency={"b1-s128-d64": 40},
    )
    assert is_region_safe(region.operations)
    assert not region.rejection_reasons

    report = DiscoveryReport.from_dict(
        {
            "schema_version": 1,
            "producer": {"name": "fastvideo", "version": "test"},
            "workload": {
                "workload_id": "ltx-t2v-480p",
                "model_id": "FastVideo/LTX2-Distilled-Diffusers",
            },
            "environment": {
                "hardware_profile_id": "test-gpu",
                "software_profile_id": "torch-test",
            },
            "total_cuda_time_us": 10000.0,
            "operators": [
                {
                    "name": "aten::mm",
                    "op_key": "aten::mm",
                    "calls": 100,
                    "cuda_time_us": 6000.0,
                    "self_cuda_time_us": 5900.0,
                },
                {
                    "name": "aten::mul",
                    "op_key": "aten::mul",
                    "calls": 40,
                    "cuda_time_us": 400.0,
                    "self_cuda_time_us": 400.0,
                },
            ],
            "regions": [region.as_dict()],
            "graph_breaks": [
                {
                    "scope": "dit.forward",
                    "reason": "data-dependent branching",
                    "count": 2,
                }
            ],
            "unsupported": [
                {
                    "op_name": "custom::flash",
                    "reason": "unsupported custom operator",
                    "count": 1,
                }
            ],
        }
    )
    ranked = report.ranked_operators()
    assert ranked[0].op_key == "aten::mm"
    assert ranked[0].impact_pct(report.total_cuda_time_us) == pytest.approx(59.0)
    assert report.graph_breaks[0].reason.startswith("data-dependent")

    path = tmp_path / "discovery.json"
    write_discovery_report(report, path)
    loaded = load_discovery_report(path)
    assert loaded.as_dict() == report.as_dict()


def test_reject_collectives_and_data_dependent_ops():
    reasons = reject_region(
        ["aten::mul", "aten::all_reduce", "aten::item", "custom::foo"]
    )
    assert any("collective" in r for r in reasons)
    assert any("data-dependent" in r for r in reasons)
    assert any("custom" in r for r in reasons)
    assert not is_region_safe(["aten::mul", "c10d::all_reduce_"])


def test_allow_elementwise_chain():
    assert is_region_safe(
        ["aten::mul", "aten::add", "aten::layer_norm", "aten::silu"]
    )


def test_discovery_rejects_prompt_payload():
    payload = {
        "schema_version": 1,
        "producer": {"name": "fastvideo", "version": "test"},
        "workload": {"workload_id": "x", "model_id": "m"},
        "environment": {"hardware_profile_id": "h", "software_profile_id": "s"},
        "total_cuda_time_us": 1.0,
        "operators": [],
        "regions": [],
    }
    payload["operators"] = [
        {
            "name": "aten::add",
            "op_key": "aten::add",
            "calls": 1,
            "cuda_time_us": 1.0,
            "self_cuda_time_us": 1.0,
            "attributes": {"prompt": "secret"},
        }
    ]
    with pytest.raises(DiscoveryError, match="secret fields"):
        DiscoveryReport.from_dict(payload)


def test_wan_like_elementwise_region_is_safe_but_low_impact():
    """Wan fusion shapes are discoverable; ranking must still use e2e share."""
    region = GraphRegion.build(
        name="wan.gated_residual_norm",
        operations=[
            "aten::mul",
            "aten::add",
            "aten::layer_norm",
        ],
        inputs=[
            TensorMeta(
                "residual",
                (1, 20280, 1536),
                (31150080, 1536, 1),
                "bfloat16",
                "cuda",
            ),
            TensorMeta(
                "gate",
                (1, 1, 1536),
                (1536, 1536, 1),
                "float32",
                "cuda",
            ),
        ],
        cuda_time_us=50.0,
        calls=40,
        pattern_family="residual_gate_norm",
    )
    assert is_region_safe(region.operations)
    # ~0.5% of a 10ms e2e window => below default 0.5% search floor when
    # optimistic reducible fraction is considered in WS3.
    assert region.impact_pct(10_000.0) == pytest.approx(0.5)
