"""CPU tests for model-independent Dynamo/FX region capture."""

from __future__ import annotations

import torch
import torch.nn as nn

from autokernel.discovery import (
    DiscoveryReport,
    RegionCaptureSession,
    capture_model_regions,
    capture_module_region,
    is_region_safe,
    reject_region,
)


class _PureBlock(nn.Module):
    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x * scale + x)


class _MutatingBlock(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.add_(1.0)
        return x


class _TinyStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = _PureBlock()
        self.block1 = _PureBlock()

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x = self.block0(x, scale)
        x = self.block1(x, scale)
        return x


def test_cpu_module_produces_graph_region_with_ops_and_deps():
    module = _PureBlock()
    x = torch.randn(2, 8)
    scale = torch.randn(2, 8)
    result = capture_module_region(
        module,
        (x, scale),
        name="test.pure_block",
        parent_module="blocks.0",
    )
    assert result.region is not None
    region = result.region
    assert region.operations
    assert region.inputs
    assert region.outputs
    assert region.parent_module == "blocks.0"
    assert region.fingerprint
    # Dependencies should encode at least one edge for a multi-op graph.
    assert isinstance(region.dependencies, tuple)
    # Metadata only — no forbidden keys
    payload = region.as_dict()
    assert "values" not in str(payload).lower() or "values" not in payload
    for key in ("weights", "prompt", "tensor_values", "source_code"):
        assert key not in payload


def test_equivalent_graphs_stable_fingerprints():
    module = _PureBlock()
    a = capture_module_region(
        module,
        (torch.randn(2, 8), torch.randn(2, 8)),
        name="test.pure_block",
        parent_module="blocks.0",
    )
    b = capture_module_region(
        module,
        (torch.randn(2, 8), torch.randn(2, 8)),
        name="test.pure_block",
        parent_module="blocks.0",
    )
    assert a.region is not None and b.region is not None
    # Same structure + same shapes/dtypes => same fingerprint (values ignored).
    assert a.region.operations == b.region.operations
    assert a.region.fingerprint == b.region.fingerprint


def test_fingerprint_changes_with_shape():
    module = _PureBlock()
    small = capture_module_region(
        module,
        (torch.randn(2, 8), torch.randn(2, 8)),
        name="test.pure_block",
        parent_module="blocks.0",
    )
    large = capture_module_region(
        module,
        (torch.randn(4, 16), torch.randn(4, 16)),
        name="test.pure_block",
        parent_module="blocks.0",
    )
    assert small.region is not None and large.region is not None
    assert small.region.fingerprint != large.region.fingerprint


def test_mutation_rejected_fail_closed():
    module = _MutatingBlock()
    result = capture_module_region(
        module,
        (torch.randn(2, 4),),
        name="test.mutating",
        parent_module="bad",
    )
    # Either trace fails or region is rejected for in-place mutation.
    if result.region is not None:
        assert result.region.rejection_reasons
        assert not is_region_safe(result.region.operations)
    else:
        assert result.graph_breaks


def test_collectives_and_custom_ops_rejected():
    reasons = reject_region(
        ["aten::mul", "c10d::all_reduce_", "custom::flash_attn"]
    )
    assert any("collective" in r for r in reasons)
    assert any("custom" in r for r in reasons)


def test_repeated_module_capture_session_builds_report():
    model = _TinyStack()
    x = torch.randn(2, 8)
    scale = torch.randn(2, 8)

    with capture_model_regions(
        model,
        predicate=lambda name, _m: name in {"block0", "block1"},
        name_prefix="dit",
    ) as session:
        # Multiple calls → aggregated call counts / shape frequency.
        for _ in range(3):
            model(x, scale)

    regions = session.regions()
    assert regions
    # Same pure block structure should collapse to one fingerprint class
    # (two scopes may still share ops; at least one region with calls>=3).
    assert any(region.calls >= 3 for region in regions) or any(
        sum((region.shape_frequency or {}).values()) >= 3 for region in regions
    )

    report = session.to_discovery_report(
        workload={"workload_id": "unit-fx", "model_id": "toy"},
        total_cuda_time_us=0.0,
    )
    assert isinstance(report, DiscoveryReport)
    assert report.regions
    # Graph breaks / unsupported must be present as fields (may be empty for pure).
    assert isinstance(report.graph_breaks, tuple)
    assert isinstance(report.unsupported, tuple)
    # Round-trip
    again = DiscoveryReport.from_dict(report.as_dict())
    assert [r.fingerprint for r in again.regions] == [
        r.fingerprint for r in report.regions
    ]


def test_graph_breaks_visible_for_untraceable_control_flow():
    class _DataDependent(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.sum().item() > 0:
                return x * 2
            return x

    result = capture_module_region(
        _DataDependent(),
        (torch.randn(2, 2),),
        name="test.data_dependent",
    )
    # symbolic_trace typically fails on data-dependent Python control flow
    assert result.region is None or result.graph_breaks or result.region.rejection_reasons
    if result.region is None:
        assert result.graph_breaks
        assert any("fx_trace_failed" in b.reason for b in result.graph_breaks)


def test_session_register_explicit_module():
    block = _PureBlock()
    session = RegionCaptureSession(name_prefix="leaf")
    session.register_module(block, scope="blocks.0")
    for _ in range(2):
        block(torch.randn(1, 4), torch.randn(1, 4))
    session.close()
    regions = session.regions()
    assert len(regions) == 1
    assert regions[0].calls == 2
    assert regions[0].parent_module == "blocks.0"
    assert regions[0].shape_frequency
