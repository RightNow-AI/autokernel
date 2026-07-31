"""Tests for offline correlation between profiler rows and FX regions."""

from __future__ import annotations

import pytest

from autokernel.discovery import (
    DiscoveryReport,
    GraphRegion,
    OperatorHotspot,
    correlate_discovery_report,
    correlate_profiler_to_regions,
)


def _tensor(name: str = "x", shape: tuple[int, ...] = (1, 128, 64)) -> GraphRegion:
    """Helper to create a simple tensor metadata for testing."""
    from autokernel.discovery import TensorMeta
    return TensorMeta(
        name=name,
        shape=shape,
        stride=(8192, 64, 1),
        dtype="bfloat16",
        device_type="cuda",
    )


def test_correlate_profiler_to_regions_basic_match():
    """Test basic profiler-to-region correlation with exact parent module match."""
    # Create profiler rows
    profiler_rows = [
        OperatorHotspot(
            name="aten::mm",
            op_key="aten::mm",
            calls=100,
            cuda_time_us=6000.0,
            self_cuda_time_us=5900.0,
            parent_module="blocks.0.attn",
        ),
        OperatorHotspot(
            name="aten::mul",
            op_key="aten::mul",
            calls=40,
            cuda_time_us=400.0,
            self_cuda_time_us=400.0,
            parent_module="blocks.0.norm",
        ),
    ]

    # Create FX regions
    fx_regions = [
        GraphRegion.build(
            name="attention",
            operations=["aten::mm", "aten::add"],
            inputs=[_tensor("q"), _tensor("k")],
            parent_module="blocks.0.attn",
            calls=100,
        ),
        GraphRegion.build(
            name="normalization",
            operations=["aten::mul", "aten::layer_norm"],
            inputs=[_tensor("x")],
            parent_module="blocks.0.norm",
            calls=40,
        ),
    ]

    # Correlate
    correlated = correlate_profiler_to_regions(
        profiler_rows,
        fx_regions,
        total_cuda_time_us=10000.0,
    )

    # Should have 2 regions (no unmatched rows)
    assert len(correlated) == 2

    # Check that timing data was populated
    attn_region = next(r for r in correlated if r.name == "attention")
    assert attn_region.self_cuda_time_us == 5900.0
    assert attn_region.calls == 100
    assert attn_region.attributes["matched_profiler_rows"] == 1

    norm_region = next(r for r in correlated if r.name == "normalization")
    assert norm_region.self_cuda_time_us == 400.0
    assert norm_region.calls == 40
    assert norm_region.attributes["matched_profiler_rows"] == 1


def test_correlate_unmatched_profiler_rows():
    """Test that unmatched profiler rows are tracked in a special region."""
    profiler_rows = [
        OperatorHotspot(
            name="aten::mm",
            op_key="aten::mm",
            calls=100,
            cuda_time_us=6000.0,
            self_cuda_time_us=5900.0,
            parent_module="blocks.0.attn",
        ),
        OperatorHotspot(
            name="custom::unknown_op",
            op_key="custom::unknown_op",
            calls=10,
            cuda_time_us=100.0,
            self_cuda_time_us=100.0,
            parent_module="unknown.module",
        ),
    ]

    fx_regions = [
        GraphRegion.build(
            name="attention",
            operations=["aten::mm", "aten::add"],
            inputs=[_tensor("q"), _tensor("k")],
            parent_module="blocks.0.attn",
            calls=100,
        ),
    ]

    correlated = correlate_profiler_to_regions(
        profiler_rows,
        fx_regions,
        total_cuda_time_us=10000.0,
    )

    # Should have 2 regions: one matched, one unmatched
    assert len(correlated) == 2

    # Check unmatched region
    unmatched = next(r for r in correlated if r.name == "unmatched_profiler_rows")
    assert unmatched.self_cuda_time_us == 100.0
    assert unmatched.calls == 10
    assert "no_fx_region_match" in unmatched.rejection_reasons
    assert unmatched.attributes["unmatched_row_count"] == 1


def test_deduplicate_equivalent_regions():
    """Test that equivalent regions are deduplicated by fingerprint."""
    profiler_rows = [
        OperatorHotspot(
            name="aten::mul",
            op_key="aten::mul",
            calls=80,
            cuda_time_us=800.0,
            self_cuda_time_us=800.0,
            parent_module="blocks.0.norm",
        ),
    ]

    # Create two equivalent regions (same operations, inputs, parent_module, etc.)
    # Note: fingerprint includes parent_module, so they must be identical to deduplicate
    fx_regions = [
        GraphRegion.build(
            name="norm_0",
            operations=["aten::mul", "aten::layer_norm"],
            inputs=[_tensor("x")],
            parent_module="blocks.0.norm",
            calls=40,
            shape_frequency={"shape1": 40},
        ),
        GraphRegion.build(
            name="norm_1",
            operations=["aten::mul", "aten::layer_norm"],
            inputs=[_tensor("x")],
            parent_module="blocks.0.norm",  # Same parent_module for same fingerprint
            calls=40,
            shape_frequency={"shape1": 40},
        ),
    ]

    correlated = correlate_profiler_to_regions(
        profiler_rows,
        fx_regions,
        total_cuda_time_us=10000.0,
    )

    # Should deduplicate to one region
    assert len(correlated) == 1
    
    # Should aggregate calls and shape frequencies
    region = correlated[0]
    assert region.calls == 80  # Aggregated from both regions
    assert region.shape_frequency == {"shape1": 80}  # Aggregated


def test_exclusive_time_without_double_counting():
    """Test that nested scopes are not double-counted."""
    # Simulate nested profiler rows
    profiler_rows = [
        OperatorHotspot(
            name="blocks.0.forward",
            op_key="blocks.0.forward",
            calls=10,
            cuda_time_us=5000.0,
            self_cuda_time_us=1000.0,  # Exclusive time (excluding children)
            parent_module="blocks.0",
        ),
        OperatorHotspot(
            name="aten::mm",
            op_key="aten::mm",
            calls=10,
            cuda_time_us=4000.0,
            self_cuda_time_us=4000.0,
            parent_module="blocks.0.attn",
        ),
    ]

    fx_regions = [
        GraphRegion.build(
            name="block_forward",
            operations=["aten::mm", "aten::add"],
            inputs=[_tensor("x")],
            parent_module="blocks.0",
            calls=10,
        ),
    ]

    correlated = correlate_profiler_to_regions(
        profiler_rows,
        fx_regions,
        total_cuda_time_us=10000.0,
    )

    region = correlated[0]
    # Should use exclusive time to avoid double-counting
    # The region should get the appropriate time based on matched rows
    assert region.self_cuda_time_us > 0
    # Total should not exceed the sum of exclusive times
    assert region.self_cuda_time_us <= 5000.0


def test_synthetic_cpu_fixtures_produce_timed_regions():
    """Test that synthetic CPU fixtures produce non-zero timed regions."""
    # Create profiler rows with synthetic CPU timing
    profiler_rows = [
        OperatorHotspot(
            name="aten::mul",
            op_key="aten::mul",
            calls=50,
            cuda_time_us=0.0,  # CPU-only
            self_cuda_time_us=0.0,
            cpu_time_us=500.0,  # CPU time
            parent_module="cpu.module",
        ),
    ]

    fx_regions = [
        GraphRegion.build(
            name="cpu_region",
            operations=["aten::mul", "aten::add"],
            inputs=[_tensor("x")],
            parent_module="cpu.module",
            calls=50,
        ),
    ]

    correlated = correlate_profiler_to_regions(
        profiler_rows,
        fx_regions,
        total_cuda_time_us=0.0,  # CPU-only
    )

    assert len(correlated) == 1
    region = correlated[0]
    # Should still have calls even if CUDA time is zero
    assert region.calls == 50
    # CPU regions should be tracked even with zero CUDA time
    assert region.attributes["matched_profiler_rows"] == 1


def test_correlate_discovery_report_integration():
    """Test the high-level correlate_discovery_report function."""
    profiler_export_rows = [
        {
            "name": "aten::mm",
            "op_key": "aten::mm",
            "calls": 100,
            "cuda_time_us": 6000.0,
            "self_cuda_time_us": 5900.0,
            "parent_module": "blocks.0.attn",
        },
    ]

    fx_report = DiscoveryReport.from_dict(
        {
            "schema_version": 1,
            "producer": {"name": "fastvideo", "version": "test"},
            "workload": {"workload_id": "test", "model_id": "test"},
            "environment": {"hardware_profile_id": "cpu", "software_profile_id": "test"},
            "total_cuda_time_us": 10000.0,
            "operators": [],
            "regions": [
                GraphRegion.build(
                    name="attention",
                    operations=["aten::mm", "aten::add"],
                    inputs=[_tensor("q"), _tensor("k")],
                    parent_module="blocks.0.attn",
                    calls=100,
                ).as_dict()
            ],
            "graph_breaks": [],
            "unsupported": [],
        }
    )

    correlated_report = correlate_discovery_report(profiler_export_rows, fx_report)

    # Should have operators from profiler
    assert len(correlated_report.operators) == 1
    assert correlated_report.operators[0].name == "aten::mm"

    # Should have populated regions
    assert len(correlated_report.regions) == 1
    assert correlated_report.regions[0].self_cuda_time_us == 5900.0

    # Other fields should be preserved
    assert correlated_report.workload == fx_report.workload
    assert correlated_report.environment == fx_report.environment


def test_correlate_discovery_report_keeps_unmatched_as_diagnostic():
    profiler_export_rows = [
        {
            "name": "aten::unmatched",
            "calls": 3,
            "cuda_time_us": 30.0,
            "self_cuda_time_us": 30.0,
        },
    ]
    fx_report = DiscoveryReport.from_dict(
        {
            "schema_version": 1,
            "producer": {"name": "fastvideo", "version": "test"},
            "workload": {"workload_id": "test", "model_id": "test"},
            "environment": {"hardware_profile_id": "cpu"},
            "total_cuda_time_us": 30.0,
            "operators": [],
            "regions": [
                GraphRegion.build(
                    name="captured",
                    operations=["aten::add"],
                    inputs=[_tensor("x")],
                ).as_dict(),
            ],
        }
    )

    correlated = correlate_discovery_report(
        profiler_export_rows,
        fx_report,
    )

    assert all(
        region.name != "unmatched_profiler_rows"
        for region in correlated.regions
    )
    assert any(
        item.op_name == "profiler::unmatched"
        for item in correlated.unsupported
    )


def test_confidence_calculation():
    """Test that confidence is calculated correctly."""
    profiler_rows = [
        OperatorHotspot(
            name="aten::mul",
            op_key="aten::mul",
            calls=100,
            cuda_time_us=1000.0,
            self_cuda_time_us=1000.0,
            parent_module="blocks.0.norm",
        ),
    ]

    fx_regions = [
        GraphRegion.build(
            name="safe_region",
            operations=["aten::mul", "aten::add"],  # Safe operations
            inputs=[_tensor("x")],
            parent_module="blocks.0.norm",
            calls=100,
        ),
    ]

    correlated = correlate_profiler_to_regions(
        profiler_rows,
        fx_regions,
        total_cuda_time_us=10000.0,
    )

    region = correlated[0]
    # Should have high confidence: safe + matched + high call count
    assert region.attributes["confidence"] >= 0.8


def test_e2e_improvement_estimation():
    """Test that end-to-end improvement is estimated correctly."""
    profiler_rows = [
        OperatorHotspot(
            name="aten::mm",
            op_key="aten::mm",
            calls=100,
            cuda_time_us=2000.0,  # 20% of total
            self_cuda_time_us=2000.0,
            parent_module="blocks.0.attn",
        ),
    ]

    fx_regions = [
        GraphRegion.build(
            name="attention",
            operations=["aten::mm", "aten::add"],
            inputs=[_tensor("x")],
            parent_module="blocks.0.attn",
            calls=100,
        ),
    ]

    correlated = correlate_profiler_to_regions(
        profiler_rows,
        fx_regions,
        total_cuda_time_us=10000.0,
    )

    region = correlated[0]
    # 20% e2e share * 0.9 reducible fraction = 18% max improvement
    assert region.attributes["e2e_share_pct"] == 20.0
    assert region.attributes["estimated_max_e2e_improvement"] == pytest.approx(0.18, rel=0.01)


def test_shape_frequency_aggregation():
    """Test that shape frequencies are aggregated correctly."""
    profiler_rows = [
        OperatorHotspot(
            name="aten::mul",
            op_key="aten::mul",
            calls=50,
            cuda_time_us=500.0,
            self_cuda_time_us=500.0,
            input_shapes=[(2, 128)],  # Shape metadata
            parent_module="blocks.0.norm",
        ),
    ]

    fx_regions = [
        GraphRegion.build(
            name="norm",
            operations=["aten::mul", "aten::layer_norm"],
            inputs=[_tensor("x", (2, 128))],
            parent_module="blocks.0.norm",
            calls=50,
            shape_frequency={"2x128": 30},  # Existing shape frequency
        ),
    ]

    correlated = correlate_profiler_to_regions(
        profiler_rows,
        fx_regions,
        total_cuda_time_us=10000.0,
    )

    region = correlated[0]
    # Should merge shape frequencies from both sources
    assert region.shape_frequency is not None
    # Should have at least the original frequency
    assert sum(region.shape_frequency.values()) >= 30


def test_rejection_reasons_aggregation():
    """Test that rejection reasons are aggregated from multiple sources."""
    profiler_rows = [
        OperatorHotspot(
            name="aten::mul_",
            op_key="aten::mul_",
            calls=10,
            cuda_time_us=100.0,
            self_cuda_time_us=100.0,
            parent_module="blocks.0.norm",
        ),
    ]

    fx_regions = [
        GraphRegion.build(
            name="unsafe_region",
            operations=["aten::mul_"],  # In-place operation
            inputs=[_tensor("x")],
            parent_module="blocks.0.norm",
            calls=10,
            rejection_reasons=("custom_reason",),
        ),
    ]

    correlated = correlate_profiler_to_regions(
        profiler_rows,
        fx_regions,
        total_cuda_time_us=10000.0,
    )

    region = correlated[0]
    # Should have both safety rejection and custom reason
    assert len(region.rejection_reasons) >= 1
    # Should include in-place mutation rejection
    assert any("in-place" in reason.lower() for reason in region.rejection_reasons)


def test_hierarchical_parent_module_matching():
    """Test that hierarchical parent module relationships are matched."""
    profiler_rows = [
        OperatorHotspot(
            name="aten::mm",
            op_key="aten::mm",
            calls=100,
            cuda_time_us=5000.0,
            self_cuda_time_us=5000.0,
            parent_module="blocks.0.attn.q_proj",  # More specific
        ),
    ]

    fx_regions = [
        GraphRegion.build(
            name="attention",
            operations=["aten::mm", "aten::add"],
            inputs=[_tensor("x")],
            parent_module="blocks.0.attn",  # Less specific
            calls=100,
        ),
    ]

    correlated = correlate_profiler_to_regions(
        profiler_rows,
        fx_regions,
        total_cuda_time_us=10000.0,
    )

    # Should still match due to hierarchical relationship
    assert len(correlated) == 1
    assert correlated[0].attributes["matched_profiler_rows"] == 1
