"""Offline correlation between profiler rows and captured FX regions.

This module implements the logic to match profiler operator rows with FX graph
regions, aggregate timing data, and populate DiscoveryReport with correlated
metrics without double-counting nested scopes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .profiler_parse import parse_key_averages_rows
from .ranking import optimistic_e2e_improvement
from .safety import reject_region
from .types import (
    DiscoveryReport,
    GraphRegion,
    OperatorHotspot,
)


@dataclass(frozen=True)
class ScopeMatch:
    """Result of matching a profiler row to a region."""

    profiler_row: OperatorHotspot
    region: GraphRegion
    confidence: float
    match_reason: str


@dataclass
class RegionAccumulator:
    """Aggregate data for regions with the same fingerprint."""

    base_region: GraphRegion
    total_cuda_time_us: float = 0.0
    total_self_cuda_time_us: float = 0.0
    total_calls: int = 0
    shape_frequencies: dict[str, int] = field(default_factory=dict)
    matched_profiler_rows: list[OperatorHotspot] = field(default_factory=list)
    unmatched_scopes: list[str] = field(default_factory=list)
    capture_failures: list[str] = field(default_factory=list)


def _match_scope_to_region(
    profiler_row: OperatorHotspot,
    region: GraphRegion,
) -> ScopeMatch | None:
    """Attempt to match a profiler row to a region based on scope heuristics.

    Matching strategies (in order of precedence):
    1. Exact parent_module match
    2. Op key appears in region operations
    3. Name similarity between profiler row and region

    Returns None if no reasonable match is found.
    """
    # Strategy 1: Exact parent_module match
    if profiler_row.parent_module and region.parent_module:
        if profiler_row.parent_module == region.parent_module:
            return ScopeMatch(
                profiler_row=profiler_row,
                region=region,
                confidence=0.9,
                match_reason="exact_parent_module",
            )
        # Check for hierarchical relationship (e.g., "blocks.0.attn" matches "blocks.0")
        if (profiler_row.parent_module.startswith(region.parent_module + ".") or
            region.parent_module.startswith(profiler_row.parent_module + ".")):
            return ScopeMatch(
                profiler_row=profiler_row,
                region=region,
                confidence=0.7,
                match_reason="hierarchical_parent_module",
            )

    # Strategy 2: Op key appears in region operations
    if profiler_row.op_key in region.operations:
        return ScopeMatch(
            profiler_row=profiler_row,
            region=region,
            confidence=0.6,
            match_reason="op_key_in_operations",
        )

    # Strategy 3: Name similarity
    profiler_name_lower = profiler_row.name.lower()
    region_name_lower = region.name.lower()
    if (profiler_name_lower in region_name_lower or
        region_name_lower in profiler_name_lower):
        return ScopeMatch(
            profiler_row=profiler_row,
            region=region,
            confidence=0.5,
            match_reason="name_similarity",
        )

    return None


def _calculate_exclusive_time(
    profiler_row: OperatorHotspot,
    parent_time_us: float,
) -> tuple[float, float]:
    """Calculate exclusive CUDA time avoiding double-counting.

    Returns (cuda_time_us, self_cuda_time_us) where:
    - cuda_time_us is the total time (inclusive)
    - self_cuda_time_us is exclusive time, capped at parent_time_us if nested
    """
    # Use the profiler's self_cuda_time_us as the exclusive time
    exclusive = profiler_row.self_cuda_time_us
    
    # If this is nested within a parent region, ensure we don't exceed parent time
    if parent_time_us > 0 and exclusive > parent_time_us:
        # This can happen if profiler attribution is inconsistent
        # Conservative approach: cap at parent time
        exclusive = parent_time_us
    
    return profiler_row.cuda_time_us, exclusive


def _aggregate_region_timing(
    region: GraphRegion,
    profiler_rows: Sequence[OperatorHotspot],
) -> tuple[float, float, int]:
    """Aggregate timing data for a region from matched profiler rows.

    Returns (total_cuda_time_us, total_self_cuda_time_us, total_calls).
    Uses exclusive time to avoid double-counting nested scopes.
    """
    if not profiler_rows:
        return region.cuda_time_us, region.self_cuda_time_us, region.calls

    # Sum exclusive times to avoid double-counting
    total_cuda = sum(row.cuda_time_us for row in profiler_rows)
    total_self_cuda = sum(row.self_cuda_time_us for row in profiler_rows)
    total_calls = sum(row.calls for row in profiler_rows)

    # Fall back to region's own timing if no profiler data
    if total_self_cuda == 0:
        total_self_cuda = region.self_cuda_time_us
    if total_cuda == 0:
        total_cuda = region.cuda_time_us
    if total_calls == 0:
        total_calls = region.calls

    return total_cuda, total_self_cuda, total_calls


def _deduplicate_regions_by_fingerprint(
    regions: Sequence[GraphRegion],
) -> dict[str, RegionAccumulator]:
    """Group equivalent regions by their stable graph fingerprint."""
    accumulators: dict[str, RegionAccumulator] = {}

    for region in regions:
        fingerprint = region.fingerprint
        if fingerprint not in accumulators:
            accumulators[fingerprint] = RegionAccumulator(base_region=region)
        
        # Merge shape frequencies
        if region.shape_frequency:
            for shape_key, count in region.shape_frequency.items():
                accumulators[fingerprint].shape_frequencies[shape_key] = (
                    accumulators[fingerprint].shape_frequencies.get(shape_key, 0) + count
                )

    return accumulators


def correlate_profiler_to_regions(
    profiler_rows: Sequence[OperatorHotspot],
    fx_regions: Sequence[GraphRegion],
    *,
    total_cuda_time_us: float,
) -> tuple[GraphRegion, ...]:
    """Correlate profiler rows with FX regions and populate timing data.

    This is the main entry point for offline correlation. It:
    1. Matches profiler rows to regions based on scope heuristics
    2. Calculates exclusive CUDA time without double-counting nested scopes
    3. Deduplicates equivalent regions using stable graph fingerprints
    4. Aggregates timing, call counts, and shape frequencies
    5. Computes confidence and rejection reasons
    6. Returns populated GraphRegion tuples

    Args:
        profiler_rows: OperatorHotspot rows from the profiler
        fx_regions: GraphRegion instances from FX capture
        total_cuda_time_us: Total end-to-end CUDA time for percentage calculations

    Returns:
        Tuple of GraphRegion instances with populated timing and metadata
    """
    # Step 1: Group regions by fingerprint for deduplication
    fingerprint_groups = _deduplicate_regions_by_fingerprint(fx_regions)

    # Step 2: Match profiler rows to regions
    matched_regions: dict[str, list[ScopeMatch]] = defaultdict(list)
    unmatched_rows: list[OperatorHotspot] = []

    for row in profiler_rows:
        best_match: ScopeMatch | None = None
        best_confidence = 0.0

        # Try to match against each region group (use first region as representative)
        for fingerprint, accumulator in fingerprint_groups.items():
            match = _match_scope_to_region(row, accumulator.base_region)
            if match and match.confidence > best_confidence:
                best_match = match
                best_confidence = match.confidence

        if best_match:
            matched_regions[best_match.region.fingerprint].append(best_match)
        else:
            unmatched_rows.append(row)

    # Step 3: Aggregate timing data and build final regions
    final_regions: list[GraphRegion] = []

    for fingerprint, accumulator in fingerprint_groups.items():
        matches = matched_regions.get(fingerprint, [])
        matched_rows = [m.profiler_row for m in matches]

        # Aggregate timing from matched profiler rows
        total_cuda, total_self_cuda, total_calls = _aggregate_region_timing(
            accumulator.base_region,
            matched_rows,
        )

        # Merge shape frequencies
        merged_shape_freq = dict(accumulator.shape_frequencies)
        for match in matches:
            if match.profiler_row.input_shapes:
                # Create a shape key from input shapes
                shape_key = "|".join(
                    "x".join(str(d) for d in shape)
                    for shape in match.profiler_row.input_shapes
                )
                merged_shape_freq[shape_key] = (
                    merged_shape_freq.get(shape_key, 0) + match.profiler_row.calls
                )

        # Calculate confidence
        safety_reasons = tuple(reject_region(accumulator.base_region.operations))
        confidence = 0.4  # Base confidence
        if not safety_reasons:
            confidence += 0.3  # Safety bonus
        if matches:
            confidence += 0.2  # Profiler match bonus
        if total_calls >= 8:
            confidence += 0.1  # Call count bonus
        if merged_shape_freq:
            confidence += 0.1  # Shape frequency bonus
        confidence = min(1.0, confidence)

        # Calculate percentage of end-to-end time
        e2e_share = 0.0
        if total_cuda_time_us > 0:
            e2e_share = 100.0 * total_self_cuda / total_cuda_time_us

        # Estimate maximum end-to-end improvement
        estimated_improvement = optimistic_e2e_improvement(e2e_share / 100.0)

        # Build rejection reasons
        rejection_reasons = list(safety_reasons)
        if not matches:
            rejection_reasons.append("no_profiler_match")
        if accumulator.base_region.rejection_reasons:
            rejection_reasons.extend(accumulator.base_region.rejection_reasons)

        # Create the populated region
        populated_region = GraphRegion.build(
            name=accumulator.base_region.name,
            operations=accumulator.base_region.operations,
            inputs=accumulator.base_region.inputs,
            outputs=accumulator.base_region.outputs,
            dependencies=accumulator.base_region.dependencies,
            parent_module=accumulator.base_region.parent_module,
            safe_constants=accumulator.base_region.safe_constants,
            pattern_family=accumulator.base_region.pattern_family,
            rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
            calls=total_calls,
            shape_frequency=merged_shape_freq or None,
            cuda_time_us=total_cuda,
            self_cuda_time_us=total_self_cuda,
            attributes={
                **(accumulator.base_region.attributes or {}),
                "e2e_share_pct": round(e2e_share, 4),
                "estimated_max_e2e_improvement": round(estimated_improvement, 4),
                "confidence": round(confidence, 4),
                "matched_profiler_rows": len(matched_rows),
            },
        )

        final_regions.append(populated_region)

    # Track unmatched profiler rows and capture failures in a special region
    if unmatched_rows:
        # Create a synthetic region to capture unmatched profiler data
        unmatched_region = GraphRegion.build(
            name="unmatched_profiler_rows",
            operations=[row.op_key for row in unmatched_rows],
            inputs=[],  # No input metadata for unmatched rows
            outputs=[],
            dependencies=[],
            parent_module=None,
            safe_constants=None,
            pattern_family=None,
            rejection_reasons=("no_fx_region_match",),
            calls=sum(row.calls for row in unmatched_rows),
            shape_frequency=None,
            cuda_time_us=sum(row.cuda_time_us for row in unmatched_rows),
            self_cuda_time_us=sum(row.self_cuda_time_us for row in unmatched_rows),
            attributes={
                "unmatched_row_count": len(unmatched_rows),
                "unmatched_row_names": [row.name for row in unmatched_rows],
            },
        )
        final_regions.append(unmatched_region)

    return tuple(final_regions)


def correlate_discovery_report(
    profiler_export_rows: Sequence[Mapping[str, Any]],
    fx_discovery_report: DiscoveryReport,
) -> DiscoveryReport:
    """Correlate profiler data with an existing FX discovery report.

    This function takes profiler export rows and a discovery report containing
    FX-captured regions, correlates them, and returns a new discovery report with
    populated timing data.

    Args:
        profiler_export_rows: Raw profiler export rows (list of dicts)
        fx_discovery_report: Discovery report from FX capture (CPU-only)

    Returns:
        New DiscoveryReport with correlated timing data
    """
    # Parse profiler rows
    profiler_operators = parse_key_averages_rows(profiler_export_rows)

    # Correlate profiler rows with FX regions
    populated_regions = correlate_profiler_to_regions(
        profiler_operators,
        fx_discovery_report.regions,
        total_cuda_time_us=fx_discovery_report.total_cuda_time_us,
    )
    unmatched = next(
        (
            region
            for region in populated_regions
            if region.name == "unmatched_profiler_rows"
        ),
        None,
    )
    searchable_regions = tuple(
        region
        for region in populated_regions
        if region.name != "unmatched_profiler_rows"
    )
    unsupported = [
        item.as_dict() for item in fx_discovery_report.unsupported
    ]
    if unmatched is not None:
        attributes = unmatched.attributes or {}
        unmatched_count = int(attributes.get("unmatched_row_count", 0))
        unsupported.append(
            {
                "op_name": "profiler::unmatched",
                "reason": (
                    f"{unmatched_count} profiler row(s) did not match a "
                    "captured FX region"
                ),
                "count": max(unmatched_count, 1),
                "scope": "profiler_correlation",
            }
        )

    # Create new discovery report with populated regions
    return DiscoveryReport.from_dict(
        {
            "schema_version": fx_discovery_report.schema_version,
            "producer": dict(fx_discovery_report.producer),
            "workload": dict(fx_discovery_report.workload),
            "environment": dict(fx_discovery_report.environment),
            "total_cuda_time_us": fx_discovery_report.total_cuda_time_us,
            "operators": [op.as_dict() for op in profiler_operators],
            "regions": [region.as_dict() for region in searchable_regions],
            "graph_breaks": [item.as_dict() for item in fx_discovery_report.graph_breaks],
            "unsupported": unsupported,
        }
    )
