"""Candidate impact ranking and Amdahl-style end-to-end ceilings.

Uses measured CUDA time share and an optimistic reducible fraction to decide
whether a region is worth searching. Defaults match the universal plan:

- impact floor: do not search when optimistic e2e improvement is below 0.5%
- promotion preference: candidates that can plausibly exceed 1% e2e gain
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .safety import is_region_safe, reject_region
from .types import GraphRegion, OperatorHotspot


DEFAULT_IMPACT_FLOOR = 0.005  # 0.5% optimistic end-to-end
DEFAULT_PROMOTION_TARGET = 0.01  # 1%
DEFAULT_REDUCIBLE_FRACTION = 0.9  # optimistic upper bound on kernel speedup


@dataclass(frozen=True)
class RankedCandidate:
    """One discovery region with model-level value estimates."""

    region: GraphRegion
    share_of_e2e: float
    estimated_reducible_fraction: float
    estimated_max_e2e_improvement: float
    confidence: float
    search_worthy: bool
    meets_promotion_target: bool
    rejection_reasons: tuple[str, ...]
    pattern_family: str | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.region.name,
            "fingerprint": self.region.fingerprint,
            "share_of_e2e": self.share_of_e2e,
            "estimated_reducible_fraction": self.estimated_reducible_fraction,
            "estimated_max_e2e_improvement": self.estimated_max_e2e_improvement,
            "confidence": self.confidence,
            "search_worthy": self.search_worthy,
            "meets_promotion_target": self.meets_promotion_target,
            "rejection_reasons": list(self.rejection_reasons),
            "pattern_family": self.pattern_family or self.region.pattern_family,
            "calls": self.region.calls,
            "cuda_time_us": self.region.cuda_time_us,
        }


def e2e_share(cuda_time_us: float, total_cuda_time_us: float) -> float:
    if total_cuda_time_us <= 0:
        return 0.0
    return max(0.0, float(cuda_time_us) / float(total_cuda_time_us))


def optimistic_e2e_improvement(
    share: float,
    *,
    reducible_fraction: float = DEFAULT_REDUCIBLE_FRACTION,
) -> float:
    """Amdahl-style upper bound: share * reducible_fraction.

    If a region is 2% of e2e and we can remove 90% of its time, the model
    improves by at most ~1.8%.
    """
    if reducible_fraction < 0.0 or reducible_fraction > 1.0:
        raise ValueError("reducible_fraction must be in [0, 1]")
    return max(0.0, share * reducible_fraction)


def classify_pattern_family(operations: Sequence[str]) -> str:
    """Coarse family label for reports; not a correctness claim."""
    ops = " ".join(operations).lower()
    if any(tok in ops for tok in ("layer_norm", "rms_norm", "native_layer_norm")):
        if any(tok in ops for tok in ("mul", "add")):
            return "residual_gate_norm"
        return "normalization"
    if any(tok in ops for tok in ("silu", "gelu", "relu", "sigmoid")):
        return "activation_epilogue"
    if any(tok in ops for tok in ("permute", "transpose", "contiguous", "clone", "to")):
        if all(
            any(x in o for x in ("permute", "transpose", "contiguous", "clone", "to", "view", "reshape"))
            for o in operations
        ):
            return "layout_cast_copy"
    if any(tok in ops for tok in ("mul", "add", "sub", "div")):
        return "elementwise_chain"
    return "unknown"


def rank_regions(
    regions: Sequence[GraphRegion],
    *,
    total_cuda_time_us: float,
    impact_floor: float = DEFAULT_IMPACT_FLOOR,
    promotion_target: float = DEFAULT_PROMOTION_TARGET,
    reducible_fraction: float = DEFAULT_REDUCIBLE_FRACTION,
) -> tuple[RankedCandidate, ...]:
    """Rank graph regions by optimistic end-to-end value."""
    ranked: list[RankedCandidate] = []
    for region in regions:
        share = e2e_share(region.cuda_time_us, total_cuda_time_us)
        safety_reasons = tuple(reject_region(region.operations))
        if region.rejection_reasons:
            safety_reasons = tuple(
                dict.fromkeys([*region.rejection_reasons, *safety_reasons])
            )
        improvement = optimistic_e2e_improvement(
            share, reducible_fraction=reducible_fraction
        )
        safe = is_region_safe(region.operations) and not safety_reasons
        # Confidence: higher when more calls and pure allowlist.
        confidence = 0.4
        if safe:
            confidence += 0.3
        if region.calls >= 8:
            confidence += 0.2
        if region.shape_frequency:
            confidence += 0.1
        confidence = min(1.0, confidence)

        search_worthy = safe and improvement >= impact_floor
        reasons = list(safety_reasons)
        if not safe:
            pass
        elif improvement < impact_floor:
            reasons.append(
                f"below_impact_floor: optimistic e2e {improvement:.4f} "
                f"< floor {impact_floor:.4f}"
            )
            search_worthy = False

        family = region.pattern_family or classify_pattern_family(
            region.operations
        )
        ranked.append(
            RankedCandidate(
                region=region,
                share_of_e2e=share,
                estimated_reducible_fraction=reducible_fraction,
                estimated_max_e2e_improvement=improvement,
                confidence=confidence,
                search_worthy=search_worthy,
                meets_promotion_target=improvement >= promotion_target,
                rejection_reasons=tuple(reasons),
                pattern_family=family,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.estimated_max_e2e_improvement,
            -item.share_of_e2e,
            -item.confidence,
            item.region.name,
        )
    )
    return tuple(ranked)


def rank_operators(
    operators: Sequence[OperatorHotspot],
    *,
    total_cuda_time_us: float,
) -> tuple[tuple[OperatorHotspot, float], ...]:
    """Return operators sorted by self CUDA e2e share (descending).

    ``cuda_time_us`` is inclusive in torch profiler exports, so ranking with it
    counts the same device work once for every enclosing record_function or
    custom-op scope. ``total_cuda_time_us`` is the sum of self CUDA time and
    operator shares must use the same accounting basis.
    """
    rows = [
        (op, e2e_share(op.self_cuda_time_us, total_cuda_time_us))
        for op in operators
    ]
    rows.sort(key=lambda item: (-item[1], -item[0].calls, item[0].name))
    return tuple(rows)
