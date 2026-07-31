"""Allowlist and rejection checks for captured graph regions.

Fail closed: mutation, data-dependent Python control flow, collectives,
unknown aliasing, and unsupported custom ops are rejected before search.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# Minimal pure ATen/elementwise subset useful for early LTX candidates.
# Expand from real profile evidence, not from a desire to support all of PyTorch.
ALLOWED_ATEN_OPS: frozenset[str] = frozenset(
    {
        "aten::add",
        "aten::mul",
        "aten::sub",
        "aten::div",
        "aten::neg",
        "aten::exp",
        "aten::silu",
        "aten::gelu",
        "aten::relu",
        "aten::sigmoid",
        "aten::tanh",
        "aten::rsqrt",
        "aten::sqrt",
        "aten::pow",
        "aten::mean",
        "aten::var",
        "aten::layer_norm",
        "aten::rms_norm",
        "aten::native_layer_norm",
        "aten::to",
        "aten::clone",
        "aten::contiguous",
        "aten::view",
        "aten::reshape",
        "aten::permute",
        "aten::transpose",
        "aten::unsqueeze",
        "aten::squeeze",
        "aten::cat",
        "aten::stack",
        "aten::expand",
        "aten::broadcast_to",
        "aten::type_as",
    }
)

REJECT_SUBSTRINGS: tuple[tuple[str, str], ...] = (
    ("c10d", "collective communication"),
    ("all_reduce", "collective communication"),
    ("all_gather", "collective communication"),
    ("reduce_scatter", "collective communication"),
    ("barrier", "collective communication"),
    ("aten::item", "data-dependent host sync"),
    ("aten::nonzero", "data-dependent indexing"),
    ("aten::unique", "data-dependent indexing"),
    ("aten::argsort", "data-dependent ordering"),
    ("aten::index_put", "mutation / complex indexing"),
    ("aten::scatter", "mutation / scatter"),
    ("aten::sort", "data-dependent ordering"),
    ("aten::randint", "rng / nondeterminism boundary"),
    ("aten::rand", "rng / nondeterminism boundary"),
    ("prims::", "unsupported prims op"),
)


def normalize_op_name(op_name: str) -> str:
    name = op_name.strip()
    if name.startswith("aten.") and not name.startswith("aten::"):
        name = "aten::" + name[len("aten.") :]
    # Strip schema overloads: aten::add.Tensor -> aten::add
    if "." in name and name.startswith("aten::"):
        base, _sep, _rest = name.partition(".")
        # Keep dtype/device markers that use a single segment after :: only when
        # they are overload names (add.Tensor). Nested module names stay intact.
        if _rest and "." not in base:
            name = base
    return name


def reject_region(
    operations: Sequence[str],
    *,
    allowlist: Iterable[str] | None = None,
) -> list[str]:
    """Return rejection reasons for a candidate op sequence (empty if safe)."""
    allowed = frozenset(allowlist) if allowlist is not None else ALLOWED_ATEN_OPS
    reasons: list[str] = []
    if not operations:
        return ["empty operation sequence"]

    for op in operations:
        normalized = normalize_op_name(op)
        if normalized.startswith("aten::") and normalized.endswith("_"):
            reasons.append(f"{normalized}: in-place mutation")
            continue
        lower = normalized.lower()
        for token, reason in REJECT_SUBSTRINGS:
            if token in lower:
                reasons.append(f"{normalized}: {reason}")
                break
        else:
            if normalized not in allowed and not normalized.startswith("aten::"):
                reasons.append(f"{normalized}: unsupported custom operator")
            elif normalized not in allowed:
                reasons.append(f"{normalized}: not in pure-tensor allowlist")
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            unique.append(reason)
    return unique


def is_region_safe(
    operations: Sequence[str],
    *,
    allowlist: Iterable[str] | None = None,
) -> bool:
    return not reject_region(operations, allowlist=allowlist)
