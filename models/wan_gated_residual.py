"""Wan post-MLP gated residual fusion specification."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from autokernel.specs import (
    DT_BYTES,
    EdgeCase,
    KernelSpec,
    Tolerance,
    resolve_torch_dtype,
    size,
)

_HERE = Path(__file__).resolve().parent
STARTER_KERNEL = _HERE.parent / "kernels" / "wan_gated_residual.py"


def wan_gated_residual_ref(residual: Any, x: Any, gate: Any) -> Any:
    """Apply Wan's FP32 gated residual update and model-dtype boundary."""
    return (
        residual.float() + x.float() * gate[:, None, :].float()
    ).to(residual.dtype)


def gen_wan_gated_residual_inputs(
    size_map: Mapping[str, int],
    dtype: Any,
    device: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate deterministic residual inputs and an FP32 gate."""
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    torch_dtype = resolve_torch_dtype(dtype)
    batch = size_map["batch"]
    tokens = size_map["tokens"]
    hidden = size_map["hidden"]
    return {
        "residual": torch.randn(
            batch,
            tokens,
            hidden,
            device=device,
            dtype=torch_dtype,
            generator=generator,
        ),
        "x": torch.randn(
            batch,
            tokens,
            hidden,
            device=device,
            dtype=torch_dtype,
            generator=generator,
        ),
        "gate": torch.randn(
            batch,
            hidden,
            device=device,
            dtype=torch.float32,
            generator=generator,
        ),
    }


SPEC = KernelSpec(
    name="wan_gated_residual",
    reference_fn=wan_gated_residual_ref,
    input_generator=gen_wan_gated_residual_inputs,
    sizes={
        "small": {"batch": 1, "tokens": 20280, "hidden": 1536},
        "medium": {"batch": 1, "tokens": 32760, "hidden": 1536},
        "large": {"batch": 1, "tokens": 20280, "hidden": 5120},
    },
    dtypes=("bfloat16", "float16"),
    tolerances={
        "bfloat16": Tolerance(atol=2e-2, rtol=2e-2),
        "float16": Tolerance(atol=3e-3, rtol=3e-3),
    },
    flops_fn=2 * size("batch") * size("tokens") * size("hidden"),
    bytes_fn=(
        3 * size("batch") * size("tokens") * size("hidden") * DT_BYTES
        + 4 * size("batch") * size("hidden")
    ),
    edge_cases=(
        EdgeCase(
            name="non_power_of_two",
            size={"batch": 1, "tokens": 257, "hidden": 1537},
        ),
        EdgeCase(
            name="batched",
            size={"batch": 2, "tokens": 511, "hidden": 1536},
        ),
    ),
    shape_keys=("batch", "tokens", "hidden"),
    shape_aliases={
        "B": "batch",
        "S": "tokens",
        "D": "hidden",
        "batch": "batch",
        "tokens": "tokens",
        "hidden": "hidden",
    },
    starter_kernels={"triton": STARTER_KERNEL},
    speedup_estimate="1.2-2x versus eager PyTorch",
)
