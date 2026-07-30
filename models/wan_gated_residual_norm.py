"""Wan self-attention residual/normalization fusion specification.

This models the transition used after self-attention in every Wan transformer
block:

1. apply the per-channel attention gate and update the residual stream;
2. compute affine LayerNorm in FP32; and
3. cast both returned streams back to the model dtype.

The production shapes cover Wan 2.1 1.3B and 14B at common 480p token counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from autokernel.specs import (
    DT_BYTES,
    EdgeCase,
    KernelSpec,
    Tolerance,
    resolve_torch_dtype,
    size,
)

_HERE = Path(__file__).resolve().parent
STARTER_KERNEL = _HERE.parent / "kernels" / "wan_gated_residual_norm.py"


def wan_gated_residual_norm_ref(
    residual: Any,
    x: Any,
    gate: Any,
    weight: Any,
    bias: Any,
) -> tuple[Any, Any]:
    """Return ``(normalized, updated_residual)`` with Wan's dtype boundaries."""
    import torch.nn.functional as F

    output_dtype = residual.dtype
    updated_fp32 = residual.float() + x.float() * gate[:, None, :].float()
    normalized_fp32 = F.layer_norm(
        updated_fp32,
        (updated_fp32.shape[-1],),
        weight.float(),
        bias.float(),
        1e-6,
    )
    return normalized_fp32.to(output_dtype), updated_fp32.to(output_dtype)


def gen_wan_gated_residual_norm_inputs(
    size_map: Mapping[str, int],
    dtype: Any,
    device: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate deterministic model-dtype activations and FP32 modulation."""
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    torch_dtype = resolve_torch_dtype(dtype)
    batch = size_map["batch"]
    tokens = size_map["tokens"]
    hidden = size_map["hidden"]
    residual = torch.randn(
        batch,
        tokens,
        hidden,
        device=device,
        dtype=torch_dtype,
        generator=generator,
    )
    x = torch.randn(
        batch,
        tokens,
        hidden,
        device=device,
        dtype=torch_dtype,
        generator=generator,
    )
    gate = torch.randn(
        batch,
        hidden,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    weight = torch.randn(
        hidden,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    bias = torch.randn(
        hidden,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    return {
        "residual": residual,
        "x": x,
        "gate": gate,
        "weight": weight,
        "bias": bias,
    }


SPEC = KernelSpec(
    name="wan_gated_residual_norm",
    reference_fn=wan_gated_residual_norm_ref,
    input_generator=gen_wan_gated_residual_norm_inputs,
    sizes={
        # Wan 2.1 T2V 1.3B, 480p, common 49-frame token count.
        "small": {"batch": 1, "tokens": 20280, "hidden": 1536},
        # Wan 2.1 T2V 1.3B, 480p, common 81-frame token count.
        "medium": {"batch": 1, "tokens": 32760, "hidden": 1536},
        # Wan 2.1 14B, 480p. Sequence parallelism reduces tokens per rank.
        "large": {"batch": 1, "tokens": 20280, "hidden": 5120},
    },
    dtypes=("bfloat16", "float16"),
    tolerances={
        "bfloat16": Tolerance(atol=2e-2, rtol=2e-2),
        "float16": Tolerance(atol=3e-3, rtol=3e-3),
    },
    # Residual update, mean/variance, normalization, and affine transform.
    flops_fn=10 * size("batch") * size("tokens") * size("hidden"),
    # Activations: residual + x reads and two output writes. Gate, weight, and
    # bias are FP32 vectors and are amortized across the token dimension.
    bytes_fn=(
        4 * size("batch") * size("tokens") * size("hidden") * DT_BYTES
        + 4 * (size("batch") + 2) * size("hidden")
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
    speedup_estimate="7.9-9.1x on GB200 versus eager PyTorch",
)
