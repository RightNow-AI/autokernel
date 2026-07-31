"""Triton baseline for Wan's gated residual + affine LayerNorm transition."""

KERNEL_TYPE = "wan_gated_residual_norm"

import torch
import triton
import triton.language as tl


@triton.jit
def _wan_gated_residual_norm_kernel(
    residual_ptr,
    x_ptr,
    gate_ptr,
    weight_ptr,
    bias_ptr,
    normalized_ptr,
    updated_ptr,
    tokens,
    hidden: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // tokens
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden
    row_offsets = row * hidden + offsets
    channel_offsets = batch * hidden + offsets

    residual = tl.load(
        residual_ptr + row_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(
        gate_ptr + channel_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    updated = residual + x * gate

    mean = tl.sum(updated, axis=0) / hidden
    centered = tl.where(mask, updated - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / hidden
    normalized = centered * tl.rsqrt(variance + 1e-6)

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    normalized = normalized * weight + bias

    tl.store(normalized_ptr + row_offsets, normalized, mask=mask)
    tl.store(updated_ptr + row_offsets, updated, mask=mask)


def kernel_fn(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse Wan's post-self-attention transition into one GPU launch."""
    if not residual.is_cuda:
        raise ValueError("wan_gated_residual_norm requires CUDA tensors")
    if residual.ndim != 3 or x.shape != residual.shape:
        raise ValueError("residual and x must have matching [B, S, D] shapes")
    if not residual.is_contiguous() or not x.is_contiguous():
        raise ValueError("residual and x must be contiguous")
    if any(
        tensor.device != residual.device
        for tensor in (x, gate, weight, bias)
    ):
        raise ValueError("all inputs must be on the residual device")
    if (
        not gate.is_contiguous()
        or not weight.is_contiguous()
        or not bias.is_contiguous()
    ):
        raise ValueError("gate, weight, and bias must be contiguous")

    batch, tokens, hidden = residual.shape
    if gate.shape != (batch, hidden):
        raise ValueError(f"gate must have shape {(batch, hidden)}")
    if weight.shape != (hidden,) or bias.shape != (hidden,):
        raise ValueError(f"weight and bias must have shape {(hidden,)}")
    if hidden > 65536:
        raise ValueError("hidden dimension exceeds the Triton baseline limit")

    normalized = torch.empty_like(residual)
    updated = torch.empty_like(residual)
    block_size = triton.next_power_of_2(hidden)
    num_warps = 4 if block_size <= 2048 else 8
    _wan_gated_residual_norm_kernel[(batch * tokens,)](
        residual,
        x,
        gate,
        weight,
        bias,
        normalized,
        updated,
        tokens=tokens,
        hidden=hidden,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return normalized, updated
