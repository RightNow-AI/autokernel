"""Triton baseline for Wan's modulated pre-attention LayerNorm."""

KERNEL_TYPE = "wan_modulated_layer_norm"

import torch
import triton
import triton.language as tl


@triton.jit
def _wan_modulated_layer_norm_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    output_ptr,
    tokens,
    hidden: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // tokens
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden
    row_offsets = row * hidden + offsets
    modulation_offsets = batch * hidden + offsets

    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / hidden
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / hidden
    normalized = centered * tl.rsqrt(variance + 1e-6)
    scale = tl.load(
        scale_ptr + modulation_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    shift = tl.load(
        shift_ptr + modulation_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    output = normalized * (1.0 + scale) + shift
    tl.store(output_ptr + row_offsets, output, mask=mask)


def kernel_fn(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """Fuse Wan's non-affine LayerNorm and FP32 modulation."""
    if not x.is_cuda:
        raise ValueError("wan_modulated_layer_norm requires CUDA tensors")
    if x.ndim != 3:
        raise ValueError("x must have shape [B, S, D]")
    if any(tensor.device != x.device for tensor in (scale, shift)):
        raise ValueError("all inputs must be on the x device")
    if not all(tensor.is_contiguous() for tensor in (x, scale, shift)):
        raise ValueError("all inputs must be contiguous")
    batch, tokens, hidden = x.shape
    if scale.shape != (batch, hidden) or shift.shape != (batch, hidden):
        raise ValueError(f"scale and shift must have shape {(batch, hidden)}")
    if hidden > 65536:
        raise ValueError("hidden dimension exceeds the Triton baseline limit")

    output = torch.empty_like(x)
    block_size = triton.next_power_of_2(hidden)
    num_warps = 4 if block_size <= 2048 else 8
    _wan_modulated_layer_norm_kernel[(batch * tokens,)](
        x,
        scale,
        shift,
        output,
        tokens=tokens,
        hidden=hidden,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output
