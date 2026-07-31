"""Triton baseline for Wan's post-MLP gated residual update."""

KERNEL_TYPE = "wan_gated_residual"

import torch
import triton
import triton.language as tl


@triton.jit
def _wan_gated_residual_kernel(
    residual_ptr,
    x_ptr,
    gate_ptr,
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
    gate_offsets = batch * hidden + offsets
    residual = tl.load(
        residual_ptr + row_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(output_ptr + row_offsets, residual + x * gate, mask=mask)


def kernel_fn(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Fuse Wan's MLP gate and residual update into one launch."""
    if not residual.is_cuda:
        raise ValueError("wan_gated_residual requires CUDA tensors")
    if residual.ndim != 3 or x.shape != residual.shape:
        raise ValueError("residual and x must have matching [B, S, D] shapes")
    if any(tensor.device != residual.device for tensor in (x, gate)):
        raise ValueError("all inputs must be on the residual device")
    if not all(tensor.is_contiguous() for tensor in (residual, x, gate)):
        raise ValueError("all inputs must be contiguous")
    batch, tokens, hidden = residual.shape
    if gate.shape != (batch, hidden):
        raise ValueError(f"gate must have shape {(batch, hidden)}")
    if hidden > 65536:
        raise ValueError("hidden dimension exceeds the Triton baseline limit")

    output = torch.empty_like(residual)
    block_size = triton.next_power_of_2(hidden)
    num_warps = 4 if block_size <= 2048 else 8
    _wan_gated_residual_kernel[(batch * tokens,)](
        residual,
        x,
        gate,
        output,
        tokens=tokens,
        hidden=hidden,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output
