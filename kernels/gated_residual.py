"""
AutoKernel -- Gated Residual Fusion kernel.
Computes:
  output = residual + gate * update
with support for broadcasted gate (e.g. [B, 1, D] or [B, D]).
"""

KERNEL_TYPE = "gated_residual"

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def gated_residual_kernel(
    RES_ptr,
    GATE_ptr,
    UPD_ptr,
    OUT_ptr,
    stride_res_row,
    stride_gate_row,
    stride_upd_row,
    stride_out_row,
    N,
    seq_len,
    IS_GATE_BROADCAST: tl.constexpr,
    IS_GATE_GLOBAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-parallel gated residual kernel. One program per row."""
    row_idx = tl.program_id(0)

    row_start_res = RES_ptr + row_idx * stride_res_row
    row_start_upd = UPD_ptr + row_idx * stride_upd_row
    row_start_out = OUT_ptr + row_idx * stride_out_row

    if IS_GATE_GLOBAL:
        gate_offset = 0
    elif IS_GATE_BROADCAST:
        batch_idx = row_idx // seq_len
        gate_offset = batch_idx * stride_gate_row
    else:
        gate_offset = row_idx * stride_gate_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    res = tl.load(row_start_res + col_offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GATE_ptr + gate_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
    upd = tl.load(row_start_upd + col_offsets, mask=mask, other=0.0).to(tl.float32)

    out = res + gate * upd
    tl.store(row_start_out + col_offsets, out, mask=mask)


def kernel_fn(
    residual: torch.Tensor,
    gate: torch.Tensor,
    update: torch.Tensor,
) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.gated_residual_ref signature."""
    assert residual.is_cuda and gate.is_cuda and update.is_cuda
    orig_shape = residual.shape
    N = residual.shape[-1]
    seq_len = residual.shape[-2] if residual.ndim >= 2 else 1
    M = residual.numel() // N

    res_2d = residual.reshape(M, N)
    upd_2d = update.reshape(M, N)
    out_2d = torch.empty_like(res_2d)

    is_gate_global = (gate.numel() == N)
    is_gate_broadcast = False
    if not is_gate_global:
        if residual.ndim >= 2 and gate.shape[0] == residual.shape[0] and (gate.numel() // N) == residual.shape[0]:
            is_gate_broadcast = True
            gate_stride_row = gate.stride(0)
        else:
            gate_2d = gate.reshape(M, N)
            gate_stride_row = gate_2d.stride(0)
    else:
        gate_stride_row = 0

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)

    gated_residual_kernel[grid](
        res_2d,
        gate,
        upd_2d,
        out_2d,
        res_2d.stride(0),
        gate_stride_row,
        upd_2d.stride(0),
        out_2d.stride(0),
        N,
        seq_len,
        IS_GATE_BROADCAST=is_gate_broadcast,
        IS_GATE_GLOBAL=is_gate_global,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_2d.reshape(orig_shape)
