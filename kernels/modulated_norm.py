"""
AutoKernel -- Modulated Normalization kernel (LayerNorm and RMSNorm)
for Diffusion and Video Transformer models (DiT, PixArt, Wan, HunyuanVideo, Flux).

Fuses:
  norm(x) * (1 + scale) + shift
and optionally:
  updated = residual + gate * update
  norm(updated) * (1 + scale) + shift
in a single memory-efficient pass.
"""

KERNEL_TYPE = "modulated_norm"

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def modulated_norm_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    B_ptr,
    SCALE_ptr,
    SHIFT_ptr,
    stride_x_row,
    stride_y_row,
    stride_scale_row,
    stride_shift_row,
    N,
    seq_len,
    eps,
    IS_RMSNORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
    IS_BATCH_BROADCAST: tl.constexpr,
    IS_GLOBAL_SCALE: tl.constexpr,
    IS_GLOBAL_SHIFT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-parallel modulated LayerNorm / RMSNorm kernel. One program per row."""
    row_idx = tl.program_id(0)

    row_start_x = X_ptr + row_idx * stride_x_row
    row_start_y = Y_ptr + row_idx * stride_y_row

    # Determine row offsets for broadcasted scale and shift
    if IS_BATCH_BROADCAST:
        batch_idx = row_idx // seq_len
        scale_offset = batch_idx * stride_scale_row
        shift_offset = batch_idx * stride_shift_row
    else:
        scale_offset = row_idx * stride_scale_row
        shift_offset = row_idx * stride_shift_row

    if IS_GLOBAL_SCALE:
        scale_offset = 0
    if IS_GLOBAL_SHIFT:
        shift_offset = 0

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load input into float32 for numerical stability
    x = tl.load(row_start_x + col_offsets, mask=mask, other=0.0).to(tl.float32)

    if IS_RMSNORM:
        sq_mean = tl.sum(x * x, axis=0) / N
        inv_rms = 1.0 / tl.sqrt(sq_mean + eps)
        x_norm = x * inv_rms
    else:
        # LayerNorm: mean and variance
        mean = tl.sum(x, axis=0) / N
        x_centered = tl.where(mask, x - mean, 0.0)
        variance = tl.sum(x_centered * x_centered, axis=0) / N
        inv_std = 1.0 / tl.sqrt(variance + eps)
        x_norm = x_centered * inv_std

    # Optional affine weights
    if HAS_WEIGHT:
        w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        x_norm = x_norm * w
    if HAS_BIAS:
        b = tl.load(B_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        x_norm = x_norm + b

    # Apply modulation: norm * (1 + scale) + shift
    if HAS_SCALE:
        scale = tl.load(SCALE_ptr + scale_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
        mod_scale = 1.0 + scale
    else:
        mod_scale = 1.0

    if HAS_SHIFT:
        shift = tl.load(SHIFT_ptr + shift_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
    else:
        shift = 0.0

    y = x_norm * mod_scale + shift

    # Store result (converts to output dtype automatically via store)
    tl.store(row_start_y + col_offsets, y, mask=mask)


@triton.jit
def fused_gated_residual_modulated_norm_kernel(
    RES_ptr,
    GATE_ptr,
    UPD_ptr,
    RES_OUT_ptr,
    Y_ptr,
    W_ptr,
    B_ptr,
    SCALE_ptr,
    SHIFT_ptr,
    stride_res_row,
    stride_gate_row,
    stride_upd_row,
    stride_res_out_row,
    stride_y_row,
    stride_scale_row,
    stride_shift_row,
    N,
    seq_len,
    eps,
    IS_RMSNORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
    IS_BATCH_BROADCAST: tl.constexpr,
    IS_GATE_BATCH_BROADCAST: tl.constexpr,
    IS_GATE_GLOBAL: tl.constexpr,
    IS_GLOBAL_SCALE: tl.constexpr,
    IS_GLOBAL_SHIFT: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused gated-residual + modulated norm kernel."""
    row_idx = tl.program_id(0)

    row_start_res = RES_ptr + row_idx * stride_res_row
    row_start_upd = UPD_ptr + row_idx * stride_upd_row
    row_start_y = Y_ptr + row_idx * stride_y_row

    if IS_GATE_GLOBAL:
        gate_offset = 0
    elif IS_GATE_BATCH_BROADCAST:
        batch_idx = row_idx // seq_len
        gate_offset = batch_idx * stride_gate_row
    else:
        gate_offset = row_idx * stride_gate_row

    if IS_BATCH_BROADCAST:
        batch_idx = row_idx // seq_len
        scale_offset = batch_idx * stride_scale_row
        shift_offset = batch_idx * stride_shift_row
    else:
        scale_offset = row_idx * stride_scale_row
        shift_offset = row_idx * stride_shift_row

    if IS_GLOBAL_SCALE:
        scale_offset = 0
    if IS_GLOBAL_SHIFT:
        shift_offset = 0

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # 1. Compute fused gated residual: res + gate * upd
    res = tl.load(row_start_res + col_offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GATE_ptr + gate_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
    upd = tl.load(row_start_upd + col_offsets, mask=mask, other=0.0).to(tl.float32)

    x = res + gate * upd

    if STORE_RESIDUAL_OUT:
        row_start_res_out = RES_OUT_ptr + row_idx * stride_res_out_row
        tl.store(row_start_res_out + col_offsets, x, mask=mask)

    # 2. Normalization
    if IS_RMSNORM:
        sq_mean = tl.sum(x * x, axis=0) / N
        inv_rms = 1.0 / tl.sqrt(sq_mean + eps)
        x_norm = x * inv_rms
    else:
        mean = tl.sum(x, axis=0) / N
        x_centered = tl.where(mask, x - mean, 0.0)
        variance = tl.sum(x_centered * x_centered, axis=0) / N
        inv_std = 1.0 / tl.sqrt(variance + eps)
        x_norm = x_centered * inv_std

    if HAS_WEIGHT:
        w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        x_norm = x_norm * w
    if HAS_BIAS:
        b = tl.load(B_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        x_norm = x_norm + b

    # 3. Modulation
    if HAS_SCALE:
        scale = tl.load(SCALE_ptr + scale_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
        mod_scale = 1.0 + scale
    else:
        mod_scale = 1.0

    if HAS_SHIFT:
        shift = tl.load(SHIFT_ptr + shift_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
    else:
        shift = 0.0

    y = x_norm * mod_scale + shift
    tl.store(row_start_y + col_offsets, y, mask=mask)


def kernel_fn(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    shift: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    norm_type: str = "layernorm",
) -> torch.Tensor:
    """
    Entry point called by bench.py. Matches reference.modulated_norm_ref signature.
    """
    assert x.is_cuda
    orig_shape = x.shape
    N = x.shape[-1]
    seq_len = x.shape[-2] if x.ndim >= 2 else 1
    M = x.numel() // N

    x_2d = x.reshape(M, N)
    y_2d = torch.empty_like(x_2d)

    is_rmsnorm = (norm_type == "rmsnorm")
    has_weight = weight is not None
    has_bias = bias is not None and not is_rmsnorm
    has_scale = scale is not None
    has_shift = shift is not None

    w_ptr = weight if has_weight else x
    b_ptr = bias if has_bias else x

    # Broadcast detection for scale and shift
    is_batch_broadcast = False
    is_global_scale = False
    is_global_shift = False

    scale_stride_row = 0
    if has_scale:
        if scale.numel() == N:
            is_global_scale = True
        elif x.ndim >= 2 and scale.shape[0] == x.shape[0] and (scale.numel() // N) == x.shape[0]:
            is_batch_broadcast = True
            scale_stride_row = scale.stride(0)
        else:
            scale_2d = scale.reshape(M, N)
            scale_stride_row = scale_2d.stride(0)
    scale_ptr = scale if has_scale else x

    shift_stride_row = 0
    if has_shift:
        if shift.numel() == N:
            is_global_shift = True
        elif x.ndim >= 2 and shift.shape[0] == x.shape[0] and (shift.numel() // N) == x.shape[0]:
            is_batch_broadcast = True
            shift_stride_row = shift.stride(0)
        else:
            shift_2d = shift.reshape(M, N)
            shift_stride_row = shift_2d.stride(0)
    shift_ptr = shift if has_shift else x

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)

    modulated_norm_kernel[grid](
        x_2d,
        y_2d,
        w_ptr,
        b_ptr,
        scale_ptr,
        shift_ptr,
        x_2d.stride(0),
        y_2d.stride(0),
        scale_stride_row,
        shift_stride_row,
        N,
        seq_len,
        eps,
        IS_RMSNORM=is_rmsnorm,
        HAS_WEIGHT=has_weight,
        HAS_BIAS=has_bias,
        HAS_SCALE=has_scale,
        HAS_SHIFT=has_shift,
        IS_BATCH_BROADCAST=is_batch_broadcast,
        IS_GLOBAL_SCALE=is_global_scale,
        IS_GLOBAL_SHIFT=is_global_shift,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y_2d.reshape(orig_shape)


def fused_gated_residual_modulated_norm(
    residual: torch.Tensor,
    gate: torch.Tensor,
    update: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    shift: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    norm_type: str = "layernorm",
    return_residual: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Combined fused gated-residual and modulated normalization.
    """
    assert residual.is_cuda and gate.is_cuda and update.is_cuda
    orig_shape = residual.shape
    N = residual.shape[-1]
    seq_len = residual.shape[-2] if residual.ndim >= 2 else 1
    M = residual.numel() // N

    res_2d = residual.reshape(M, N)
    upd_2d = update.reshape(M, N)
    y_2d = torch.empty_like(res_2d)
    res_out_2d = torch.empty_like(res_2d) if return_residual else y_2d

    is_rmsnorm = (norm_type == "rmsnorm")
    has_weight = weight is not None
    has_bias = bias is not None and not is_rmsnorm
    has_scale = scale is not None
    has_shift = shift is not None

    w_ptr = weight if has_weight else residual
    b_ptr = bias if has_bias else residual

    is_gate_global = (gate.numel() == N)
    is_gate_batch_broadcast = False
    gate_stride_row = 0
    if not is_gate_global:
        if residual.ndim >= 2 and gate.shape[0] == residual.shape[0] and (gate.numel() // N) == residual.shape[0]:
            is_gate_batch_broadcast = True
            gate_stride_row = gate.stride(0)
        else:
            gate_2d = gate.reshape(M, N)
            gate_stride_row = gate_2d.stride(0)

    is_batch_broadcast = False
    is_global_scale = False
    is_global_shift = False

    scale_stride_row = 0
    if has_scale:
        if scale.numel() == N:
            is_global_scale = True
        elif residual.ndim >= 2 and scale.shape[0] == residual.shape[0] and (scale.numel() // N) == residual.shape[0]:
            is_batch_broadcast = True
            scale_stride_row = scale.stride(0)
        else:
            scale_2d = scale.reshape(M, N)
            scale_stride_row = scale_2d.stride(0)
    scale_ptr = scale if has_scale else residual

    shift_stride_row = 0
    if has_shift:
        if shift.numel() == N:
            is_global_shift = True
        elif residual.ndim >= 2 and shift.shape[0] == residual.shape[0] and (shift.numel() // N) == residual.shape[0]:
            is_batch_broadcast = True
            shift_stride_row = shift.stride(0)
        else:
            shift_2d = shift.reshape(M, N)
            shift_stride_row = shift_2d.stride(0)
    shift_ptr = shift if has_shift else residual

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)

    fused_gated_residual_modulated_norm_kernel[grid](
        res_2d,
        gate,
        upd_2d,
        res_out_2d,
        y_2d,
        w_ptr,
        b_ptr,
        scale_ptr,
        shift_ptr,
        res_2d.stride(0),
        gate_stride_row,
        upd_2d.stride(0),
        res_out_2d.stride(0),
        y_2d.stride(0),
        scale_stride_row,
        shift_stride_row,
        N,
        seq_len,
        eps,
        IS_RMSNORM=is_rmsnorm,
        HAS_WEIGHT=has_weight,
        HAS_BIAS=has_bias,
        HAS_SCALE=has_scale,
        HAS_SHIFT=has_shift,
        IS_BATCH_BROADCAST=is_batch_broadcast,
        IS_GATE_BATCH_BROADCAST=is_gate_batch_broadcast,
        IS_GATE_GLOBAL=is_gate_global,
        IS_GLOBAL_SCALE=is_global_scale,
        IS_GLOBAL_SHIFT=is_global_shift,
        STORE_RESIDUAL_OUT=return_residual,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    y = y_2d.reshape(orig_shape)
    res_out = res_out_2d.reshape(orig_shape) if return_residual else None
    return y, res_out
