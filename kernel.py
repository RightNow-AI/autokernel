"""
AutoKernel -- Extracted kernel from model profiling.
Op type: fused_mlp
Rank: 30 (0.7% of GPU time)
Model shape: batch=2048, dim=2048, hidden=5504

This kernel was extracted from profiling transformers.
The agent optimizes this to maximize throughput at the model-specific shapes.
"""

KERNEL_TYPE = "fused_mlp"

# Model-specific shapes (the shapes that matter for THIS model)
MODEL_SHAPES = {'batch': 2048, 'dim': 2048, 'hidden': 5504}

# Benchmark config (self-describing -- bench.py can load this dynamically)
TEST_SIZES = [
    ("model_primary", {'batch': 2048, 'dim': 2048, 'hidden': 5504}),
    # Also test nearby sizes for robustness
    ("model_half", {'batch': 1024, 'dim': 1024, 'hidden': 2752}),
    ("model_double", {'batch': 4096, 'dim': 4096, 'hidden': 11008}),
]

TOLERANCES = {'float16': {'atol': 0.01, 'rtol': 0.01}, 'bfloat16': {'atol': 0.1, 'rtol': 0.1}, 'float32': {'atol': 0.0001, 'rtol': 0.0001}}


def FLOPS_FN(s):
    return 2 * s["batch"] * s["dim"] * s["hidden"] * 3


def BYTES_FN(s, dt_bytes):
    return (s["batch"] * s["dim"] + s["hidden"] * s["dim"] * 3 + s["batch"] * s["dim"]) * dt_bytes


# ======================================================================
# Triton kernel code (from kernels/fused_mlp.py)
# ======================================================================

import torch
import triton
import triton.language as tl


@triton.jit
def fused_gate_up_kernel(
    X_ptr,
    W_gate_ptr,
    W_up_ptr,
    Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wgk, stride_wgn,
    stride_wuk, stride_wun,
    stride_om, stride_on,
    USE_SILU: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel: computes activation(X @ W_gate^T) * (X @ W_up^T).
    W_gate and W_up are [intermediate_size, hidden_size] (transposed access).
    X is [M, K], output is [M, N] where N = intermediate_size, K = hidden_size.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers for X
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk

    # Pointers for W_gate (shape [K, N] after transpose -- stored as [N, K])
    wg_ptrs = W_gate_ptr + offs_k[:, None] * stride_wgk + offs_n[None, :] * stride_wgn
    # Pointers for W_up
    wu_ptrs = W_up_ptr + offs_k[:, None] * stride_wuk + offs_n[None, :] * stride_wun

    # Accumulators
    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        k_offs = k_start + offs_k
        # Load X tile
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W_gate tile
        wg_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        wg = tl.load(wg_ptrs, mask=wg_mask, other=0.0)

        # Load W_up tile
        wu_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        wu = tl.load(wu_ptrs, mask=wu_mask, other=0.0)

        acc_gate += tl.dot(x, wg)
        acc_up += tl.dot(x, wu)

        x_ptrs += BLOCK_SIZE_K * stride_xk
        wg_ptrs += BLOCK_SIZE_K * stride_wgk
        wu_ptrs += BLOCK_SIZE_K * stride_wuk

    # Apply activation to gate and multiply with up
    if USE_SILU:
        # SiLU(x) = x * sigmoid(x)
        gate_activated = acc_gate * tl.sigmoid(acc_gate)
    else:
        # GELU approximation
        gate_activated = 0.5 * acc_gate * (1.0 + tl.math.tanh(0.7978845608 * (acc_gate + 0.044715 * acc_gate * acc_gate * acc_gate)))

    result = gate_activated * acc_up

    # Store
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, result.to(Out_ptr.dtype.element_ty), mask=out_mask)


def kernel_fn(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    """
    Entry point called by bench.py. Must match reference.fused_mlp_ref signature.

    SwiGLU MLP:
      hidden = activation(x @ w_gate.T) * (x @ w_up.T)
      out = hidden @ w_down.T

    Args:
        x: [batch, hidden_size] or [batch, seq_len, hidden_size]
        w_gate: [intermediate_size, hidden_size]
        w_up: [intermediate_size, hidden_size]
        w_down: [hidden_size, intermediate_size]
        activation: "silu" or "gelu"
    """
    assert x.device.type in ("cuda", "xpu")

    # Handle multi-dim input
    orig_shape = x.shape
    if x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    M, K = x.shape
    N, K2 = w_gate.shape
    assert K == K2, f"Hidden dim mismatch: x has {K}, w_gate has {K2}"
    assert w_up.shape == (N, K), f"w_up shape mismatch"

    hidden = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # W_gate and W_up are [N, K]. We access them as transposed: X[M,K] @ W^T[K,N]
    # So stride_wgk corresponds to stride along the K dimension (stride(1) of [N,K])
    # and stride_wgn corresponds to stride along N dimension (stride(0) of [N,K])
    fused_gate_up_kernel[grid](
        x,
        w_gate,
        w_up,
        hidden,
        M, N, K,
        x.stride(0), x.stride(1),
        w_gate.stride(1), w_gate.stride(0),  # transposed: K-stride, N-stride
        w_up.stride(1), w_up.stride(0),      # transposed: K-stride, N-stride
        hidden.stride(0), hidden.stride(1),
        USE_SILU=(activation == "silu"),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    # Down projection (not fused -- separate matmul)
    # hidden: [M, N], w_down: [hidden_size, intermediate_size]
    # out = hidden @ w_down.T
    out = hidden @ w_down.t()

    if len(orig_shape) > 2:
        out = out.view(*orig_shape[:-1], out.shape[-1])

    return out
