"""
AutoKernel -- GQA prefill attention kernel (one prefill chunk against a long KV cache).

Current kernel: block-wise online softmax (Flash Attention forward) with grouped KV heads
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

The chunk's queries sit at the end of the sequence: query i is at position kv_len - q_len + i and sees keys
0 .. kv_len - q_len + i (bottom-right causal). Query head h reads KV head h // (heads_q // heads_kv).
Each program loads only the KV blocks some of its queries can see, and masks only the blocks that cross
the causal edge or the end of the keys. Known headroom: every query head re-reads its KV head; packing a
group's query heads into one block shares those loads.

K and V are either contiguous [batch, heads_kv, kv_len, head_dim] or, with block_table, the engine's page
pools [num_pages, page_size, heads_kv, head_dim]: key j of sequence b is in page block_table[b, j // page_size]
at slot j % page_size. The paged path gathers every key's row through the table; a block of keys that stays
inside one page could be loaded as one contiguous tile instead.
"""

KERNEL_TYPE = "gqa_prefill_attention"

import math
import os

import torch
import triton
import triton.language as tl


@triton.jit
def gqa_prefill_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, BT_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_btz,
    M_size, N_size, GROUP, PAGE_SIZE,
    sm_scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGED: tl.constexpr,
):
    """One program per (query block, query head, batch).

    PAGED: K and V are page pools; stride_kz / stride_kn (and the V ones) are the page and slot strides.
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_z = tl.program_id(2)
    kv_h = pid_h // GROUP
    offset = N_size - M_size  # absolute position of the chunk's first query

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q_ptrs = (Q_ptr + pid_z * stride_qz + pid_h * stride_qh
              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < M_size, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634  # fold log2(e) in: exp2 is cheaper than exp

    # Every query of this block sees keys [0, first_hidden); the block's last query sees up to kv_end - 1.
    first_hidden = offset + pid_m * BLOCK_M + 1
    kv_end = tl.minimum(N_size, offset + (pid_m + 1) * BLOCK_M)
    if PAGED:
        k_base = K_ptr + kv_h * stride_kh
        v_base = V_ptr + kv_h * stride_vh
    else:
        k_base = K_ptr + pid_z * stride_kz + kv_h * stride_kh
        v_base = V_ptr + pid_z * stride_vz + kv_h * stride_vh
    for start_n in range(0, kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        in_keys = offs_n < N_size
        if PAGED:
            page = tl.load(BT_ptr + pid_z * stride_btz + offs_n // PAGE_SIZE, mask=in_keys, other=0).to(tl.int64)
            slot = offs_n % PAGE_SIZE
            k_rows = page * stride_kz + slot * stride_kn
            v_rows = page * stride_vz + slot * stride_vn
        else:
            k_rows = offs_n * stride_kn
            v_rows = offs_n * stride_vn
        k = tl.load(k_base + k_rows[:, None] + offs_d[None, :] * stride_kk, mask=in_keys[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        if start_n + BLOCK_N > first_hidden:  # crosses the causal edge or the end of the keys
            visible = (offs_n[None, :] <= offset + offs_m[:, None]) & in_keys[None, :]
            qk = tl.where(visible, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_base + v_rows[:, None] + offs_d[None, :] * stride_vk, mask=in_keys[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = (O_ptr + pid_z * stride_oz + pid_h * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=offs_m[:, None] < M_size)


def kernel_fn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, block_table: torch.Tensor = None,
              kv_len: int = None, sm_scale: float = None) -> torch.Tensor:
    """
    Entry point called by bench.py. Must match reference.gqa_prefill_attention_ref.

    Args:
        Q: [batch, heads_q, q_len, head_dim], the chunk's queries
        K: [batch, heads_kv, kv_len, head_dim], the cached prefix plus the chunk; with block_table, a page pool
           [num_pages, page_size, heads_kv, head_dim]
        V: same layout as K
        block_table: optional int32 [batch, pages]; key j of sequence b is in page block_table[b, j // page_size]
        kv_len: valid keys per sequence, required with block_table
        sm_scale: softmax scale factor, default 1/sqrt(head_dim)
    """
    assert (Q.is_cuda and K.is_cuda and V.is_cuda) or os.environ.get("TRITON_INTERPRET") == "1"
    Z, H, M_size, D = Q.shape
    assert K.shape == V.shape and K.dim() == 4
    paged = block_table is not None
    if paged:
        assert kv_len is not None and block_table.dim() == 2 and block_table.shape[0] == Z
        page_size, H_kv, N_size = K.shape[1], K.shape[2], int(kv_len)
        assert block_table.shape[1] * page_size >= N_size
        # Pools are [page, slot, head, dim]: pass page and slot strides where the contiguous layout has
        # batch and position strides.
        k_strides = (K.stride(0), K.stride(2), K.stride(1), K.stride(3))
        v_strides = (V.stride(0), V.stride(2), V.stride(1), V.stride(3))
        block_table = block_table.contiguous()
        bt, bt_stride = block_table, block_table.stride(0)
    else:
        page_size, H_kv, N_size = 1, K.shape[1], K.shape[2]
        k_strides, v_strides = K.stride(), V.stride()
        bt, bt_stride = K, 0  # never read: the contiguous path has no block table
    assert H % H_kv == 0 and 1 <= M_size <= N_size
    assert D in (16, 32, 64, 128, 256), f"Head dim {D} not supported, must be a power of 2 in [16..256]"
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    O = torch.empty_like(Q)
    BLOCK_M = 64
    BLOCK_N = 32 if D == 256 else 64
    grid = (triton.cdiv(M_size, BLOCK_M), H, Z)
    gqa_prefill_attention_kernel[grid](
        Q, K, V, O, bt,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        *k_strides,
        *v_strides,
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        bt_stride,
        M_size, N_size, H // H_kv, page_size,
        sm_scale,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        PAGED=paged,
        num_warps=8 if D == 256 else 4,
        num_stages=2,
    )
    return O
