"""
Reference implementations -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. These are the oracles that the benchmark harness checks against.
"""

import torch
import torch.nn.functional as F

# Matrix Multiplication
def matmul_ref(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Standard matrix multiplication. A @ B."""
    return torch.matmul(A, B)

# Softmax
def softmax_ref(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Standard softmax along dim."""
    return F.softmax(x, dim=dim)

# Layer Normalization
def layernorm_ref(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Layer normalization over last dimension."""
    normalized_shape = x.shape[-1:]
    return F.layer_norm(x, normalized_shape, weight, bias, eps)

# RMS Normalization
def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization."""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight

# Flash Attention
def flash_attention_ref(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = True, sm_scale: float = None) -> torch.Tensor:
    """Standard scaled dot-product attention."""
    if sm_scale is None:
        sm_scale = Q.shape[-1] ** -0.5
    attn = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    if causal:
        seq_len_q, seq_len_k = Q.shape[-2], K.shape[-2]
        mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=Q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, V)

# Fused MLP (SwiGLU-style)
def fused_mlp_ref(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, activation: str = "silu") -> torch.Tensor:
    """SwiGLU-style fused MLP: down(activation(gate(x)) * up(x))."""
    gate = x @ w_gate.T
    up = x @ w_up.T
    if activation == "silu":
        gate = F.silu(gate)
    elif activation == "gelu":
        gate = F.gelu(gate)
    elif activation == "relu2":
        gate = F.relu(gate) ** 2
    return (gate * up) @ w_down.T

# Cross Entropy Loss
def cross_entropy_ref(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard cross entropy loss."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

# Rotary Position Embedding
def rotary_embedding_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)

# Parallel Reductions
def reduce_sum_ref(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sum reduction."""
    return x.sum(dim=dim)

def reduce_max_ref(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Max reduction."""
    return x.max(dim=dim).values

# GQA chunked prefill attention: one prefill chunk against a long KV cache
def gqa_prefill_attention_ref(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, sm_scale: float = None,
                              q_tile: int = 1024, kv_tile: int = 8192, causal: bool = True,
                              block_table: torch.Tensor = None, kv_len: int = None) -> torch.Tensor:
    """Exact float32 oracle for one chunk of a long causal prefill with grouped-query attention.

    Q: [batch, heads_q, q_len, head_dim], the chunk's queries at absolute positions kv_len - q_len .. kv_len - 1.
    K, V: [batch, heads_kv, kv_len, head_dim], every key the chunk attends to: the cached prefix plus the chunk.
    Paged KV (block_table given): K, V are the engine's page pools [num_pages, page_size, heads_kv, head_dim];
    key j of sequence b sits in page block_table[b, j // page_size] at slot j % page_size, and kv_len is the
    number of valid keys per sequence (the last page may be partly filled). Other pages and slots are never read.
    heads_q is a multiple of heads_kv; query head h reads KV head h // (heads_q // heads_kv).
    The causal mask is bottom-right aligned: query i sees keys 0 .. kv_len - q_len + i (causal=False: every key).
    Online softmax over KV tiles in float32, so the q_len x kv_len score matrix is never materialized
    (a 32K-token chunk against a 1M-token cache would need 128 GiB per head).
    """
    batch, heads_q, q_len, head_dim = Q.shape
    if K.shape != V.shape or K.dim() != 4 or K.shape[3] != head_dim:
        raise ValueError("K and V must be 4-D tensors of one shape with Q's head_dim")
    if block_table is None:
        if K.shape[0] != batch or kv_len not in (None, K.shape[2]):
            raise ValueError("K and V must be [batch, heads_kv, kv_len, head_dim] matching Q")
        heads_kv, kv_len = K.shape[1], K.shape[2]

        def kv_rows(T, b, g, k0, k1):  # keys k0 .. k1 - 1 of KV head g: [k1 - k0, head_dim]
            return T[b, g, k0:k1]
    else:
        page_size, heads_kv = K.shape[1], K.shape[2]
        if (block_table.dim() != 2 or block_table.shape[0] != batch or kv_len is None
                or block_table.shape[1] * page_size < kv_len):
            raise ValueError("paged K and V need kv_len and a [batch, pages] block_table that covers kv_len keys")

        def kv_rows(T, b, g, k0, k1):
            keys = torch.arange(k0, k1, device=T.device)
            return T[block_table[b, keys // page_size].long(), keys % page_size, g]
    if heads_kv < 1 or heads_q % heads_kv or not 1 <= q_len <= kv_len:
        raise ValueError("need heads_q % heads_kv == 0 and 1 <= q_len <= kv_len")
    group = heads_q // heads_kv
    scale = head_dim ** -0.5 if sm_scale is None else sm_scale
    offset = kv_len - q_len
    out = torch.empty_like(Q)
    for b in range(batch):
        for g in range(heads_kv):
            heads = slice(g * group, (g + 1) * group)
            for q0 in range(0, q_len, q_tile):
                q1 = min(q0 + q_tile, q_len)
                q = Q[b, heads, q0:q1].float() * scale  # [group, rows, head_dim]
                rows = q1 - q0
                m = torch.full((group, rows), float("-inf"), device=Q.device)
                l = torch.zeros((group, rows), device=Q.device)
                acc = torch.zeros((group, rows, head_dim), device=Q.device)
                last_key = offset + q1 - 1 if causal else kv_len - 1  # the last key this tile may see
                for k0 in range(0, last_key + 1, kv_tile):
                    k1 = min(k0 + kv_tile, last_key + 1)
                    s = torch.matmul(q, kv_rows(K, b, g, k0, k1).float().transpose(0, 1))  # [group, rows, keys]
                    if causal and k1 - 1 > offset + q0:  # this tile holds keys some queries must not see
                        limit = torch.arange(q0, q1, device=Q.device)[:, None] + offset
                        keys = torch.arange(k0, k1, device=Q.device)[None, :]
                        s = s.masked_fill(keys > limit, float("-inf"))
                    m_new = torch.maximum(m, s.amax(dim=-1))
                    p = torch.exp(s - m_new[..., None])
                    alpha = torch.exp(m - m_new)
                    l = l * alpha + p.sum(dim=-1)
                    acc = acc * alpha[..., None] + torch.matmul(p, kv_rows(V, b, g, k0, k1).float())
                    m = m_new
                out[b, heads, q0:q1] = (acc / l[..., None]).to(Q.dtype)
    return out
