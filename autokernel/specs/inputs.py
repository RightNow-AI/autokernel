"""Deterministic input generators for the built-in operations.

Moved verbatim (same tensor creation order, same seeding) out of ``bench.py`` so
one specification owns both the shapes and the inputs for an operation.

Every generator has the signature::

    generator(size, dtype, device, seed=42) -> dict[str, Any]

``dtype`` accepts a canonical dtype name (``"float16"``) or a ``torch.dtype``.
``torch`` is imported inside the generators so importing this module -- and
therefore discovering specifications -- never pulls in torch or a GPU context.
"""

from __future__ import annotations

from typing import Any, Mapping

from .dtypes import resolve_torch_dtype

__all__ = [
    "gen_cross_entropy_inputs",
    "gen_flash_attention_inputs",
    "gen_fused_mlp_inputs",
    "gen_layernorm_inputs",
    "gen_matmul_inputs",
    "gen_reduce_inputs",
    "gen_rmsnorm_inputs",
    "gen_rotary_embedding_inputs",
    "gen_softmax_inputs",
]

SizeMap = Mapping[str, int]


def _prepare(dtype: Any, seed: int):
    """Resolve the dtype and seed the global RNG (deterministic per seed)."""
    import torch

    torch.manual_seed(seed)
    return resolve_torch_dtype(dtype)


def gen_matmul_inputs(size: SizeMap, dtype: Any, device: str, seed: int = 42) -> dict:
    import torch

    dtype = _prepare(dtype, seed)
    M, N, K = size["M"], size["N"], size["K"]
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    return {"A": A, "B": B}


def gen_softmax_inputs(size: SizeMap, dtype: Any, device: str, seed: int = 42) -> dict:
    import torch

    dtype = _prepare(dtype, seed)
    rows, cols = size["rows"], size["cols"]
    x = torch.randn(rows, cols, device=device, dtype=dtype)
    return {"x": x}


def gen_layernorm_inputs(size: SizeMap, dtype: Any, device: str, seed: int = 42) -> dict:
    import torch

    dtype = _prepare(dtype, seed)
    batch, dim = size["batch"], size["dim"]
    x = torch.randn(batch, dim, device=device, dtype=dtype)
    weight = torch.ones(dim, device=device, dtype=dtype)
    bias = torch.zeros(dim, device=device, dtype=dtype)
    return {"x": x, "weight": weight, "bias": bias}


def gen_flash_attention_inputs(size: SizeMap, dtype: Any, device: str, seed: int = 42) -> dict:
    import torch

    dtype = _prepare(dtype, seed)
    batch, heads = size["batch"], size["heads"]
    seq_len, head_dim = size["seq_len"], size["head_dim"]
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    return {"Q": Q, "K": K, "V": V}


def gen_fused_mlp_inputs(size: SizeMap, dtype: Any, device: str, seed: int = 42) -> dict:
    import torch

    dtype = _prepare(dtype, seed)
    batch, dim, hidden = size["batch"], size["dim"], size["hidden"]
    x = torch.randn(batch, dim, device=device, dtype=dtype)
    w_gate = torch.randn(hidden, dim, device=device, dtype=dtype) * 0.02
    w_up = torch.randn(hidden, dim, device=device, dtype=dtype) * 0.02
    w_down = torch.randn(dim, hidden, device=device, dtype=dtype) * 0.02
    return {"x": x, "w_gate": w_gate, "w_up": w_up, "w_down": w_down}


def gen_cross_entropy_inputs(size: SizeMap, dtype: Any, device: str, seed: int = 42) -> dict:
    import torch

    dtype = _prepare(dtype, seed)
    batch, vocab = size["batch"], size["vocab"]
    logits = torch.randn(batch, vocab, device=device, dtype=dtype)
    targets = torch.randint(0, vocab, (batch,), device=device, dtype=torch.long)
    return {"logits": logits, "targets": targets}


def gen_rotary_embedding_inputs(size: SizeMap, dtype: Any, device: str, seed: int = 42) -> dict:
    import torch

    dtype = _prepare(dtype, seed)
    batch, heads = size["batch"], size["heads"]
    seq_len, head_dim = size["seq_len"], size["head_dim"]
    x = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    half_dim = head_dim // 2
    cos = torch.randn(seq_len, half_dim, device=device, dtype=dtype)
    sin = torch.randn(seq_len, half_dim, device=device, dtype=dtype)
    return {"x": x, "cos": cos, "sin": sin}


def gen_rmsnorm_inputs(size: SizeMap, dtype: Any, device: str, seed: int = 42) -> dict:
    import torch

    dtype = _prepare(dtype, seed)
    M, N = size["M"], size["N"]
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    return {"x": x, "weight": weight}


def gen_reduce_inputs(size: SizeMap, dtype: Any, device: str, seed: int = 42) -> dict:
    import torch

    dtype = _prepare(dtype, seed)
    M, N = size["M"], size["N"]
    x = torch.randn(M, N, device=device, dtype=dtype)
    return {"x": x}
