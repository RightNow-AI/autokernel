#!/usr/bin/env python3
"""
bench.py -- AutoKernel benchmark harness (FIXED -- the agent NEVER modifies this file).

Handles:
  1. GPU hardware detection and roofline modelling
  2. Correctness verification (5 stages)
  3. Performance benchmarking (Triton do_bench)
  4. Structured, greppable output for the agent loop

Usage:
  uv run bench.py                        # benchmark kernel.py using its KERNEL_TYPE
  uv run bench.py --kernel matmul        # force kernel type
  uv run bench.py --quick                # skip stages 3-5, bench only large size
  uv run bench.py --profile              # emit torch profiler trace
  uv run bench.py --sizes large          # benchmark only 'large' size
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Timeout helper (cross-platform)
# ---------------------------------------------------------------------------

class BenchTimeoutError(Exception):
    pass


class _Timeout:
    """Context-manager wall-clock timeout. Works on both Unix (SIGALRM) and
    Windows (thread-based fallback)."""

    def __init__(self, seconds: int):
        self.seconds = seconds
        self._use_signal = hasattr(signal, "SIGALRM")

    # --- signal-based (Unix) -------------------------------------------
    def _handler(self, signum, frame):
        raise BenchTimeoutError(f"Timed out after {self.seconds}s")

    def __enter__(self):
        if self._use_signal:
            self._old = signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        else:
            import threading
            self._timer = threading.Timer(self.seconds, self._timeout_thread)
            self._timer.daemon = True
            self._timed_out = False
            self._timer.start()
        return self

    def __exit__(self, *exc):
        if self._use_signal:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)
        else:
            self._timer.cancel()
        return False

    def _timeout_thread(self):
        self._timed_out = True
        # On Windows we cannot forcefully interrupt the main thread the same
        # way SIGALRM does.  We set a flag; callers that iterate can check it.
        # For truly blocking GPU calls, this will not help -- but at least
        # the outer try/except will catch it after the call returns.
        import _thread
        _thread.interrupt_main()


# =========================================================================
# 1. GPU HARDWARE DETECTION
# =========================================================================

@dataclass
class GPUSpec:
    name: str = "Unknown"
    sm_count: int = 0
    memory_gb: float = 0.0
    peak_tflops_fp16: float = 0.0
    peak_tflops_bf16: float = 0.0
    peak_tflops_fp32: float = 0.0
    peak_bandwidth_gb_s: float = 0.0
    l2_cache_mb: float = 0.0
    compute_capability: Tuple[int, int] = (0, 0)


# Known GPU database: name_fragment -> (peak_fp16_tflops, peak_bandwidth_gb_s, l2_cache_mb)
_KNOWN_GPUS: Dict[str, Tuple[float, float, float]] = {
    "H100 SXM":   (989.5,  3352.0, 50.0),
    "H100 PCIe":  (756.0,  2039.0, 50.0),
    "H100":       (756.0,  2039.0, 50.0),   # fallback for H100 variants
    "A100-SXM":   (312.0,  2039.0, 40.0),
    "A100-PCIE":  (312.0,  1935.0, 40.0),
    "A100":       (312.0,  2039.0, 40.0),   # fallback
    "L40S":       (362.05, 864.0,  48.0),
    "L4":         (121.0,  300.0,  48.0),
    "A10":        (125.0,  600.0,  6.0),
    "4090":       (330.0,  1008.0, 72.0),
    "4080":       (305.0,  716.8,  64.0),
    "3090":       (142.0,  936.2,  6.0),
    "3080":       (119.5,  760.3,  5.0),
    # AMD Instinct GPUs
    "MI300X":     (1307.4, 5300.0, 256.0),
    "MI325X":     (1307.4, 6000.0, 256.0),
    "MI350X":     (2300.0, 8000.0, 256.0),
    "MI355X":     (2300.0, 8000.0, 256.0),
}

# AMD GPU database keyed by gcnArchName prefix for ROCm detection.
# ROCm may report an empty device name; gcnArchName is always available.
_KNOWN_AMD_GPUS: Dict[str, Tuple[str, float, float, float]] = {
    # gcnArchName prefix -> (display_name, peak_fp16_tflops, peak_bw_gb_s, l2_mb)
    "gfx942": ("AMD Instinct MI300X", 1307.4, 5300.0, 256.0),
    "gfx950": ("AMD Instinct MI350X", 2300.0, 8000.0, 256.0),
}


def detect_gpu() -> GPUSpec:
    """Auto-detect current GPU and return its spec."""
    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected, using dummy spec")
        return GPUSpec()

    props = torch.cuda.get_device_properties(0)
    name = props.name
    sm_count = props.multi_processor_count
    memory_gb = round(props.total_memory / (1024 ** 3), 1)
    cc = (props.major, props.minor)

    # On ROCm, device name may be empty; try gcnArchName-based lookup first
    gcn_arch = getattr(props, 'gcnArchName', '')
    if gcn_arch and not name:
        # ROCm: resolve name and specs via gcnArchName
        matched_amd = None
        for arch_prefix, amd_specs in _KNOWN_AMD_GPUS.items():
            if gcn_arch.startswith(arch_prefix):
                matched_amd = amd_specs
                break
        if matched_amd is not None:
            name, peak_fp16, peak_bw, l2 = matched_amd
        else:
            name = f"AMD GPU ({gcn_arch})"

    # Try to match a known GPU by name
    matched = None
    for fragment, specs in _KNOWN_GPUS.items():
        if fragment in name:
            matched = specs
            break

    if matched is not None:
        peak_fp16, peak_bw, l2 = matched
    else:
        # Estimate from hardware counters
        if hasattr(props, 'clock_rate') and props.clock_rate > 0:
            # NVIDIA path: fp16 tensor cores estimate
            ops_per_clock_per_sm = 256 if cc[0] >= 8 else 128
            clock_ghz = props.clock_rate / 1e6  # clock_rate is in kHz
            peak_fp16 = sm_count * ops_per_clock_per_sm * clock_ghz * 2 / 1e3
            peak_bw = props.clock_rate / 1e6 * 256 / 8 * 2
            peak_bw = max(peak_bw, 500.0)
        else:
            # ROCm fallback: no clock_rate available
            peak_fp16 = 500.0  # conservative estimate
            peak_bw = 2000.0   # conservative estimate
        l2 = props.L2_cache_size / (1024 * 1024) if hasattr(props, 'L2_cache_size') else 0.0

    # Derive bf16 and fp32 from fp16
    # For Ampere/Hopper: bf16 ~ fp16, fp32 ~ fp16/2
    peak_bf16 = peak_fp16
    peak_fp32 = peak_fp16 / 2.0

    return GPUSpec(
        name=name,
        sm_count=sm_count,
        memory_gb=memory_gb,
        peak_tflops_fp16=peak_fp16,
        peak_tflops_bf16=peak_bf16,
        peak_tflops_fp32=peak_fp32,
        peak_bandwidth_gb_s=peak_bw,
        l2_cache_mb=l2,
        compute_capability=cc,
    )


# =========================================================================
# 2. INPUT GENERATORS (deterministic via manual_seed)
# =========================================================================

def gen_matmul_inputs(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    M, N, K = size["M"], size["N"], size["K"]
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    return {"A": A, "B": B}


def gen_softmax_inputs(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    rows, cols = size["rows"], size["cols"]
    x = torch.randn(rows, cols, device=device, dtype=dtype)
    return {"x": x}


def gen_layernorm_inputs(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    batch, dim = size["batch"], size["dim"]
    x = torch.randn(batch, dim, device=device, dtype=dtype)
    weight = torch.ones(dim, device=device, dtype=dtype)
    bias = torch.zeros(dim, device=device, dtype=dtype)
    return {"x": x, "weight": weight, "bias": bias}


def gen_flash_attention_inputs(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    batch, heads, seq_len, head_dim = size["batch"], size["heads"], size["seq_len"], size["head_dim"]
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    return {"Q": Q, "K": K, "V": V}


def gen_fused_mlp_inputs(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    batch, dim, hidden = size["batch"], size["dim"], size["hidden"]
    x = torch.randn(batch, dim, device=device, dtype=dtype)
    w_gate = torch.randn(hidden, dim, device=device, dtype=dtype) * 0.02
    w_up = torch.randn(hidden, dim, device=device, dtype=dtype) * 0.02
    w_down = torch.randn(dim, hidden, device=device, dtype=dtype) * 0.02
    return {"x": x, "w_gate": w_gate, "w_up": w_up, "w_down": w_down}


def gen_cross_entropy_inputs(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    batch, vocab = size["batch"], size["vocab"]
    logits = torch.randn(batch, vocab, device=device, dtype=dtype)
    targets = torch.randint(0, vocab, (batch,), device=device, dtype=torch.long)
    return {"logits": logits, "targets": targets}


def gen_rotary_embedding_inputs(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    batch, heads, seq_len, head_dim = size["batch"], size["heads"], size["seq_len"], size["head_dim"]
    x = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    half_dim = head_dim // 2
    cos = torch.randn(seq_len, half_dim, device=device, dtype=dtype)
    sin = torch.randn(seq_len, half_dim, device=device, dtype=dtype)
    return {"x": x, "cos": cos, "sin": sin}


def gen_rmsnorm_inputs(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    M, N = size["M"], size["N"]
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    return {"x": x, "weight": weight}


def gen_reduce_inputs(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    M, N = size["M"], size["N"]
    x = torch.randn(M, N, device=device, dtype=dtype)
    return {"x": x}


# =========================================================================
# 3. REFERENCE WRAPPERS
# =========================================================================
# Thin wrappers that call reference.py functions with the right dict keys.

def _ref_matmul(inputs: dict) -> torch.Tensor:
    import reference
    return reference.matmul_ref(inputs["A"], inputs["B"])

def _ref_softmax(inputs: dict) -> torch.Tensor:
    import reference
    return reference.softmax_ref(inputs["x"])

def _ref_layernorm(inputs: dict) -> torch.Tensor:
    import reference
    return reference.layernorm_ref(inputs["x"], inputs["weight"], inputs["bias"])

def _ref_flash_attention(inputs: dict) -> torch.Tensor:
    import reference
    return reference.flash_attention_ref(inputs["Q"], inputs["K"], inputs["V"])

def _ref_fused_mlp(inputs: dict) -> torch.Tensor:
    import reference
    return reference.fused_mlp_ref(inputs["x"], inputs["w_gate"], inputs["w_up"], inputs["w_down"])

def _ref_cross_entropy(inputs: dict) -> torch.Tensor:
    import reference
    return reference.cross_entropy_ref(inputs["logits"], inputs["targets"])

def _ref_rotary_embedding(inputs: dict) -> torch.Tensor:
    import reference
    return reference.rotary_embedding_ref(inputs["x"], inputs["cos"], inputs["sin"])

def _ref_rmsnorm(inputs: dict) -> torch.Tensor:
    import reference
    return reference.rmsnorm_ref(inputs["x"], inputs["weight"])

def _ref_reduce(inputs: dict) -> torch.Tensor:
    import reference
    return reference.reduce_sum_ref(inputs["x"], dim=-1)


# =========================================================================
# 4. KERNEL CONFIGS
# =========================================================================

def _dtype_bytes(dtype: torch.dtype) -> int:
    """Return byte-width for a dtype."""
    return torch.tensor([], dtype=dtype).element_size()


KERNEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------
    # MATMUL
    # -----------------------------------------------------------------
    "matmul": {
        "test_sizes": [
            ("tiny",    {"M": 128,  "N": 128,  "K": 128}),
            ("small",   {"M": 512,  "N": 512,  "K": 512}),
            ("medium",  {"M": 1024, "N": 1024, "K": 1024}),
            ("large",   {"M": 2048, "N": 2048, "K": 2048}),
            ("xlarge",  {"M": 4096, "N": 4096, "K": 4096}),
            ("tall",    {"M": 8192, "N": 1024, "K": 1024}),
            ("wide",    {"M": 1024, "N": 8192, "K": 1024}),
            ("deep_k",  {"M": 1024, "N": 1024, "K": 8192}),
            ("llm_qkv", {"M": 4096, "N": 4096, "K": 512}),
            ("llm_mlp", {"M": 4096, "N": 11008, "K": 4096}),
        ],
        "test_dtypes": [torch.float16, torch.bfloat16, torch.float32],
        "tolerances": {
            torch.float16:  {"atol": 1e-2, "rtol": 1e-2},
            torch.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
            torch.float32:  {"atol": 1e-4, "rtol": 1e-4},
        },
        "flops_fn": lambda s: 2 * s["M"] * s["N"] * s["K"],
        "bytes_fn": lambda s, dt: (s["M"] * s["K"] + s["K"] * s["N"] + s["M"] * s["N"]) * _dtype_bytes(dt),
        "input_generator": gen_matmul_inputs,
        "reference_fn": _ref_matmul,
        "edge_sizes": [
            ("edge_1023",  {"M": 1023, "N": 1023, "K": 1023}),
            ("edge_4097",  {"M": 4097, "N": 4097, "K": 512}),
            ("edge_1537",  {"M": 1537, "N": 1537, "K": 1537}),
        ],
    },

    # -----------------------------------------------------------------
    # SOFTMAX
    # -----------------------------------------------------------------
    "softmax": {
        "test_sizes": [
            ("tiny",    {"rows": 32,    "cols": 128}),
            ("small",   {"rows": 256,   "cols": 512}),
            ("medium",  {"rows": 1024,  "cols": 1024}),
            ("large",   {"rows": 4096,  "cols": 4096}),
            ("xlarge",  {"rows": 8192,  "cols": 8192}),
            ("wide",    {"rows": 1024,  "cols": 32768}),
            ("narrow",  {"rows": 32768, "cols": 128}),
            ("vocab",   {"rows": 4096,  "cols": 50257}),
        ],
        "test_dtypes": [torch.float16, torch.bfloat16, torch.float32],
        "tolerances": {
            torch.float16:  {"atol": 1e-3, "rtol": 1e-3},
            torch.bfloat16: {"atol": 2e-3, "rtol": 2e-3},
            torch.float32:  {"atol": 1e-5, "rtol": 1e-5},
        },
        "flops_fn": lambda s: 5 * s["rows"] * s["cols"],  # exp + sub + sum + div + max
        "bytes_fn": lambda s, dt: 2 * s["rows"] * s["cols"] * _dtype_bytes(dt),  # read + write
        "input_generator": gen_softmax_inputs,
        "reference_fn": _ref_softmax,
        "edge_sizes": [
            ("edge_1023",  {"rows": 1023, "cols": 1023}),
            ("edge_4097",  {"rows": 4097, "cols": 4097}),
            ("edge_50257", {"rows": 1024, "cols": 50257}),
        ],
    },

    # -----------------------------------------------------------------
    # LAYERNORM
    # -----------------------------------------------------------------
    "layernorm": {
        "test_sizes": [
            ("tiny",    {"batch": 32,    "dim": 128}),
            ("small",   {"batch": 256,   "dim": 512}),
            ("medium",  {"batch": 1024,  "dim": 1024}),
            ("large",   {"batch": 4096,  "dim": 2048}),
            ("xlarge",  {"batch": 8192,  "dim": 4096}),
            ("wide",    {"batch": 1024,  "dim": 8192}),
            ("llm_7b",  {"batch": 4096,  "dim": 4096}),
            ("llm_13b", {"batch": 4096,  "dim": 5120}),
        ],
        "test_dtypes": [torch.float16, torch.bfloat16, torch.float32],
        "tolerances": {
            torch.float16:  {"atol": 1e-3, "rtol": 1e-3},
            torch.bfloat16: {"atol": 2e-3, "rtol": 2e-3},
            torch.float32:  {"atol": 1e-5, "rtol": 1e-5},
        },
        "flops_fn": lambda s: 8 * s["batch"] * s["dim"],  # mean, var, norm, scale, shift
        "bytes_fn": lambda s, dt: (2 * s["batch"] * s["dim"] + 2 * s["dim"]) * _dtype_bytes(dt),
        "input_generator": gen_layernorm_inputs,
        "reference_fn": _ref_layernorm,
        "edge_sizes": [
            ("edge_1023",  {"batch": 1023, "dim": 1023}),
            ("edge_4097",  {"batch": 4097, "dim": 4097}),
        ],
    },

    # -----------------------------------------------------------------
    # FLASH ATTENTION
    # -----------------------------------------------------------------
    "flash_attention": {
        "test_sizes": [
            ("tiny",    {"batch": 1, "heads": 4,  "seq_len": 64,   "head_dim": 64}),
            ("small",   {"batch": 2, "heads": 8,  "seq_len": 256,  "head_dim": 64}),
            ("medium",  {"batch": 2, "heads": 16, "seq_len": 512,  "head_dim": 64}),
            ("large",   {"batch": 2, "heads": 32, "seq_len": 1024, "head_dim": 64}),
            ("xlarge",  {"batch": 2, "heads": 32, "seq_len": 2048, "head_dim": 64}),
            ("long",    {"batch": 1, "heads": 32, "seq_len": 4096, "head_dim": 64}),
            ("gqa",     {"batch": 2, "heads": 32, "seq_len": 1024, "head_dim": 128}),
            ("llm_7b",  {"batch": 1, "heads": 32, "seq_len": 2048, "head_dim": 128}),
        ],
        "test_dtypes": [torch.float16, torch.bfloat16],
        "tolerances": {
            torch.float16:  {"atol": 1e-2, "rtol": 1e-2},
            torch.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
            torch.float32:  {"atol": 1e-4, "rtol": 1e-4},
        },
        # 4*N*S^2*D FLOPs (Q@K^T + softmax + attn@V)
        "flops_fn": lambda s: 4 * s["batch"] * s["heads"] * (s["seq_len"] ** 2) * s["head_dim"],
        "bytes_fn": lambda s, dt: 3 * s["batch"] * s["heads"] * s["seq_len"] * s["head_dim"] * _dtype_bytes(dt) + \
                                   s["batch"] * s["heads"] * s["seq_len"] * s["head_dim"] * _dtype_bytes(dt),
        "input_generator": gen_flash_attention_inputs,
        "reference_fn": _ref_flash_attention,
        "edge_sizes": [
            ("edge_127",  {"batch": 1, "heads": 8, "seq_len": 127,  "head_dim": 64}),
            ("edge_1023", {"batch": 1, "heads": 8, "seq_len": 1023, "head_dim": 64}),
        ],
    },

    # -----------------------------------------------------------------
    # FUSED MLP (SwiGLU)
    # -----------------------------------------------------------------
    "fused_mlp": {
        "test_sizes": [
            ("tiny",    {"batch": 32,   "dim": 128,  "hidden": 256}),
            ("small",   {"batch": 256,  "dim": 512,  "hidden": 1024}),
            ("medium",  {"batch": 1024, "dim": 1024, "hidden": 2048}),
            ("large",   {"batch": 2048, "dim": 2048, "hidden": 5504}),
            ("xlarge",  {"batch": 4096, "dim": 4096, "hidden": 11008}),
            ("llm_7b",  {"batch": 2048, "dim": 4096, "hidden": 11008}),
            ("llm_13b", {"batch": 2048, "dim": 5120, "hidden": 13824}),
        ],
        "test_dtypes": [torch.float16, torch.bfloat16, torch.float32],
        "tolerances": {
            torch.float16:  {"atol": 1e-2, "rtol": 1e-2},
            torch.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
            torch.float32:  {"atol": 1e-4, "rtol": 1e-4},
        },
        # gate_proj + up_proj + silu + mul + down_proj
        "flops_fn": lambda s: 2 * s["batch"] * s["dim"] * s["hidden"] * 3,
        "bytes_fn": lambda s, dt: (s["batch"] * s["dim"] + s["hidden"] * s["dim"] * 3 + s["batch"] * s["dim"]) * _dtype_bytes(dt),
        "input_generator": gen_fused_mlp_inputs,
        "reference_fn": _ref_fused_mlp,
        "edge_sizes": [
            ("edge_1023", {"batch": 1023, "dim": 1024, "hidden": 2048}),
            ("edge_4097", {"batch": 4097, "dim": 512,  "hidden": 1024}),
        ],
    },

    # -----------------------------------------------------------------
    # CROSS ENTROPY
    # -----------------------------------------------------------------
    "cross_entropy": {
        "test_sizes": [
            ("tiny",    {"batch": 32,    "vocab": 256}),
            ("small",   {"batch": 256,   "vocab": 1024}),
            ("medium",  {"batch": 1024,  "vocab": 4096}),
            ("large",   {"batch": 4096,  "vocab": 32000}),
            ("xlarge",  {"batch": 8192,  "vocab": 50257}),
            ("llama",   {"batch": 4096,  "vocab": 32000}),
            ("gpt2",    {"batch": 4096,  "vocab": 50257}),
        ],
        "test_dtypes": [torch.float16, torch.bfloat16, torch.float32],
        "tolerances": {
            torch.float16:  {"atol": 1e-2, "rtol": 1e-2},
            torch.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
            torch.float32:  {"atol": 1e-5, "rtol": 1e-5},
        },
        # log_softmax + nll
        "flops_fn": lambda s: 4 * s["batch"] * s["vocab"],
        "bytes_fn": lambda s, dt: (s["batch"] * s["vocab"] + s["batch"]) * _dtype_bytes(dt),
        "input_generator": gen_cross_entropy_inputs,
        "reference_fn": _ref_cross_entropy,
        "edge_sizes": [
            ("edge_1023",  {"batch": 1023, "vocab": 32000}),
            ("edge_50257", {"batch": 4096, "vocab": 50257}),
        ],
    },

    # -----------------------------------------------------------------
    # ROTARY EMBEDDING
    # -----------------------------------------------------------------
    "rotary_embedding": {
        "test_sizes": [
            ("tiny",    {"batch": 1, "heads": 4,  "seq_len": 64,   "head_dim": 64}),
            ("small",   {"batch": 2, "heads": 8,  "seq_len": 256,  "head_dim": 64}),
            ("medium",  {"batch": 2, "heads": 16, "seq_len": 512,  "head_dim": 64}),
            ("large",   {"batch": 2, "heads": 32, "seq_len": 1024, "head_dim": 128}),
            ("xlarge",  {"batch": 2, "heads": 32, "seq_len": 2048, "head_dim": 128}),
            ("llm_7b",  {"batch": 1, "heads": 32, "seq_len": 2048, "head_dim": 128}),
            ("llm_13b", {"batch": 1, "heads": 40, "seq_len": 2048, "head_dim": 128}),
        ],
        "test_dtypes": [torch.float16, torch.bfloat16, torch.float32],
        "tolerances": {
            torch.float16:  {"atol": 1e-3, "rtol": 1e-3},
            torch.bfloat16: {"atol": 2e-3, "rtol": 2e-3},
            torch.float32:  {"atol": 1e-5, "rtol": 1e-5},
        },
        # mul + add per element, x2 (cos and sin parts)
        "flops_fn": lambda s: 6 * s["batch"] * s["heads"] * s["seq_len"] * s["head_dim"],
        "bytes_fn": lambda s, dt: (s["batch"] * s["heads"] * s["seq_len"] * s["head_dim"] * 2 +
                                    s["seq_len"] * s["head_dim"]) * _dtype_bytes(dt),
        "input_generator": gen_rotary_embedding_inputs,
        "reference_fn": _ref_rotary_embedding,
        "edge_sizes": [
            ("edge_127",  {"batch": 1, "heads": 8, "seq_len": 127,  "head_dim": 64}),
            ("edge_1023", {"batch": 1, "heads": 8, "seq_len": 1023, "head_dim": 128}),
        ],
    },

    # -----------------------------------------------------------------
    # RMSNORM
    # -----------------------------------------------------------------
    "rmsnorm": {
        "test_sizes": [
            ("small",   {"M": 1024, "N": 768}),
            ("medium",  {"M": 4096, "N": 1024}),
            ("large",   {"M": 4096, "N": 4096}),
            ("llama",   {"M": 2048, "N": 4096}),
        ],
        "test_dtypes": [torch.float16, torch.bfloat16],
        "tolerances": {
            torch.float16:  {"atol": 1e-2, "rtol": 1e-2},
            torch.bfloat16: {"atol": 1e-1, "rtol": 5e-2},
        },
        "flops_fn": lambda s: 6 * s["M"] * s["N"],  # square, mean, sqrt, div, mul
        "bytes_fn": lambda s, dt: (2 * s["M"] * s["N"] + s["N"]) * torch.tensor([], dtype=dt).element_size(),
        "input_generator": gen_rmsnorm_inputs,
        "reference_fn": _ref_rmsnorm,
        "edge_sizes": [
            ("edge_1023", {"M": 1023, "N": 768}),
            ("edge_4097", {"M": 4097, "N": 1024}),
        ],
    },

    # -----------------------------------------------------------------
    # REDUCE (sum along last dim)
    # -----------------------------------------------------------------
    "reduce": {
        "test_sizes": [
            ("small",   {"M": 1024, "N": 1024}),
            ("medium",  {"M": 4096, "N": 4096}),
            ("large",   {"M": 8192, "N": 8192}),
            ("wide",    {"M": 1024, "N": 32768}),
        ],
        "test_dtypes": [torch.float16, torch.bfloat16],
        "tolerances": {
            torch.float16:  {"atol": 1e-2, "rtol": 1e-2},
            torch.bfloat16: {"atol": 1e-1, "rtol": 5e-2},
        },
        "flops_fn": lambda s: s["M"] * s["N"],  # N additions per row
        "bytes_fn": lambda s, dt: (s["M"] * s["N"] + s["M"]) * torch.tensor([], dtype=dt).element_size(),
        "input_generator": gen_reduce_inputs,
        "reference_fn": _ref_reduce,
        "edge_sizes": [
            ("edge_1023", {"M": 1023, "N": 1024}),
            ("edge_4097", {"M": 4096, "N": 4097}),
        ],
    },
}


# =========================================================================
# 5. CORRECTNESS TESTING (5 stages)
# =========================================================================

def _compare(output: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict:
    """Compare two tensors and return statistics."""
    if output.shape != expected.shape:
        return {
            "match": False,
            "reason": f"shape mismatch: {output.shape} vs {expected.shape}",
            "max_abs_error": float("inf"),
            "mean_abs_error": float("inf"),
            "pct_within_tol": 0.0,
        }

    # Cast both to float32 for comparison
    out_f = output.float()
    exp_f = expected.float()

    abs_diff = (out_f - exp_f).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    # Percentage of elements within tolerance
    within = (abs_diff <= atol + rtol * exp_f.abs()).float().mean().item() * 100.0

    match = torch.allclose(out_f, exp_f, atol=atol, rtol=rtol)
    return {
        "match": match,
        "reason": "" if match else f"max_abs_error={max_abs:.6e} exceeds tol(atol={atol}, rtol={rtol})",
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "pct_within_tol": within,
    }


def _has_nan_inf(t: torch.Tensor) -> bool:
    """Check for NaN or Inf."""
    return bool(torch.isnan(t).any().item() or torch.isinf(t).any().item())


def run_correctness(kernel_fn: Callable, config: dict, quick: bool = False) -> dict:
    """Run all correctness stages. Returns dict with results."""
    device = "cuda"
    results = {
        "smoke_test": "SKIP",
        "shape_sweep": "SKIP",
        "numerical_stability": "SKIP",
        "determinism": "SKIP",
        "edge_cases": "SKIP",
        "correctness": "FAIL",
    }
    details = []
    all_pass = True

    gen_fn = config["input_generator"]
    ref_fn = config["reference_fn"]
    sizes = config["test_sizes"]
    dtypes = config["test_dtypes"]
    tols = config["tolerances"]

    # ------------------------------------------------------------------
    # Stage 1: SMOKE TEST -- tiny input, tight tolerance
    # ------------------------------------------------------------------
    print("\n--- Stage 1: Smoke Test ---")
    try:
        tiny_label, tiny_size = sizes[0]
        # Use first dtype
        dtype0 = dtypes[0]
        inputs = gen_fn(tiny_size, dtype0, device, seed=42)
        expected = ref_fn(inputs)
        with _Timeout(30):
            output = kernel_fn(**inputs)

        if _has_nan_inf(output):
            results["smoke_test"] = "FAIL"
            details.append(f"  smoke: NaN/Inf in output")
            all_pass = False
            print(f"  FAIL: NaN/Inf in output")
        else:
            tol = tols.get(dtype0, {"atol": 1e-2, "rtol": 1e-2})
            cmp = _compare(output, expected, **tol)
            if cmp["match"]:
                results["smoke_test"] = "PASS"
                print(f"  PASS (max_abs_error={cmp['max_abs_error']:.6e})")
            else:
                results["smoke_test"] = "FAIL"
                details.append(f"  smoke: {cmp['reason']}")
                all_pass = False
                print(f"  FAIL: {cmp['reason']}")
    except BenchTimeoutError:
        results["smoke_test"] = "FAIL"
        details.append("  smoke: TIMEOUT")
        all_pass = False
        print("  FAIL: TIMEOUT")
    except torch.cuda.OutOfMemoryError:
        results["smoke_test"] = "FAIL"
        details.append("  smoke: OOM")
        all_pass = False
        print("  FAIL: OOM on tiny input")
    except Exception as e:
        results["smoke_test"] = "FAIL"
        details.append(f"  smoke: CRASH ({type(e).__name__}: {e})")
        all_pass = False
        print(f"  FAIL: CRASH ({type(e).__name__}: {e})")

    # If smoke fails, abort early
    if results["smoke_test"] == "FAIL":
        results["correctness"] = "FAIL"
        results["details"] = details
        print(f"\ncorrectness: FAIL (smoke test failed, aborting remaining stages)")
        return results

    # ------------------------------------------------------------------
    # Stage 2: SHAPE SWEEP -- all sizes x all dtypes
    # ------------------------------------------------------------------
    print("\n--- Stage 2: Shape Sweep ---")
    sweep_pass = True
    sweep_count = 0
    sweep_fail_count = 0
    worst_error = 0.0
    worst_case = ""

    for label, sz in sizes:
        for dtype in dtypes:
            sweep_count += 1
            try:
                inputs = gen_fn(sz, dtype, device, seed=42)
                expected = ref_fn(inputs)
                with _Timeout(30):
                    output = kernel_fn(**inputs)

                if _has_nan_inf(output):
                    sweep_pass = False
                    sweep_fail_count += 1
                    details.append(f"  sweep {label}/{dtype}: NaN/Inf")
                    print(f"  FAIL: {label} {dtype} -> NaN/Inf")
                    continue

                tol = tols.get(dtype, {"atol": 1e-2, "rtol": 1e-2})
                cmp = _compare(output, expected, **tol)

                if cmp["max_abs_error"] > worst_error:
                    worst_error = cmp["max_abs_error"]
                    worst_case = f"{label}/{dtype}"

                if not cmp["match"]:
                    sweep_pass = False
                    sweep_fail_count += 1
                    details.append(f"  sweep {label}/{dtype}: {cmp['reason']}")
                    print(f"  FAIL: {label} {dtype} -> {cmp['reason']}")
                else:
                    print(f"  PASS: {label} {dtype} (max_err={cmp['max_abs_error']:.2e}, within_tol={cmp['pct_within_tol']:.1f}%)")

            except torch.cuda.OutOfMemoryError:
                # OOM on larger sizes is acceptable -- just skip
                print(f"  SKIP: {label} {dtype} -> OOM")
                torch.cuda.empty_cache()
                continue
            except BenchTimeoutError:
                sweep_pass = False
                sweep_fail_count += 1
                details.append(f"  sweep {label}/{dtype}: TIMEOUT")
                print(f"  FAIL: {label} {dtype} -> TIMEOUT")
            except Exception as e:
                sweep_pass = False
                sweep_fail_count += 1
                details.append(f"  sweep {label}/{dtype}: {type(e).__name__}: {e}")
                print(f"  FAIL: {label} {dtype} -> {type(e).__name__}: {e}")
            finally:
                torch.cuda.empty_cache()

    if sweep_pass:
        results["shape_sweep"] = f"PASS ({sweep_count} configs, worst_err={worst_error:.2e} at {worst_case})"
        print(f"  shape_sweep: PASS ({sweep_count} configs, worst_err={worst_error:.2e})")
    else:
        results["shape_sweep"] = f"FAIL ({sweep_fail_count}/{sweep_count} failed)"
        all_pass = False
        print(f"  shape_sweep: FAIL ({sweep_fail_count}/{sweep_count} failed)")

    # ------------------------------------------------------------------
    # Stages 3-5: Skip in --quick mode
    # ------------------------------------------------------------------
    if quick:
        results["numerical_stability"] = "SKIP (quick mode)"
        results["determinism"] = "SKIP (quick mode)"
        results["edge_cases"] = "SKIP (quick mode)"
        results["correctness"] = "PASS" if all_pass else "FAIL"
        results["details"] = details
        print(f"\ncorrectness: {results['correctness']} (quick mode: stages 3-5 skipped)")
        return results

    # ------------------------------------------------------------------
    # Stage 3: NUMERICAL STABILITY -- adversarial inputs
    # ------------------------------------------------------------------
    print("\n--- Stage 3: Numerical Stability ---")
    stability_pass = True
    # Use medium-sized config and first dtype for stability tests
    stab_size = None
    for label, sz in sizes:
        if label == "small":
            stab_size = sz
            break
    if stab_size is None:
        stab_size = sizes[min(1, len(sizes) - 1)][1]
    stab_dtype = dtypes[0]

    # Generate adversarial input variants
    adversarial_cases = [
        ("near_max", lambda t: t * 60000.0 if t.dtype == torch.float16 else t * 1e30),
        ("near_zero", lambda t: t * 1e-6),
        ("mixed_scale", lambda t: t * torch.where(torch.rand_like(t.float()).to(t.dtype) > 0.5,
                                                    torch.tensor(1e3, device=t.device, dtype=t.dtype),
                                                    torch.tensor(1e-3, device=t.device, dtype=t.dtype))),
        ("all_zeros", lambda t: torch.zeros_like(t)),
        ("all_same", lambda t: torch.ones_like(t) * 0.5),
    ]

    for case_name, transform_fn in adversarial_cases:
        try:
            inputs = gen_fn(stab_size, stab_dtype, device, seed=42)
            # Apply transform to all float tensors in inputs
            transformed = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    transformed[k] = transform_fn(v)
                else:
                    transformed[k] = v

            expected = ref_fn(transformed)
            with _Timeout(30):
                output = kernel_fn(**transformed)

            if _has_nan_inf(output) and not _has_nan_inf(expected):
                stability_pass = False
                details.append(f"  stability {case_name}: NaN/Inf (reference is clean)")
                print(f"  FAIL: {case_name} -> NaN/Inf (reference is clean)")
            elif _has_nan_inf(output) and _has_nan_inf(expected):
                # Both have NaN/Inf -- acceptable (e.g. overflow in near_max)
                print(f"  PASS: {case_name} -> both have NaN/Inf (expected overflow)")
            else:
                tol = tols.get(stab_dtype, {"atol": 1e-2, "rtol": 1e-2})
                # Relax tolerances for adversarial inputs
                relaxed_atol = tol["atol"] * 10
                relaxed_rtol = tol["rtol"] * 10
                cmp = _compare(output, expected, atol=relaxed_atol, rtol=relaxed_rtol)
                if cmp["match"]:
                    print(f"  PASS: {case_name} (max_err={cmp['max_abs_error']:.2e})")
                else:
                    stability_pass = False
                    details.append(f"  stability {case_name}: {cmp['reason']}")
                    print(f"  FAIL: {case_name} -> {cmp['reason']}")

        except torch.cuda.OutOfMemoryError:
            print(f"  SKIP: {case_name} -> OOM")
            torch.cuda.empty_cache()
        except BenchTimeoutError:
            stability_pass = False
            details.append(f"  stability {case_name}: TIMEOUT")
            print(f"  FAIL: {case_name} -> TIMEOUT")
        except Exception as e:
            stability_pass = False
            details.append(f"  stability {case_name}: {type(e).__name__}: {e}")
            print(f"  FAIL: {case_name} -> {type(e).__name__}: {e}")
        finally:
            torch.cuda.empty_cache()

    results["numerical_stability"] = "PASS" if stability_pass else "FAIL"
    if not stability_pass:
        all_pass = False
    print(f"  numerical_stability: {results['numerical_stability']}")

    # ------------------------------------------------------------------
    # Stage 4: DETERMINISM -- same input 3 times, bitwise identical
    # ------------------------------------------------------------------
    print("\n--- Stage 4: Determinism ---")
    determinism_pass = True
    try:
        det_size = stab_size
        det_dtype = dtypes[0]
        inputs = gen_fn(det_size, det_dtype, device, seed=42)

        outputs = []
        for i in range(3):
            # Re-generate with same seed to ensure identical inputs
            inputs_i = gen_fn(det_size, det_dtype, device, seed=42)
            with _Timeout(30):
                out_i = kernel_fn(**inputs_i)
            outputs.append(out_i)

        for i in range(1, 3):
            if not torch.equal(outputs[0], outputs[i]):
                determinism_pass = False
                diff = (outputs[0].float() - outputs[i].float()).abs()
                details.append(f"  determinism: run 0 vs run {i} differ (max_diff={diff.max().item():.6e})")
                print(f"  FAIL: run 0 vs run {i} differ (max_diff={diff.max().item():.6e})")

        if determinism_pass:
            print("  PASS: 3 runs are bitwise identical")
        results["determinism"] = "PASS" if determinism_pass else "FAIL"
    except Exception as e:
        results["determinism"] = f"FAIL ({type(e).__name__})"
        all_pass = False
        details.append(f"  determinism: {type(e).__name__}: {e}")
        print(f"  FAIL: {type(e).__name__}: {e}")
    finally:
        torch.cuda.empty_cache()

    if not determinism_pass:
        all_pass = False

    # ------------------------------------------------------------------
    # Stage 5: EDGE CASES -- non-power-of-2 sizes
    # ------------------------------------------------------------------
    print("\n--- Stage 5: Edge Cases ---")
    edge_pass = True
    edge_sizes = config.get("edge_sizes", [])
    if not edge_sizes:
        results["edge_cases"] = "SKIP (no edge sizes defined)"
        print("  SKIP: no edge sizes defined")
    else:
        for label, sz in edge_sizes:
            for dtype in dtypes[:1]:  # test with first dtype only for speed
                try:
                    inputs = gen_fn(sz, dtype, device, seed=42)
                    expected = ref_fn(inputs)
                    with _Timeout(30):
                        output = kernel_fn(**inputs)

                    if _has_nan_inf(output) and not _has_nan_inf(expected):
                        edge_pass = False
                        details.append(f"  edge {label}: NaN/Inf")
                        print(f"  FAIL: {label} -> NaN/Inf")
                    else:
                        tol = tols.get(dtype, {"atol": 1e-2, "rtol": 1e-2})
                        cmp = _compare(output, expected, **tol)
                        if cmp["match"]:
                            print(f"  PASS: {label} (max_err={cmp['max_abs_error']:.2e})")
                        else:
                            edge_pass = False
                            details.append(f"  edge {label}: {cmp['reason']}")
                            print(f"  FAIL: {label} -> {cmp['reason']}")

                except torch.cuda.OutOfMemoryError:
                    print(f"  SKIP: {label} -> OOM")
                    torch.cuda.empty_cache()
                except BenchTimeoutError:
                    edge_pass = False
                    details.append(f"  edge {label}: TIMEOUT")
                    print(f"  FAIL: {label} -> TIMEOUT")
                except Exception as e:
                    edge_pass = False
                    details.append(f"  edge {label}: {type(e).__name__}: {e}")
                    print(f"  FAIL: {label} -> {type(e).__name__}: {e}")
                finally:
                    torch.cuda.empty_cache()

        results["edge_cases"] = "PASS" if edge_pass else "FAIL"
        if not edge_pass:
            all_pass = False
        print(f"  edge_cases: {results['edge_cases']}")

    # Final verdict
    results["correctness"] = "PASS" if all_pass else "FAIL"
    results["details"] = details
    print(f"\ncorrectness: {results['correctness']}")
    return results


# =========================================================================
# 6. PERFORMANCE BENCHMARKING
# =========================================================================

def _do_bench(fn: Callable, warmup: int = 25, rep: int = 100) -> float:
    """Benchmark a function and return median time in milliseconds.
    Uses triton.testing.do_bench if available, otherwise manual implementation."""
    try:
        from triton.testing import do_bench
        ms = do_bench(fn, warmup=warmup, rep=rep)
        return ms
    except ImportError:
        pass

    # Fallback: manual benchmark
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]  # median


def run_performance(kernel_fn: Callable, config: dict, gpu: GPUSpec,
                    sizes_filter: str = "all") -> dict:
    """Run performance benchmarks. Returns dict with metrics."""
    device = "cuda"
    gen_fn = config["input_generator"]
    ref_fn = config["reference_fn"]
    flops_fn = config["flops_fn"]
    bytes_fn = config["bytes_fn"]
    dtypes = config["test_dtypes"]

    # Select benchmark size
    sizes = config["test_sizes"]
    bench_sizes = []
    if sizes_filter == "all":
        bench_sizes = sizes
    else:
        for label, sz in sizes:
            if label == sizes_filter:
                bench_sizes = [(label, sz)]
                break
        if not bench_sizes:
            # If filter doesn't match, use 'large' or the biggest available
            for label, sz in sizes:
                if label == "large":
                    bench_sizes = [(label, sz)]
                    break
            if not bench_sizes:
                bench_sizes = [sizes[-1]]

    # Find the primary benchmark size (large or biggest)
    primary_label = None
    primary_size = None
    for label, sz in sizes:
        if label == "large":
            primary_label = label
            primary_size = sz
            break
    if primary_size is None:
        primary_label, primary_size = sizes[-1]

    dtype = dtypes[0]  # primary dtype for benchmarking

    all_results = []
    primary_result = None

    for label, sz in bench_sizes:
        print(f"\n  Benchmarking: {label} ...")
        try:
            flops = flops_fn(sz)
            nbytes = bytes_fn(sz, dtype)

            inputs = gen_fn(sz, dtype, device, seed=42)

            # Benchmark kernel
            with _Timeout(30):
                kernel_ms = _do_bench(lambda: kernel_fn(**inputs), warmup=25, rep=100)

            # Benchmark PyTorch reference
            with _Timeout(30):
                ref_ms = _do_bench(lambda: ref_fn(inputs), warmup=25, rep=100)

            # Compute metrics
            kernel_us = kernel_ms * 1000.0
            ref_us = ref_ms * 1000.0
            throughput_tflops = flops / (kernel_ms / 1000.0) / 1e12 if kernel_ms > 0 else 0.0
            bandwidth_gb_s = nbytes / (kernel_ms / 1000.0) / 1e9 if kernel_ms > 0 else 0.0
            ref_throughput_tflops = flops / (ref_ms / 1000.0) / 1e12 if ref_ms > 0 else 0.0

            # Roofline analysis
            arithmetic_intensity = flops / nbytes if nbytes > 0 else 0.0
            ridge_point = (gpu.peak_tflops_fp16 * 1e12) / (gpu.peak_bandwidth_gb_s * 1e9) if gpu.peak_bandwidth_gb_s > 0 else 0.0

            if arithmetic_intensity < ridge_point:
                bottleneck = "memory_bound"
                pct_peak_bandwidth = (bandwidth_gb_s / gpu.peak_bandwidth_gb_s * 100.0) if gpu.peak_bandwidth_gb_s > 0 else 0.0
                pct_peak_compute = (throughput_tflops / gpu.peak_tflops_fp16 * 100.0) if gpu.peak_tflops_fp16 > 0 else 0.0
            else:
                bottleneck = "compute_bound"
                pct_peak_compute = (throughput_tflops / gpu.peak_tflops_fp16 * 100.0) if gpu.peak_tflops_fp16 > 0 else 0.0
                pct_peak_bandwidth = (bandwidth_gb_s / gpu.peak_bandwidth_gb_s * 100.0) if gpu.peak_bandwidth_gb_s > 0 else 0.0

            speedup = ref_ms / kernel_ms if kernel_ms > 0 else 0.0

            entry = {
                "label": label,
                "size": sz,
                "dtype": str(dtype),
                "flops": flops,
                "bytes": nbytes,
                "kernel_latency_us": kernel_us,
                "pytorch_latency_us": ref_us,
                "throughput_tflops": throughput_tflops,
                "bandwidth_gb_s": bandwidth_gb_s,
                "ref_throughput_tflops": ref_throughput_tflops,
                "pct_peak_compute": pct_peak_compute,
                "pct_peak_bandwidth": pct_peak_bandwidth,
                "arithmetic_intensity": arithmetic_intensity,
                "ridge_point": ridge_point,
                "bottleneck": bottleneck,
                "speedup_vs_pytorch": speedup,
            }
            all_results.append(entry)

            if label == primary_label:
                primary_result = entry

            print(f"    kernel: {kernel_us:.2f} us | pytorch: {ref_us:.2f} us | "
                  f"speedup: {speedup:.3f}x | {throughput_tflops:.3f} TFLOPS | "
                  f"{pct_peak_compute:.1f}% peak")

        except torch.cuda.OutOfMemoryError:
            print(f"    SKIP: {label} -> OOM")
            torch.cuda.empty_cache()
        except BenchTimeoutError:
            print(f"    SKIP: {label} -> TIMEOUT")
        except Exception as e:
            print(f"    ERROR: {label} -> {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    # If we didn't bench the primary size, use the last successful one
    if primary_result is None and all_results:
        primary_result = all_results[-1]

    return {
        "primary": primary_result,
        "all": all_results,
    }


# =========================================================================
# 7. PROFILER (optional)
# =========================================================================

def run_profile(kernel_fn: Callable, config: dict):
    """Run torch profiler and save a trace."""
    device = "cuda"
    gen_fn = config["input_generator"]
    sizes = config["test_sizes"]

    # Use 'medium' or first size
    prof_size = None
    for label, sz in sizes:
        if label == "medium":
            prof_size = sz
            break
    if prof_size is None:
        prof_size = sizes[0][1]

    dtype = config["test_dtypes"][0]
    inputs = gen_fn(prof_size, dtype, device, seed=42)

    trace_dir = "./traces"
    os.makedirs(trace_dir, exist_ok=True)

    print("\n=== PROFILING ===")
    print(f"Profiling with size: {prof_size}, dtype: {dtype}")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        # Warmup
        for _ in range(5):
            kernel_fn(**inputs)
        torch.cuda.synchronize()
        # Profiled runs
        for _ in range(10):
            kernel_fn(**inputs)
        torch.cuda.synchronize()

    trace_path = os.path.join(trace_dir, "kernel_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"profile_trace: {trace_path}")

    # Print summary table
    try:
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=20))
    except Exception:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


# =========================================================================
# 8. MAIN -- orchestrate everything and produce structured output
# =========================================================================

def main():
    t_start = time.time()

    parser = argparse.ArgumentParser(description="AutoKernel benchmark harness")
    parser.add_argument("--kernel", type=str, default=None,
                        help="Kernel type to benchmark (default: read from kernel.py)")
    parser.add_argument("--sizes", type=str, default="all",
                        help="Which sizes to benchmark: small|medium|large|all (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: skip correctness stages 3-5, bench only large size")
    parser.add_argument("--profile", action="store_true",
                        help="Enable torch profiler trace")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Import the kernel module
    # ------------------------------------------------------------------
    print("=" * 60)
    print("AutoKernel Benchmark Harness")
    print("=" * 60)

    kernel_module = None
    kernel_fn = None
    kernel_type = args.kernel

    try:
        # Add cwd to path so 'import kernel' works
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        # Also add the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        kernel_module = importlib.import_module("kernel")
        kernel_fn = kernel_module.kernel_fn

        if kernel_type is None:
            kernel_type = getattr(kernel_module, "KERNEL_TYPE", None)
            if kernel_type is None:
                print("ERROR: kernel.py has no KERNEL_TYPE attribute and --kernel not specified")
                sys.exit(1)

        print(f"kernel_type: {kernel_type}")
        print(f"kernel_module: kernel.py loaded successfully")

    except SyntaxError as e:
        print(f"\nERROR: kernel.py has a syntax error:")
        print(f"  {e}")
        traceback.print_exc()
        print(f"\ncorrectness: FAIL")
        print(f"throughput_tflops: 0.000")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Failed to import kernel.py:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"\ncorrectness: FAIL")
        print(f"throughput_tflops: 0.000")
        sys.exit(1)

    # Validate kernel type
    if kernel_type not in KERNEL_CONFIGS:
        print(f"\nERROR: Unknown kernel type '{kernel_type}'")
        print(f"  Available: {', '.join(KERNEL_CONFIGS.keys())}")
        print(f"\ncorrectness: FAIL")
        print(f"throughput_tflops: 0.000")
        sys.exit(1)

    config = KERNEL_CONFIGS[kernel_type]

    # ------------------------------------------------------------------
    # GPU Detection
    # ------------------------------------------------------------------
    gpu = detect_gpu()

    print(f"\n=== GPU INFO ===")
    print(f"gpu_name: {gpu.name}")
    print(f"gpu_sm_count: {gpu.sm_count}")
    print(f"gpu_memory_gb: {gpu.memory_gb}")
    print(f"gpu_peak_tflops_fp16: {gpu.peak_tflops_fp16}")
    print(f"gpu_peak_tflops_bf16: {gpu.peak_tflops_bf16}")
    print(f"gpu_peak_tflops_fp32: {gpu.peak_tflops_fp32}")
    print(f"gpu_peak_bandwidth_gb_s: {gpu.peak_bandwidth_gb_s}")
    print(f"gpu_l2_cache_mb: {gpu.l2_cache_mb}")
    print(f"gpu_compute_capability: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------
    print(f"\n=== CORRECTNESS ===")
    try:
        correctness_results = run_correctness(kernel_fn, config, quick=args.quick)
    except Exception as e:
        print(f"\nFATAL: Correctness testing crashed: {type(e).__name__}: {e}")
        traceback.print_exc()
        correctness_results = {"correctness": "FAIL", "smoke_test": "CRASH", "shape_sweep": "CRASH",
                               "numerical_stability": "CRASH", "determinism": "CRASH", "edge_cases": "CRASH"}

    print(f"\n--- Correctness Summary ---")
    print(f"smoke_test: {correctness_results.get('smoke_test', 'N/A')}")
    print(f"shape_sweep: {correctness_results.get('shape_sweep', 'N/A')}")
    print(f"numerical_stability: {correctness_results.get('numerical_stability', 'N/A')}")
    print(f"determinism: {correctness_results.get('determinism', 'N/A')}")
    print(f"edge_cases: {correctness_results.get('edge_cases', 'N/A')}")
    print(f"correctness: {correctness_results['correctness']}")

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------
    # Determine primary size info for the header
    _perf_sizes = config["test_sizes"]
    _perf_primary_label = None
    _perf_primary_size = None
    for _pl, _ps in _perf_sizes:
        if _pl == "large":
            _perf_primary_label = _pl
            _perf_primary_size = _ps
            break
    if _perf_primary_size is None:
        _perf_primary_label, _perf_primary_size = _perf_sizes[-1]
    _perf_dtype = config["test_dtypes"][0]
    _size_params = ", ".join(f"{k}={v}" for k, v in _perf_primary_size.items())
    print(f"\n=== PERFORMANCE ({_perf_primary_label}: {_size_params}, dtype={_perf_dtype}) ===")

    perf_results = {"primary": None, "all": []}
    peak_vram_mb = 0.0
    try:
        sizes_filter = args.sizes
        if args.quick:
            sizes_filter = "large"
        torch.cuda.reset_peak_memory_stats()
        perf_results = run_performance(kernel_fn, config, gpu, sizes_filter=sizes_filter)
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception as e:
        print(f"\nFATAL: Performance benchmarking crashed: {type(e).__name__}: {e}")
        traceback.print_exc()

    primary = perf_results.get("primary")
    if primary is not None:
        print(f"\n--- Performance Summary (primary: {primary['label']}) ---")
        print(f"latency_us: {primary['kernel_latency_us']:.2f}")
        print(f"latency_ms: {primary['kernel_latency_us'] / 1000.0:.4f}")
        print(f"throughput_tflops: {primary['throughput_tflops']:.3f}")
        print(f"bandwidth_gb_s: {primary['bandwidth_gb_s']:.1f}")
        print(f"pct_peak_compute: {primary['pct_peak_compute']:.1f}%")
        print(f"pct_peak_bandwidth: {primary['pct_peak_bandwidth']:.1f}%")
        print(f"arithmetic_intensity: {primary['arithmetic_intensity']:.2f}")
        print(f"ridge_point: {primary['ridge_point']:.2f}")
        print(f"bottleneck: {primary['bottleneck']}")
        print(f"flops: {primary['flops']}")
        print(f"bytes: {primary['bytes']}")
        print(f"peak_vram_mb: {peak_vram_mb:.1f}")

        print(f"\n=== COMPARISON VS PYTORCH ===")
        print(f"pytorch_latency_us: {primary['pytorch_latency_us']:.2f}")
        print(f"pytorch_latency_ms: {primary['pytorch_latency_us'] / 1000.0:.4f}")
        print(f"kernel_latency_us: {primary['kernel_latency_us']:.2f}")
        print(f"kernel_latency_ms: {primary['kernel_latency_us'] / 1000.0:.4f}")
        print(f"speedup_vs_pytorch: {primary['speedup_vs_pytorch']:.3f}x")
        print(f"pytorch_tflops: {primary['ref_throughput_tflops']:.3f}")
        print(f"kernel_tflops: {primary['throughput_tflops']:.3f}")
    else:
        print(f"\nlatency_us: 0.00")
        print(f"latency_ms: 0.0000")
        print(f"throughput_tflops: 0.000")
        print(f"bandwidth_gb_s: 0.0")
        print(f"pct_peak_compute: 0.0%")
        print(f"pct_peak_bandwidth: 0.0%")
        print(f"peak_vram_mb: {peak_vram_mb:.1f}")
        print(f"\n=== COMPARISON VS PYTORCH ===")
        print(f"pytorch_latency_us: 0.00")
        print(f"pytorch_latency_ms: 0.0000")
        print(f"kernel_latency_us: 0.00")
        print(f"kernel_latency_ms: 0.0000")
        print(f"speedup_vs_pytorch: 0.000x")

    # ------------------------------------------------------------------
    # All sizes summary table
    # ------------------------------------------------------------------
    all_perf = perf_results.get("all", [])
    if len(all_perf) > 1:
        print(f"\n=== SIZE SWEEP ===")
        print(f"{'size':<12} {'kernel_us':>12} {'pytorch_us':>12} {'speedup':>10} {'tflops':>10} {'%peak':>8}")
        print("-" * 66)
        for entry in all_perf:
            print(f"{entry['label']:<12} {entry['kernel_latency_us']:>12.2f} "
                  f"{entry['pytorch_latency_us']:>12.2f} {entry['speedup_vs_pytorch']:>9.3f}x "
                  f"{entry['throughput_tflops']:>10.3f} {entry['pct_peak_compute']:>7.1f}%")

    # ------------------------------------------------------------------
    # Profiling (optional)
    # ------------------------------------------------------------------
    if args.profile:
        try:
            run_profile(kernel_fn, config)
        except Exception as e:
            print(f"\nWARNING: Profiling failed: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Final summary (key greppable lines)
    # ------------------------------------------------------------------
    t_elapsed = time.time() - t_start
    throughput = primary["throughput_tflops"] if primary else 0.0

    print(f"\n=== FINAL ===")
    print(f"kernel_type: {kernel_type}")
    print(f"correctness: {correctness_results['correctness']}")
    print(f"throughput_tflops: {throughput:.3f}")
    if primary:
        print(f"speedup_vs_pytorch: {primary['speedup_vs_pytorch']:.3f}x")
        print(f"pct_peak_compute: {primary['pct_peak_compute']:.1f}%")
    else:
        print(f"speedup_vs_pytorch: 0.000x")
        print(f"pct_peak_compute: 0.0%")
    print(f"bench_time_seconds: {t_elapsed:.1f}")

    if t_elapsed > 90:
        print(f"WARNING: bench.py took {t_elapsed:.1f}s (budget: 90s)")


if __name__ == "__main__":
    main()
