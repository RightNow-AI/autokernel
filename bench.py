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
  uv run bench.py --page-size 16         # paged-KV ops: every size reads 16-token pages through a block table
  uv run bench.py --calibration calibrate.json   # also report % of the peak calibrate.py measured on this GPU
"""

from __future__ import annotations

import argparse
import importlib
import json
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


def _load_stdlib_profile() -> None:
    """Register the standard-library profile module before anything imports it.

    Running a script from this directory puts it first on sys.path, where profile.py (AutoKernel's profiler)
    shadows the standard module that cProfile imports. torch._dynamo imports cProfile, so torch.compile and
    torch.nn.attention.bias would otherwise fail with "module 'profile' has no attribute 'run'".
    """
    if "profile" in sys.modules:
        return
    import importlib.util
    import sysconfig
    spec = importlib.util.spec_from_file_location(
        "profile", os.path.join(sysconfig.get_paths()["stdlib"], "profile.py"))
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["profile"] = module
    spec.loader.exec_module(module)


_load_stdlib_profile()

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
# Fragments are matched in order, so GB300/GB200 must come before B300/B200.
# l2_cache_mb = 0.0 means "read L2 size from the device".
# Blackwell: dense FP16/BF16 = half of NVIDIA's sparse HGX/NVL72 table figure.
# Check a new node with `uv run calibrate.py`.
_KNOWN_GPUS: Dict[str, Tuple[float, float, float]] = {
    "GB300":      (2500.0, 8000.0, 0.0),
    "GB200":      (2500.0, 8000.0, 0.0),
    "B300":       (2250.0, 8000.0, 0.0),
    "B200":       (2250.0, 8000.0, 0.0),
    "H200":       (989.5,  4800.0, 50.0),
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
        if l2 <= 0.0:
            l2 = props.L2_cache_size / (1024 * 1024) if hasattr(props, 'L2_cache_size') else 0.0
    else:
        print(f"WARNING: {name} is not in the GPU database; roofline peaks are estimates. "
              f"Run `uv run calibrate.py` and add an entry.")
        if hasattr(props, 'clock_rate') and props.clock_rate > 0:
            # NVIDIA path: fp16 tensor cores estimate
            ops_per_clock_per_sm = 256 if cc[0] >= 8 else 128
            clock_ghz = props.clock_rate / 1e6  # clock_rate is in kHz
            peak_fp16 = sm_count * ops_per_clock_per_sm * clock_ghz * 2 / 1e3
            mem_clock_khz = getattr(props, 'memory_clock_rate', 0)
            bus_width_bits = getattr(props, 'memory_bus_width', 0)
            if mem_clock_khz > 0 and bus_width_bits > 0:
                # Same formula as the CUDA deviceQuery sample (DDR: x2)
                peak_bw = 2.0 * mem_clock_khz * (bus_width_bits / 8) / 1e6
            else:
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


def gen_gqa_prefill_attention_inputs(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    """One prefill chunk of q_len queries against kv_len keys (the cached prefix plus the chunk).

    Queries are scaled so attention scores have a standard deviation of 4. Random unscaled inputs spread
    attention over thousands of keys at long kv_len, which shrinks the output far below the tolerance;
    peaked attention keeps it O(1), so a kernel that drops distant keys or attends past the causal edge fails.

    A size with page_size > 0 gets the keys and values in the engine's paged layout instead (paged_kv_cache):
    the same values, so a paged size and a contiguous size of one shape have the same expected output.
    """
    torch.manual_seed(seed)
    batch, heads_q, heads_kv = size["batch"], size["heads_q"], size["heads_kv"]
    q_len, kv_len, head_dim = size["q_len"], size["kv_len"], size["head_dim"]
    Q = (torch.randn(batch, heads_q, q_len, head_dim, device=device, dtype=torch.float32) * 4.0).to(dtype)
    K = torch.randn(batch, heads_kv, kv_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads_kv, kv_len, head_dim, device=device, dtype=dtype)
    if not size.get("page_size"):
        return {"Q": Q, "K": K, "V": V}
    return {"Q": Q, **paged_kv_cache(K, V, size["page_size"])}


PAGE_POISON = 100.0  # scale of the finite values in the reserved page and in unused slots


def paged_kv_cache(K: torch.Tensor, V: torch.Tensor, page_size: int) -> dict:
    """Contiguous keys and values [batch, heads_kv, kv_len, head_dim] scattered into an engine-style page pool.

    Returns K and V as pools [num_pages, page_size, heads_kv, head_dim] (vLLM's per-layer KV layout), an int32
    block_table [batch, pages + 1] and kv_len. Page 0 is reserved like vLLM's null block: only the table's
    spare last column points to it. The other pages are a random permutation, so a kernel that ignores the
    table reads the wrong keys. Page 0 and the unused tail slots of each sequence's last page hold large finite
    values: a kernel that attends to them (one that takes pages x page_size for kv_len, say) misses the
    tolerance by orders of magnitude, while one that loads them and gives them zero weight stays exact.
    """
    batch, heads_kv, kv_len, head_dim = K.shape
    if page_size < 1:
        raise ValueError("page_size must be at least 1")
    pages = -(-kv_len // page_size)
    shape = (1 + batch * pages, page_size, heads_kv, head_dim)
    k_pool = torch.randn(shape, device=K.device, dtype=K.dtype) * PAGE_POISON
    v_pool = torch.randn(shape, device=V.device, dtype=V.dtype) * PAGE_POISON
    table = torch.zeros(batch, pages + 1, dtype=torch.int32)
    table[:, :pages] = (torch.randperm(batch * pages) + 1).view(batch, pages)
    table = table.to(K.device)
    keys = torch.arange(kv_len, device=K.device)
    for b in range(batch):
        page, slot = table[b, keys // page_size].long(), keys % page_size
        k_pool[page, slot] = K[b].transpose(0, 1)
        v_pool[page, slot] = V[b].transpose(0, 1)
    return {"K": k_pool, "V": v_pool, "block_table": table, "kv_len": kv_len}


def gather_paged_kv(pool: torch.Tensor, block_table: torch.Tensor, kv_len: int) -> torch.Tensor:
    """A page pool [num_pages, page_size, heads_kv, head_dim] back to contiguous [batch, heads_kv, kv_len, head_dim]."""
    page_size = pool.shape[1]
    keys = torch.arange(kv_len, device=pool.device)
    pages = block_table[:, keys // page_size].long()  # [batch, kv_len]
    return pool[pages, keys % page_size].permute(0, 2, 1, 3).contiguous()


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

def _ref_gqa_prefill_attention(inputs: dict) -> torch.Tensor:
    import reference
    return reference.gqa_prefill_attention_ref(inputs["Q"], inputs["K"], inputs["V"],
                                               block_table=inputs.get("block_table"), kv_len=inputs.get("kv_len"))

def _contiguous_gqa_inputs(inputs: dict) -> dict:
    """Paged inputs as contiguous Q, K, V holding the same values (contiguous inputs are returned as they are).

    bench.py calls this once per size before timing the baseline, so the gather is never inside a timed call:
    a paged kernel is timed against SDPA on the same keys laid out contiguously.
    """
    if inputs.get("block_table") is None:
        return inputs
    table, kv_len = inputs["block_table"], inputs["kv_len"]
    return {"Q": inputs["Q"], "K": gather_paged_kv(inputs["K"], table, kv_len),
            "V": gather_paged_kv(inputs["V"], table, kv_len)}

def _sdpa_gqa_prefill_attention(inputs: dict) -> torch.Tensor:
    """Timing baseline: PyTorch SDPA with a bottom-right causal mask (the float32 oracle is far too slow to time)."""
    from torch.nn.attention.bias import causal_lower_right
    inputs = _contiguous_gqa_inputs(inputs)
    Q, K, V = inputs["Q"], inputs["K"], inputs["V"]
    mask = causal_lower_right(Q.shape[-2], K.shape[-2])
    try:
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, enable_gqa=True)
    except RuntimeError:
        # Some backends take grouped KV heads only when expanded to one KV head per query head.
        group = Q.shape[1] // K.shape[1]
        return F.scaled_dot_product_attention(Q, K.repeat_interleave(group, dim=1),
                                              V.repeat_interleave(group, dim=1), attn_mask=mask)

def _gqa_prefill_visible_pairs(s: dict) -> int:
    """(query, key) pairs the bottom-right causal mask keeps: query i sees kv_len - q_len + i + 1 keys."""
    return s["q_len"] * s["kv_len"] - s["q_len"] * (s["q_len"] - 1) // 2

def calibrated_attention_peak(calibration: dict, size: dict, dtype: torch.dtype) -> Optional[dict]:
    """The highest dense-attention TFLOP/s calibrate.py measured at this size's head_dim and dtype, or None.

    The highest one is used so a weak calibration shape never inflates the percentage. Dense attention is the
    ceiling for causal prefill counted over visible pairs only: a causal kernel skips masked tiles but cannot
    do the visible ones faster than dense attention does.
    """
    name, best = str(dtype).replace("torch.", ""), None
    for entry in calibration.get("attention") or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("shape"), dict):
            continue
        tflops = entry.get("best_tflops")
        if (entry.get("dtype") == name and entry["shape"].get("head_dim") == size["head_dim"]
                and type(tflops) in (int, float) and math.isfinite(tflops) and tflops > 0
                and (best is None or tflops > best["tflops"])):
            best = {"tflops": tflops, "backend": entry.get("best_backend"), "shape": entry["shape"]}
    return best

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
    # GQA PREFILL ATTENTION (one prefill chunk against a long KV cache)
    # -----------------------------------------------------------------
    # Q: [batch, heads_q, q_len, head_dim] at positions kv_len - q_len .. kv_len - 1; K, V: [batch, heads_kv,
    # kv_len, head_dim]. Bottom-right causal mask. The oracle is an exact float32 online softmax; the timing
    # baseline is PyTorch SDPA (baseline_fn), not the engine's own prefill kernel: measure against that in the
    # engine (engine/prefill_ttft.py) before claiming a serving win.
    # A size with page_size reads K and V from an engine-style page pool through a block table instead
    # (kernel_fn gets block_table and kv_len too); --page-size N gives every size the engine's page size.
    "gqa_prefill_attention": {
        "test_sizes": [
            ("tiny",          {"batch": 1, "heads_q": 8,  "heads_kv": 1, "q_len": 64,    "kv_len": 64,      "head_dim": 128}),
            ("small",         {"batch": 2, "heads_q": 8,  "heads_kv": 2, "q_len": 256,   "kv_len": 2048,    "head_dim": 128}),
            ("medium",        {"batch": 1, "heads_q": 16, "heads_kv": 2, "q_len": 2048,  "kv_len": 16384,   "head_dim": 256}),
            ("large",         {"batch": 1, "heads_q": 16, "heads_kv": 2, "q_len": 8192,  "kv_len": 131072,  "head_dim": 256}),
            ("chunk16k_256k", {"batch": 1, "heads_q": 16, "heads_kv": 2, "q_len": 16384, "kv_len": 262144,  "head_dim": 256}),
            ("chunk8k_1m",    {"batch": 1, "heads_q": 16, "heads_kv": 2, "q_len": 8192,  "kv_len": 1048576, "head_dim": 256}),
            ("chunk32k_1m",   {"batch": 1, "heads_q": 16, "heads_kv": 2, "q_len": 32768, "kv_len": 1048576, "head_dim": 256}),
            ("d128_8k_1m",    {"batch": 1, "heads_q": 32, "heads_kv": 4, "q_len": 8192,  "kv_len": 1048576, "head_dim": 128}),
            # Two page sizes, so a kernel cannot hard-code one
            ("paged_8k_128k", {"batch": 1, "heads_q": 16, "heads_kv": 2, "q_len": 8192,  "kv_len": 131072,  "head_dim": 256, "page_size": 16}),
            ("paged_32k_1m",  {"batch": 1, "heads_q": 16, "heads_kv": 2, "q_len": 32768, "kv_len": 1048576, "head_dim": 256, "page_size": 64}),
        ],
        "test_dtypes": [torch.bfloat16, torch.float16],
        "tolerances": {
            torch.float16:  {"atol": 1e-2, "rtol": 1e-2},
            torch.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
            torch.float32:  {"atol": 1e-4, "rtol": 1e-4},
        },
        # 2 FLOPs per multiply-add in Q@K^T and in P@V, over the visible (query, key) pairs only
        "flops_fn": lambda s: 4 * s["batch"] * s["heads_q"] * s["head_dim"] * _gqa_prefill_visible_pairs(s),
        "bytes_fn": lambda s, dt: (2 * s["batch"] * s["heads_q"] * s["q_len"]
                                   + 2 * s["batch"] * s["heads_kv"] * s["kv_len"]) * s["head_dim"] * _dtype_bytes(dt),
        "input_generator": gen_gqa_prefill_attention_inputs,
        "reference_fn": _ref_gqa_prefill_attention,
        "baseline_fn": _sdpa_gqa_prefill_attention,
        "baseline_inputs": _contiguous_gqa_inputs,
        "paged_kv": True,
        "calibrated_peak_fn": calibrated_attention_peak,
        "edge_sizes": [
            ("edge_q1",         {"batch": 1, "heads_q": 16, "heads_kv": 2, "q_len": 1,    "kv_len": 4097, "head_dim": 256}),
            ("edge_odd",        {"batch": 1, "heads_q": 16, "heads_kv": 2, "q_len": 1000, "kv_len": 5003, "head_dim": 256}),
            ("edge_full",       {"batch": 1, "heads_q": 16, "heads_kv": 2, "q_len": 1500, "kv_len": 1500, "head_dim": 256}),
            # A page size that is not a power of two, two block-table rows and a last page holding 11 of 48 keys
            ("edge_paged_tail", {"batch": 2, "heads_q": 16, "heads_kv": 2, "q_len": 1000, "kv_len": 5003, "head_dim": 256, "page_size": 48}),
            # One query; the last page holds a single key
            ("edge_paged_q1",   {"batch": 1, "heads_q": 16, "heads_kv": 2, "q_len": 1,    "kv_len": 4097, "head_dim": 256, "page_size": 16}),
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


def with_page_size(config: dict, page_size: int) -> dict:
    """The op's sizes with one paged-KV page size for all of them (0 = contiguous K/V).

    For matching a kernel to the engine's attention page size. Sizes that become identical run once, under
    their first label.
    """
    if not config.get("paged_kv"):
        raise ValueError("--page-size applies only to ops that read a paged KV cache (gqa_prefill_attention)")
    if page_size < 0:
        raise ValueError("--page-size must be 0 (contiguous K/V) or a positive number of tokens")

    def apply(sizes):
        seen, out = set(), []
        for label, size in sizes:
            size = {k: v for k, v in size.items() if k != "page_size"}
            if page_size:
                size["page_size"] = page_size
            key = tuple(sorted(size.items()))
            if key not in seen:
                seen.add(key)
                out.append((label, size))
        return out
    return {**config, "test_sizes": apply(config["test_sizes"]), "edge_sizes": apply(config.get("edge_sizes", []))}


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
                    sizes_filter: str = "all", calibration: Optional[dict] = None) -> dict:
    """Run performance benchmarks. Returns dict with metrics.

    calibration: calibrate.py's JSON from this GPU. An op with a calibrated_peak_fn then also reports
    pct_calibrated_peak, its throughput as a percentage of the measured peak for that op (attention for
    gqa_prefill_attention) instead of only the database's tensor-core peak.
    """
    device = "cuda"
    gen_fn = config["input_generator"]
    ref_fn = config["reference_fn"]
    flops_fn = config["flops_fn"]
    # An op may time a faster baseline than its correctness oracle (an exact float32 oracle can take seconds).
    base_fn = config.get("baseline_fn") or ref_fn
    # ...and give it its inputs in another layout, built once per size outside the timed calls.
    base_inputs_fn = config.get("baseline_inputs") or (lambda inputs: inputs)
    peak_fn = config.get("calibrated_peak_fn") if calibration else None
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
            base_inputs = base_inputs_fn(inputs)
            with _Timeout(30):
                ref_ms = _do_bench(lambda: base_fn(base_inputs), warmup=25, rep=100)

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
            peak = peak_fn(calibration, sz, dtype) if peak_fn else None

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
                "calibrated_peak": peak,
                "pct_calibrated_peak": 100.0 * throughput_tflops / peak["tflops"] if peak else None,
            }
            all_results.append(entry)

            if label == primary_label:
                primary_result = entry

            calibrated = (f" | {entry['pct_calibrated_peak']:.1f}% of calibrated peak" if peak
                          else " | calibrated peak: absent" if peak_fn else "")
            print(f"    kernel: {kernel_us:.2f} us | pytorch: {ref_us:.2f} us | "
                  f"speedup: {speedup:.3f}x | {throughput_tflops:.3f} TFLOPS | "
                  f"{pct_peak_compute:.1f}% peak{calibrated}")

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
    parser.add_argument("--page-size", type=int, default=None,
                        help="Paged-KV page size in tokens for every size of an op that reads a paged KV cache "
                             "(gqa_prefill_attention): use the engine's attention page size. 0 = contiguous K/V. "
                             "Default: each size's own")
    parser.add_argument("--calibration", default=None, metavar="JSON",
                        help="calibrate.py --out file from this GPU: also report pct_calibrated_peak, the "
                             "throughput as a percentage of the measured peak for the op (gqa_prefill_attention: "
                             "the best dense-attention TFLOP/s at its head_dim)")
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
    if args.page_size is not None:
        try:
            config = with_page_size(config, args.page_size)
        except ValueError as e:
            print(f"\nERROR: {e}")
            print(f"\ncorrectness: FAIL")
            print(f"throughput_tflops: 0.000")
            sys.exit(1)
        print(f"page_size: {args.page_size or 'contiguous'} (every size)")

    # ------------------------------------------------------------------
    # GPU Detection
    # ------------------------------------------------------------------
    gpu = detect_gpu()
    calibration = None
    if args.calibration:
        try:
            with open(args.calibration, encoding="utf-8-sig") as f:
                calibration = json.load(f)
            if not isinstance(calibration, dict):
                raise ValueError("not a JSON object")
        except (OSError, ValueError) as e:
            print(f"\nERROR: could not read the calibration file {args.calibration}: {type(e).__name__}")
            print(f"\ncorrectness: FAIL")
            print(f"throughput_tflops: 0.000")
            sys.exit(1)
        if calibration.get("gpu") != gpu.name:
            print(f"WARNING: {args.calibration} was measured on {calibration.get('gpu')!r}, this GPU is "
                  f"{gpu.name!r}: its peaks are not used")
            calibration = None
        elif not config.get("calibrated_peak_fn"):
            print(f"WARNING: {kernel_type} has no calibrated peak; --calibration is ignored")
            calibration = None

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
        perf_results = run_performance(kernel_fn, config, gpu, sizes_filter=sizes_filter, calibration=calibration)
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
        if calibration is not None:
            peak = primary["calibrated_peak"]
            if peak:
                s = peak["shape"]
                print(f"pct_calibrated_peak: {primary['pct_calibrated_peak']:.1f}%")
                print(f"calibrated_peak_tflops: {peak['tflops']:.1f} ({peak['backend']}, dense "
                      f"{s.get('q_len')}x{s.get('kv_len')} {s.get('heads_q')}/{s.get('heads_kv')} d{s.get('head_dim')})")
            else:
                print("pct_calibrated_peak: absent (no calibrated peak at this head_dim and dtype)")
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
        cal_col = calibration is not None
        print(f"{'size':<12} {'kernel_us':>12} {'pytorch_us':>12} {'speedup':>10} {'tflops':>10} {'%peak':>8}"
              + (f" {'%cal':>8}" if cal_col else ""))
        print("-" * (75 if cal_col else 66))
        for entry in all_perf:
            pct_cal = entry.get("pct_calibrated_peak")
            cal = (f" {pct_cal:>7.1f}%" if pct_cal is not None else f" {'absent':>8}") if cal_col else ""
            print(f"{entry['label']:<12} {entry['kernel_latency_us']:>12.2f} "
                  f"{entry['pytorch_latency_us']:>12.2f} {entry['speedup_vs_pytorch']:>9.3f}x "
                  f"{entry['throughput_tflops']:>10.3f} {entry['pct_peak_compute']:>7.1f}%{cal}")

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
        if primary.get("pct_calibrated_peak") is not None:
            print(f"pct_calibrated_peak: {primary['pct_calibrated_peak']:.1f}%")
    else:
        print(f"speedup_vs_pytorch: 0.000x")
        print(f"pct_peak_compute: 0.0%")
    print(f"bench_time_seconds: {t_elapsed:.1f}")

    if t_elapsed > 90:
        print(f"WARNING: bench.py took {t_elapsed:.1f}s (budget: 90s)")


if __name__ == "__main__":
    main()
