#!/usr/bin/env python3
"""
AutoKernel Model Profiler -- Profile any PyTorch model to identify bottleneck kernels.

Usage:
    uv run profile.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,2048 --dtype float16
    uv run profile.py --model models/gpt2.py --class-name GPT2 --input-shape 1,1024
    uv run profile.py --module transformers --class-name AutoModelForCausalLM --pretrained meta-llama/Llama-2-7b-hf --input-shape 1,2048

Output: profile_report.json in workspace/ directory
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import logging
import os
import pickle
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARMUP_ITERS = 5
PROFILE_ITERS = 10

WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")

# Kernel classification rules: list of (pattern_fragments, op_type)
# Checked in order; first match wins.
_KERNEL_CLASSIFICATION: List[Tuple[List[str], str]] = [
    (["flash", "fmha"],                       "flash_attention"),
    (["attention"],                            "flash_attention"),
    (["gemm", "matmul", "cublas"],             "matmul"),
    (["softmax"],                              "softmax"),
    (["layer_norm", "layernorm"],              "layernorm"),
    (["rms_norm", "rmsnorm"],                  "rmsnorm"),
    (["gelu", "silu", "mlp"],                  "fused_mlp"),
    (["cross_entropy", "nll"],                 "cross_entropy"),
    (["rotary", "rope"],                       "rotary_embedding"),
    (["reduce", "all_reduce"],                 "reduce"),
]

# Op types that have a matching kernels/*.py file in AutoKernel.
_SUPPORTED_OP_TYPES: set[str] = set()


def _discover_supported_op_types() -> set[str]:
    """Scan kernels/ directory for supported kernel types."""
    kernels_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
    supported = set()
    if os.path.isdir(kernels_dir):
        for fname in os.listdir(kernels_dir):
            if fname.endswith(".py") and fname != "__init__.py":
                supported.add(fname[:-3])  # e.g. "matmul", "softmax"
    return supported


# ---------------------------------------------------------------------------
# GPU detection -- import from bench.py if available, else standalone fallback
# ---------------------------------------------------------------------------

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


def _fallback_detect_gpu() -> GPUSpec:
    """Standalone GPU detection when bench.py is not importable."""
    if not torch.cuda.is_available():
        return GPUSpec()

    props = torch.cuda.get_device_properties(0)
    name = props.name
    sm_count = props.multi_processor_count
    memory_gb = round(props.total_memory / (1024 ** 3), 1)
    cc = (props.major, props.minor)

    # Known GPUs: name_fragment -> (peak_fp16_tflops, peak_bandwidth_gb_s, l2_cache_mb)
    _KNOWN_GPUS: Dict[str, Tuple[float, float, float]] = {
        "H100 SXM":  (989.5, 3352.0, 50.0),
        "H100 PCIe": (756.0, 2039.0, 50.0),
        "H100":      (756.0, 2039.0, 50.0),
        "A100-SXM":  (312.0, 2039.0, 40.0),
        "A100-PCIE": (312.0, 1935.0, 40.0),
        "A100":      (312.0, 2039.0, 40.0),
        "L40S":      (362.05, 864.0, 48.0),
        "L4":        (121.0, 300.0, 48.0),
        "A10":       (125.0, 600.0, 6.0),
        "4090":      (330.0, 1008.0, 72.0),
        "4080":      (305.0, 716.8, 64.0),
        "3090":      (142.0, 936.2, 6.0),
        "3080":      (119.5, 760.3, 5.0),
        # AMD Instinct GPUs
        "MI300X":    (1307.4, 5300.0, 256.0),
        "MI325X":    (1307.4, 6000.0, 256.0),
        "MI350X":    (2300.0, 8000.0, 256.0),
        "MI355X":    (2300.0, 8000.0, 256.0),
    }

    # On ROCm, device name may be empty; try gcnArchName-based lookup
    gcn_arch = getattr(props, 'gcnArchName', '')
    if gcn_arch and not name:
        _AMD_ARCHS = {
            "gfx942": ("AMD Instinct MI300X", 1307.4, 5300.0, 256.0),
            "gfx950": ("AMD Instinct MI350X", 2300.0, 8000.0, 256.0),
        }
        for arch_prefix, amd_specs in _AMD_ARCHS.items():
            if gcn_arch.startswith(arch_prefix):
                name, peak_fp16_val, peak_bw_val, l2_val = amd_specs
                break

    matched = None
    for fragment, specs in _KNOWN_GPUS.items():
        if fragment in name:
            matched = specs
            break

    if matched is not None:
        peak_fp16, peak_bw, l2 = matched
    else:
        if hasattr(props, 'clock_rate') and props.clock_rate > 0:
            ops_per_clock_per_sm = 256 if cc[0] >= 8 else 128
            clock_ghz = props.clock_rate / 1e6
            peak_fp16 = sm_count * ops_per_clock_per_sm * clock_ghz * 2 / 1e3
            peak_bw = max(props.clock_rate / 1e6 * 256 / 8 * 2, 500.0)
        else:
            # ROCm fallback: no clock_rate available
            peak_fp16 = 500.0
            peak_bw = 2000.0
        l2 = props.L2_cache_size / (1024 * 1024) if hasattr(props, "L2_cache_size") else 0.0

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


def detect_gpu() -> GPUSpec:
    """Try to import detect_gpu from bench.py; fall back to standalone."""
    try:
        bench_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")
        spec = importlib.util.spec_from_file_location("bench", bench_path)
        if spec and spec.loader:
            bench_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bench_mod)  # type: ignore[union-attr]
            gpu = bench_mod.detect_gpu()  # type: ignore[attr-defined]
            # Convert to our local GPUSpec (fields are identical)
            return GPUSpec(
                name=gpu.name,
                sm_count=gpu.sm_count,
                memory_gb=gpu.memory_gb,
                peak_tflops_fp16=gpu.peak_tflops_fp16,
                peak_tflops_bf16=gpu.peak_tflops_bf16,
                peak_tflops_fp32=gpu.peak_tflops_fp32,
                peak_bandwidth_gb_s=gpu.peak_bandwidth_gb_s,
                l2_cache_mb=gpu.l2_cache_mb,
                compute_capability=gpu.compute_capability,
            )
    except Exception:
        pass
    return _fallback_detect_gpu()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    dtype_str = dtype_str.lower().strip()
    if dtype_str not in mapping:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}'. Supported: {list(mapping.keys())}"
        )
    return mapping[dtype_str]


def _load_model_from_file(model_path: str, class_name: str) -> nn.Module:
    """Load a model class from a Python file and instantiate it."""
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    module_name = Path(model_path).stem
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    if not hasattr(module, class_name):
        available = [n for n in dir(module) if not n.startswith("_")]
        raise AttributeError(
            f"Class '{class_name}' not found in {model_path}. Available: {available}"
        )

    cls = getattr(module, class_name)
    try:
        model = cls()
    except TypeError as e:
        raise RuntimeError(
            f"Could not instantiate {class_name}() with no arguments: {e}. "
            "If the model requires config, provide a factory function or use --module."
        ) from e
    return model


def _load_model_from_module(
    module_name: str, class_name: str, pretrained: Optional[str] = None
) -> nn.Module:
    """Load a model from an installed Python module (e.g. transformers)."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module '{module_name}'. Is it installed? Error: {e}"
        ) from e

    if not hasattr(module, class_name):
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_name}'."
        )

    cls = getattr(module, class_name)

    if pretrained:
        # HuggingFace-style: cls.from_pretrained(...)
        if hasattr(cls, "from_pretrained"):
            try:
                model = cls.from_pretrained(pretrained, torch_dtype="auto")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load pretrained model '{pretrained}' via "
                    f"{class_name}.from_pretrained(): {e}"
                ) from e
        else:
            raise RuntimeError(
                f"{class_name} does not have a from_pretrained() method."
            )
    else:
        try:
            model = cls()
        except TypeError as e:
            raise RuntimeError(
                f"Could not instantiate {class_name}() with no arguments: {e}. "
                "Use --pretrained for HuggingFace models."
            ) from e

    return model


def load_model(args: argparse.Namespace) -> Tuple[nn.Module, str]:
    """Load model according to CLI args. Returns (model, description_string)."""
    if args.model:
        model = _load_model_from_file(args.model, args.class_name)
        desc = f"{args.class_name} from {args.model}"
    elif args.module:
        model = _load_model_from_module(args.module, args.class_name, args.pretrained)
        desc = f"{args.class_name} from {args.module}"
        if args.pretrained:
            desc += f" (pretrained: {args.pretrained})"
    else:
        raise ValueError("Must specify either --model <file> or --module <package>")

    return model, desc


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

def _is_language_model(model: nn.Module) -> bool:
    """Heuristic: does the model expect input_ids (integer tokens)?"""
    # Check common class names
    cls_name = type(model).__name__.lower()
    lm_indicators = [
        "causal", "lm", "gpt", "llama", "bert", "t5", "opt", "falcon",
        "mistral", "gemma", "phi", "qwen", "codegen", "bloom", "mpt",
        "seq2seq",
    ]
    if any(ind in cls_name for ind in lm_indicators):
        return True

    # Check if model has an embedding layer as first child
    for name, child in model.named_children():
        child_name = type(child).__name__.lower()
        if "embed" in name.lower() or "embedding" in child_name:
            return True
        break  # only check first child

    # Check forward signature for 'input_ids'
    try:
        sig = inspect.signature(model.forward)
        if "input_ids" in sig.parameters:
            return True
    except (ValueError, TypeError):
        pass

    return False


def generate_input(
    model: nn.Module,
    input_shape: List[int],
    dtype: torch.dtype,
    device: str,
) -> Dict[str, Any]:
    """Generate appropriate sample input for the model."""
    if _is_language_model(model):
        # Language model: generate integer token IDs
        batch = input_shape[0] if len(input_shape) >= 1 else 1
        seq_len = input_shape[1] if len(input_shape) >= 2 else 512
        input_ids = torch.randint(0, 32000, (batch, seq_len), device=device, dtype=torch.long)
        return {"input_ids": input_ids}
    else:
        # Generic model: generate float tensor of given shape
        x = torch.randn(*input_shape, device=device, dtype=dtype)
        return {"x": x}


def _try_forward(
    model: nn.Module,
    inputs: Dict[str, Any],
) -> bool:
    """Attempt a forward pass. Returns True on success."""
    try:
        if "input_ids" in inputs:
            model(input_ids=inputs["input_ids"])
        elif "x" in inputs:
            model(inputs["x"])
        else:
            model(**inputs)
        return True
    except Exception:
        return False


def _prepare_model_and_input(
    model: nn.Module,
    input_shape: List[int],
    dtype: torch.dtype,
    device: str,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Move model to device, generate input, validate forward pass.
    Handles OOM by trying smaller batch sizes."""
    model = model.to(device=device)
    # Cast model dtype if not float32
    if dtype in (torch.float16, torch.bfloat16):
        model = model.to(dtype=dtype)
    model.eval()

    original_batch = input_shape[0] if len(input_shape) >= 1 else 1

    for attempt_batch in [original_batch, max(1, original_batch // 2), 1]:
        current_shape = [attempt_batch] + input_shape[1:]
        inputs = generate_input(model, current_shape, dtype, device)

        try:
            with torch.no_grad():
                success = _try_forward(model, inputs)
            if success:
                if attempt_batch != original_batch:
                    print(
                        f"  NOTE: Reduced batch size from {original_batch} to "
                        f"{attempt_batch} to fit in GPU memory."
                    )
                return model, inputs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt_batch == 1:
                raise RuntimeError(
                    "Model does not fit in GPU memory even with batch_size=1. "
                    f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                )
            print(
                f"  OOM with batch_size={attempt_batch}, trying smaller..."
            )
            continue
        except Exception as e:
            # Forward pass failed for a non-OOM reason -- try passing x as positional arg
            if "x" in inputs:
                try:
                    with torch.no_grad():
                        model(inputs["x"])
                    if attempt_batch != original_batch:
                        print(
                            f"  NOTE: Reduced batch size from {original_batch} to "
                            f"{attempt_batch}."
                        )
                    return model, inputs
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue
                except Exception:
                    pass
            raise RuntimeError(
                f"Forward pass failed: {e}\n"
                "Check that --input-shape and --class-name are correct."
            ) from e

    raise RuntimeError("Could not run forward pass with any batch size.")


# ---------------------------------------------------------------------------
# Kernel classification
# ---------------------------------------------------------------------------

def classify_kernel(kernel_name: str) -> str:
    """Map a CUDA kernel name to an AutoKernel op type."""
    name_lower = kernel_name.lower()

    for fragments, op_type in _KERNEL_CLASSIFICATION:
        for frag in fragments:
            if frag in name_lower:
                return op_type

    # Check for standalone "mm" -- common in cuBLAS kernel names like
    # "void cutlass::...sgemm..." or names containing "_mm_" or ending in "mm".
    # Avoid false positives from words like "command", "summary", "commit".
    if "mm" in name_lower:
        if re.search(r"(?:^|[^a-z])mm(?:$|[^a-z])", name_lower):
            return "matmul"

    return "other"


def is_autokernel_supported(op_type: str) -> bool:
    """Check if this op type has a matching kernels/*.py implementation."""
    return op_type in _SUPPORTED_OP_TYPES


# ---------------------------------------------------------------------------
# Roofline estimation
# ---------------------------------------------------------------------------

def estimate_roofline_position(
    kernel_name: str,
    op_type: str,
    gpu_time_us: float,
    gpu: GPUSpec,
) -> str:
    """Rough heuristic: is this kernel compute-bound or memory-bound?"""
    compute_bound_ops = {"matmul", "flash_attention"}
    memory_bound_ops = {"softmax", "layernorm", "rmsnorm", "reduce", "rotary_embedding",
                        "fused_mlp", "cross_entropy"}

    if op_type in compute_bound_ops:
        return "compute-bound"
    elif op_type in memory_bound_ops:
        return "memory-bound"
    else:
        # For 'other', long-running kernels are more likely compute-bound
        if gpu_time_us > 100.0:
            return "likely compute-bound"
        else:
            return "likely memory-bound"


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

@dataclass
class KernelRecord:
    """Aggregated stats for one kernel (name + shape combo)."""
    name: str
    op_type: str
    gpu_time_us: float
    call_count: int
    input_shapes: str  # string representation of shapes
    roofline: str = ""
    supported: bool = False


def _run_forward(model: nn.Module, inputs: Dict[str, Any]) -> None:
    """Run a single forward pass with the correct calling convention."""
    if "input_ids" in inputs:
        model(input_ids=inputs["input_ids"])
    elif "x" in inputs:
        try:
            model(inputs["x"])
        except TypeError:
            model(**inputs)
    else:
        model(**inputs)


def profile_model(
    model: nn.Module,
    inputs: Dict[str, Any],
    warmup_iters: int = WARMUP_ITERS,
    profile_iters: int = PROFILE_ITERS,
    export_trace: bool = False,
    memory_snapshot: bool = False,
) -> Tuple[List[KernelRecord], Dict[str, Any]]:
    """Profile the model and return a list of KernelRecords sorted by GPU time desc.

    Returns:
        Tuple of (records, extras) where extras contains paths to exported
        artifacts and optional HTA analysis results.
    """
    extras: Dict[str, Any] = {}

    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    trace_path = os.path.join(WORKSPACE_DIR, "trace.json")
    snapshot_path = os.path.join(WORKSPACE_DIR, "memory_snapshot.pickle")

    # --- Warmup ---
    with torch.no_grad():
        for _ in range(warmup_iters):
            _run_forward(model, inputs)

    torch.cuda.synchronize()

    # --- Start memory recording if requested ---
    if memory_snapshot:
        try:
            torch.cuda.memory._record_memory_history(max_entries=100000)
        except Exception as e:
            print(f"  WARNING: Could not start memory history recording: {e}")
            memory_snapshot = False

    # --- Profile ---
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            for _ in range(profile_iters):
                _run_forward(model, inputs)
                torch.cuda.synchronize()

    # --- Export Chrome trace ---
    if export_trace:
        try:
            prof.export_chrome_trace(trace_path)
            extras["trace_path"] = trace_path
        except Exception as e:
            print(f"  WARNING: Could not export Chrome trace: {e}")

    # --- Capture memory snapshot ---
    if memory_snapshot:
        try:
            snapshot = torch.cuda.memory._snapshot()
            with open(snapshot_path, "wb") as snap_f:
                pickle.dump(snapshot, snap_f)
            extras["memory_snapshot_path"] = snapshot_path
        except Exception as e:
            print(f"  WARNING: Could not capture memory snapshot: {e}")
        finally:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass

    # --- Optional HTA analysis ---
    if export_trace and "trace_path" in extras:
        try:
            from HolisticTraceAnalysis import TraceAnalysis  # type: ignore[import-untyped]
            trace_dir = os.path.dirname(extras["trace_path"])
            analyzer = TraceAnalysis(trace_dir=trace_dir)
            temporal = analyzer.get_temporal_breakdown()
            kernel_breakdown = analyzer.get_gpu_kernel_breakdown()
            extras["hta_temporal_breakdown"] = str(temporal)
            extras["hta_kernel_breakdown"] = str(kernel_breakdown)
        except ImportError:
            pass  # HTA not installed, skip
        except Exception as e:
            print(f"  WARNING: HTA analysis failed: {e}")

    # --- Extract kernel averages ---
    key_averages = prof.key_averages(group_by_input_shape=True)

    records: List[KernelRecord] = []
    for evt in key_averages:
        # We only care about events that ran on CUDA
        cuda_time_us = getattr(evt, "self_device_time_total", None) or getattr(evt, "self_cuda_time_total", 0)
        if cuda_time_us <= 0:
            continue

        name = evt.key
        op_type = classify_kernel(name)

        # Build shape info string
        shape_str = ""
        if evt.input_shapes:
            try:
                shape_str = str(evt.input_shapes)
            except Exception:
                shape_str = ""

        records.append(KernelRecord(
            name=name,
            op_type=op_type,
            gpu_time_us=cuda_time_us,
            call_count=evt.count,
            input_shapes=shape_str,
        ))

    # Sort by GPU time descending
    records.sort(key=lambda r: r.gpu_time_us, reverse=True)

    return records, extras


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _priority_label(pct: float) -> str:
    if pct >= 10.0:
        return "HIGH"
    elif pct >= 3.0:
        return "MEDIUM"
    else:
        return "LOW"


def build_report(
    records: List[KernelRecord],
    gpu: GPUSpec,
    args: argparse.Namespace,
    model_desc: str,
) -> Dict[str, Any]:
    """Build the profile_report.json structure."""
    total_gpu_time_us = sum(r.gpu_time_us for r in records)
    total_gpu_time_ms = total_gpu_time_us / 1000.0

    # Annotate records with roofline + supported
    for r in records:
        r.roofline = estimate_roofline_position(r.name, r.op_type, r.gpu_time_us, gpu)
        r.supported = is_autokernel_supported(r.op_type)

    # Build top_kernels list
    top_kernels = []
    cumulative_pct = 0.0
    for i, r in enumerate(records):
        pct = (r.gpu_time_us / total_gpu_time_us * 100.0) if total_gpu_time_us > 0 else 0.0
        cumulative_pct += pct
        top_kernels.append({
            "rank": i + 1,
            "name": r.name,
            "op_type": r.op_type,
            "shape_info": r.input_shapes,
            "gpu_time_ms": round(r.gpu_time_us / 1000.0, 3),
            "call_count": r.call_count,
            "avg_time_us": round(r.gpu_time_us / max(r.call_count, 1), 2),
            "pct_total": round(pct, 1),
            "cumulative_pct": round(cumulative_pct, 1),
            "roofline": r.roofline,
            "autokernel_supported": r.supported,
            "optimization_priority": _priority_label(pct),
        })

    # Optimization summary
    supported_time_us = sum(r.gpu_time_us for r in records if r.supported)
    supported_pct = (supported_time_us / total_gpu_time_us * 100.0) if total_gpu_time_us > 0 else 0.0

    top5_time_us = sum(r.gpu_time_us for r in records[:5])
    top5_pct = (top5_time_us / total_gpu_time_us * 100.0) if total_gpu_time_us > 0 else 0.0

    # Estimated max speedup via Amdahl's law:
    # If supported kernels can be made ~3x faster on average:
    # S = 1 / ((1 - f) + f/s) where f = supported fraction, s = per-kernel speedup
    f = supported_pct / 100.0
    s = 3.0  # assume each supported kernel can be made 3x faster on average
    if f > 0:
        amdahl_speedup = 1.0 / ((1.0 - f) + f / s)
    else:
        amdahl_speedup = 1.0

    input_shape = [int(x) for x in args.input_shape.split(",")]

    report = {
        "model": args.model or args.module,
        "class_name": args.class_name,
        "input_shape": input_shape,
        "dtype": args.dtype,
        "gpu_name": gpu.name,
        "gpu_peak_tflops_fp16": gpu.peak_tflops_fp16,
        "gpu_peak_bandwidth_gb_s": gpu.peak_bandwidth_gb_s,
        "total_gpu_time_ms": round(total_gpu_time_ms, 3),
        "total_kernels": len(records),
        "profile_iters": PROFILE_ITERS,
        "top_kernels": top_kernels,
        "optimization_summary": {
            "supported_kernels_pct": round(supported_pct, 1),
            "top5_pct": round(top5_pct, 1),
            "estimated_max_speedup": (
                f"{amdahl_speedup:.1f}x "
                f"(Amdahl's law, f={f:.0%} supported, {s:.0f}x per-kernel)"
            ),
        },
    }

    return report


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def print_report(
    records: List[KernelRecord],
    report: Dict[str, Any],
    model_desc: str,
) -> None:
    """Pretty-print the profiling results to the terminal."""
    print()
    print("=" * 60)
    print("  AutoKernel Profiler")
    print("=" * 60)
    print(f"  Model: {model_desc}")
    print(f"  Input: shape={report['input_shape']}, dtype={report['dtype']}")
    print(f"  GPU:   {report['gpu_name']}")
    print(f"  Total GPU time: {report['total_gpu_time_ms']:.3f} ms "
          f"({report['total_kernels']} kernels, {PROFILE_ITERS} iterations)")
    print()

    # Kernel ranking table
    print("=" * 60)
    print("  KERNEL RANKING (by GPU time)")
    print("=" * 60)
    header = (
        f"{'Rank':>4} | {'Op Type':<20} | {'GPU Time (ms)':>13} | "
        f"{'Calls':>5} | {'Pct':>6} | {'Cumul':>6} | Supported"
    )
    print(header)
    print("-" * len(header))

    display_count = min(len(records), 20)
    for i in range(display_count):
        k = report["top_kernels"][i]
        sup_label = "YES" if k["autokernel_supported"] else "no"
        print(
            f"{k['rank']:>4} | {k['op_type']:<20} | "
            f"{k['gpu_time_ms']:>13.3f} | "
            f"{k['call_count']:>5} | "
            f"{k['pct_total']:>5.1f}% | "
            f"{k['cumulative_pct']:>5.1f}% | "
            f"{sup_label}"
        )

    remaining = len(records) - display_count
    if remaining > 0:
        remaining_pct = sum(
            report["top_kernels"][i]["pct_total"]
            for i in range(display_count, len(report["top_kernels"]))
        )
        print(f"  ... ({remaining} more kernels, {remaining_pct:.1f}% of total)")

    # Optimization summary
    summary = report["optimization_summary"]
    print()
    print("=" * 60)
    print("  OPTIMIZATION PLAN")
    print("=" * 60)

    supported_types = set()
    for r in records:
        if r.supported:
            supported_types.add(r.op_type)

    type_count = len(supported_types)
    type_word = "type" if type_count == 1 else "types"
    print(
        f"  AutoKernel can optimize {summary['supported_kernels_pct']:.1f}% of GPU time "
        f"({type_count} kernel {type_word})."
    )
    print(
        f"  Top-5 kernels account for {summary['top5_pct']:.1f}% of total GPU time."
    )
    print(f"  Estimated max speedup: {summary['estimated_max_speedup']}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AutoKernel Model Profiler -- identify GPU kernel bottlenecks "
            "in any PyTorch model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run profile.py --model models/llama_7b.py "
            "--class-name LlamaModel --input-shape 1,2048 --dtype float16\n"
            "  uv run profile.py --module transformers "
            "--class-name AutoModelForCausalLM "
            "--pretrained meta-llama/Llama-2-7b-hf --input-shape 1,2048\n"
            "  uv run profile.py --model my_net.py "
            "--class-name MyNet --input-shape 8,3,224,224 --dtype float32\n"
        ),
    )

    # Model source
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a Python file containing the model class.",
    )
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        help="Python module to import the model from (e.g. 'transformers').",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        required=True,
        help="Name of the model class to instantiate.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Pretrained model name/path for HuggingFace from_pretrained().",
    )

    # Input configuration
    parser.add_argument(
        "--input-shape",
        type=str,
        required=True,
        help="Comma-separated input shape, e.g. '1,2048' or '8,3,224,224'.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Data type: float16, bfloat16, float32 (default: float16).",
    )

    # Profiling configuration
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=WARMUP_ITERS,
        help=f"Number of warmup iterations (default: {WARMUP_ITERS}).",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=PROFILE_ITERS,
        help=f"Number of measured iterations (default: {PROFILE_ITERS}).",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: workspace/profile_report.json).",
    )

    # Advanced profiling exports
    parser.add_argument(
        "--export-trace",
        action="store_true",
        default=False,
        help="Export Chrome trace JSON for HTA/trace-blame analysis (saved to workspace/trace.json).",
    )
    parser.add_argument(
        "--memory-snapshot",
        action="store_true",
        default=False,
        help="Capture CUDA memory snapshot for mosaic analysis (saved to workspace/memory_snapshot.pickle).",
    )
    parser.add_argument(
        "--torch-compile-log",
        action="store_true",
        default=False,
        help="Save torch.compile logs for tlparse analysis (saved to workspace/compile_log.txt).",
    )

    args = parser.parse_args()

    if not args.model and not args.module:
        parser.error("Must specify either --model <file.py> or --module <package>.")

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    global WARMUP_ITERS, PROFILE_ITERS

    args = parse_args()
    WARMUP_ITERS = args.warmup_iters
    PROFILE_ITERS = args.profile_iters

    # Discover supported kernel types from kernels/ directory
    global _SUPPORTED_OP_TYPES
    _SUPPORTED_OP_TYPES = _discover_supported_op_types()

    # Parse input shape
    try:
        input_shape = [int(x.strip()) for x in args.input_shape.split(",")]
    except ValueError:
        print(
            f"ERROR: Invalid --input-shape '{args.input_shape}'. "
            "Expected comma-separated integers."
        )
        return 1

    # Resolve dtype
    try:
        dtype = _resolve_dtype(args.dtype)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. The profiler requires a GPU.")
        return 1

    device = "cuda"

    # Detect GPU
    gpu = detect_gpu()

    print()
    print("=== AutoKernel Profiler ===")

    # Load model
    print("Loading model...")
    try:
        model, model_desc = load_model(args)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        traceback.print_exc()
        return 1

    print(f"  Model: {model_desc}")
    print(f"  Input: shape={input_shape}, dtype={args.dtype}")
    print(f"  GPU:   {gpu.name}")
    print()

    # Prepare model and input
    print("Preparing model and input...")
    try:
        model, inputs = _prepare_model_and_input(model, input_shape, dtype, device)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("ERROR: GPU out of memory. Try a smaller --input-shape or batch size.")
        return 1

    # Enable torch.compile logging if requested
    compile_log_path = os.path.join(WORKSPACE_DIR, "compile_log.txt")
    compile_log_handler: Optional[logging.FileHandler] = None
    if args.torch_compile_log:
        os.makedirs(WORKSPACE_DIR, exist_ok=True)
        try:
            torch._logging.set_logs(dynamo=logging.DEBUG, inductor=logging.DEBUG)  # type: ignore[attr-defined]
            compile_log_handler = logging.FileHandler(compile_log_path, mode="w")
            compile_log_handler.setLevel(logging.DEBUG)
            for logger_name in ["torch._dynamo", "torch._inductor"]:
                lg = logging.getLogger(logger_name)
                lg.addHandler(compile_log_handler)
                lg.setLevel(logging.DEBUG)
            print("  torch.compile logging enabled.")
        except Exception as e:
            print(f"  WARNING: Could not enable torch.compile logging: {e}")
            compile_log_handler = None

    # Profile
    print(
        f"Profiling... ({WARMUP_ITERS} warmup + "
        f"{PROFILE_ITERS} measured iterations)"
    )
    try:
        records, extras = profile_model(
            model, inputs, WARMUP_ITERS, PROFILE_ITERS,
            export_trace=args.export_trace,
            memory_snapshot=args.memory_snapshot,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(
            "ERROR: GPU out of memory during profiling. "
            "Try a smaller --input-shape."
        )
        return 1
    except Exception as e:
        print(f"ERROR during profiling: {e}")
        traceback.print_exc()
        return 1

    if not records:
        print(
            "WARNING: No CUDA kernels were captured. "
            "The model may not use GPU operations."
        )
        print("Check that the model runs on GPU and the input shape is correct.")
        return 1

    # Finalize torch.compile log
    if compile_log_handler is not None:
        for logger_name in ["torch._dynamo", "torch._inductor"]:
            lg = logging.getLogger(logger_name)
            lg.removeHandler(compile_log_handler)
        compile_log_handler.close()
        if os.path.isfile(compile_log_path) and os.path.getsize(compile_log_path) > 0:
            extras["compile_log_path"] = compile_log_path
        else:
            print("  NOTE: torch.compile log is empty (no torch.compile calls detected).")

    # Build report
    report = build_report(records, gpu, args, model_desc)

    # Add HTA results to report if available
    if "hta_temporal_breakdown" in extras:
        report["hta_temporal_breakdown"] = extras["hta_temporal_breakdown"]
    if "hta_kernel_breakdown" in extras:
        report["hta_kernel_breakdown"] = extras["hta_kernel_breakdown"]

    # Print to terminal
    print_report(records, report, model_desc)

    # Save JSON
    output_path = args.output
    if output_path is None:
        os.makedirs(WORKSPACE_DIR, exist_ok=True)
        output_path = os.path.join(WORKSPACE_DIR, "profile_report.json")
    else:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Profile saved to {output_path}")

    # Print exported artifacts and next-step suggestions
    if extras:
        print()
        print("=" * 60)
        print("  EXPORTED ARTIFACTS")
        print("=" * 60)
        if "trace_path" in extras:
            print(f"  Chrome trace:     {extras['trace_path']}")
        if "memory_snapshot_path" in extras:
            print(f"  Memory snapshot:  {extras['memory_snapshot_path']}")
        if "compile_log_path" in extras:
            print(f"  Compile log:      {extras['compile_log_path']}")
        if "hta_temporal_breakdown" in extras:
            print("  HTA analysis:     included in profile_report.json")

        print()
        print("  Suggested next steps:")
        if "trace_path" in extras:
            print(f"    - View trace in Chrome:  chrome://tracing  (load {extras['trace_path']})")
            print("    - Run HTA analysis:      pip install HolisticTraceAnalysis && analyze trace")
            print("    - Run trace-blame:       trace-blame " + extras["trace_path"])
        if "memory_snapshot_path" in extras:
            print(f"    - Analyze with mosaic:   python -m torch.cuda.memory._snapshot {extras['memory_snapshot_path']}")
        if "compile_log_path" in extras:
            print(f"    - Parse with tlparse:    tlparse {extras['compile_log_path']}")

    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
