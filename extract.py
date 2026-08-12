#!/usr/bin/env python3
"""
AutoKernel Kernel Extractor -- Generate baseline kernels from profiling results.

Usage:
    uv run extract.py                          # extract from workspace/profile_report.json
    uv run extract.py --top 5                  # extract only top-5 kernels
    uv run extract.py --kernel-type matmul     # extract only matmul kernels
    uv run extract.py --report path/to/report.json
    uv run extract.py --backend cuda           # use CUDA C++ starter kernels instead of Triton
    uv run extract.py --dtype bfloat16         # override the profiled dtype
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(SCRIPT_DIR, "workspace")
KERNELS_DIR = os.path.join(SCRIPT_DIR, "kernels")
DEFAULT_REPORT_PATH = os.path.join(WORKSPACE_DIR, "profile_report.json")
OPTIMIZATION_PLAN_PATH = os.path.join(WORKSPACE_DIR, "optimization_plan.json")


# ---------------------------------------------------------------------------
# Shape key mappings per kernel type
# ---------------------------------------------------------------------------
# Each entry maps op_type -> list of (shape_key_aliases...) so we can parse
# various shape_info string formats from profile_report.json.

SHAPE_KEYS: Dict[str, List[str]] = {
    "matmul":            ["M", "N", "K"],
    "flash_attention":   ["B", "H", "N", "D"],
    "layernorm":         ["M", "N"],
    "softmax":           ["M", "N"],
    "cross_entropy":     ["batch", "vocab"],
    "fused_mlp":         ["M", "N", "K"],
    "rmsnorm":           ["M", "N"],
    "reduce":            ["M", "N"],
    "rotary_embedding":  ["B", "H", "N", "D"],
}

# Aliases: profile_report.json may use different key names than bench.py
# Map from alias -> canonical bench.py key, per op_type.
SHAPE_ALIAS_MAP: Dict[str, Dict[str, str]] = {
    "matmul": {},
    "flash_attention": {
        "B": "batch", "H": "heads", "N": "seq_len", "S": "seq_len", "D": "head_dim",
        "batch": "batch", "heads": "heads", "seq_len": "seq_len", "head_dim": "head_dim",
    },
    "layernorm": {
        "M": "batch", "N": "dim", "rows": "batch", "cols": "dim",
        "batch": "batch", "dim": "dim",
    },
    "softmax": {
        "M": "rows", "N": "cols", "rows": "rows", "cols": "cols",
    },
    "cross_entropy": {
        "batch": "batch", "vocab": "vocab",
    },
    "fused_mlp": {
        "M": "batch", "N": "hidden", "K": "dim",
        "batch": "batch", "dim": "dim", "hidden": "hidden",
    },
    "rmsnorm": {
        "M": "M", "N": "N",
    },
    "reduce": {
        "M": "M", "N": "N",
    },
    "rotary_embedding": {
        "B": "batch", "H": "heads", "N": "seq_len", "S": "seq_len", "D": "head_dim",
        "batch": "batch", "heads": "heads", "seq_len": "seq_len", "head_dim": "head_dim",
    },
}

# Default tolerances per op_type (matching bench.py structure, serialized for template)
TOLERANCES_MAP: Dict[str, Dict[str, Dict[str, float]]] = {
    "matmul": {
        "float16":  {"atol": 1e-2, "rtol": 1e-2},
        "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
        "float32":  {"atol": 1e-4, "rtol": 1e-4},
    },
    "flash_attention": {
        "float16":  {"atol": 1e-2, "rtol": 1e-2},
        "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
        "float32":  {"atol": 1e-4, "rtol": 1e-4},
    },
    "layernorm": {
        "float16":  {"atol": 1e-3, "rtol": 1e-3},
        "bfloat16": {"atol": 2e-3, "rtol": 2e-3},
        "float32":  {"atol": 1e-5, "rtol": 1e-5},
    },
    "softmax": {
        "float16":  {"atol": 1e-3, "rtol": 1e-3},
        "bfloat16": {"atol": 2e-3, "rtol": 2e-3},
        "float32":  {"atol": 1e-5, "rtol": 1e-5},
    },
    "cross_entropy": {
        "float16":  {"atol": 1e-2, "rtol": 1e-2},
        "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
        "float32":  {"atol": 1e-5, "rtol": 1e-5},
    },
    "fused_mlp": {
        "float16":  {"atol": 1e-2, "rtol": 1e-2},
        "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
        "float32":  {"atol": 1e-4, "rtol": 1e-4},
    },
    "rmsnorm": {
        "float16":  {"atol": 1e-2, "rtol": 1e-2},
        "bfloat16": {"atol": 1e-1, "rtol": 5e-2},
    },
    "reduce": {
        "float16":  {"atol": 1e-2, "rtol": 1e-2},
        "bfloat16": {"atol": 1e-1, "rtol": 5e-2},
    },
    "rotary_embedding": {
        "float16":  {"atol": 1e-3, "rtol": 1e-3},
        "bfloat16": {"atol": 2e-3, "rtol": 2e-3},
        "float32":  {"atol": 1e-5, "rtol": 1e-5},
    },
}

# Dtypes swept per op_type, mirroring bench.py's built-in test_dtypes.
# The profiled model's dtype is moved to the front of this list, since bench.py
# measures performance with test_dtypes[0].
TEST_DTYPES_MAP: Dict[str, List[str]] = {
    "matmul":           ["float16", "bfloat16", "float32"],
    "softmax":          ["float16", "bfloat16", "float32"],
    "layernorm":        ["float16", "bfloat16", "float32"],
    "flash_attention":  ["float16", "bfloat16"],
    "fused_mlp":        ["float16", "bfloat16", "float32"],
    "cross_entropy":    ["float16", "bfloat16", "float32"],
    "rotary_embedding": ["float16", "bfloat16", "float32"],
    "rmsnorm":          ["float16", "bfloat16"],
    "reduce":           ["float16", "bfloat16"],
}

# Canonical dtype names, so --dtype bf16 and --dtype bfloat16 agree.
DTYPE_ALIASES: Dict[str, str] = {
    "float16": "float16", "fp16": "float16", "half": "float16",
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float32": "float32", "fp32": "float32", "float": "float32",
}

# Fallback tolerances when a model's dtype has no entry in TOLERANCES_MAP.
DEFAULT_TOLERANCE_BY_DTYPE: Dict[str, Dict[str, float]] = {
    "float16":  {"atol": 1e-2, "rtol": 1e-2},
    "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
    "float32":  {"atol": 1e-4, "rtol": 1e-4},
}

# FLOPS formulas as source strings, per op_type
FLOPS_FN_SRC: Dict[str, str] = {
    "matmul":           'return 2 * s["M"] * s["N"] * s["K"]',
    "flash_attention":  'return 4 * s["batch"] * s["heads"] * (s["seq_len"] ** 2) * s["head_dim"]',
    "layernorm":        'return 8 * s["batch"] * s["dim"]',
    "softmax":          'return 5 * s["rows"] * s["cols"]',
    "cross_entropy":    'return 4 * s["batch"] * s["vocab"]',
    "fused_mlp":        'return 2 * s["batch"] * s["dim"] * s["hidden"] * 3',
    "rmsnorm":          'return 6 * s["M"] * s["N"]',
    "reduce":           'return s["M"] * s["N"]',
    "rotary_embedding": 'return 6 * s["batch"] * s["heads"] * s["seq_len"] * s["head_dim"]',
}

# BYTES formulas as source strings, per op_type (dt_bytes is passed in)
BYTES_FN_SRC: Dict[str, str] = {
    "matmul":           'return (s["M"] * s["K"] + s["K"] * s["N"] + s["M"] * s["N"]) * dt_bytes',
    "flash_attention":  'return 4 * s["batch"] * s["heads"] * s["seq_len"] * s["head_dim"] * dt_bytes',
    "layernorm":        'return (2 * s["batch"] * s["dim"] + 2 * s["dim"]) * dt_bytes',
    "softmax":          'return 2 * s["rows"] * s["cols"] * dt_bytes',
    "cross_entropy":    'return (s["batch"] * s["vocab"] + s["batch"]) * dt_bytes',
    "fused_mlp":        'return (s["batch"] * s["dim"] + s["hidden"] * s["dim"] * 3 + s["batch"] * s["dim"]) * dt_bytes',
    "rmsnorm":          'return (2 * s["M"] * s["N"] + s["N"]) * dt_bytes',
    "reduce":           'return (s["M"] * s["N"] + s["M"]) * dt_bytes',
    "rotary_embedding": 'return (s["batch"] * s["heads"] * s["seq_len"] * s["head_dim"] * 2 + s["seq_len"] * s["head_dim"]) * dt_bytes',
}

# Speedup potential heuristic per op_type
SPEEDUP_ESTIMATES: Dict[str, str] = {
    "matmul":           "2-3x",
    "flash_attention":  "2-4x",
    "layernorm":        "1.5-3x",
    "softmax":          "1.5-3x",
    "cross_entropy":    "1.5-2x",
    "fused_mlp":        "2-3x",
    "rmsnorm":          "1.5-3x",
    "reduce":           "1.5-2x",
    "rotary_embedding": "1.5-2x",
}


# ---------------------------------------------------------------------------
# Shape parsing
# ---------------------------------------------------------------------------

def parse_shape_info(shape_info_str: str, op_type: str) -> Optional[Dict[str, int]]:
    """
    Parse a shape_info string like "M=4096, N=4096, K=4096" into a dict.

    Handles various formats:
      - "M=4096, N=4096, K=4096"
      - "B=1, H=32, N=4096, D=128"
      - "batch=4096, vocab=32000"
      - "rows=4096, cols=4096"

    Returns None if parsing fails.
    """
    if not shape_info_str or not isinstance(shape_info_str, str):
        return None

    # Match key=value pairs
    pairs = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\d+)", shape_info_str)
    if not pairs:
        return None

    raw = {k: int(v) for k, v in pairs}

    # Map to canonical bench.py keys using alias map
    alias_map = SHAPE_ALIAS_MAP.get(op_type, {})
    if alias_map:
        canonical = {}
        for k, v in raw.items():
            mapped_key = alias_map.get(k, k)
            canonical[mapped_key] = v
        return canonical
    else:
        return raw


def shape_to_display(shape: Dict[str, int]) -> str:
    """Convert a shape dict to a display string like 'M=4096, N=4096, K=4096'."""
    return ", ".join(f"{k}={v}" for k, v in shape.items())


def scale_shape(shape: Dict[str, int], factor: float) -> Dict[str, int]:
    """
    Scale all shape dimensions by a factor, rounding to nearest integer.
    Ensures all values are at least 1.
    """
    return {k: max(1, int(round(v * factor))) for k, v in shape.items()}


def get_default_shape(op_type: str) -> Dict[str, int]:
    """
    Return a reasonable default shape for a given op_type when parsing fails.
    Based on the 'large' size from bench.py KERNEL_CONFIGS.
    """
    defaults: Dict[str, Dict[str, int]] = {
        "matmul":           {"M": 2048, "N": 2048, "K": 2048},
        "flash_attention":  {"batch": 2, "heads": 32, "seq_len": 1024, "head_dim": 64},
        "layernorm":        {"batch": 4096, "dim": 2048},
        "softmax":          {"rows": 4096, "cols": 4096},
        "cross_entropy":    {"batch": 4096, "vocab": 32000},
        "fused_mlp":        {"batch": 2048, "dim": 2048, "hidden": 5504},
        "rmsnorm":          {"M": 4096, "N": 4096},
        "reduce":           {"M": 4096, "N": 4096},
        "rotary_embedding": {"batch": 2, "heads": 32, "seq_len": 1024, "head_dim": 128},
    }
    return defaults.get(op_type, {"M": 2048, "N": 2048})


# ---------------------------------------------------------------------------
# Kernel file generation
# ---------------------------------------------------------------------------

def normalize_dtype(dtype_str: Optional[str]) -> str:
    """Canonicalize a dtype name from the profile report. Defaults to float16."""
    if not dtype_str:
        return "float16"
    return DTYPE_ALIASES.get(str(dtype_str).lower().strip(), "float16")


def order_test_dtypes(op_type: str, model_dtype: str) -> List[str]:
    """Return the dtype sweep for an op with the model's dtype first.

    bench.py benchmarks with test_dtypes[0], so a bf16 model must not be
    measured in fp16.
    """
    candidates = list(TEST_DTYPES_MAP.get(op_type, ["float16", "bfloat16"]))
    return [model_dtype] + [d for d in candidates if d != model_dtype]


def read_starter_kernel(op_type: str, backend: str = "triton") -> Optional[str]:
    """Read the starter kernel file. Returns None if not found.

    For backend='triton': reads from kernels/{op_type}.py
    For backend='cuda':   reads from kernels/cuda/{op_type}.py
    """
    if backend == "cuda":
        path = os.path.join(KERNELS_DIR, "cuda", f"{op_type}.py")
    else:
        path = os.path.join(KERNELS_DIR, f"{op_type}.py")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_kernel_body(starter_code: str) -> str:
    """
    Extract the Triton kernel code from a starter file, stripping the
    original module docstring and KERNEL_TYPE declaration (which we replace
    in the template header).

    Returns everything from the first 'import' statement onward.
    """
    lines = starter_code.split("\n")

    # Find the first import line
    import_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_idx = i
            break

    if import_idx is not None:
        return "\n".join(lines[import_idx:])
    else:
        # Fallback: return everything after KERNEL_TYPE line
        for i, line in enumerate(lines):
            if line.strip().startswith("KERNEL_TYPE"):
                return "\n".join(lines[i + 1:])
        return starter_code


def generate_kernel_file(
    op_type: str,
    rank: int,
    pct_total: float,
    model_shape: Dict[str, int],
    model_name: str,
    gpu_time_ms: float,
    starter_code: str,
    backend: str = "triton",
    model_dtype: str = "float16",
) -> str:
    """Generate the complete kernel file content for extraction."""

    half_shape = scale_shape(model_shape, 0.5)
    double_shape = scale_shape(model_shape, 2.0)

    shape_display = shape_to_display(model_shape)
    half_display = shape_to_display(half_shape)
    double_display = shape_to_display(double_shape)

    test_dtypes = order_test_dtypes(op_type, model_dtype)

    tolerances = dict(TOLERANCES_MAP.get(op_type, DEFAULT_TOLERANCE_BY_DTYPE))
    # Every dtype in the sweep needs a tolerance entry.
    for dt in test_dtypes:
        if dt not in tolerances:
            tolerances[dt] = DEFAULT_TOLERANCE_BY_DTYPE.get(
                dt, {"atol": 1e-2, "rtol": 1e-2}
            )

    flops_fn_body = FLOPS_FN_SRC.get(op_type, 'return 0')
    bytes_fn_body = BYTES_FN_SRC.get(op_type, 'return 0')

    # Extract the kernel code body (imports + jit functions + kernel_fn)
    kernel_body = extract_kernel_body(starter_code)

    # Build the file
    lines = []

    # Header docstring
    lines.append('"""')
    lines.append(f"AutoKernel -- Extracted kernel from model profiling.")
    lines.append(f"Op type: {op_type}")
    lines.append(f"Rank: {rank} ({pct_total}% of GPU time)")
    lines.append(f"Model shape: {shape_display}")
    lines.append(f"Model dtype: {model_dtype}")
    lines.append(f"")
    lines.append(f"This kernel was extracted from profiling {model_name}.")
    lines.append(f"The agent optimizes this to maximize throughput at the model-specific shapes.")
    lines.append('"""')
    lines.append("")

    # KERNEL_TYPE and BACKEND
    lines.append(f'KERNEL_TYPE = "{op_type}"')
    if backend == "cuda":
        lines.append(f'BACKEND = "cuda"')
    lines.append("")

    # Model-specific shapes
    lines.append("# Model-specific shapes (the shapes that matter for THIS model)")
    lines.append(f"MODEL_SHAPES = {repr(model_shape)}")
    lines.append("")

    # Benchmark config
    lines.append("# Benchmark config -- bench.py loads these and overlays them onto its")
    lines.append("# built-in config for this kernel type, so the model's own shapes and")
    lines.append("# dtype are what get benchmarked.")
    lines.append("TEST_SIZES = [")
    lines.append(f'    ("model_primary", {repr(model_shape)}),')
    lines.append(f"    # Also test nearby sizes for robustness")
    lines.append(f'    ("model_half", {repr(half_shape)}),')
    lines.append(f'    ("model_double", {repr(double_shape)}),')
    lines.append("]")
    lines.append("")

    # Dtypes (model dtype first -- bench.py measures with TEST_DTYPES[0])
    lines.append("# Model dtype first: bench.py measures performance with TEST_DTYPES[0].")
    lines.append(f"TEST_DTYPES = {repr(test_dtypes)}")
    lines.append("")

    # Tolerances
    lines.append(f"TOLERANCES = {repr(tolerances)}")
    lines.append("")

    # FLOPS function
    lines.append("")
    lines.append("def FLOPS_FN(s):")
    lines.append(f"    {flops_fn_body}")
    lines.append("")

    # BYTES function
    lines.append("")
    lines.append("def BYTES_FN(s, dt_bytes):")
    lines.append(f"    {bytes_fn_body}")
    lines.append("")

    # Separator
    lines.append("")
    lines.append(f"# {'=' * 70}")
    backend_label = "CUDA C++" if backend == "cuda" else "Triton"
    backend_dir = f"kernels/cuda/{op_type}.py" if backend == "cuda" else f"kernels/{op_type}.py"
    lines.append(f"# {backend_label} kernel code (from {backend_dir})")
    lines.append(f"# {'=' * 70}")
    lines.append("")

    # Kernel body
    lines.append(kernel_body)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profile reading and validation
# ---------------------------------------------------------------------------

def load_profile_report(path: str) -> Optional[Dict[str, Any]]:
    """Load and validate the profile report JSON. Returns None on failure."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"ERROR: Failed to read profile report: {e}")
        return None


def get_supported_kernels(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract the list of supported (autokernel_supported=True) kernels from
    the profile report, sorted by rank.
    """
    kernels = report.get("top_kernels", report.get("kernels", report.get("bottleneck_kernels", [])))
    supported = []
    for k in kernels:
        if k.get("autokernel_supported", False):
            supported.append(k)

    # Sort by rank if available, otherwise by gpu_time_ms descending
    supported.sort(key=lambda x: x.get("rank", x.get("gpu_time_ms", 0)))
    # Ensure rank ordering (lower rank = higher priority)
    for i, k in enumerate(supported):
        if "rank" not in k:
            k["rank"] = i + 1

    return supported


# ---------------------------------------------------------------------------
# Optimization plan generation
# ---------------------------------------------------------------------------

def generate_optimization_plan(
    extracted: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the optimization_plan.json data structure."""
    kernels_to_optimize = []
    total_pct = 0.0

    for entry in extracted:
        total_pct += entry["pct_total"]
        kernels_to_optimize.append({
            "rank": entry["rank"],
            "file": entry["output_file"],
            "op_type": entry["op_type"],
            "model_shape": entry["model_shape"],
            "gpu_time_ms": entry["gpu_time_ms"],
            "pct_total": entry["pct_total"],
            "estimated_speedup_potential": SPEEDUP_ESTIMATES.get(
                entry["op_type"], "1.5-2x"
            ),
        })

    return {
        "kernels_to_optimize": kernels_to_optimize,
        "total_optimization_targets": len(kernels_to_optimize),
        "covered_gpu_time_pct": round(total_pct, 1),
    }


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------

def extract_kernels(
    report_path: str,
    top_n: Optional[int] = None,
    kernel_type_filter: Optional[str] = None,
    backend: str = "triton",
    dtype_override: Optional[str] = None,
) -> None:
    """Main extraction pipeline."""

    backend_label = "CUDA C++" if backend == "cuda" else "Triton"
    print(f"=== AutoKernel Kernel Extractor ({backend_label}) ===")
    print()

    # -- Load profile report --
    print(f"Reading profile from {report_path}...")
    report = load_profile_report(report_path)
    if report is None:
        print(f"ERROR: Profile report not found at {report_path}")
        print(f"       Run the profiler first: uv run profile.py")
        sys.exit(1)

    # -- Get model name --
    model_name = report.get("model_name", report.get("model", "unknown model"))

    # -- Get model dtype (bench.py measures with TEST_DTYPES[0]) --
    model_dtype = normalize_dtype(dtype_override or report.get("dtype"))
    print(f"Model dtype: {model_dtype}")

    # -- Get supported kernels --
    supported = get_supported_kernels(report)
    if not supported:
        print("ERROR: No supported kernels found in profile report.")
        print("       Ensure the profiler marks kernels with autokernel_supported=True.")
        sys.exit(1)

    # -- Apply filters --
    if kernel_type_filter:
        supported = [k for k in supported if k.get("op_type") == kernel_type_filter]
        if not supported:
            print(f"WARNING: No kernels of type '{kernel_type_filter}' found in profile report.")
            sys.exit(1)

    if top_n is not None:
        supported = supported[:top_n]

    print(f"Found {len(supported)} supported kernels to extract.")
    print()

    # -- Ensure workspace directory exists --
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    # -- Extract each kernel --
    print("Extracting kernels:")
    extracted = []
    skipped = 0

    for idx, kernel_info in enumerate(supported):
        rank = kernel_info.get("rank", idx + 1)
        op_type = kernel_info.get("op_type", "unknown")
        pct_total = kernel_info.get("pct_total", kernel_info.get("pct_gpu_time", 0.0))
        gpu_time_ms = kernel_info.get("gpu_time_ms", kernel_info.get("total_gpu_time_ms", 0.0))
        shape_info_str = kernel_info.get("shape_info", kernel_info.get("shape", ""))

        # Parse model shape
        model_shape = parse_shape_info(shape_info_str, op_type)
        if model_shape is None:
            # Try to use a "shapes" dict directly if provided
            if isinstance(kernel_info.get("shapes"), dict):
                model_shape = kernel_info["shapes"]
            else:
                print(f"  WARNING: Could not parse shape for {op_type} (rank {rank}), "
                      f"using default shapes.")
                model_shape = get_default_shape(op_type)

        # Read starter kernel
        starter_code = read_starter_kernel(op_type, backend=backend)
        if starter_code is None:
            starter_dir = "kernels/cuda" if backend == "cuda" else "kernels"
            print(f"  WARNING: No starter kernel found at {starter_dir}/{op_type}.py -- skipping.")
            skipped += 1
            continue

        # Generate output filename
        output_filename = f"kernel_{op_type}_{rank}.py"
        output_path = os.path.join(WORKSPACE_DIR, output_filename)
        # Relative path for display and plan
        output_relpath = f"workspace/{output_filename}"

        # Generate the customized kernel file
        kernel_content = generate_kernel_file(
            op_type=op_type,
            rank=rank,
            pct_total=pct_total,
            model_shape=model_shape,
            model_name=model_name,
            gpu_time_ms=gpu_time_ms,
            starter_code=starter_code,
            backend=backend,
            model_dtype=model_dtype,
        )

        # Write to workspace
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(kernel_content)

        # Print progress
        position = idx + 1
        total = len(supported)
        shape_display = shape_to_display(model_shape)
        print(f"  [{position}/{total}] {op_type} (rank {rank}, {pct_total}%) "
              f"-> {output_relpath}")
        print(f"        Model shape: {shape_display}")
        starter_dir = "kernels/cuda" if backend == "cuda" else "kernels"
        print(f"        Based on: {starter_dir}/{op_type}.py")
        print()

        extracted.append({
            "rank": rank,
            "op_type": op_type,
            "pct_total": pct_total,
            "gpu_time_ms": gpu_time_ms,
            "model_shape": model_shape,
            "output_file": output_relpath,
        })

    if not extracted:
        print("ERROR: No kernels were successfully extracted.")
        if skipped > 0:
            print(f"       {skipped} kernel(s) skipped due to missing starter files.")
        sys.exit(1)

    # -- Generate optimization plan --
    plan = generate_optimization_plan(extracted)
    with open(OPTIMIZATION_PLAN_PATH, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=4)
    print(f"Optimization plan saved to workspace/optimization_plan.json")

    # -- Print next steps --
    print()
    top_kernel = extracted[0]
    top_file = top_kernel["output_file"]
    print("Next steps:")
    print(f"  1. Copy a kernel to kernel.py: cp {top_file} kernel.py")
    print(f"  2. Run benchmark: uv run bench.py")
    print(f"  3. Start optimizing (or let the agent do it via program.md)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoKernel Kernel Extractor -- Generate baseline kernels from profiling results.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=DEFAULT_REPORT_PATH,
        help="Path to profile_report.json (default: workspace/profile_report.json)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Extract only the top-N kernels by rank",
    )
    parser.add_argument(
        "--kernel-type",
        type=str,
        default=None,
        help="Extract only kernels of this type (e.g., matmul, flash_attention)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["triton", "cuda"],
        default="triton",
        help="Backend for starter kernels: 'triton' (default) or 'cuda' (native CUDA C++)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Override the dtype recorded in the profile report (e.g. bfloat16). "
             "This becomes TEST_DTYPES[0] in the generated kernels, which is what "
             "bench.py measures with.",
    )

    args = parser.parse_args()

    extract_kernels(
        report_path=args.report,
        top_n=args.top,
        kernel_type_filter=args.kernel_type,
        backend=args.backend,
        dtype_override=args.dtype,
    )


if __name__ == "__main__":
    main()
