#!/usr/bin/env python3
"""
AutoKernel Kernel Extractor -- Generate baseline kernels from profiling results.

Usage:
    uv run extract.py                          # extract from workspace/profile_report.json
    uv run extract.py --top 5                  # extract only top-5 kernels
    uv run extract.py --kernel-type matmul     # extract only matmul kernels
    uv run extract.py --report path/to/report.json
    uv run extract.py --backend cuda           # use CUDA C++ starter kernels instead of Triton
    uv run extract.py --spec path/spec.py:SPEC # extract an external KernelSpec

Operation metadata (shape aliases, tolerances, FLOP/byte accounting, starter
kernels, speedup estimates) comes from autokernel/specs/, not from this file.
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

# The package lives next to this script; make sure it is importable when
# extract.py is invoked from another working directory.
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from autokernel.specs import (  # noqa: E402  (path bootstrap must run first)
    KernelRegistry,
    KernelSpec,
    SpecLoadError,
    SpecValidationError,
    create_builtin_registry,
    resolve_spec,
    serialize_accounting,
)


# ---------------------------------------------------------------------------
# Shape parsing
# ---------------------------------------------------------------------------

def parse_shape_info(shape_info_str: str, spec: KernelSpec) -> Optional[Dict[str, int]]:
    """
    Parse a shape_info string like "M=4096, N=4096, K=4096" into a dict.

    Handles various formats:
      - "M=4096, N=4096, K=4096"
      - "B=1, H=32, N=4096, D=128"
      - "batch=4096, vocab=32000"
      - "rows=4096, cols=4096"

    Profiler key spellings are mapped to the specification's canonical size keys
    through ``spec.shape_aliases``.

    Returns None if parsing fails.
    """
    if not shape_info_str or not isinstance(shape_info_str, str):
        return None

    # Match key=value pairs
    pairs = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\d+)", shape_info_str)
    if not pairs:
        return None

    raw = {k: int(v) for k, v in pairs}

    # Map to the spec's canonical size keys using its alias map
    alias_map = spec.shape_aliases
    if alias_map:
        canonical = {}
        for k, v in raw.items():
            mapped_key = alias_map.get(k, k)
            canonical[mapped_key] = v
        return canonical
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


def get_default_shape(spec: KernelSpec) -> Dict[str, int]:
    """Fallback shape for a specification when a profiled shape cannot be parsed."""
    return spec.extraction_shape()


# ---------------------------------------------------------------------------
# Kernel file generation
# ---------------------------------------------------------------------------

def read_starter_kernel(spec: KernelSpec, backend: str = "triton") -> Optional[str]:
    """Read the starter kernel declared by a specification. None if absent."""
    path = spec.starter_kernel(backend)
    if path is None or not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def starter_kernel_display(spec: KernelSpec, backend: str = "triton") -> str:
    """Repository-relative starter kernel path, for logs and generated headers."""
    path = spec.starter_kernel(backend)
    if path is None:
        return f"<no {backend} starter kernel declared for {spec.name}>"
    try:
        return str(path.relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path)


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


def _accounting_body(fn: object, spec: KernelSpec, spec_locator: Optional[str]) -> str:
    """Return the body of a generated accounting function.

    Accounting :class:`~autokernel.specs.accounting.Expression` objects serialize
    to a plain numeric expression. Opaque callables (an external spec supplying
    an ordinary Python function) cannot be serialized safely, so the generated
    file resolves them from the specification at runtime instead of this script
    guessing a formula or ``eval``-ing spec content.
    """
    source = serialize_accounting(fn)
    if source is not None:
        return f"return {source}"
    if spec_locator:
        return (
            f"raise NotImplementedError(\n"
            f"        'accounting for {spec.name} is not serializable; load it from '\n"
            f"        \"the spec: autokernel.specs.load_spec({spec_locator!r})\"\n"
            f"    )"
        )
    return (
        f"raise NotImplementedError(\n"
        f"        'accounting for {spec.name} is not serializable; read it from the '\n"
        f"        'registered KernelSpec instead'\n"
        f"    )"
    )


def generate_kernel_file(
    spec: KernelSpec,
    rank: int,
    pct_total: float,
    model_shape: Dict[str, int],
    model_name: str,
    gpu_time_ms: float,
    starter_code: str,
    backend: str = "triton",
    spec_locator: Optional[str] = None,
) -> str:
    """Generate the complete kernel file content for extraction."""

    op_type = spec.name
    half_shape = scale_shape(model_shape, 0.5)
    double_shape = scale_shape(model_shape, 2.0)

    shape_display = shape_to_display(model_shape)
    half_display = shape_to_display(half_shape)
    double_display = shape_to_display(double_shape)

    tolerances = {
        dtype: tol.as_dict() for dtype, tol in spec.tolerances.items()
    }

    flops_fn_body = _accounting_body(spec.flops_fn, spec, spec_locator)
    bytes_fn_body = _accounting_body(spec.bytes_fn, spec, spec_locator)

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
    lines.append(f"")
    lines.append(f"This kernel was extracted from profiling {model_name}.")
    lines.append(f"The agent optimizes this to maximize throughput at the model-specific shapes.")
    lines.append('"""')
    lines.append("")

    # KERNEL_TYPE and BACKEND
    lines.append(f'KERNEL_TYPE = "{op_type}"')
    if backend == "cuda":
        lines.append(f'BACKEND = "cuda"')
    if spec_locator:
        lines.append(f'KERNEL_SPEC = "{spec_locator}"')
    lines.append("")

    # Model-specific shapes
    lines.append("# Model-specific shapes (the shapes that matter for THIS model)")
    lines.append(f"MODEL_SHAPES = {repr(model_shape)}")
    lines.append("")

    # Benchmark config
    lines.append("# Benchmark config (self-describing -- bench.py can load this dynamically)")
    lines.append("TEST_SIZES = [")
    lines.append(f'    ("model_primary", {repr(model_shape)}),')
    lines.append(f"    # Also test nearby sizes for robustness")
    lines.append(f'    ("model_half", {repr(half_shape)}),')
    lines.append(f'    ("model_double", {repr(double_shape)}),')
    lines.append("]")
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
    backend_dir = starter_kernel_display(spec, backend)
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
            "estimated_speedup_potential": entry.get(
                "estimated_speedup_potential", "1.5-2x"
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

def _synthetic_report_entry(spec: KernelSpec) -> Dict[str, Any]:
    """A single extraction target derived from a specification alone.

    Used when ``--spec`` is supplied without a matching profile report entry, so
    an external operation can be turned into a starter kernel file without
    profiling a model first.
    """
    return {
        "rank": 1,
        "op_type": spec.name,
        "pct_total": 0.0,
        "gpu_time_ms": 0.0,
        "shapes": spec.extraction_shape(),
        "autokernel_supported": True,
    }


def extract_kernels(
    report_path: str,
    top_n: Optional[int] = None,
    kernel_type_filter: Optional[str] = None,
    backend: str = "triton",
    spec_locator: Optional[str] = None,
    spec_override: bool = False,
) -> None:
    """Main extraction pipeline."""

    backend_label = "CUDA C++" if backend == "cuda" else "Triton"
    print(f"=== AutoKernel Kernel Extractor ({backend_label}) ===")
    print()

    # -- Resolve operation specifications ---------------------------------
    # Precedence: --spec, then --kernel-type, then whatever the report names.
    registry: KernelRegistry = create_builtin_registry()
    external_spec: Optional[KernelSpec] = None
    if spec_locator:
        try:
            external_spec, registry = resolve_spec(
                spec_locator=spec_locator, registry=registry, override=spec_override
            )
        except (SpecLoadError, SpecValidationError) as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        kernel_type_filter = external_spec.name
        print(f"Using external kernel spec: {spec_locator} (operation "
              f"'{external_spec.name}')")
        print()

    # -- Load profile report --
    print(f"Reading profile from {report_path}...")
    report = load_profile_report(report_path)
    if report is None:
        if external_spec is None:
            print(f"ERROR: Profile report not found at {report_path}")
            print(f"       Run the profiler first: uv run profile.py")
            sys.exit(1)
        print(f"  No profile report at {report_path}; extracting "
              f"'{external_spec.name}' from its specification instead.")
        report = {"model_name": f"{external_spec.name} spec", "top_kernels": []}

    # -- Get model name --
    model_name = report.get("model_name", report.get("model", "unknown model"))

    # -- Get supported kernels --
    supported = get_supported_kernels(report)

    # -- Apply filters --
    if kernel_type_filter:
        supported = [k for k in supported if k.get("op_type") == kernel_type_filter]

    if not supported:
        if external_spec is not None:
            supported = [_synthetic_report_entry(external_spec)]
            print(f"  Profile report has no '{external_spec.name}' entries; using the "
                  f"specification's default shape.")
        elif kernel_type_filter:
            print(f"WARNING: No kernels of type '{kernel_type_filter}' found in profile report.")
            sys.exit(1)
        else:
            print("ERROR: No supported kernels found in profile report.")
            print("       Ensure the profiler marks kernels with autokernel_supported=True.")
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

        # Look up the specification that owns this operation
        if not registry.contains(op_type):
            print(f"  WARNING: No kernel specification registered for '{op_type}' "
                  f"-- skipping. Registered: {', '.join(registry.list_names())}")
            skipped += 1
            continue
        spec = registry.get(op_type)
        entry_locator = spec_locator if external_spec is not None and spec is external_spec else None

        # Parse model shape
        model_shape = parse_shape_info(shape_info_str, spec)
        if model_shape is None:
            # Try to use a "shapes" dict directly if provided
            if isinstance(kernel_info.get("shapes"), dict):
                model_shape = kernel_info["shapes"]
            else:
                print(f"  WARNING: Could not parse shape for {op_type} (rank {rank}), "
                      f"using default shapes.")
                model_shape = get_default_shape(spec)

        # Read starter kernel
        starter_code = read_starter_kernel(spec, backend=backend)
        if starter_code is None:
            print(f"  WARNING: No {backend} starter kernel declared by the "
                  f"'{op_type}' spec -- skipping.")
            skipped += 1
            continue

        # Generate output filename
        output_filename = f"kernel_{op_type}_{rank}.py"
        output_path = os.path.join(WORKSPACE_DIR, output_filename)
        # Relative path for display and plan
        output_relpath = f"workspace/{output_filename}"

        # Generate the customized kernel file
        kernel_content = generate_kernel_file(
            spec=spec,
            rank=rank,
            pct_total=pct_total,
            model_shape=model_shape,
            model_name=model_name,
            gpu_time_ms=gpu_time_ms,
            starter_code=starter_code,
            backend=backend,
            spec_locator=entry_locator,
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
        print(f"        Based on: {starter_kernel_display(spec, backend)}")
        print()

        extracted.append({
            "rank": rank,
            "op_type": op_type,
            "pct_total": pct_total,
            "gpu_time_ms": gpu_time_ms,
            "model_shape": model_shape,
            "output_file": output_relpath,
            "estimated_speedup_potential": spec.speedup_estimate or "1.5-2x",
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
        "--spec",
        type=str,
        default=None,
        help="External KernelSpec locator, e.g. 'path/to/spec.py:SPEC' or "
             "'package.module:SPEC'. Takes precedence over --kernel-type.",
    )
    parser.add_argument(
        "--spec-override",
        action="store_true",
        help="Allow --spec to replace a built-in operation of the same name",
    )

    args = parser.parse_args()

    extract_kernels(
        report_path=args.report,
        top_n=args.top,
        kernel_type_filter=args.kernel_type,
        backend=args.backend,
        spec_locator=args.spec,
        spec_override=args.spec_override,
    )


if __name__ == "__main__":
    main()
