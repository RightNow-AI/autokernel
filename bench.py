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
  uv run bench.py --spec path/spec.py:SPEC   # benchmark an external KernelSpec
  uv run bench.py --quick                # skip stages 3-5, bench only large size
  uv run bench.py --profile              # emit torch profiler trace
  uv run bench.py --sizes large          # benchmark only 'large' size

Operation metadata (sizes, dtypes, tolerances, edge cases, FLOP/byte accounting)
lives in autokernel/specs/, not in this file.
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
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

# The package lives next to this script; make sure it is importable when bench.py
# is invoked from another working directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from autokernel.specs import (  # noqa: E402  (path bootstrap must run first)
    KernelRegistry,
    KernelSpec,
    SpecLoadError,
    SpecNotFoundError,
    SpecValidationError,
    create_builtin_registry,
    dtype_bytes,
    resolve_spec,
    resolve_torch_dtype,
)
from autokernel.verification import (  # noqa: E402
    TreeComparison,
    check_backward,
    check_compile,
    collect_environment_metadata,
    compare_deterministic,
    compare_output_trees,
    result_envelope,
    tree_has_nan_or_inf,
    write_result_atomic,
)
from autokernel.verification.corpus import (  # noqa: E402
    CorpusError,
    load_shape_corpus,
    validate_corpus_against_spec,
    weighted_aggregate,
)

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
# 2. OPERATION METADATA (from the KernelSpec registry)
# =========================================================================
# Input generators, reference wiring, sizes, dtypes, tolerances, edge cases and
# FLOP/byte accounting are owned by autokernel/specs/. This file only translates
# canonical dtype names into torch dtypes at the runtime boundary.


def _dtype_bytes(dtype: torch.dtype) -> int:
    """Return byte-width for a dtype."""
    return torch.tensor([], dtype=dtype).element_size()


def _spec_sizes(spec: KernelSpec) -> List[Tuple[str, Dict[str, int]]]:
    """Ordered ``(label, size)`` pairs, as the harness has always consumed them."""
    return list(spec.size_items())


def _spec_dtypes(spec: KernelSpec) -> List[torch.dtype]:
    """Declared dtypes as torch dtypes, in benchmark order."""
    return [resolve_torch_dtype(name) for name in spec.dtypes]


def _spec_tolerances(spec: KernelSpec) -> Dict[torch.dtype, Dict[str, float]]:
    """Tolerances keyed by torch dtype."""
    return {
        resolve_torch_dtype(name): tol.as_dict()
        for name, tol in spec.tolerances.items()
    }


def _spec_reference(spec: KernelSpec) -> Callable[[Dict[str, Any]], Any]:
    """Adapt ``reference_fn(**inputs)`` to the harness' ``ref_fn(inputs)`` shape."""

    def ref_fn(inputs: Dict[str, Any]) -> Any:
        return spec.reference_fn(**inputs)

    return ref_fn


def _spec_bytes_fn(spec: KernelSpec) -> Callable[[Dict[str, int], torch.dtype], Any]:
    """Adapt ``bytes_fn(size, dt_bytes)`` to a torch-dtype call site."""

    def bytes_fn(size: Dict[str, int], dtype: torch.dtype) -> Any:
        return spec.bytes_fn(size, _dtype_bytes(dtype))

    return bytes_fn


def _spec_edge_cases(spec: KernelSpec) -> List[Tuple[str, Dict[str, int]]]:
    """Edge-case ``(label, size)`` pairs for the shape-robustness stage."""
    return [(edge.name, dict(edge.size)) for edge in spec.edge_cases]


def _legacy_config(spec: KernelSpec) -> Dict[str, Any]:
    """Build the pre-registry ``KERNEL_CONFIGS`` entry for one specification."""
    return {
        "test_sizes": _spec_sizes(spec),
        "test_dtypes": _spec_dtypes(spec),
        "tolerances": _spec_tolerances(spec),
        "flops_fn": spec.flops_fn,
        "bytes_fn": _spec_bytes_fn(spec),
        "input_generator": spec.input_generator,
        "reference_fn": _spec_reference(spec),
        "edge_sizes": _spec_edge_cases(spec),
        "spec": spec,
    }


def _build_legacy_configs() -> Dict[str, Dict[str, Any]]:
    return {spec.name: _legacy_config(spec) for spec in create_builtin_registry()}


#: DEPRECATED compatibility view of the old hard-coded configuration table.
#: Derived from the built-in registry; edit autokernel/specs/builtins.py instead.
#: New code should use ``autokernel.specs.create_builtin_registry()``.
KERNEL_CONFIGS: Dict[str, Dict[str, Any]] = _build_legacy_configs()


#: Device the harness allocates on. Overridden only by CPU tests; the
#: benchmark itself always measures on the GPU.
BENCH_DEVICE = "cuda"


def resolve_operation_name(
    spec_name: Optional[str],
    kernel_arg: Optional[str],
    declared_type: Optional[str],
) -> Optional[str]:
    """Apply the operation-selection precedence.

    1. the name declared by an explicit ``--spec``;
    2. an explicit ``--kernel``;
    3. ``kernel.py::KERNEL_TYPE`` (the historical default).

    Returns None when nothing selects an operation.
    """
    return spec_name or kernel_arg or declared_type


def _get_spec_or_exit(registry: KernelRegistry, kernel_type: str) -> KernelSpec:
    """Fetch a spec from the registry, preserving the CLI failure contract."""
    try:
        return registry.get(kernel_type)
    except SpecNotFoundError:
        print(f"\nERROR: Unknown kernel type '{kernel_type}'")
        print(f"  Available: {', '.join(registry.list_names())}")
        print(f"\ncorrectness: FAIL")
        print(f"throughput_tflops: 0.000")
        sys.exit(1)


def _load_validated_corpus(args: argparse.Namespace, spec: KernelSpec):
    """Load and validate ``--shape-corpus`` against the selected spec.

    Returns the validated cases, or None when no corpus was requested. Exits
    with the greppable failure contract on any corpus error. Runs before the
    candidate module is imported and before any GPU allocation, so a malformed
    corpus never executes candidate code or touches the device.
    """
    if not args.shape_corpus:
        return None
    try:
        corpus = load_shape_corpus(args.shape_corpus)
        validate_corpus_against_spec(corpus, spec)
    except CorpusError as e:
        print(f"\nERROR: {e}")
        print(f"\ncorrectness: FAIL")
        print(f"throughput_tflops: 0.000")
        sys.exit(1)
    mode = "corpus-only" if args.shape_corpus_only else "append"
    print(f"shape_corpus: {args.shape_corpus} ({len(corpus.cases)} cases, mode={mode})")
    return corpus.cases


# =========================================================================
# 3. CORRECTNESS TESTING (5 stages)
# =========================================================================

def _compare_outputs(output: Any, expected: Any, spec: KernelSpec, *, relax: float = 1.0) -> TreeComparison:
    """Compare candidate and reference output trees using the spec's policy.

    Single-tensor outputs behave exactly as the historical ``_compare`` did:
    the tolerance declared for the benchmark dtype is applied, and the
    failure reason is unchanged. Structured outputs are compared leaf by leaf
    with stable diagnostic paths.
    """
    return compare_output_trees(
        output,
        expected,
        spec.tolerances,
        output_spec=spec.output_spec,
        relax=relax,
    )


def _record_leaves(records: List[Dict[str, Any]], stage: str, case: str, cmp: TreeComparison) -> None:
    """Collect per-leaf comparison details for the structured result artifact."""
    for leaf in cmp.leaf_records():
        records.append({"stage": stage, "case": case, **leaf})


def run_correctness(
    kernel_fn: Callable,
    spec: KernelSpec,
    quick: bool = False,
    corpus_cases: Optional[Sequence] = None,
    corpus_only: bool = False,
) -> dict:
    """Run all correctness stages. Returns dict with results.

    ``corpus_cases`` appends validated production shapes to the stage-2 shape
    sweep; ``corpus_only`` replaces the built-in size/dtype sweep with them.
    """
    device = BENCH_DEVICE
    results = {
        "smoke_test": "SKIP",
        "shape_sweep": "SKIP",
        "numerical_stability": "SKIP",
        "determinism": "SKIP",
        "edge_cases": "SKIP",
        "correctness": "FAIL",
    }
    details = []
    leaf_records: List[Dict[str, Any]] = []
    all_pass = True

    gen_fn = spec.input_generator
    ref_fn = _spec_reference(spec)
    sizes = _spec_sizes(spec)
    dtypes = _spec_dtypes(spec)

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

        if tree_has_nan_or_inf(output):
            results["smoke_test"] = "FAIL"
            details.append(f"  smoke: NaN/Inf in output")
            all_pass = False
            print(f"  FAIL: NaN/Inf in output")
        else:
            cmp = _compare_outputs(output, expected, spec)
            _record_leaves(leaf_records, "smoke", tiny_label, cmp)
            if cmp.match:
                results["smoke_test"] = "PASS"
                print(f"  PASS (max_abs_error={cmp.worst_abs_error:.6e})")
            else:
                results["smoke_test"] = "FAIL"
                details.append(f"  smoke: {cmp.reason}")
                all_pass = False
                print(f"  FAIL: {cmp.reason}")
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
        results["leaf_details"] = leaf_records
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

    sweep_configs: List[Tuple[str, Dict[str, int], torch.dtype]] = []
    if not corpus_only:
        for label, sz in sizes:
            for dtype in dtypes:
                sweep_configs.append((label, sz, dtype))
    if corpus_cases:
        for case in corpus_cases:
            case_dtype = resolve_torch_dtype(case.dtype) if case.dtype else dtypes[0]
            sweep_configs.append((case.name, dict(case.size), case_dtype))

    for label, sz, dtype in sweep_configs:
            sweep_count += 1
            try:
                inputs = gen_fn(sz, dtype, device, seed=42)
                expected = ref_fn(inputs)
                with _Timeout(30):
                    output = kernel_fn(**inputs)

                if tree_has_nan_or_inf(output):
                    sweep_pass = False
                    sweep_fail_count += 1
                    details.append(f"  sweep {label}/{dtype}: NaN/Inf")
                    print(f"  FAIL: {label} {dtype} -> NaN/Inf")
                    continue

                cmp = _compare_outputs(output, expected, spec)
                _record_leaves(leaf_records, "sweep", f"{label}/{dtype}", cmp)

                if cmp.worst_abs_error > worst_error:
                    worst_error = cmp.worst_abs_error
                    worst_case = f"{label}/{dtype}"

                if not cmp.match:
                    sweep_pass = False
                    sweep_fail_count += 1
                    details.append(f"  sweep {label}/{dtype}: {cmp.reason}")
                    print(f"  FAIL: {label} {dtype} -> {cmp.reason}")
                else:
                    print(f"  PASS: {label} {dtype} (max_err={cmp.worst_abs_error:.2e}, within_tol={cmp.worst_pct_within_tol:.1f}%)")

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
        results["leaf_details"] = leaf_records
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

            if tree_has_nan_or_inf(output) and not tree_has_nan_or_inf(expected):
                stability_pass = False
                details.append(f"  stability {case_name}: NaN/Inf (reference is clean)")
                print(f"  FAIL: {case_name} -> NaN/Inf (reference is clean)")
            elif tree_has_nan_or_inf(output) and tree_has_nan_or_inf(expected):
                # Both have NaN/Inf -- acceptable (e.g. overflow in near_max)
                print(f"  PASS: {case_name} -> both have NaN/Inf (expected overflow)")
            else:
                # Relax tolerances for adversarial inputs
                cmp = _compare_outputs(output, expected, spec, relax=10.0)
                _record_leaves(leaf_records, "stability", case_name, cmp)
                if cmp.match:
                    print(f"  PASS: {case_name} (max_err={cmp.worst_abs_error:.2e})")
                else:
                    stability_pass = False
                    details.append(f"  stability {case_name}: {cmp.reason}")
                    print(f"  FAIL: {case_name} -> {cmp.reason}")

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
            cmp = compare_deterministic(outputs[0], outputs[i], output_spec=spec.output_spec)
            _record_leaves(leaf_records, "determinism", f"run_0_vs_{i}", cmp)
            if not cmp.match:
                determinism_pass = False
                failure = cmp.first_failure()
                if failure is not None and failure.max_abs_error is not None:
                    where = "" if failure.path == "output" else f" at {failure.path}"
                    message = f"run 0 vs run {i} differ{where} (max_diff={failure.max_abs_error:.6e})"
                else:
                    message = f"run 0 vs run {i} differ ({cmp.reason})"
                details.append(f"  determinism: {message}")
                print(f"  FAIL: {message}")

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
    edge_cases = spec.edge_cases
    if not edge_cases:
        results["edge_cases"] = "SKIP (no edge sizes defined)"
        print("  SKIP: no edge sizes defined")
    else:
        for edge in edge_cases:
            label = edge.name
            sz = dict(edge.size)
            # An edge case may pin its own dtype; otherwise use the primary
            # dtype only, for speed.
            dtype = resolve_torch_dtype(edge.dtype) if edge.dtype else dtypes[0]
            try:
                inputs = gen_fn(sz, dtype, device, seed=edge.seed)
                if edge.input_transform is not None:
                    inputs = edge.input_transform(inputs)
                expected = ref_fn(inputs)
                with _Timeout(30):
                    output = kernel_fn(**inputs)

                if tree_has_nan_or_inf(output) and not tree_has_nan_or_inf(expected):
                    edge_pass = False
                    details.append(f"  edge {label}: NaN/Inf")
                    print(f"  FAIL: {label} -> NaN/Inf")
                else:
                    cmp = _compare_outputs(output, expected, spec)
                    _record_leaves(leaf_records, "edge", label, cmp)
                    if cmp.match:
                        print(f"  PASS: {label} (max_err={cmp.worst_abs_error:.2e})")
                    else:
                        edge_pass = False
                        details.append(f"  edge {label}: {cmp.reason}")
                        print(f"  FAIL: {label} -> {cmp.reason}")

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
    results["leaf_details"] = leaf_records
    print(f"\ncorrectness: {results['correctness']}")
    return results


def run_backward_check(kernel_fn: Callable, spec: KernelSpec) -> dict:
    """Opt-in gradient verification. Prints the greppable verdict line and
    returns the structured report."""
    print(f"\n=== BACKWARD CORRECTNESS ===")
    report = check_backward(kernel_fn, spec, device=BENCH_DEVICE)
    if report.status == "PASS":
        print(f"  upstream outputs: {', '.join(report.output_paths)}")
        for record in report.gradients:
            print(f"  grad[{record.input_name}]: match "
                  f"(max_err={record.max_abs_error:.2e}, mean_err={record.mean_abs_error:.2e})")
    else:
        print(f"  FAIL: {report.reason}")
        for record in report.gradients:
            if record.status != "match":
                print(f"  grad[{record.input_name}]: {record.status}: {record.reason}")
    print(f"BACKWARD_CORRECTNESS: {report.status}")
    return report.as_dict()


def run_compile_check(kernel_fn: Callable, spec: KernelSpec) -> dict:
    """Run compile verification outside all timed performance regions."""
    print(f"\n=== COMPILE CORRECTNESS ===")
    report = check_compile(kernel_fn, spec, device=BENCH_DEVICE)
    for case in report.cases:
        detail = f": {case.reason}" if case.reason else ""
        print(f"  {case.label}: {case.status}{detail}")
    if report.reason and report.status != "PASS":
        print(f"  {report.reason}")
    print(f"COMPILE_CORRECTNESS: {report.status}")
    return report.as_dict()


# =========================================================================
# 4. PERFORMANCE BENCHMARKING
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


def run_performance(kernel_fn: Callable, spec: KernelSpec, gpu: GPUSpec,
                    sizes_filter: str = "all",
                    corpus_cases: Optional[Sequence] = None,
                    corpus_only: bool = False) -> dict:
    """Run performance benchmarks. Returns dict with metrics.

    ``corpus_cases`` appends production shapes (each benchmarked once; their
    weights feed aggregate reporting, not repetition). ``corpus_only``
    benchmarks only those shapes.
    """
    device = BENCH_DEVICE
    gen_fn = spec.input_generator
    ref_fn = _spec_reference(spec)
    flops_fn = spec.flops_fn
    bytes_fn = _spec_bytes_fn(spec)
    dtypes = _spec_dtypes(spec)

    # Select benchmark size
    sizes = _spec_sizes(spec)
    bench_sizes = []
    if corpus_only:
        bench_sizes = []
    elif sizes_filter == "all":
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

    # (label, size, dtype, weight, source) benchmark configurations. Built-in
    # sizes carry weight 1; corpus cases carry their declared weight, which is
    # used only for aggregate reporting, never to repeat benchmark loops.
    bench_configs: List[Tuple[str, Dict[str, int], torch.dtype, int, str]] = [
        (label, sz, dtype, 1, "builtin") for label, sz in bench_sizes
    ]
    if corpus_cases:
        for case in corpus_cases:
            case_dtype = resolve_torch_dtype(case.dtype) if case.dtype else dtype
            bench_configs.append((case.name, dict(case.size), case_dtype, case.weight, "corpus"))

    all_results = []
    primary_result = None

    for label, sz, dtype, weight, source in bench_configs:
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
                "weight": weight,
                "source": source,
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

    # Weighted aggregate reporting for corpus cases, grouped per dtype so
    # results from different dtypes are never mixed into one aggregate.
    corpus_summary = None
    corpus_entries = [entry for entry in all_results if entry["source"] == "corpus"]
    if corpus_entries:
        weighted = weighted_aggregate(
            {
                "dtype": entry["dtype"],
                "weight": entry["weight"],
                "kernel_ms": entry["kernel_latency_us"] / 1000.0,
                "ref_ms": entry["pytorch_latency_us"] / 1000.0,
            }
            for entry in corpus_entries
        )
        corpus_summary = {"cases": corpus_entries, "weighted": weighted}
        print(f"\n  === SHAPE CORPUS: weighted aggregates ===")
        for dtype_name, agg in weighted.items():
            print(f"  dtype={dtype_name}: cases={agg['cases']}, total_weight={agg['weight']}")
            print(f"    weighted_kernel_latency_us: {agg['kernel_ms'] * 1000.0:.2f}")
            print(f"    weighted_pytorch_latency_us: {agg['ref_ms'] * 1000.0:.2f}")
            print(f"    weighted_speedup_vs_pytorch: {agg['speedup']:.3f}x")

    return {
        "primary": primary_result,
        "all": all_results,
        "corpus": corpus_summary,
    }


# =========================================================================
# 5. PROFILER (optional)
# =========================================================================

def run_profile(kernel_fn: Callable, spec: KernelSpec):
    """Run torch profiler and save a trace."""
    device = BENCH_DEVICE
    gen_fn = spec.input_generator
    sizes = _spec_sizes(spec)

    # Use 'medium' or first size
    prof_size = None
    for label, sz in sizes:
        if label == "medium":
            prof_size = sz
            break
    if prof_size is None:
        prof_size = sizes[0][1]

    dtype = _spec_dtypes(spec)[0]
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
# 6. MAIN -- orchestrate everything and produce structured output
# =========================================================================

def main():
    t_start = time.time()

    parser = argparse.ArgumentParser(description="AutoKernel benchmark harness")
    parser.add_argument("--kernel", type=str, default=None,
                        help="Kernel type to benchmark (default: read from kernel.py)")
    parser.add_argument("--spec", type=str, default=None,
                        help="External KernelSpec locator, e.g. "
                             "'path/to/spec.py:SPEC' or 'package.module:SPEC'. "
                             "Takes precedence over --kernel.")
    parser.add_argument("--spec-override", action="store_true",
                        help="Allow --spec to replace a built-in operation of the same name")
    parser.add_argument("--sizes", type=str, default="all",
                        help="Which sizes to benchmark: small|medium|large|all (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: skip correctness stages 3-5, bench only large size")
    parser.add_argument("--profile", action="store_true",
                        help="Enable torch profiler trace")
    parser.add_argument("--shape-corpus", type=str, default=None, metavar="PATH",
                        help="Versioned JSON shape corpus; validated cases are "
                             "appended to the built-in sweep and benchmarked once each")
    parser.add_argument("--shape-corpus-only", action="store_true",
                        help="Benchmark only the --shape-corpus cases, skipping the "
                             "built-in size sweep (requires --shape-corpus)")
    parser.add_argument("--check-backward", action="store_true",
                        help="Also verify gradients against the reference "
                             "(requires the spec to declare a backward_spec; "
                             "correctness-only, no performance claims)")
    parser.add_argument("--check-compile", action="store_true",
                        help="Verify torch.compile parity using the spec's compile settings "
                             "(correctness-only; compilation is never timed)")
    parser.add_argument("--result-json", type=str,
                        default=os.path.join(_SCRIPT_DIR, "workspace", "bench_result.json"),
                        metavar="PATH",
                        help="Atomic machine-readable result path "
                             "(default: workspace/bench_result.json)")
    args = parser.parse_args()
    if args.shape_corpus_only and not args.shape_corpus:
        parser.error("--shape-corpus-only requires --shape-corpus PATH")

    # ------------------------------------------------------------------
    # Import the kernel module
    # ------------------------------------------------------------------
    print("=" * 60)
    print("AutoKernel Benchmark Harness")
    print("=" * 60)

    kernel_module = None
    kernel_fn = None
    kernel_type = args.kernel

    # ------------------------------------------------------------------
    # Resolve the operation specification.
    # Precedence: --spec, then --kernel, then kernel.py::KERNEL_TYPE.
    # Loading happens after argument parsing, so `bench.py --help` never
    # imports an external specification.
    # ------------------------------------------------------------------
    registry: KernelRegistry = create_builtin_registry()
    spec: Optional[KernelSpec] = None
    if args.spec:
        try:
            spec, registry = resolve_spec(
                spec_locator=args.spec,
                registry=registry,
                override=args.spec_override,
            )
        except (SpecLoadError, SpecValidationError) as e:
            print(f"\nERROR: {e}")
            print(f"\ncorrectness: FAIL")
            print(f"throughput_tflops: 0.000")
            sys.exit(1)
        kernel_type = spec.name
        print(f"kernel_spec: {args.spec}")

    # When the operation is already determined (--spec or --kernel), fetch its
    # spec and validate the shape corpus *before* importing the candidate
    # module: malformed metadata must fail without executing candidate code
    # and before any GPU allocation.
    kernel_type = spec.name if spec is not None else args.kernel
    spec_locked = kernel_type is not None
    corpus_cases = None
    if spec_locked:
        if spec is None:
            spec = _get_spec_or_exit(registry, kernel_type)
        corpus_cases = _load_validated_corpus(args, spec)

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

        declared_type = getattr(kernel_module, "KERNEL_TYPE", None)
        if spec_locked:
            resolved = kernel_type
        else:
            resolved = resolve_operation_name(None, args.kernel, declared_type)
        if resolved is None:
            print("ERROR: kernel.py has no KERNEL_TYPE attribute and --kernel not specified")
            sys.exit(1)
        if declared_type is not None and declared_type != resolved:
            print(f"WARNING: kernel.py declares KERNEL_TYPE '{declared_type}' but "
                  f"'{resolved}' was requested; benchmarking kernel_fn against "
                  f"'{resolved}'")
        kernel_type = resolved

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

    # The default selection path (kernel.py::KERNEL_TYPE) resolves the spec
    # and validates the corpus only after the candidate import.
    if not spec_locked:
        spec = _get_spec_or_exit(registry, kernel_type)
        corpus_cases = _load_validated_corpus(args, spec)

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
        correctness_results = run_correctness(
            kernel_fn, spec, quick=args.quick,
            corpus_cases=corpus_cases, corpus_only=args.shape_corpus_only,
        )
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
    print(f"FORWARD_CORRECTNESS: {correctness_results['correctness']}")

    # ------------------------------------------------------------------
    # Backward verification (opt-in; correctness-only, never timed)
    # ------------------------------------------------------------------
    backward_result = None
    backward_requested = args.check_backward or (
        spec.backward_spec is not None and spec.backward_spec.enabled_by_default
    )
    if backward_requested:
        try:
            backward_result = run_backward_check(kernel_fn, spec)
        except Exception as e:
            print(f"\nFATAL: Backward verification crashed: {type(e).__name__}: {e}")
            traceback.print_exc()
            backward_result = {"status": "FAIL", "reason": f"crash: {type(e).__name__}: {e}"}
            print(f"BACKWARD_CORRECTNESS: FAIL")

    # ------------------------------------------------------------------
    # Compile verification (opt-in; correctness-only, never timed)
    # ------------------------------------------------------------------
    compile_result = None
    compile_requested = args.check_compile or (
        spec.compile_spec is not None and spec.compile_spec.enabled
    )
    if compile_requested:
        try:
            compile_result = run_compile_check(kernel_fn, spec)
        except Exception as e:
            print(f"\nFATAL: Compile verification crashed: {type(e).__name__}: {e}")
            traceback.print_exc()
            compile_result = {
                "status": "FAIL",
                "reason": f"crash: {type(e).__name__}: {e}",
            }
            print(f"COMPILE_CORRECTNESS: FAIL")

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------
    # Determine primary size info for the header
    _perf_sizes = _spec_sizes(spec)
    _perf_primary_label = None
    _perf_primary_size = None
    for _pl, _ps in _perf_sizes:
        if _pl == "large":
            _perf_primary_label = _pl
            _perf_primary_size = _ps
            break
    if _perf_primary_size is None:
        _perf_primary_label, _perf_primary_size = _perf_sizes[-1]
    _perf_dtype = _spec_dtypes(spec)[0]
    _size_params = ", ".join(f"{k}={v}" for k, v in _perf_primary_size.items())
    print(f"\n=== PERFORMANCE ({_perf_primary_label}: {_size_params}, dtype={_perf_dtype}) ===")

    perf_results = {"primary": None, "all": []}
    peak_vram_mb = 0.0
    try:
        sizes_filter = args.sizes
        if args.quick:
            sizes_filter = "large"
        torch.cuda.reset_peak_memory_stats()
        perf_results = run_performance(
            kernel_fn, spec, gpu, sizes_filter=sizes_filter,
            corpus_cases=corpus_cases, corpus_only=args.shape_corpus_only,
        )
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
            run_profile(kernel_fn, spec)
        except Exception as e:
            print(f"\nWARNING: Profiling failed: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Final summary (key greppable lines)
    # ------------------------------------------------------------------
    t_elapsed = time.time() - t_start
    throughput = primary["throughput_tflops"] if primary else 0.0

    corpus_identity = None
    if args.shape_corpus:
        corpus_identity = {
            "source": os.path.abspath(args.shape_corpus),
            "mode": "only" if args.shape_corpus_only else "append",
            "cases": [
                {
                    "name": case.name,
                    "size": dict(case.size),
                    "dtype": case.dtype,
                    "weight": case.weight,
                    "tags": list(case.tags),
                }
                for case in (corpus_cases or ())
            ],
        }
    result_payload = result_envelope(
        kernel_type,
        environment=collect_environment_metadata(BENCH_DEVICE),
        request={
            "spec": args.spec,
            "sizes": args.sizes,
            "quick": args.quick,
            "profile": args.profile,
            "check_backward": backward_requested,
            "check_compile": compile_requested,
        },
        gpu=asdict(gpu),
        shape_corpus=corpus_identity,
        forward=correctness_results,
        backward=backward_result,
        compile=compile_result,
        performance={
            **perf_results,
            "peak_vram_mb": peak_vram_mb,
        },
        bench_time_seconds=t_elapsed,
    )
    try:
        result_path = write_result_atomic(args.result_json, result_payload)
        print(f"result_json: {result_path}")
    except Exception as e:
        print(f"WARNING: Failed to write result JSON: {type(e).__name__}: {e}")

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
