#!/usr/bin/env python3
"""
KernelBench Evaluation Harness -- Correctness + Performance for KernelBench problems.

Fully compatible with the KernelBench evaluation protocol:
  - 5 random input trials for correctness (atol=1e-2, rtol=1e-2)
  - 3 warmup + 100 timed runs for performance
  - Speedup vs PyTorch reference (Model)

Additional AutoKernel features:
  - Numerical stability probing (NaN/Inf detection)
  - Determinism checks (3 runs with same seed must be bitwise identical)
  - VRAM monitoring
  - Timeout protection (30s per kernel call)
  - Greppable summary output (for log parsing by the agent)

Usage:
    uv run kernelbench/bench_kb.py                     # Full evaluation
    uv run kernelbench/bench_kb.py --quick              # Quick (3 trials, 30 timed)
    uv run kernelbench/bench_kb.py --correctness-only   # Skip performance
    uv run kernelbench/bench_kb.py --n-timed 200        # More timing iterations
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import signal
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = PROJECT_DIR / "workspace"
KB_ACTIVE_DIR = WORKSPACE_DIR / "kb_active"
KERNEL_PY = PROJECT_DIR / "kernel.py"

# Ensure project root is on sys.path (for `from kernels.cuda._compile import ...`)
sys.path.insert(0, str(PROJECT_DIR))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2
DEFAULT_N_CORRECTNESS = 5
DEFAULT_N_WARMUP = 3
DEFAULT_N_TIMED = 100
TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

class _Timeout:
    """Context manager that raises TimeoutError after `seconds`."""

    def __init__(self, seconds: float, msg: str = "timeout"):
        self.seconds = seconds
        self.msg = msg
        self._use_signal = hasattr(signal, "SIGALRM")  # Unix only

    def __enter__(self):
        if self._use_signal:
            self._prev = signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(int(self.seconds))
        else:
            self._timer = threading.Timer(self.seconds, self._thread_raise)
            self._timer.start()
        return self

    def __exit__(self, *_):
        if self._use_signal:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._prev)
        else:
            self._timer.cancel()

    def _handler(self, signum, frame):
        raise TimeoutError(self.msg)

    def _thread_raise(self):
        # Thread-based fallback for Windows -- limited, can't interrupt GPU ops
        pass


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def _load_module_from_path(path: Path, module_name: str):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_reference():
    """Load the reference Model, get_inputs, get_init_inputs from workspace."""
    ref_path = KB_ACTIVE_DIR / "reference.py"
    if not ref_path.exists():
        print("ERROR: No active KernelBench problem.")
        print("       Run: uv run kernelbench/bridge.py setup --level 1 --problem 1")
        sys.exit(1)
    mod = _load_module_from_path(ref_path, "_kb_reference")
    Model = getattr(mod, "Model", None)
    get_inputs = getattr(mod, "get_inputs", None)
    get_init_inputs = getattr(mod, "get_init_inputs", None)
    if Model is None or get_inputs is None:
        print("ERROR: reference.py must define Model, get_inputs(), get_init_inputs().")
        sys.exit(1)
    if get_init_inputs is None:
        get_init_inputs = lambda: []
    return Model, get_inputs, get_init_inputs


def load_kernel():
    """Load ModelNew from kernel.py."""
    if not KERNEL_PY.exists():
        print("ERROR: kernel.py not found.")
        print("       Run: uv run kernelbench/bridge.py setup --level 1 --problem 1")
        sys.exit(1)
    mod = _load_module_from_path(KERNEL_PY, "_kb_kernel")
    ModelNew = getattr(mod, "ModelNew", None)
    if ModelNew is None:
        print("ERROR: kernel.py must define a ModelNew class.")
        sys.exit(1)
    get_inputs = getattr(mod, "get_inputs", None)
    get_init_inputs = getattr(mod, "get_init_inputs", None)
    problem_meta = getattr(mod, "KERNELBENCH_PROBLEM", {})
    return ModelNew, get_inputs, get_init_inputs, problem_meta


def load_metadata() -> Dict[str, Any]:
    """Load active problem metadata."""
    meta_path = KB_ACTIVE_DIR / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# Tensor comparison
# ---------------------------------------------------------------------------

def _has_nan_inf(t) -> bool:
    import torch
    return bool(torch.isnan(t).any() or torch.isinf(t).any())


def _compare(output, expected, atol: float, rtol: float) -> Dict[str, Any]:
    """Compare two tensors. Returns match info."""
    import torch

    result: Dict[str, Any] = {
        "match": False,
        "reason": "",
        "max_abs_error": float("inf"),
        "mean_abs_error": float("inf"),
    }

    # Shape check
    if output.shape != expected.shape:
        result["reason"] = f"shape mismatch: {output.shape} vs {expected.shape}"
        return result

    # Cast to float32 for comparison
    out_f = output.detach().float().cpu()
    exp_f = expected.detach().float().cpu()

    # NaN/Inf symmetry
    out_nan = torch.isnan(out_f)
    exp_nan = torch.isnan(exp_f)
    if out_nan.any() or exp_nan.any():
        if not torch.equal(out_nan, exp_nan):
            result["reason"] = "NaN position mismatch"
            return result
        mask = ~out_nan
        if mask.any():
            out_f = out_f[mask]
            exp_f = exp_f[mask]
        else:
            result["match"] = True
            result["reason"] = "all NaN (matching)"
            result["max_abs_error"] = 0.0
            result["mean_abs_error"] = 0.0
            return result

    abs_err = (out_f - exp_f).abs()
    result["max_abs_error"] = float(abs_err.max())
    result["mean_abs_error"] = float(abs_err.mean())

    if torch.allclose(out_f, exp_f, atol=atol, rtol=rtol):
        result["match"] = True
        result["reason"] = "PASS"
    else:
        result["reason"] = (
            f"tolerance exceeded: max_abs={result['max_abs_error']:.6e}, "
            f"mean_abs={result['mean_abs_error']:.6e} (atol={atol}, rtol={rtol})"
        )

    return result


def _compare_outputs(output, expected, atol: float, rtol: float) -> Dict[str, Any]:
    """Compare outputs: tensors, tuples/lists of tensors, or scalars."""
    import torch

    if isinstance(output, torch.Tensor) and isinstance(expected, torch.Tensor):
        return _compare(output, expected, atol, rtol)

    if isinstance(output, (tuple, list)) and isinstance(expected, (tuple, list)):
        if len(output) != len(expected):
            return {"match": False, "reason": f"output count mismatch: {len(output)} vs {len(expected)}"}
        worst = {"match": True, "reason": "PASS", "max_abs_error": 0.0, "mean_abs_error": 0.0}
        for i, (o, e) in enumerate(zip(output, expected)):
            r = _compare_outputs(o, e, atol, rtol)
            if not r["match"]:
                return {"match": False, "reason": f"output[{i}]: {r['reason']}"}
            worst["max_abs_error"] = max(worst["max_abs_error"], r.get("max_abs_error", 0))
        return worst

    # Scalar comparison
    try:
        diff = abs(float(output) - float(expected))
        if diff <= atol:
            return {"match": True, "reason": "PASS", "max_abs_error": diff, "mean_abs_error": diff}
    except (TypeError, ValueError):
        pass

    return {"match": False, "reason": f"incomparable types: {type(output)} vs {type(expected)}"}


# ---------------------------------------------------------------------------
# Correctness evaluation
# ---------------------------------------------------------------------------

def run_correctness(
    model_new,
    model_ref,
    get_inputs_fn: Callable,
    n_trials: int = DEFAULT_N_CORRECTNESS,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    KernelBench-compatible correctness checks.

    Generate n_trials random input sets, run both models, compare within (atol, rtol).
    """
    import torch

    results: Dict[str, Any] = {
        "correctness": "FAIL",
        "trials_passed": 0,
        "trials_total": n_trials,
        "worst_max_abs_error": 0.0,
        "worst_mean_abs_error": 0.0,
        "details": [],
    }

    for trial in range(n_trials):
        trial_info: Dict[str, Any] = {"trial": trial, "status": "FAIL"}
        try:
            with _Timeout(TIMEOUT_SECONDS, f"trial {trial} timed out"):
                inputs = get_inputs_fn()
                inputs_dev = [
                    inp.to(device) if isinstance(inp, torch.Tensor) else inp
                    for inp in inputs
                ]

                with torch.no_grad():
                    expected = model_ref(*inputs_dev)
                with torch.no_grad():
                    output = model_new(*inputs_dev)

                cmp = _compare_outputs(output, expected, atol, rtol)
                trial_info["max_abs_error"] = cmp.get("max_abs_error", float("inf"))
                trial_info["mean_abs_error"] = cmp.get("mean_abs_error", float("inf"))

                if cmp["match"]:
                    trial_info["status"] = "PASS"
                    results["trials_passed"] += 1
                    results["worst_max_abs_error"] = max(
                        results["worst_max_abs_error"], cmp.get("max_abs_error", 0)
                    )
                    results["worst_mean_abs_error"] = max(
                        results["worst_mean_abs_error"], cmp.get("mean_abs_error", 0)
                    )
                else:
                    trial_info["reason"] = cmp["reason"]

        except TimeoutError as e:
            trial_info["status"] = "TIMEOUT"
            trial_info["reason"] = str(e)
        except Exception as e:
            trial_info["status"] = "ERROR"
            trial_info["reason"] = f"{type(e).__name__}: {e}"

        results["details"].append(trial_info)

        # Early exit on failure
        if trial_info["status"] != "PASS":
            break

    if results["trials_passed"] == n_trials:
        results["correctness"] = "PASS"

    return results


def run_stability(
    model_new,
    get_inputs_fn: Callable,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Test numerical stability: check for NaN/Inf on normal inputs."""
    import torch

    result: Dict[str, Any] = {"stability": "PASS", "details": []}

    for trial in range(3):
        try:
            inputs = get_inputs_fn()
            inputs_dev = [
                inp.to(device) if isinstance(inp, torch.Tensor) else inp
                for inp in inputs
            ]
            with torch.no_grad():
                output = model_new(*inputs_dev)

            if isinstance(output, torch.Tensor):
                if _has_nan_inf(output):
                    result["stability"] = "WARN"
                    result["details"].append(f"trial {trial}: output contains NaN/Inf")
            elif isinstance(output, (tuple, list)):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor) and _has_nan_inf(o):
                        result["stability"] = "WARN"
                        result["details"].append(f"trial {trial}: output[{i}] contains NaN/Inf")

        except Exception as e:
            result["stability"] = "FAIL"
            result["details"].append(f"trial {trial}: {type(e).__name__}: {e}")

    return result


def run_determinism(
    model_new,
    get_inputs_fn: Callable,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run 3 times with identical inputs, check bitwise reproducibility."""
    import torch

    result: Dict[str, Any] = {"determinism": "PASS", "details": []}

    try:
        torch.manual_seed(42)
        inputs = get_inputs_fn()
        inputs_dev = [
            inp.to(device) if isinstance(inp, torch.Tensor) else inp
            for inp in inputs
        ]
        inputs_copies = [
            [
                inp.clone() if isinstance(inp, torch.Tensor) else copy.deepcopy(inp)
                for inp in inputs_dev
            ]
            for _ in range(3)
        ]

        outputs = []
        for i in range(3):
            with torch.no_grad():
                out = model_new(*inputs_copies[i])
            if isinstance(out, torch.Tensor):
                outputs.append(out.clone())
            elif isinstance(out, (tuple, list)):
                outputs.append(tuple(o.clone() if isinstance(o, torch.Tensor) else o for o in out))
            else:
                outputs.append(out)

        for i in range(1, 3):
            if isinstance(outputs[0], torch.Tensor):
                if not torch.equal(outputs[0], outputs[i]):
                    result["determinism"] = "WARN"
                    result["details"].append(f"run 0 vs run {i}: not bitwise identical")
            elif isinstance(outputs[0], (tuple, list)):
                for j, (a, b) in enumerate(zip(outputs[0], outputs[i])):
                    if isinstance(a, torch.Tensor) and not torch.equal(a, b):
                        result["determinism"] = "WARN"
                        result["details"].append(
                            f"run 0 vs run {i}, output[{j}]: not bitwise identical"
                        )

    except Exception as e:
        result["determinism"] = "ERROR"
        result["details"].append(f"{type(e).__name__}: {e}")

    return result


# ---------------------------------------------------------------------------
# Performance evaluation
# ---------------------------------------------------------------------------

def run_performance(
    model_new,
    model_ref,
    get_inputs_fn: Callable,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_timed: int = DEFAULT_N_TIMED,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    KernelBench-compatible performance benchmarking via CUDA event timing.
    Returns speedup = ref_time / kernel_time.
    """
    import torch

    result: Dict[str, Any] = {
        "kernel_time_ms": 0.0,
        "reference_time_ms": 0.0,
        "speedup": 0.0,
        "kernel_times": [],
        "reference_times": [],
    }

    torch.manual_seed(0)
    inputs = get_inputs_fn()
    inputs_dev = [
        inp.to(device) if isinstance(inp, torch.Tensor) else inp
        for inp in inputs
    ]

    def _time_model(model, inputs_list, n_warm, n_iter):
        """Time a model using CUDA events."""
        for _ in range(n_warm):
            with torch.no_grad():
                model(*inputs_list)
        torch.cuda.synchronize()

        times = []
        for _ in range(n_iter):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                model(*inputs_list)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return times

    try:
        ref_times = _time_model(model_ref, inputs_dev, n_warmup, n_timed)
        result["reference_times"] = ref_times
        result["reference_time_ms"] = _robust_median(ref_times)

        kernel_times = _time_model(model_new, inputs_dev, n_warmup, n_timed)
        result["kernel_times"] = kernel_times
        result["kernel_time_ms"] = _robust_median(kernel_times)

        if result["kernel_time_ms"] > 0:
            result["speedup"] = result["reference_time_ms"] / result["kernel_time_ms"]

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def _robust_median(times: List[float]) -> float:
    """Trimmed median: median of the middle 80% of measurements."""
    if not times:
        return 0.0
    s = sorted(times)
    n = len(s)
    trim = max(1, n // 10)
    trimmed = s[trim:n - trim] if n > 2 * trim else s
    mid = len(trimmed) // 2
    if len(trimmed) % 2 == 0:
        return (trimmed[mid - 1] + trimmed[mid]) / 2
    return trimmed[mid]


# ---------------------------------------------------------------------------
# VRAM monitoring
# ---------------------------------------------------------------------------

def get_vram_usage() -> Dict[str, float]:
    """Get current GPU VRAM usage in MB."""
    import torch
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "reserved_mb": torch.cuda.memory_reserved() / 1e6,
        "peak_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KernelBench Evaluation -- Correctness + Performance",
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 trials, 30 timed runs")
    parser.add_argument("--correctness-only", action="store_true",
                        help="Skip performance benchmarking")
    parser.add_argument("--n-trials", type=int, default=None,
                        help=f"Correctness trials (default: {DEFAULT_N_CORRECTNESS})")
    parser.add_argument("--n-timed", type=int, default=None,
                        help=f"Timed iterations (default: {DEFAULT_N_TIMED})")
    parser.add_argument("--n-warmup", type=int, default=DEFAULT_N_WARMUP,
                        help=f"Warmup iterations (default: {DEFAULT_N_WARMUP})")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL,
                        help=f"Absolute tolerance (default: {DEFAULT_ATOL})")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL,
                        help=f"Relative tolerance (default: {DEFAULT_RTOL})")
    parser.add_argument("--skip-stability", action="store_true",
                        help="Skip stability test")
    parser.add_argument("--skip-determinism", action="store_true",
                        help="Skip determinism test")

    args = parser.parse_args()
    n_trials = args.n_trials or (3 if args.quick else DEFAULT_N_CORRECTNESS)
    n_timed = args.n_timed or (30 if args.quick else DEFAULT_N_TIMED)

    import torch
    use_xpu = (not torch.cuda.is_available()) and hasattr(torch, "xpu") and torch.xpu.is_available()
    device = "xpu" if use_xpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: No CUDA/XPU GPU detected. Results on CPU are not meaningful.")

    meta = load_metadata()
    uid = meta.get("uid", "unknown")
    name = meta.get("name", "unknown")
    level = meta.get("level", "?")

    print("=" * 65)
    print(f"KernelBench Evaluation: {uid} -- {name}")
    print(f"Level: {level} | Device: {device}")
    print("=" * 65)
    print()

    # ---- Load reference ----
    print("Loading reference model...")
    Model, ref_get_inputs, ref_get_init_inputs = load_reference()
    ref_init_args = ref_get_init_inputs()
    model_ref = Model(*ref_init_args)
    if hasattr(model_ref, "to"):
        model_ref = model_ref.to(device)
    if hasattr(model_ref, "eval"):
        model_ref = model_ref.eval()

    # ---- Load kernel ----
    print("Loading ModelNew from kernel.py...")
    ModelNew, kern_get_inputs, kern_get_init_inputs, problem_meta = load_kernel()

    if kern_get_init_inputs is not None:
        try:
            init_args = kern_get_init_inputs()
        except Exception:
            init_args = ref_init_args
    else:
        init_args = ref_init_args

    try:
        model_new = ModelNew(*init_args)
    except Exception as e:
        print(f"\nFATAL: ModelNew instantiation failed: {e}")
        traceback.print_exc()
        _print_summary("FAIL", 0.0, uid, name, {"correctness": "FAIL"})
        sys.exit(1)

    if hasattr(model_new, "to"):
        model_new = model_new.to(device)
    if hasattr(model_new, "eval"):
        model_new = model_new.eval()

    get_inputs_fn = kern_get_inputs or ref_get_inputs

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device == "xpu":
        torch.xpu.reset_peak_memory_stats()

    # ---- Stage 1: Correctness ----
    print(f"\n--- Stage 1: Correctness ({n_trials} trials, atol={args.atol}, rtol={args.rtol}) ---")
    correctness = run_correctness(
        model_new, model_ref, get_inputs_fn,
        n_trials=n_trials, atol=args.atol, rtol=args.rtol, device=device,
    )
    status = correctness["correctness"]
    print(f"  Trials passed: {correctness['trials_passed']}/{correctness['trials_total']}")
    if status == "PASS":
        print(f"  Worst max_abs_error: {correctness['worst_max_abs_error']:.6e}")
        print(f"  Worst mean_abs_error: {correctness['worst_mean_abs_error']:.6e}")
    else:
        for d in correctness["details"]:
            if d["status"] != "PASS":
                print(f"  Trial {d['trial']}: {d['status']} -- {d.get('reason', '?')}")
    print(f"  Result: {status}")

    # ---- Stage 2: Stability ----
    stability = {"stability": "SKIP"}
    if not args.skip_stability and status == "PASS":
        print("\n--- Stage 2: Numerical Stability ---")
        stability = run_stability(model_new, get_inputs_fn, device=device)
        print(f"  Result: {stability['stability']}")
        for d in stability.get("details", []):
            print(f"  {d}")

    # ---- Stage 3: Determinism ----
    determinism = {"determinism": "SKIP"}
    if not args.skip_determinism and status == "PASS":
        print("\n--- Stage 3: Determinism ---")
        determinism = run_determinism(model_new, get_inputs_fn, device=device)
        print(f"  Result: {determinism['determinism']}")
        for d in determinism.get("details", []):
            print(f"  {d}")

    # ---- Stage 4: Performance ----
    perf: Dict[str, Any] = {"speedup": 0.0}
    if not args.correctness_only and status == "PASS":
        print(f"\n--- Stage 4: Performance ({args.n_warmup} warmup + {n_timed} timed) ---")
        perf = run_performance(
            model_new, model_ref, get_inputs_fn,
            n_warmup=args.n_warmup, n_timed=n_timed, device=device,
        )
        print(f"  Reference time: {perf['reference_time_ms']:.4f} ms")
        print(f"  Kernel time:    {perf['kernel_time_ms']:.4f} ms")
        print(f"  Speedup:        {perf['speedup']:.3f}x")
        if "error" in perf:
            print(f"  Error:          {perf['error']}")
        if perf.get("kernel_times"):
            kt = perf["kernel_times"]
            print(f"  Kernel stats:   median={statistics.median(kt):.4f}ms, "
                  f"std={statistics.stdev(kt) if len(kt) > 1 else 0:.4f}ms, "
                  f"min={min(kt):.4f}ms, max={max(kt):.4f}ms")
    elif status != "PASS":
        print("\n--- Stage 4: Performance SKIPPED (correctness failed) ---")

    vram = get_vram_usage() if device == "cuda" else {}

    _print_summary(
        status, perf.get("speedup", 0.0), uid, name,
        {**correctness, **stability, **determinism, **perf, **vram},
    )
    _save_results(uid, correctness, stability, determinism, perf, vram, meta)


def _print_summary(
    correctness_status: str, speedup: float,
    uid: str, name: str, data: Dict[str, Any],
) -> None:
    """Print greppable summary for agent log parsing."""
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"problem: {uid}")
    print(f"name: {name}")
    print(f"correctness: {correctness_status}")
    print(f"speedup: {speedup:.3f}x")
    print(f"kernel_time_ms: {data.get('kernel_time_ms', 0):.4f}")
    print(f"reference_time_ms: {data.get('reference_time_ms', 0):.4f}")
    print(f"stability: {data.get('stability', 'SKIP')}")
    print(f"determinism: {data.get('determinism', 'SKIP')}")
    print(f"peak_vram_mb: {data.get('peak_mb', 0):.1f}")
    print(f"worst_max_abs_error: {data.get('worst_max_abs_error', 0):.6e}")
    for threshold in [1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0]:
        passes = correctness_status == "PASS" and speedup >= threshold
        print(f"fast_{threshold}: {'PASS' if passes else 'FAIL'}")
    print("=" * 65)


def _save_results(uid, correctness, stability, determinism, perf, vram, meta):
    """Append results to workspace/kb_active/results.json."""
    results_path = KB_ACTIVE_DIR / "results.json"
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "uid": uid,
        "correctness": correctness.get("correctness", "FAIL"),
        "speedup": perf.get("speedup", 0.0),
        "kernel_time_ms": perf.get("kernel_time_ms", 0.0),
        "reference_time_ms": perf.get("reference_time_ms", 0.0),
        "worst_max_abs_error": correctness.get("worst_max_abs_error", 0.0),
        "stability": stability.get("stability", "SKIP"),
        "determinism": determinism.get("determinism", "SKIP"),
        "peak_vram_mb": vram.get("peak_mb", 0.0),
    }
    history = []
    if results_path.exists():
        try:
            history = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []
    history.append(entry)
    results_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
