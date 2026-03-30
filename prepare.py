"""
AutoKernel -- One-time setup and baseline benchmarking.

Verifies environment (CUDA, Triton, PyTorch), generates deterministic test data,
runs a smoke test on the current kernel, and benchmarks PyTorch reference
implementations so that future experiments have a cached baseline to compare
against.

Usage:
    uv run prepare.py
"""

import json
import os
import sys

import torch

# ---------------------------------------------------------------------------
# Constants (shared with bench.py -- keep in sync)
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autokernel")
TEST_DATA_DIR = os.path.join(CACHE_DIR, "test_data")
BASELINES_PATH = os.path.join(CACHE_DIR, "baselines.json")

# Matmul test sizes (must match bench.py)
MATMUL_SIZES = [
    ("tiny",    {"M": 128,  "N": 128,  "K": 128}),
    ("small",   {"M": 512,  "N": 512,  "K": 512}),
    ("medium",  {"M": 1024, "N": 1024, "K": 1024}),
    ("large",   {"M": 2048, "N": 2048, "K": 2048}),
    ("xlarge",  {"M": 4096, "N": 4096, "K": 4096}),
]

TEST_DTYPES = [torch.float16, torch.bfloat16]

# Number of warmup and benchmark iterations for baseline timing
_WARMUP_ITERS = 25
_BENCH_ITERS = 100

# Deterministic seed for reproducibility
_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dtype_tag(dtype: torch.dtype) -> str:
    """Short string tag for a dtype, e.g. 'fp16', 'bf16'."""
    return {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}[dtype]


def _matmul_flops(M: int, N: int, K: int) -> int:
    """FLOPs for a single matmul C[M,N] = A[M,K] @ B[K,N]."""
    return 2 * M * N * K


def _benchmark_fn(fn, *args, warmup: int = _WARMUP_ITERS, iters: int = _BENCH_ITERS):
    """
    Benchmark *fn* using CUDA events. Returns median latency in microseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    torch.cuda.synchronize()
    for i in range(iters):
        start_events[i].record()
        fn(*args)
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    return median_ms * 1000.0  # convert to microseconds


# ---------------------------------------------------------------------------
# Step 1-4: Environment verification
# ---------------------------------------------------------------------------

def verify_environment() -> None:
    """Print GPU specs, PyTorch version, Triton version. Exit on failure."""

    print("=== AutoKernel Setup ===\n")

    # -- CUDA & GPU --
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. A CUDA-capable GPU is required.")
        sys.exit(1)

    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    props = torch.cuda.get_device_properties(device)
    mem_gb = props.total_memory / (1024 ** 3)
    sm_count = props.multi_processor_count
    cc_major = props.major
    cc_minor = props.minor

    # Runtime version: CUDA toolkit or ROCm/HIP version
    cuda_version = torch.version.cuda or "unknown"
    hip_version = getattr(torch.version, "hip", None)
    if hip_version:
        cuda_version = f"ROCm {hip_version}"

    # GPU driver version -- try nvidia-smi first, then rocm-smi as fallback
    import subprocess
    driver_str = "unknown"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            driver_str = result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    if driver_str == "unknown":
        try:
            result = subprocess.run(
                ["rocm-smi", "--showdriverversion"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if "Driver" in line:
                        driver_str = line.split(":")[-1].strip()
                        break
        except Exception:
            pass

    print(f"GPU: {gpu_name}")
    print(f"  Memory: {mem_gb:.1f} GB")
    print(f"  SM Count: {sm_count}")
    print(f"  Compute Capability: {cc_major}.{cc_minor}")
    print(f"  Driver: {driver_str}")
    print(f"  CUDA: {cuda_version}")
    print()

    # -- PyTorch --
    print(f"PyTorch: {torch.__version__}")

    # -- Triton --
    try:
        import triton
        print(f"Triton: {triton.__version__}")
    except ImportError:
        print("ERROR: Triton is not installed. Install with: pip install triton")
        sys.exit(1)

    print()


# ---------------------------------------------------------------------------
# Step 5-6: Generate & cache test data
# ---------------------------------------------------------------------------

def generate_test_data() -> None:
    """Generate deterministic test tensors for all sizes and dtypes."""

    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    print("Generating test data...")

    gen = torch.Generator(device="cpu")

    for size_name, dims in MATMUL_SIZES:
        M, N, K = dims["M"], dims["N"], dims["K"]
        for dtype in TEST_DTYPES:
            tag = _dtype_tag(dtype)
            label = f"  matmul/{size_name}/{tag}"

            save_dir = os.path.join(TEST_DATA_DIR, "matmul", size_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{tag}.pt")

            if os.path.exists(save_path):
                print(f"{label} ... cached")
                continue

            # Deterministic generation -- seed is fixed per (size, dtype) pair
            gen.manual_seed(_SEED)
            A = torch.randn(M, K, generator=gen, dtype=dtype)
            B = torch.randn(K, N, generator=gen, dtype=dtype)

            torch.save({"A": A, "B": B}, save_path)
            print(f"{label} ... ok")

    print()


# ---------------------------------------------------------------------------
# Step 7: Smoke test
# ---------------------------------------------------------------------------

def smoke_test() -> None:
    """Import kernel.py, run on tiny input, check correctness."""

    print("Smoke test...")

    # Import kernel
    try:
        import kernel  # noqa: F401
        print("  Import kernel.py: ok")
    except Exception as e:
        print(f"  Import kernel.py: FAIL ({e})")
        sys.exit(1)

    # Detect the kernel type from the module. Only run the full smoke test
    # for matmul kernels -- other kernel types have different calling
    # conventions and input shapes that we cannot generically test here.
    kernel_type = getattr(kernel, "KERNEL_TYPE", None)
    if kernel_type is None:
        # Try to infer from module contents
        try:
            if hasattr(kernel, "kernel_fn"):
                kernel_type = "unknown"
            else:
                kernel_type = "unknown"
        except Exception:
            kernel_type = "unknown"

    if kernel_type != "matmul" and kernel_type != "unknown":
        print(f"  Kernel type is '{kernel_type}' (not matmul) -- skipping matmul smoke test.")
        print(f"  Smoke test: SKIP (kernel-type-specific smoke test not implemented)")
        print()
        return

    if kernel_type != "matmul":
        # kernel_type is "unknown" -- try the matmul smoke test but do not
        # fail hard if the calling convention does not match.
        print(f"  Kernel type not declared -- attempting matmul smoke test...")

    # Import reference
    try:
        from reference import matmul_ref
    except Exception as e:
        print(f"  Import reference.py: FAIL ({e})")
        sys.exit(1)

    # Run kernel on tiny fp16 input
    dtype = torch.float16
    size_name = "tiny"
    dims = dict(MATMUL_SIZES)[size_name]
    M, N, K = dims["M"], dims["N"], dims["K"]

    gen = torch.Generator(device="cpu")
    gen.manual_seed(_SEED)
    A = torch.randn(M, K, generator=gen, dtype=dtype).cuda()
    B = torch.randn(K, N, generator=gen, dtype=dtype).cuda()

    try:
        C_kernel = kernel.kernel_fn(A, B)
        torch.cuda.synchronize()
        print(f"  Run kernel (tiny, fp16): ok")
    except Exception as e:
        if kernel_type == "unknown":
            print(f"  Run kernel (tiny, fp16): SKIP (not a matmul kernel? error: {e})")
            print()
            return
        print(f"  Run kernel (tiny, fp16): FAIL ({e})")
        sys.exit(1)

    # Correctness check
    C_ref = matmul_ref(A, B)
    torch.cuda.synchronize()

    # For fp16 matmul, use relaxed tolerance
    atol = 1e-2
    rtol = 1e-2
    if torch.allclose(C_kernel, C_ref, atol=atol, rtol=rtol):
        print("  Correctness check: PASS")
    else:
        max_diff = (C_kernel - C_ref).abs().max().item()
        print(f"  Correctness check: FAIL (max diff = {max_diff:.6f}, atol={atol}, rtol={rtol})")
        # Don't exit -- let the user decide

    print()


# ---------------------------------------------------------------------------
# Step 8: Benchmark PyTorch baselines
# ---------------------------------------------------------------------------

def benchmark_baselines() -> dict:
    """Benchmark torch.matmul at all sizes and dtypes. Returns results dict."""

    print("Benchmarking PyTorch baselines...")
    results = {}

    for size_name, dims in MATMUL_SIZES:
        M, N, K = dims["M"], dims["N"], dims["K"]
        flops = _matmul_flops(M, N, K)

        for dtype in TEST_DTYPES:
            tag = _dtype_tag(dtype)

            # Load cached test data if available, else generate on the fly
            save_path = os.path.join(TEST_DATA_DIR, "matmul", size_name, f"{tag}.pt")
            if os.path.exists(save_path):
                data = torch.load(save_path, weights_only=True)
                A = data["A"].cuda()
                B = data["B"].cuda()
            else:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(_SEED)
                A = torch.randn(M, K, generator=gen, dtype=dtype).cuda()
                B = torch.randn(K, N, generator=gen, dtype=dtype).cuda()

            latency_us = _benchmark_fn(torch.matmul, A, B)
            tflops = flops / (latency_us * 1e-6) / 1e12

            key = f"matmul_{size_name}_{tag}"
            results[key] = {
                "kernel_type": "matmul",
                "size": size_name,
                "dtype": tag,
                "M": M, "N": N, "K": K,
                "latency_us": round(latency_us, 2),
                "throughput_tflops": round(tflops, 3),
            }

            print(f"  matmul {size_name} {tag}: {tflops:.1f} TFLOPS ({latency_us:.2f} us)")

            # Free GPU memory
            del A, B
            torch.cuda.empty_cache()

    print()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Step 1-4: Verify environment
    verify_environment()

    # Step 5: Create cache directories
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    # Step 6: Generate test data
    generate_test_data()

    # Step 7: Smoke test
    smoke_test()

    # Step 8: Benchmark baselines
    baselines = benchmark_baselines()

    # Save baselines
    with open(BASELINES_PATH, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"Baselines saved to {BASELINES_PATH}")

    # Step 9: Summary
    print()
    print("Ready to run experiments!")


if __name__ == "__main__":
    main()
