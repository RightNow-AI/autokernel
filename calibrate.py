"""
AutoKernel -- GPU calibration: measure achieved peaks and compare to the GPU database.

bench.py and profile.py compute pct_peak and the roofline from the peaks in
_KNOWN_GPUS. On a new GPU, run this once to check those peaks against what the
hardware actually delivers:

  - HBM bandwidth: a large device-to-device copy (read + write bytes)
  - Dense FP16/BF16 tensor throughput: a large torch.matmul (cuBLAS)

A sane entry measures roughly 80-95% of the database value on both. A result
far above 100% means the database entry is too low and every pct_peak is
inflated; far below means the entry is too high or the GPU is throttled.

Usage:
    uv run calibrate.py
    uv run calibrate.py --gemm-size 16384 --copy-gb 8
"""

import argparse
import json
import sys

import torch

from bench import _do_bench, detect_gpu


def measure_bandwidth_gb_s(copy_gb: float) -> float:
    n_bytes = int(copy_gb * (1024 ** 3))
    src = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    ms = _do_bench(lambda: dst.copy_(src), warmup=10, rep=50)
    # A copy reads every byte once and writes it once
    return 2 * n_bytes / (ms / 1e3) / 1e9


def measure_gemm_tflops(size: int, dtype: torch.dtype) -> float:
    a = torch.randn(size, size, dtype=dtype, device="cuda")
    b = torch.randn(size, size, dtype=dtype, device="cuda")
    ms = _do_bench(lambda: torch.matmul(a, b), warmup=10, rep=50)
    return 2 * size ** 3 / (ms / 1e3) / 1e12


def main():
    parser = argparse.ArgumentParser(description="Measure achieved GPU peaks vs the AutoKernel GPU database")
    parser.add_argument("--gemm-size", type=int, default=8192, help="M=N=K of the calibration GEMM")
    parser.add_argument("--copy-gb", type=float, default=4.0, help="GiB copied in the bandwidth test")
    parser.add_argument("--json", action="store_true", help="Print one JSON object instead of a table")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA GPU available")
        sys.exit(1)

    gpu = detect_gpu()
    bw = measure_bandwidth_gb_s(args.copy_gb)
    fp16 = measure_gemm_tflops(args.gemm_size, torch.float16)
    bf16 = measure_gemm_tflops(args.gemm_size, torch.bfloat16)

    result = {
        "gpu": gpu.name,
        "compute_capability": f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}",
        "sm_count": gpu.sm_count,
        "memory_gb": gpu.memory_gb,
        "l2_cache_mb": round(gpu.l2_cache_mb, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "bandwidth_gb_s": {"measured": round(bw, 1), "database": gpu.peak_bandwidth_gb_s},
        "fp16_tflops": {"measured": round(fp16, 1), "database": gpu.peak_tflops_fp16},
        "bf16_tflops": {"measured": round(bf16, 1), "database": gpu.peak_tflops_bf16},
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"gpu: {result['gpu']} (sm_{gpu.compute_capability[0]}{gpu.compute_capability[1]}, "
          f"{gpu.sm_count} SMs, {gpu.memory_gb} GB, L2 {result['l2_cache_mb']} MB)")
    print(f"torch {result['torch']}, CUDA {result['cuda']}")
    print(f"{'metric':<16}{'measured':>12}{'database':>12}{'measured/db':>14}")
    for key, unit in (("bandwidth_gb_s", "GB/s"), ("fp16_tflops", "TFLOPS"), ("bf16_tflops", "TFLOPS")):
        m, d = result[key]["measured"], result[key]["database"]
        pct = f"{100 * m / d:.1f}%" if d > 0 else "n/a"
        print(f"{key:<16}{m:>12.1f}{d:>12.1f}{pct:>14}")


if __name__ == "__main__":
    main()
