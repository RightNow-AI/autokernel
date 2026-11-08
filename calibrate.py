"""
AutoKernel -- GPU calibration: measure achieved peaks and compare to the GPU database.

bench.py and profile.py compute pct_peak and the roofline from the peaks in
_KNOWN_GPUS. On a new GPU, run this once to check those peaks against what the
hardware actually delivers:

  - HBM bandwidth: a large device-to-device copy (read + write bytes)
  - Dense FP16/BF16 tensor throughput: a large torch.matmul (cuBLAS)
  - Attention: dense (non-causal) attention of one long-prefill chunk over a long KV cache, per PyTorch SDPA
    backend, each checked against the float32 oracle first. The best backend is the attainable attention
    peak that % of SOL for a prefill-attention kernel is measured against. Default shapes are grouped-query
    chunks of 8K queries over 128K keys at head_dim 128 and 256.

A sane entry measures roughly 80-95% of the database value on both. A result
far above 100% means the database entry is too low and every pct_peak is
inflated; far below means the entry is too high or the GPU is throttled.

Usage:
    uv run calibrate.py
    uv run calibrate.py --gemm-size 16384 --copy-gb 8
    uv run calibrate.py --attention 8192x1048576:16/2:256   # q_len x kv_len : heads_q/heads_kv : head_dim
    uv run calibrate.py --attention none
"""

import argparse
import json
import sys

import torch
import torch.nn.functional as F

from bench import KERNEL_CONFIGS, _do_bench, detect_gpu, gen_gqa_prefill_attention_inputs
import reference

ATTENTION_DEFAULT = ("8192x131072:16/2:128", "8192x131072:16/2:256")
ATTENTION_BACKENDS = ("FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION")


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


def parse_attention_shape(spec: str) -> dict:
    """'8192x131072:16/2:256' -> one grouped-query chunk: q_len x kv_len : heads_q/heads_kv : head_dim."""
    try:
        lengths, heads, head_dim = spec.split(":")
        q_len, kv_len = (int(v) for v in lengths.lower().split("x"))
        heads_q, heads_kv = (int(v) for v in heads.split("/"))
        size = {"batch": 1, "heads_q": heads_q, "heads_kv": heads_kv,
                "q_len": q_len, "kv_len": kv_len, "head_dim": int(head_dim)}
    except ValueError:
        raise ValueError(f"attention shape {spec!r} is not q_len x kv_len : heads_q/heads_kv : head_dim") from None
    if min(size.values()) < 1 or heads_q % heads_kv or q_len > kv_len:
        raise ValueError(f"attention shape {spec!r} needs heads_q % heads_kv == 0 and q_len <= kv_len")
    return size


def _judge(out, run, expected, tol, bench_fn, flops, **extra) -> dict:
    """Time run only if its output matches the oracle."""
    error = (out.float() - expected).abs().max().item()
    entry = {"max_abs_error": error, **extra}
    if not torch.allclose(out.float(), expected, **tol):
        return {**entry, "wrong": True}
    ms = bench_fn(run)
    return {**entry, "ms": ms, "tflops": flops / (ms / 1e3) / 1e12}


def _flashinfer(x: dict, expected, tol, bench_fn, flops) -> dict:
    try:
        import flashinfer
    except ImportError:
        return {"unavailable": True}
    if x["Q"].shape[0] != 1:
        return {"unsupported": True}
    # [1, heads, len, head_dim] -> FlashInfer's NHD layout [len, heads, head_dim], once, outside the timed call
    q, k, v = (t[0].transpose(0, 1).contiguous() for t in (x["Q"], x["K"], x["V"]))

    def run():
        return flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False).transpose(0, 1).unsqueeze(0)
    try:
        out = run()
    except Exception:  # noqa: BLE001 -- an unsupported shape, dtype or GPU is reported, not raised
        return {"unsupported": True}
    return _judge(out, run, expected, tol, bench_fn, flops, kv_expanded=False)


def measure_attention(size: dict, dtype: torch.dtype = torch.bfloat16, device: str = "cuda",
                      bench_fn=None) -> dict:
    """Dense attention of one q_len chunk over kv_len keys, timed per SDPA backend and FlashInfer.

    A backend is timed only after its output matches the float32 oracle at the bench tolerance. Grouped KV
    heads go in as they are (enable_gqa); a backend that refuses them is retried with expanded heads, built
    outside the timed call and reported as kv_expanded. Unsupported and wrong backends are reported, not timed.
    FlashInfer's single-request prefill (the kernel family behind vLLM's FlashInfer backend) is measured
    when the package is installed, with layouts converted outside the timed call.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel
    bench_fn = bench_fn or (lambda fn: _do_bench(fn, warmup=25, rep=200))
    x = gen_gqa_prefill_attention_inputs(size, dtype, device)
    expected = reference.gqa_prefill_attention_ref(x["Q"], x["K"], x["V"], causal=False).float()
    tol = KERNEL_CONFIGS["gqa_prefill_attention"]["tolerances"][dtype]
    flops = 4 * size["batch"] * size["heads_q"] * size["q_len"] * size["kv_len"] * size["head_dim"]
    group = size["heads_q"] // size["heads_kv"]
    backends = {}
    for name in ATTENTION_BACKENDS:
        backend = getattr(SDPBackend, name)
        out, run, expanded = None, None, False
        for expanded in (False, True):
            if expanded and group == 1:
                break
            k, v = (x["K"], x["V"]) if not expanded else (x["K"].repeat_interleave(group, 1),
                                                         x["V"].repeat_interleave(group, 1))

            def run(k=k, v=v, expanded=expanded):
                with sdpa_kernel([backend]):
                    return F.scaled_dot_product_attention(x["Q"], k, v, enable_gqa=not expanded)
            try:
                out = run()
                break
            except RuntimeError:
                out = None
        if out is None:
            backends[name.lower()] = {"unsupported": True}
            continue
        backends[name.lower()] = _judge(out, run, expected, tol, bench_fn, flops, kv_expanded=expanded)
    backends["flashinfer"] = _flashinfer(x, expected, tol, bench_fn, flops)
    timed = {k: v for k, v in backends.items() if "tflops" in v}
    best = max(timed, key=lambda k: timed[k]["tflops"]) if timed else None
    return {"shape": size, "dtype": str(dtype).replace("torch.", ""), "flops": flops, "backends": backends,
            "best_backend": best, "best_tflops": timed[best]["tflops"] if best else None}


def main():
    parser = argparse.ArgumentParser(description="Measure achieved GPU peaks vs the AutoKernel GPU database")
    parser.add_argument("--gemm-size", type=int, default=8192, help="M=N=K of the calibration GEMM")
    parser.add_argument("--copy-gb", type=float, default=4.0, help="GiB copied in the bandwidth test")
    parser.add_argument("--attention", action="append", default=None, metavar="QxKV:HQ/HKV:D",
                        help="Long-prefill attention shape to measure (repeatable; 'none' skips the attention peak). "
                             "Default: " + " and ".join(ATTENTION_DEFAULT))
    parser.add_argument("--json", action="store_true", help="Print one JSON object instead of a table")
    args = parser.parse_args()
    specs = ATTENTION_DEFAULT if args.attention is None else [s for s in args.attention if s.lower() != "none"]
    try:
        attention_sizes = [parse_attention_shape(spec) for spec in specs]
    except ValueError as exc:
        parser.error(str(exc))

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
        "attention": [measure_attention(size) for size in attention_sizes],
    }
    for entry in result["attention"]:
        best = entry["best_tflops"]
        entry["best_pct_of_bf16_gemm_measured"] = round(100 * best / bf16, 1) if best and bf16 > 0 else None

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
    for entry in result["attention"]:
        s = entry["shape"]
        cells = []
        for name, b in entry["backends"].items():
            state = (f"{b['tflops']:.1f}" + ("*" if b["kv_expanded"] else "") if "tflops" in b
                     else "wrong" if b.get("wrong") else "unsupported")
            cells.append(f"{name} {state}")
        best = (f"best {entry['best_backend']} {entry['best_tflops']:.1f} TFLOPS = "
                f"{entry['best_pct_of_bf16_gemm_measured']}% of measured BF16 GEMM" if entry["best_backend"]
                else "no backend")
        print(f"attention {s['q_len']}x{s['kv_len']} {s['heads_q']}/{s['heads_kv']} d{s['head_dim']} "
              f"{entry['dtype']} dense: " + ", ".join(cells) + f"; {best}")
    if any(b.get("kv_expanded") for e in result["attention"] for b in e["backends"].values()):
        print("* timed with KV heads expanded to one per query head (the backend refused grouped heads)")


if __name__ == "__main__":
    main()
