# Workload-Aware Optimization Strategy

> Curated from [AMD-AGI/GEAK](https://github.com/AMD-AGI/GEAK) `workload_guidance.py`.
> This framework guides the optimization agent to prioritize profiling-driven kernel-body
> rewrites over parameter sweeps, for both Triton and HIP backends.

---

## Core Principle

**Prefer kernel-body algorithmic changes over autotune parameter sweeps.**

The GEAK project's empirical finding: pure `@triton.autotune` config sweeps or
`num_warps/num_stages/BLOCK_*` parameter searches without kernel-body changes yield
limited gains (<5%). Kernel-body rewrites (tiling, fusion, math reformulation) typically
yield 20-50%+ improvement.

---

## Triton Backend Strategy

### Bottleneck: Memory-Bound

When `bench.py` reports `bottleneck: memory` or high `% peak bandwidth`:

**Prefer First:**
1. Algorithmic kernel-body rewrites that change the reduction tree, tiling scheme, or math formulation
2. Operation fusion — merge adjacent work into the Triton kernel body to eliminate memory round-trips
3. Memory-access rewrites: better blocking, fewer redundant loads/stores, higher SRAM/L2 reuse
4. Masking, pointer-arithmetic, or load/store simplifications that reduce HBM traffic

**Consider Next:**
- Shape-specialized kernel variants for different input regimes
- Vectorized or blocked load/store patterns as part of a broader traffic reduction plan
- Kernel-body memory-layout and live-range cleanup

**Deprioritize:**
- `@triton.autotune`-only config sweeps
- Pure `num_warps / num_stages / BLOCK_*` parameter search without body change
- Python dispatch, import-routing, or wrapper-only edits

### Bottleneck: Compute-Bound

When `bench.py` reports `bottleneck: compute` or high `% peak TFLOPS`:

**Prefer First:**
1. Instruction-count reduction and control-flow simplification inside hot loops
2. MFMA / `tl.dot`-friendly reformulations, cheaper math primitives
3. Algorithmic approximations when correctness permits

**Consider Next:**
- Register-pressure and live-range reductions for better compiler scheduling
- Shape-specialized variants

**Deprioritize:**
- Same as memory-bound

### Bottleneck: Latency-Bound

When kernel is very short (small shapes, launch overhead dominates):

**Prefer First:**
1. Fuse adjacent short kernels so each launch does materially more work
2. Persistent or multi-tile kernel patterns that amortize launch overhead
3. Increase work per program

**Consider Next:**
- Shape-specialized kernel variants for small vs large shapes

---

## HIP Backend Strategy

### Bottleneck: Memory-Bound

**Prefer First:**
1. Algorithmic HIP kernel-body rewrites (search / reduction / tiling structure)
2. Coalescing, vectorized access, or LDS staging to raise effective bandwidth
3. Global-memory traffic reduction by fusing steps or recomputing cheap values

**Consider Next:**
- Wavefront-level memory-access reordering or bank-conflict reduction
- Size-specialized kernel variants

**Deprioritize:**
- Launch-config or occupancy-only tuning
- Wrapper / dispatch / copy-path edits

### Bottleneck: Compute-Bound

**Prefer First:**
1. Instruction-count reduction, branch simplification, cheaper per-thread math
2. Wave intrinsics, MFMA-friendly decomposition, unrolled inner loops

### Bottleneck: LDS-Bound

**Prefer First:**
1. LDS-bank-conflict reduction and staged-access restructuring
2. Move transient data from LDS to registers when it reduces LDS pressure

---

## Strategy Selection Flowchart

```
bench.py output
    │
    ├── bottleneck: memory
    │   ├── % peak BW > 70%  → focus on algorithmic fusion / fewer memory ops
    │   └── % peak BW < 50%  → focus on coalescing / blocking / access patterns
    │
    ├── bottleneck: compute
    │   ├── % peak TFLOPS > 60%  → near roofline, try smaller algorithmic changes
    │   └── % peak TFLOPS < 40%  → tl.dot not utilized well, check data layouts
    │
    └── bottleneck: latency
        └── focus on kernel fusion and persistent patterns
```

---

## Planning Policy (from GEAK)

1. Fill most optimization attempts with "Prefer First" strategies
2. Only add autotune / launch / wrapper attempts after at least 3 preferred-family attempts
3. Skip iteration if only low-priority ideas remain — move to next kernel
4. Each iteration should change ONE thing in the kernel body and measure its effect

---

## References

- [GEAK workload_guidance.py](https://github.com/AMD-AGI/GEAK/blob/main/src/minisweagent/agents/heterogeneous/workload_guidance.py)
- [GEAK mini_kernel_strategy_list.yaml](https://github.com/AMD-AGI/GEAK/blob/main/src/minisweagent/config/mini_kernel_strategy_list.yaml)
