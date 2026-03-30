# AMD CDNA3 (MI300X, gfx942) Optimization Reference

> Curated from [AMD-AGI/GEAK](https://github.com/AMD-AGI/GEAK) knowledge-base.
> Use this reference when optimizing Triton or HIP kernels on MI300-series GPUs.

---

## 1. Hardware Architecture

### MI300X Key Specs

| Parameter | Value |
|-----------|-------|
| Architecture | CDNA3 (3D chiplet, 5nm + 6nm) |
| Compute Units | 304 CUs |
| Matrix Cores | 1216 (MFMA engines) |
| HBM3 Memory | 192 GB |
| Memory Bandwidth | 5.3 TB/s peak |
| FP16 / BF16 Peak | 1307 TFLOPS |
| FP32 Peak | 163 TFLOPS |
| FP64 Peak | 163 TFLOPS |
| INT8 Peak | 2614 TOPS |
| LDS per CU | 64 KB |
| L2 Cache | 256 MB (shared across dies) |
| Wavefront Size | **64 threads** (not 32) |
| Max VGPRs per CU | 65536 |
| TDP | 750W |
| Interconnect | Infinity Fabric, 128 GB/s per link |

### Variant Detection

All MI300-series GPUs share gfx942 ISA and 304 CUs. Detected via `gcnArchName = "gfx942:sramecc+:xnack-"`.
Some variants (MI308XHF etc.) may report different device names but share the same architecture and specs.

### Memory Hierarchy

```
HBM3 (5.3 TB/s, 192 GB)
    ↓
Infinity Cache / L3 (shared across dies)
    ↓
L2 Cache (256 MB, shared by CU groups)
    ↓
L1 Cache (per CU, 16-32 KB)
    ↓
LDS (64 KB per CU, explicitly managed — maps to Triton shared memory)
    ↓
Registers (65536 VGPRs per CU, partitioned across active wavefronts)
```

---

## 2. Triton on MI300X: Key Differences from NVIDIA

### Wavefront = 64 (not Warp = 32)

All Triton `num_warps` values actually control wavefronts of 64 threads:
- `num_warps=4` → 4 × 64 = 256 threads per block (good default for CDNA)
- `num_warps=8` → 8 × 64 = 512 threads per block

### Recommended Autotune Configs for gfx942

```python
import triton

MI300_MATMUL_CONFIGS = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
                  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32},
                  num_warps=8, num_stages=2),
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
                  num_warps=8, num_stages=2),
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32},
                  num_warps=8, num_stages=2),
]

MI300_ELEMENTWISE_CONFIGS = [
    triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
]
```

### `waves_per_eu` Occupancy Hint

Controls how many wavefronts per execution unit the compiler targets:
```python
triton.Config({'BLOCK_SIZE': 256}, num_warps=4, waves_per_eu=2)  # compute-bound
triton.Config({'BLOCK_SIZE': 256}, num_warps=4, waves_per_eu=4)  # memory-bound
```
- `waves_per_eu=0`: auto (compiler decides)
- `waves_per_eu=2-4`: good for compute-bound kernels (more registers per wavefront)
- `waves_per_eu=4-8`: good for memory-bound kernels (more wavefronts to hide latency)

### HIP Backend Limitations

| Feature | Status |
|---------|--------|
| `tl.dot` | Works — maps to MFMA |
| `tl.trans` | Works |
| `tl.sigmoid` | Works |
| `tl.math.tanh` | **NOT available** — use `2*tl.sigmoid(2*x) - 1` |
| `tl.math.rsqrt` | Works |
| `num_stages` | **Must be >= 1** (0 crashes). Use 2 for default pipelining. |
| `tl.load(..., eviction_policy=...)` | May be ignored on HIP |

### Environment Variables for Debugging

```bash
export TRITON_PRINT_AUTOTUNING=1      # show autotune results
export TRITON_ALWAYS_COMPILE=1        # bypass cache
export TRITON_DEBUG=1                 # debug mode
export MLIR_ENABLE_DUMP=1             # dump MLIR IR
export HIP_VISIBLE_DEVICES=0          # select GPU
```

---

## 3. Occupancy Tuning on CDNA

> Source: GEAK `knowledge-base/amd-knowledge-base/layer-6-extended/optimize-guides/silu_optim/occupancy-tuning.md`

### Resource Limits per CU

| Resource | Limit | Impact |
|----------|-------|--------|
| Wavefronts per CU | 32-40 (arch dependent) | Hard cap on parallelism |
| VGPRs per CU | 65536 | Registers per wavefront = registers_per_thread × 64 |
| LDS per CU | 64 KB | Shared among all blocks on the CU |
| Wavefront size | 64 threads | Fixed |

### Occupancy Calculation

```
Occupancy = Active_Wavefronts / Max_Wavefronts_per_CU

Registers_per_Wavefront = Registers_per_Thread × 64
Max_Concurrent_Wavefronts = 65536 / Registers_per_Wavefront
```

Example: 32 registers/thread → 2048/wavefront → 32 wavefronts → ~80% occupancy.

### Occupancy Sweet Spots (Memory-Bound Kernels)

| Occupancy | Performance | When to Use |
|-----------|-------------|-------------|
| 25-40% | Sub-optimal | Avoid unless compute-bound with high ILP |
| 50-75% | Good | Balanced for most kernels |
| **75-90%** | **Sweet spot** | Memory-bound kernels (softmax, layernorm, etc.) |
| 100% | Not always best | May over-constrain register usage |

---

## 4. Memory Coalescing on CDNA

> Source: GEAK `knowledge-base/amd-knowledge-base/layer-6-extended/optimize-guides/silu_optim/memory-coalescing-hip.md`

### Transaction Sizes

| Cache Level | Line Size | Notes |
|-------------|-----------|-------|
| L1 Cache Line | 64 bytes | 16 × fp32 or 32 × bf16 |
| L2 Cache Line | 128 bytes | 32 × fp32 or 64 × bf16 |
| Optimal Transaction | 128-256 bytes | Full wavefront: 64 threads × 4 bytes = 256 bytes |

### Coalescing Efficiency

| Pattern | Efficiency | Bandwidth (MI300X) |
|---------|------------|---------------------|
| Perfect coalescing | 100% | 3700-4700 GB/s (70-90% peak) |
| Stride-2 access | 25-50% | 1300-2600 GB/s |
| Random access | 1.5% | 80-265 GB/s |

### Rules for Triton

1. Standard `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` pattern is coalesced.
2. Use vectorized loads when possible (compiler usually handles this).
3. Avoid strided pointer arithmetic that breaks contiguity.
4. BLOCK_SIZE should be a multiple of 64 (wavefront size).

---

## 5. Performance Counters for Profiling

> Source: GEAK `knowledge-base/amd-knowledge-base/layer-1-hardware/amd-gpu-arch/mi300-mi200-performance-counters.md`

### Key Counters to Collect

```bash
# Compute utilization
rocprof -i counters.txt ./your_app
# counters.txt:
# pmc : SQ_INSTS_VALU SQ_INSTS_MFMA_MOPS_FP16 SQ_INSTS_MFMA_MOPS_BF16
# pmc : SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_MFMA SQ_BUSY_CYCLES
# pmc : SQ_WAVE_CYCLES SQ_WAVES
# pmc : TCC_HIT TCC_MISS TCC_REQ
# pmc : TCC_EA_RDREQ TCC_EA_WRREQ
```

### Derived Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| GPU Utilization | `GRBM_GUI_ACTIVE / GRBM_COUNT` | >90% |
| MFMA Efficiency | `SQ_ACTIVE_INST_MFMA / SQ_BUSY_CYCLES` | >50% for matmul |
| L2 Hit Rate | `TCC_HIT / TCC_REQ` | >80% for tiled kernels |
| Wavefront Occupancy | `SQ_WAVE_CYCLES / (CUs × Max_Waves × Total_Cycles)` | >60% |
| Memory BW Utilization | `(TCC_EA_RDREQ × 32B + TCC_EA_WRREQ × 64B) / Time / Peak_BW` | >70% for mem-bound |

### Bottleneck Identification

| Indicator | Bottleneck | Action |
|-----------|-----------|--------|
| High `TCP_PENDING_STALL_CYCLES` | Memory-bound | Improve blocking, increase SRAM/L2 reuse |
| Low L2 hit rate | Memory-bound | Better tiling, prefetch |
| High `SQ_ACTIVE_INST_MFMA` | Compute-bound | Already using matrix cores well |
| Low `SPI_CSN_BUSY` | Launch-bound | Fuse kernels, use persistent patterns |
| High `SQ_LDS_BANK_CONFLICT` | LDS contention | Adjust access stride, add padding |

---

## 6. CDNA3 vs NVIDIA Architecture Comparison

| Feature | MI300X (CDNA3) | H100 (Hopper) |
|---------|---------------|---------------|
| Wavefront/Warp Size | 64 | 32 |
| FP16 Peak TFLOPS | 1307 | 989 (SXM) |
| Memory BW | 5.3 TB/s | 3.35 TB/s |
| Memory Capacity | 192 GB HBM3 | 80 GB HBM3 |
| Shared Mem / LDS | 64 KB/CU | 228 KB/SM (configurable) |
| L2 Cache | 256 MB | 50 MB |
| Matrix Ops | MFMA | WGMMA / HMMA |
| Async Copy | Compiler-managed | TMA (hardware) |
| Sparsity | Not available | 2:4 structured |

**Implication for kernel optimization**: MI300X has higher peak compute and bandwidth but less
per-CU shared memory (64 KB vs 228 KB). Use smaller tile sizes in shared memory and rely more
on L2 cache (256 MB is 5× larger than H100's).

---

## References

- [GEAK: GPU Efficient Automatic Kernel optimizer](https://github.com/AMD-AGI/GEAK)
- [CDNA3 Architecture Guide](https://www.amd.com/en/technologies/cdna)
- [MI300 Performance Counters](https://instinct.docs.amd.com/latest/gpu-arch/mi300-mi200-performance-counters.html)
- [Triton on ROCm](https://github.com/ROCm/triton)
- [ROCm HIP Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)
