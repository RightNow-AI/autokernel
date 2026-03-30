# AutoKernel -- Autonomous GPU Kernel Optimization Agent

You are an autonomous GPU kernel optimization researcher. You accept a full PyTorch
model, profile it, identify bottleneck kernels, and optimize each one in priority
order. You maximize end-to-end model speedup, not just individual kernel throughput.

---

## Overview

The workflow has three phases:

| Phase | Description | Human Involvement |
|-------|-------------|-------------------|
| **A: Model Analysis** | Profile the model, identify bottlenecks, plan | Interactive with human |
| **B: Multi-Kernel Optimization** | Optimize each kernel in priority order | Fully autonomous |
| **C: Integration** | Verify end-to-end, generate final report | Autonomous, human reviews |

A typical run covers 5-10 kernels across 10+ hours. You should expect to run 300+ experiments
total across all kernels.

---

## Phase A: Model Analysis (with human)

This phase is interactive. You work with the human to understand the model, profile it,
and agree on an optimization plan.

### A1. Human provides the model

The human gives you one of:
- A local Python file: `models/llama_7b.py` with a class name like `LlamaModel`
- A HuggingFace model: `transformers.AutoModelForCausalLM` with `meta-llama/Llama-2-7b-hf`
- An input shape: e.g., `1,2048` for batch=1, seq_len=2048

Record these details. You will use them throughout the run.

### A2. Profile the model

Run profiling to identify where the model spends its time:

```bash
uv run profile.py --model <path> --class-name <name> --input-shape <shape>
```

Or for HuggingFace models:

```bash
uv run profile.py --module transformers --class-name AutoModelForCausalLM --pretrained <model_name> --input-shape <shape>
```

Read the output. The profiler reports:
- Total model latency
- Per-op breakdown (which ops take the most time)
- Memory usage
- GPU utilization

### A3. Read the profile report

```bash
cat workspace/profile_report.json
```

Look for:
- The top 5-10 ops by time percentage
- Whether the model is compute-bound or memory-bound overall
- Which op types dominate (matmul, attention, normalization, etc.)

Present findings to the human in a clear summary:

```
Model: LlamaModel (7B params)
Input: [1, 2048], dtype=float16
Total latency: 142.5 ms

Top bottleneck ops:
  1. matmul       -- 62.3% of total (88.8 ms)  [compute-bound]
  2. attention    -- 18.1% of total (25.8 ms)  [memory-bound]
  3. layernorm    -- 8.2% of total  (11.7 ms)  [memory-bound]
  4. rmsnorm      -- 4.5% of total  (6.4 ms)   [memory-bound]
  5. rotary_emb   -- 2.1% of total  (3.0 ms)   [memory-bound]
  Remaining ops:  4.8% (6.8 ms)
```

### A4. Extract kernels for optimization

```bash
uv run extract.py --top 5
```

This extracts the top bottleneck kernels into the workspace:

```
workspace/
  kernel_matmul_1.py          -- rank 1 bottleneck
  kernel_attention_2.py       -- rank 2 bottleneck
  kernel_layernorm_3.py       -- rank 3 bottleneck
  kernel_rmsnorm_4.py         -- rank 4 bottleneck
  kernel_rotary_embedding_5.py -- rank 5 bottleneck
  orchestration_state.json    -- tracks progress across all kernels
```

### A5. Present the optimization plan

Use Amdahl's law to estimate the maximum possible speedup for each kernel:

```
Amdahl's Law Estimates (assuming 2x speedup on each kernel):
  matmul (62.3%):      model speedup = 1.45x
  + attention (18.1%): model speedup = 1.67x
  + layernorm (8.2%):  model speedup = 1.76x
  + rmsnorm (4.5%):    model speedup = 1.81x
  + rotary_emb (2.1%): model speedup = 1.83x

Recommendation: Focus on top 3 kernels (matmul, attention, layernorm).
They cover 88.6% of total latency and offer up to 1.76x end-to-end speedup.
```

### A6. Human confirms

Wait for the human to confirm the plan. They may:
- Adjust the number of kernels to optimize
- Change priority order
- Specify time budget constraints
- Request focus on specific ops

Once confirmed, proceed to Phase B.

### A7. Create the branch

```bash
git checkout -b autokernel/<tag>
```

Use a descriptive tag like `mar10-llama7b`.

### A8. Read all files for context

Read every file in the repo:
- `README.md` -- repository context and design philosophy
- `bench.py` -- fixed benchmark and correctness harness (do not modify)
- `reference.py` -- PyTorch reference implementations (do not modify)
- `prepare.py` -- one-time setup (do not modify)
- `kernel.py` -- the single file you modify for each kernel
- `kernels/` -- starter kernels for each supported type
- `verify.py` -- end-to-end verification (do not modify)
- `workspace/` -- extracted kernels and state

### A9. Verify environment

```bash
uv run prepare.py
```

Confirm that `~/.cache/autokernel/` contains `test_data/` and `baselines.json`.

### A10. Initialize results.tsv

Create `results.tsv` with just the header row:
```
experiment	tag	kernel_type	throughput_tflops	latency_us	pct_peak	speedup_vs_pytorch	correctness	peak_vram_mb	description
```
Use tabs as separators (NOT commas -- commas break in descriptions).

---

## Phase B: Multi-Kernel Optimization Loop

**This phase is fully autonomous. NEVER STOP. NEVER ASK THE HUMAN.**

You optimize each kernel in priority order. The orchestrator (`orchestrate.py`) tracks
progress across all kernels and tells you when to move on.

### B1. Check orchestrator for next kernel

```bash
uv run orchestrate.py next
```

The orchestrator returns one of:
- `NEXT: kernel_matmul_1` -- optimize this kernel next
- `CONTINUE: kernel_matmul_1` -- keep optimizing the current kernel
- `DONE` -- all kernels have reached their targets or plateaued
- `REVISIT: kernel_attention_2` -- go back to a previous kernel

### B2. Set up the kernel

Copy the target kernel into `kernel.py`:

```bash
cp workspace/kernel_<type>_<rank>.py kernel.py
```

Verify:
- `KERNEL_TYPE` in `kernel.py` matches the expected type
- `kernel_fn()` signature matches the reference in `reference.py`

### B3. Run baseline for this kernel

```bash
uv run bench.py > run.log 2>&1
```

Record this as the baseline:

```bash
uv run orchestrate.py record kernel_<type>_<rank>.py <tflops> keep "baseline"
```

### B4. Single-kernel experiment loop

**LOOP FOREVER. NEVER STOP. NEVER ASK THE HUMAN.**

Each iteration:

#### 1. Hypothesize

Think carefully about what to try next. Consider:
- What is the current bottleneck? (compute-bound vs memory-bound -- check roofline data)
- What tier of the optimization playbook should you explore?
- What worked or failed in previous experiments?
- Are there combinations of successful changes you haven't tried?

Write a brief hypothesis (1-2 sentences) about what you expect the change to do and why.

#### 2. Edit kernel.py

Make **one focused change** per experiment. Do not combine multiple unrelated optimizations
in a single experiment -- you need to know what caused the improvement or regression.

Examples of a single focused change:
- Change BLOCK_SIZE_M from 64 to 128
- Add software prefetching with `tl.prefetch`
- Switch accumulator from fp32 to tf32
- Add L2 cache swizzling to the tile index

#### 3. Commit

```bash
git add kernel.py && git commit -m "exp N: <brief hypothesis>"
```

Always commit before running. This lets you cleanly revert if the experiment fails.

#### 4. Run

```bash
uv run bench.py > run.log 2>&1
```

**IMPORTANT**: Always redirect to `run.log`. Do NOT use `tee`. Do NOT let output flood your
context window. The benchmark harness writes structured output that you parse afterward.

#### 5. Check Results

Parse the results from `run.log`. Look for these fields:

```bash
grep "correctness\|throughput_tflops\|latency_us\|speedup_vs_pytorch\|pct_peak_compute\|pct_peak_bandwidth\|bottleneck\|peak_vram_mb" run.log
```

If grep returns nothing, the run crashed. Read the traceback:
```bash
tail -n 50 run.log
```

#### 6. Decide: KEEP or REVERT

Apply these rules strictly, in order:

| Condition | Action |
|-----------|--------|
| correctness = FAIL | **REVERT** immediately. `git reset --hard HEAD~1`. Never keep an incorrect kernel. |
| correctness = PASS, throughput improved | **KEEP**. This is the new baseline. |
| correctness = PASS, throughput same or worse | **REVERT**. `git reset --hard HEAD~1`. |

"Improved" means at least 1% gain in `throughput_tflops`. Noise-level changes should be
reverted unless the code is simpler.

**Exception -- simplicity wins**: If throughput is equal but the new code is meaningfully
simpler, KEEP it.

#### 7. Record with orchestrator

```bash
uv run orchestrate.py record <file> <tflops> keep|revert "<description>"
```

#### 8. Analyze roofline

Look at the roofline data from benchmark output:
- `pct_peak_compute` -- how close to GPU's theoretical FLOPS ceiling
- `pct_peak_bandwidth` -- how close to GPU's memory bandwidth ceiling
- `bottleneck` -- whether the kernel is compute-bound or memory-bound

This tells you where to focus next:
- **Compute-bound**: Try reducing instruction count, using tf32/tensor cores, fusing operations.
- **Memory-bound**: Try improving coalescing, adding prefetching, reducing memory traffic.

#### 9. Log

Append one row to `results.tsv` with tab-separated values:

```
N	exp-tag	kernel_type	throughput	latency	pct_peak	speedup	PASS/FAIL	vram	description
```

**Note on `pct_peak`**: Use `pct_peak_compute` if compute-bound, `pct_peak_bandwidth` if
memory-bound (as reported by `bench.py`'s `bottleneck:` output).

**Do NOT commit results.tsv** -- leave it untracked by git.

#### 10. Check orchestrator

```bash
uv run orchestrate.py next
```

- If `CONTINUE` -- repeat from step 1
- If `NEXT` or `REVISIT` -- break out of this loop, save kernel, move to next

### B5. Save optimized kernel

When the orchestrator says to move on, save the optimized kernel:

```bash
cp kernel.py workspace/kernel_<type>_<rank>_optimized.py
```

### B6. Check aggregate progress

```bash
uv run orchestrate.py status
```

This shows:
- Per-kernel: baseline vs current best, speedup, experiments run
- Aggregate: estimated end-to-end model speedup (Amdahl's law)
- Remaining: which kernels are left, estimated time

### B7. Move to next kernel

Go back to step B1. Repeat until the orchestrator says `DONE` or all kernels are optimized.

### B8. Complete loop structure

```
FOR each kernel in priority order:
    B1. Check orchestrator: uv run orchestrate.py next
    B2. If DONE → go to Phase C
    B3. Copy kernel: cp workspace/kernel_{type}_{rank}.py kernel.py
    B4. Run baseline: uv run bench.py > run.log 2>&1
    B5. Record baseline: uv run orchestrate.py record <file> <tflops> keep "baseline"
    B6. LOOP (single-kernel optimization):
        - Hypothesize, edit, commit, run, check, keep/revert
        - uv run orchestrate.py record ...
        - uv run orchestrate.py next → break if not CONTINUE
    B7. Save: cp kernel.py workspace/kernel_{type}_{rank}_optimized.py
    B8. Status: uv run orchestrate.py status
    B9. Next kernel
```

---

## Phase C: Integration and Verification

After all kernels are optimized (or the orchestrator says `DONE`), verify the end-to-end result.

### C1. Run end-to-end verification

```bash
uv run verify.py --model <path> --class-name <name> --input-shape <shape>
```

Or for HuggingFace models:

```bash
uv run verify.py --module transformers --class-name AutoModelForCausalLM \
 --pretrained <model_name> --input-shape <shape>
```

The verifier:
1. Loads the model
2. Runs inference with original PyTorch ops (reference)
3. Replaces optimized modules (nn.Linear with matmul kernel, nn.LayerNorm with layernorm kernel, etc.)
4. Runs inference with optimized kernels
5. Compares outputs for correctness
6. Reports end-to-end speedup

### C2. Handle verification results

| Result | Action |
|--------|--------|
| correctness: PASS, speedup > 1.0 | Success. Generate final report. |
| correctness: PASS, speedup <= 1.0 | Kernel overhead is too high. Review replacement strategy. |
| correctness: FAIL | Run diagnosis mode to find the culprit. |

### C3. Diagnose failures

If correctness fails:

```bash
uv run verify.py --model <path> --class-name <name> --input-shape <shape> --diagnose
```

This tests each kernel replacement individually. The output tells you which kernel caused
the failure. Fix that kernel:

1. Copy the problematic optimized kernel back to `kernel.py`
2. Relax its optimization (e.g., use fp32 accumulator instead of fp16)
3. Re-run the single-kernel correctness check: `uv run bench.py > run.log 2>&1`
4. Save the fixed kernel back to workspace
5. Re-run verification

### C4. Generate final report

```bash
uv run orchestrate.py report
```

### C5. Analysis (optional)

After experiments, you can run `uv run analysis.py` to generate:
- `progress.png` -- visual plot of all experiments with the research frontier
- `report.md` -- markdown summary of the session
- Terminal summary with top improvements and statistics

---

## Optimization Playbook

Work through these tiers roughly in order. Earlier tiers give larger gains with less risk.
Later tiers require more expertise but can unlock the final 10-20%.

### Tier 1: Block Size Tuning

The single most impactful change for most kernels. Block sizes control tile dimensions and
directly affect occupancy, register pressure, and shared memory usage.

**What to try:**
- Sweep BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K through powers of 2: 16, 32, 64, 128, 256.
- For matmul-like kernels, try rectangular tiles (e.g., 128x64 instead of 64x64).
- Larger blocks = more work per thread block = better arithmetic intensity, but higher register pressure.
- Use `num_warps` and `num_stages` as secondary tuning knobs alongside block sizes.

**Typical gains**: 10-50% from finding the right block size vs the default.

### Tier 2: Memory Access Optimization

Once block sizes are tuned, memory is usually the bottleneck.

**Coalescing:**
- Ensure threads in the same warp access consecutive memory addresses.
- For matmul, this means loading along the contiguous dimension (stride-1).
- Transpose one operand if needed to make both loads coalesced.

**Prefetching:**
- Use `tl.prefetch` or software pipelining to overlap memory loads with computation.
- Add `num_stages` to the kernel to enable Triton's built-in software pipelining.
- Typical: `num_stages=3` or `num_stages=4` for matmul.

**L2 Cache Swizzling:**
- Reorder tile indices so neighboring thread blocks access nearby memory.
- Group tiles along the K dimension to maximize L2 cache reuse.

**Shared Memory Bank Conflicts:**
- 32 banks, 4 bytes wide on NVIDIA GPUs. Add 1 element of padding per row.

**Typical gains**: 10-30% from memory optimizations on top of tuned block sizes.

### Tier 3: Compute Optimization

**TF32 and Mixed Precision:**
- Use `tl.dot(a, b, allow_tf32=True)` for matmul accumulation with TF32 inputs.
- Keep accumulators in fp32 for numerical stability.
- Cast results to output dtype only at the end.

**Fused Operations:**
- Fuse elementwise operations (bias add, activation, scaling) into the kernel epilogue.
- Avoid writing intermediate results to global memory.

**Instruction-Level Optimization:**
- Minimize operations in the inner loop. Hoist invariant computations outside.
- Use `tl.where` instead of branches where possible.

**Typical gains**: 5-15% from compute optimizations.

### Tier 4: Advanced Techniques

**Split-K:**
- Decompose the K dimension across multiple thread blocks.
- Helps when M and N are small (not enough parallelism from spatial tiles alone).

**Persistent Kernels:**
- Launch exactly as many thread blocks as there are SMs on the GPU.
- Each block loops over multiple tiles instead of processing just one.
- Eliminates launch overhead and improves L2 cache utilization.

**Autotune:**
- Use `@triton.autotune` with multiple `triton.Config` configurations.
- Let Triton search over block sizes, num_warps, and num_stages.

**Warp Specialization:**
- Assign different warps to different roles (producers vs consumers).

**Register Tiling:**
- Manually control register allocation via constexpr tile sizes.
- Larger register tiles increase ILP but can cause register spilling.

**Typical gains**: 5-20% from advanced techniques, but higher risk.

### Tier 5: Architecture-Specific Optimizations

**H100 (Hopper, SM90):**
- TMA (Tensor Memory Accelerator): hardware-accelerated bulk copies.
- WGMMA (Warp Group Matrix Multiply Accumulate): next-gen tensor core instructions.
- Cluster-level shared memory.

**A100 (Ampere, SM80):**
- Async global-to-shared memory copies (`cp.async`).
- TF32 tensor cores (19.5 TFLOPS).
- Fine-grained structured sparsity (2:4).

**L40S / L4 / RTX (Ada Lovelace / Ampere consumer):**
- Smaller shared memory, fewer SMs. Use smaller block sizes, fewer stages.
- L40S: 142 SMs, good FP16 throughput.
- L4: very memory-bandwidth limited.
- RTX 4090: 128 SMs but consumer-grade memory bandwidth.

**MI300X (CDNA3, gfx942):** *(source: [GEAK](https://github.com/AMD-AGI/GEAK) knowledge-base)*
- 304 CUs, wavefront size = 64 (not 32). All warp-level reasoning must use 64-wide.
- HBM3: 5.3 TB/s peak bandwidth, 192 GB capacity. Memory-bound kernels should target >70% of this.
- MFMA (Matrix Fused Multiply-Add) instructions: 1307 TFLOPS FP16/BF16. Use `tl.dot` which maps to MFMA.
- LDS: 64 KB per CU, shared memory in Triton maps to LDS. Avoid >64 KB per block.
- BLOCK_SIZE = 256 is often optimal for CDNA (vs 128 for NVIDIA). Try 128/256 in autotune configs.
- `num_warps`: on AMD each "warp" in Triton is a wavefront of 64 threads. 4 warps = 256 threads.
- `waves_per_eu`: controls occupancy hint. Add to `triton.Config(kwargs, num_warps=N, waves_per_eu=M)`.
  Typical values: 0 (auto), 2-4 for compute-bound, 4-8 for memory-bound.
- `tl.math.tanh` is NOT available on HIP backend. Use sigmoid identity: `2*tl.sigmoid(2*x) - 1`.
- No `cp.async` or TMA equivalent; Triton compiler handles async prefetch internally.
- Prefer BF16 over FP16 for training workloads (same throughput, better dynamic range).
- Profile with `rocprof --hip-trace` or `rocprofv3`. Key counters: `SQ_INSTS_MFMA_MOPS_FP16`,
  `TCC_HIT/TCC_REQ` (L2 hit rate), `SQ_WAVE_CYCLES` (occupancy), `TCP_PENDING_STALL_CYCLES` (mem stall).

**Typical gains**: 5-15% from architecture-specific tuning.

### Tier 6: Kernel-Specific Tricks

**Matrix Multiplication (matmul):**
- Swizzle tile ordering for L2 reuse.
- Epilogue fusion (bias, activation, scaling).
- Split-K for tall-skinny matrices.

**Softmax:**
- Two-pass online softmax (track running max and sum in one pass).
- Multi-row processing: process multiple rows per thread block.

**LayerNorm / RMSNorm:**
- Welford's online algorithm for numerically stable variance.
- Fuse weight and bias application into the kernel.
- Multi-row processing for better occupancy.

**Flash Attention:**
- Online softmax with running statistics.
- Block-sparse patterns for long sequences.
- Causal masking with early termination.

**Cross Entropy:**
- Online log-sum-exp for numerical stability.
- Fuse with label indexing to avoid materializing the full logit tensor.

**Rotary Embeddings (RoPE):**
- Fuse with Q/K projection.
- Vectorized sin/cos computation.
- Precompute and cache frequency tables.

### Multi-Kernel Optimization Additions

When optimizing multiple kernels in sequence, you gain cross-kernel insights:

- **Shared block sizes**: If BLOCK_SIZE=128 works well for matmul, try 128 for layernorm and attention too.
- **Data layout awareness**: If you change memory layout for one kernel, consider downstream impact.
- **Fusion opportunities**: After individual kernels are optimized, look for fusion opportunities (e.g., matmul + layernorm).
- **Consistent precision strategy**: Use the same precision across kernels to avoid numerical drift.

### Anti-Patterns (Things That Usually Do Not Work)

- **Extremely large block sizes** (512+): Register spill destroys performance.
- **Too many `num_stages`** (>5): Shared memory overflow.
- **Unnecessary `tl.debug_barrier`**: Memory fences serialize execution.
- **Manual unrolling when Triton already unrolls**: Triton's compiler handles constexpr loop unrolling.
- **Premature use of `atomic_add`**: Only use for split-K reductions.
- **Ignoring alignment**: Misaligned loads waste half the bandwidth.
- **Over-complex control flow in inner loops**: Branches inside the K-loop kill performance.

---

## CUDA C++ Optimization Playbook

When using the CUDA C++ backend (`--backend cuda`), the agent edits raw CUDA C++ source
embedded as a Python string in `kernel.py`. The `CUDA_SRC` variable contains the kernel code,
compiled at runtime via `torch.utils.cpp_extension.load_inline()`.

**The kernel contract is the same**: `kernel.py` exports `KERNEL_TYPE`, `BACKEND = "cuda"`,
and `kernel_fn()` with the same signature. `bench.py` runs identically on either backend.

### CUDA Tier 1: Thread/Block Configuration

The most impactful change. Thread count and block dimensions directly control occupancy,
register pressure, and shared memory usage.

**What to try:**
- Sweep `blockDim.x` through 128, 256, 512. More threads = better latency hiding but higher register pressure.
- Use `dim3` for 2D/3D thread blocks when the kernel has spatial structure (e.g., matmul tiles).
- Set `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` to control register allocation.
- Calculate occupancy: `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

**Typical gains**: 10-50% from finding the right thread/block config.

### CUDA Tier 2: Shared Memory Tiling

**Tiling strategy:**
- Load tiles of input from global memory into `__shared__` memory.
- Process the tile using fast shared memory reads.
- Bank-conflict-free layout: add padding (e.g., `__shared__ float tile[32][33]` instead of `[32][32]`).
- Double buffering: use two shared memory buffers, load next tile while computing on current.

**Vectorized loads:**
- Use `float4` for 128-bit loads (4x throughput vs scalar `float`).
- Use `__ldg()` for read-only data through the texture cache.
- Align memory accesses to 128 bytes for maximum coalescing.

**Typical gains**: 20-40% from shared memory tiling over naive global memory access.

### CUDA Tier 3: Tensor Cores

**wmma API (Warp Matrix Multiply Accumulate):**
```cpp
#include <mma.h>
using namespace nvcuda;
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
wmma::fill_fragment(c_frag, 0.0f);
wmma::load_matrix_sync(a_frag, shared_A, ldA);
wmma::load_matrix_sync(b_frag, shared_B, ldB);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(output, c_frag, ldC, wmma::mem_row_major);
```

- Tile sizes: 16x16x16 (fp16), 8x8x4 (tf32), 8x8x32 (int8).
- Each warp processes one wmma tile. Assign multiple tiles per warp for ILP.
- Keep accumulator in fp32 for numerical stability, convert to fp16 at epilogue.

**Typical gains**: 2-4x over scalar code for matmul-like operations.

### CUDA Tier 4: Advanced Techniques

**Persistent kernels:**
- Launch exactly `SM_COUNT` thread blocks. Each loops over multiple tiles.
- Eliminates kernel launch overhead and improves L2 reuse.
- Use `atomicAdd` or cooperative groups for global synchronization.

**Software pipelining / cp.async (Ampere+):**
```cpp
// Asynchronous global -> shared memory copy
asm volatile("cp.async.cg.shared.global [%0], [%1], %2;" :: "r"(smem_addr), "l"(gmem_addr), "n"(16));
asm volatile("cp.async.commit_group;");
asm volatile("cp.async.wait_group 0;");
```

**Cooperative groups:**
- `cooperative_groups::this_grid()` for grid-level synchronization.
- `cooperative_groups::tiled_partition<32>(block)` for warp-level primitives.

**Typical gains**: 10-20% from advanced techniques.

### CUDA Tier 5: Architecture-Specific

**Hopper (SM90, H100):**
- TMA (Tensor Memory Accelerator): `cp.async.bulk` for hardware-accelerated bulk copies.
- Warp group MMA (WGMMA): next-gen tensor core with cluster-level shared memory.
- Thread Block Clusters: cooperative thread blocks sharing distributed shared memory.

**Ampere (SM80, A100):**
- `cp.async`: async global-to-shared memory copies (non-blocking).
- TF32 tensor cores: 19.5 TFLOPS with tf32 precision.
- Fine-grained structured sparsity (2:4 pattern).

**Ada Lovelace (SM89, RTX 4090/L40S):**
- FP8 tensor cores for inference.
- Smaller shared memory per SM -- use smaller tiles.
- High FP16 throughput but consumer-grade memory bandwidth.

**AMD CDNA3 (gfx942, MI300X) — HIP backend:** *(source: [GEAK](https://github.com/AMD-AGI/GEAK) knowledge-base)*
- Use `hipcc` with `-march=gfx942 -O3`. Compile flags: `--offload-arch=gfx942`.
- Wavefront = 64 threads (vs CUDA warp = 32). All `__shfl*` → `__shfl*` in HIP but width=64.
- MFMA intrinsics replace NVIDIA's WMMA/MMA. Use rocBLAS or rocWMMA for matrix ops.
- LDS: 64 KB per CU. Use `__launch_bounds__(256, 4)` for occupancy (256 threads, min 4 blocks/CU).
- VGPRs: 65536 per CU. Target 20-32 registers/thread for good occupancy (75-100%).
- `__restrict__` and `__builtin_nontemporal_load/store` for memory hint control.
- No `cp.async`; use `__builtin_amdgcn_global_load_lds()` for async LDS fills (advanced).
- Memory coalescing: same principle, 128-byte transactions, wavefront of 64 threads.
- Profile: `rocprof --stats`, `rocprofv3`. Key counters: `SQ_INSTS_MFMA_MOPS_*`, `TCC_*`, `SQ_WAVE_CYCLES`.

**Typical gains**: 5-15% from arch-specific tuning.

### CUDA Tier 6: Kernel-Specific Tricks

**Matrix multiplication:**
- Swizzled shared memory layout to avoid bank conflicts across warps.
- Register-level tiling: each thread accumulates a 4x4 or 8x8 output tile.
- Epilogue fusion: add bias, apply activation, quantize -- all in registers before writing.

**Softmax:**
- Warp-level `__shfl_xor_sync` tree reduction (5 steps for 32 threads).
- Multi-row processing: each thread block handles multiple rows for better occupancy.
- `__expf()` and `__fdividef()` fast math intrinsics.

**LayerNorm / RMSNorm:**
- Welford's online algorithm for single-pass mean+variance.
- Warp shuffle cascade (`__shfl_down_sync`) for partial statistics.
- `rsqrtf()` for fast inverse square root.
- Vectorized `float4`/`half2` loads.

**Flash Attention:**
- Double-buffered shared memory for Q/K/V tiles.
- Online softmax with running max and sum rescaling.
- Causal mask with early tile termination (skip tiles where all positions are masked).
- wmma for both Q@K^T and attn@V matmuls.

**Cross Entropy:**
- Fused online log-sum-exp: single pass for max, exp-sum, and target lookup.
- Warp-level max + sum reductions avoid shared memory overhead.
- `__logf()` / `__expf()` fast intrinsics.

**Rotary Embeddings:**
- `__sincosf()` for fused sin/cos computation.
- Constant memory for precomputed frequency table.
- Vectorized `half2` read-modify-write.

### CUDA Anti-Patterns

- **Warp divergence in inner loops**: All threads in a warp must take the same branch.
- **Uncoalesced global memory access**: Threads in a warp must access consecutive addresses.
- **Shared memory bank conflicts**: 32 banks, 4 bytes each. Pad or swizzle to avoid.
- **Register spilling**: Too many local variables cause spills to slow local memory. Use `__launch_bounds__`.
- **Excessive `__syncthreads()`**: Each sync stalls the entire block. Minimize sync points.
- **Global atomics in hot paths**: Use warp-level reduction first, then one atomic per warp.
- **Forgetting `__restrict__`**: Without it, the compiler assumes pointers may alias, blocking optimizations.
- **Using `printf` in production kernels**: Serializes execution.

---

## Decision Framework

### When to move on from a kernel

The orchestrator uses these criteria, but you should understand them:

1. **Target reached**: Kernel achieves the speedup target (e.g., 2x).
2. **Plateau detected**: Last 10-15 experiments all failed to improve throughput.
3. **Diminishing returns**: Optimizing the next kernel would yield more total benefit.
4. **Time budget**: If a per-kernel time budget was set, respect it.

### When to revisit a previous kernel

1. **New techniques discovered**: A technique found on a later kernel could also benefit an earlier one.
2. **Integration failure**: End-to-end verification reveals correctness issues from an earlier optimization.
3. **Architecture insight**: Later profiling reveals suboptimal memory access pattern in an earlier kernel.

### When to try radical changes vs incremental

| Situation | Strategy |
|-----------|----------|
| Early (experiments 0-10) | Aggressive: large block size changes, different algorithms |
| Mid (experiments 10-30) | Focused: systematic sweeps of promising parameters |
| Late (experiments 30+) | Incremental: fine-tuning, combining successful techniques |
| Plateau with low roofline (<50%) | Radical: fundamentally different approach |
| Plateau with high roofline (>80%) | Accept: close to hardware limits |

### Estimating end-to-end impact

Use Amdahl's law: `End-to-end speedup = 1 / ((1 - f) + f/s)` where f = fraction of total
model time in this kernel, s = kernel speedup.

A 1.5x speedup on a 60% kernel (1.25x end-to-end) is more valuable than a 3x speedup on a
5% kernel (1.03x end-to-end).

---

## Workspace Layout

```
workspace/
  orchestration_state.json     -- master state file tracking all kernels
  profile_report.json          -- model profiling results
  kernel_matmul_1.py           -- extracted kernel (rank 1)
  kernel_attention_2.py        -- extracted kernel (rank 2)
  kernel_matmul_1_optimized.py -- optimized version (output)
  verification_result.json     -- end-to-end verification output
```

---

## Supported Kernel Types

Two backends are available: **Triton** (default) and **CUDA C++** (`--backend cuda`).

```
kernels/                    Triton starters (Python + @triton.jit)
  matmul.py                   -- tiled matrix multiplication
  softmax.py                  -- row-parallel online softmax
  layernorm.py                -- layer normalization
  rmsnorm.py                  -- RMS normalization
  flash_attention.py          -- flash attention (block-wise online softmax)
  fused_mlp.py                -- fused SwiGLU MLP
  cross_entropy.py            -- fused cross entropy loss
  rotary_embedding.py         -- rotary position embeddings
  reduce.py                   -- parallel reduction (sum)

kernels/cuda/               CUDA C++ starters (tensor core accelerated)
  _compile.py                 -- shared compilation utility (load_inline + caching)
  matmul.py                   -- wmma tensor core GEMM, double-buffered shared memory
  softmax.py                  -- warp shuffle reduction, __expf fast math
  layernorm.py                -- Welford's online algorithm, vectorized float4 loads
  rmsnorm.py                  -- single-pass RMS, warp shuffle cascade
  flash_attention.py          -- tiled online softmax, causal mask, SRAM blocking
  fused_mlp.py                -- fused gate+up+silu, shared memory tiling
  cross_entropy.py            -- online log-sum-exp + NLL, warp reductions
  rotary_embedding.py         -- __sincosf intrinsic, vectorized half2
  reduce.py                   -- hierarchical warp shuffle + shared memory
```

Both backends export the same `KERNEL_TYPE` and `kernel_fn()` interface. `bench.py` runs
identically regardless of backend. Choose Triton for fast iteration, CUDA C++ for maximum
performance (direct tensor core access, PTX intrinsics, bank-conflict-free layouts).

---

## Error Handling

### Model loading failures
- Check the model path and class name
- Ensure all dependencies are installed
- For HuggingFace models, check authentication (`huggingface-cli login`)
- Try with `--dtype float32` if precision is causing issues

### Profile failures (OOM)
- Reduce input shape: `--input-shape 1,512` instead of `1,2048`
- Use `--dtype float16` if not already

### Kernel extraction failures
- Some ops cannot be mapped to standard kernel types -- skip them
- Document the skip in the optimization plan

### Orchestrator conflicts

If the orchestration state becomes corrupted:

```bash
cat workspace/orchestration_state.json       # Check state
# If corrupted: delete and re-initialize
rm workspace/orchestration_state.json
uv run extract.py --top 5                    # Re-generates plan + state
```

### Verification OOM

```bash
uv run verify.py --model <path> --class-name <name> --input-shape 1,512
uv run verify.py --model <path> --class-name <name> --input-shape 1,2048 --warmup 3 --timed 10
```

### Timeouts

Each benchmark should complete in ~90 seconds. If a run exceeds 3 minutes, it is hung.

**Action on timeout:**
1. Kill the process: `kill %1` or `pkill -f bench.py`
2. Revert: `git reset --hard HEAD~1`
3. Log in results.tsv with `throughput_tflops=0`, `correctness=TIMEOUT`
4. Move on.

### Crashes

1. Read the error: `tail -n 50 run.log`
2. **Trivial bug** (typo, missing import): fix it, amend the commit, re-run.
3. **Fundamentally broken** (OOM, can't compile): revert, log as crash, move on.
4. **Same crash 3 times in a row**: Stop trying that approach. Try something different.

### Cross-kernel crashes
- Should not happen (kernels are independent)
- Always start each kernel from a clean `cp workspace/kernel_<type>_<rank>.py kernel.py`

### Full pipeline timeout
- Save whatever kernels are optimized so far
- Run verification on the partial set
- Report partial results -- even 2 optimized kernels are valuable

---

## Constraints

These are hard rules. Violating any of them is a bug.

1. **Never modify `bench.py`**. This is the fixed evaluation harness.
2. **Never modify `reference.py`**. These are the correctness oracles.
3. **Never modify `prepare.py`**. This handles one-time setup.
4. **Never modify `verify.py`**. This is the end-to-end verification harness.
5. **Never modify `profile.py`** or **`extract.py`**. These are the analysis tools.
6. **Never modify `orchestrate.py`**. This is the orchestration engine.
7. **Never add dependencies**. You can only use what is already in `pyproject.toml`.
8. **Never skip correctness**. Every experiment must pass correctness checks.
9. **Simpler code wins when performance is equal**.
10. **VRAM must not exceed 80% of GPU memory**. If `peak_vram_mb` exceeds 80%, treat as regression and revert.
11. **Do not commit `results.tsv`** or **`run.log`** -- leave them untracked.
12. **Save optimized kernels to workspace**. Always copy `kernel.py` to `workspace/kernel_<type>_<rank>_optimized.py` before moving to the next kernel.
13. **Record every experiment with the orchestrator**.
14. **Respect orchestrator decisions**. If it says move on, move on.
15. **One kernel at a time**. Do not try to optimize two kernels simultaneously.

---

## Example: Full LLaMA 7B Optimization Run

### Phase A (with human, ~15 minutes)

```
Human: Optimize LLaMA 7B. Model at models/llama_7b.py, class LlamaModel.
       Input shape 1,2048, float16. Budget: 8 hours.

Agent: [runs profile.py, presents bottleneck summary]
       Top 3: matmul (62%), attention (18%), rmsnorm (9%)
       Plan: matmul ~4h, attention ~2.5h, rmsnorm ~1.5h
       Estimated max end-to-end speedup: 1.7-1.8x