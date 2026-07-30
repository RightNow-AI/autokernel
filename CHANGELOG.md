# Changelog

## Unreleased (downstream)

### Model optimization campaigns

- Added a versioned, metadata-only campaign contract with strict validation,
  impact ranking, legacy orchestration-plan generation, and trusted
  starter-kernel preparation through `campaign.py`
- Added a one-command, time-bounded and resumable overnight campaign runner
  with per-target benchmark instructions, durable logs, terminal receipts, and
  a consolidated morning report

### MotionKernel identity

- Renamed the downstream distribution to MotionKernel and reset its independent
  package version to `0.1.0`
- Repositioned the project around verified GPU kernel optimization for video
  generation models, with FastVideo as the first target integration
- Preserved the `autokernel` Python import namespace as a temporary
  compatibility boundary and retained the upstream MIT license and attribution

### Custom operation registry

- Added the `autokernel` package with `autokernel/specs/`: a typed `KernelSpec`
  that owns one operation's reference, deterministic inputs, sizes, dtypes,
  tolerances, edge cases, FLOP/byte accounting, profiler shape aliases and
  starter kernels
- Added `KernelRegistry` with deterministic ordering, duplicate detection and
  per-command isolation (`create_builtin_registry()`); registry discovery imports
  no `torch` and initializes no GPU, so it works on CPU-only machines
- Migrated all nine built-in operations (`matmul`, `softmax`, `layernorm`,
  `flash_attention`, `fused_mlp`, `cross_entropy`, `rotary_embedding`,
  `rmsnorm`, `reduce`) to specifications; `bench.py` and `extract.py` now read
  metadata only from those specs
- Removed the duplicated metadata maps from `extract.py` (`SHAPE_KEYS`,
  `SHAPE_ALIAS_MAP`, `TOLERANCES_MAP`, `FLOPS_FN_SRC`, `BYTES_FN_SRC`,
  `SPEEDUP_ESTIMATES`, hard-coded default shapes). FLOP/byte accounting is a
  serializable expression tree instead of stored Python source strings
- Added `--spec LOCATOR` and `--spec-override` to `bench.py` and `extract.py`.
  Precedence is `--spec`, then `--kernel`, then `kernel.py::KERNEL_TYPE`, so
  existing invocations are unchanged. `--help` never imports an external spec
- Added `examples/custom_ops/add.py` (external spec) and
  `examples/custom_ops/add_kernel.py` (its starter kernel)
- Added a CPU test suite (`uv run pytest -m "not gpu"`) that freezes the
  built-in metadata against its pre-refactor values, plus a `dev` extra and a
  CI job to run it. GPU tests are marked `gpu`
- `bench.py` keeps a deprecated `KERNEL_CONFIGS` view derived from the registry
  for out-of-tree callers
- `Tolerance` rejects negative, NaN, and infinite `atol`/`rtol` values, so a
  malformed specification cannot silently disable the correctness gate
- All declared built-in, edge-case and default-shape dimensions must be
  positive integers; empty starter-kernel mappings are explicitly supported
  for benchmark-only external specifications

### Generalized verification

- Added deterministic comparison for tensor, tuple, list, dictionary,
  named-tuple and nested output trees, including exact metadata comparison and
  per-leaf NaN, infinity and error diagnostics
- Added versioned production shape corpora, append and corpus-only benchmark
  modes, and weighted aggregates that remain separated by dtype
- Added optional `BackwardSpec` gradient verification with deterministic
  upstream gradients and per-input diagnostics
- Added optional `CompileSpec` verification and `--check-compile`; candidates
  compile with full-graph mode by default, run at least twice, reuse one
  compiled callable for dynamic shapes and compare through the normal output
  tree gate outside performance timing
- Added `FORWARD_CORRECTNESS`, `BACKWARD_CORRECTNESS` and
  `COMPILE_CORRECTNESS` console verdicts
- Added schema-versioned, atomic JSON results under
  `workspace/bench_result.json`, configurable with `--result-json`
- Added `examples/custom_ops/affine.py`, its candidate and a metadata-only
  shape corpus as a structured-output, backward and compile fixture
- Made the float32 matmul starter request IEEE dot inputs instead of Triton's
  TF32 default, kept strict BF16 LayerNorm parity through an explicit PyTorch
  fallback pending a Welford Triton implementation, and made the affine
  fixture's residual rounding stable under Inductor fusion
- Kept the top-level `profile.py` CLI compatible with the standard-library
  `profile` API so importing `cProfile` and initializing `torch.compile` from
  the repository root no longer fails

### Wan kernel fusion

- Added the first production video-DiT operation specification: Wan's
  post-self-attention gated residual update plus FP32 affine LayerNorm
- Added specifications, production corpora, and Triton starters for Wan's
  modulated pre-attention LayerNorm and post-MLP gated residual, completing the
  first three-target Wan optimization pack
- Added a metadata-only shape corpus covering Wan 2.1 1.3B and 14B at common
  480p token counts, including four-way sequence-parallel layouts
- Added a structured-output Triton baseline that returns both the normalized
  activation and updated residual stream in the model dtype
- Validated the full production corpus on GB200 with all correctness stages
  passing and a weighted 8.638x operator speedup over eager PyTorch

## v1.3.0 -- 2026-03-13

### AMD ROCm GPU Support (PR #3 by @andyluo7)

- Added AMD Instinct MI300X, MI325X, MI350X, MI355X to GPU database with correct peak FP16 TFLOPS, memory bandwidth, and L2 cache specs
- Added `gcnArchName`-based GPU detection for ROCm (device name is often empty on ROCm; `gcnArchName` like `gfx942` is always available)
- Guarded `clock_rate` access behind `hasattr` + `> 0` check (ROCm devices report `clock_rate=0`)
- Applied same fixes to `profile.py` fallback detector
- Tested on AMD Instinct MI300X (gfx942, ROCm 6.3) and MI350X (gfx950, ROCm 7.2)

### Bug Fixes

- **Fixed `verify.py` SyntaxError on Python 3.13+**: moved `global` declaration before variable usage in `main()` -- file would not even import on Python 3.13/3.14
- **Fixed CUDA flash_attention `sm_scale` parameter being ignored**: the `sm_scale` argument was accepted but the kernel hardcoded `rsqrtf(D)` instead of using it -- now correctly passes `sm_scale` through to the CUDA kernel
- **Fixed CUDA cross_entropy returning wrong dtype**: loss was cast back to input dtype instead of always returning `float32` (matching `F.cross_entropy` behavior)
- **Fixed Triton rotary_embedding broadcasting truncation**: `cos`/`sin` repeat used integer division which truncated when `n_rows` was not a multiple of `cos.shape[0]` -- now uses ceiling division and slices to exact size
- **Fixed Triton reduce output shape for non-last-dim reductions**: after permuting to move the reduce dim last, the output was reshaped using the original dim order instead of the permuted order

## v1.2.0 -- 2026-03-12

### Enhanced Profiler (Issue #1)

- Added `--export-trace` flag to export Chrome trace JSON for HTA/trace-blame analysis
- Added `--memory-snapshot` flag to capture CUDA memory snapshots for mosaic analysis
- Added `--torch-compile-log` flag to save torch.compile logs for tlparse analysis
- Added optional HTA (Holistic Trace Analysis) integration -- when installed, runs temporal and kernel breakdown analysis
- Added exported artifacts summary with suggested next steps for each tool
- Added `HolisticTraceAnalysis` as optional `profiling` dependency

### HuggingFace Kernels Export (Issue #2)

- Added `export_hf.py` -- exports optimized kernels to HuggingFace Kernels format
- Supports CUDA C++ kernels: auto-extracts CUDA source, parses function signatures, generates `build.toml` + `torch_binding.cpp` + `__init__.py`
- Supports Triton kernels: packages as a Python module with pyproject.toml
- Generates ready-to-upload project structure compatible with `kernels upload` CLI
- Added `kernels` and `huggingface-hub` as optional `hf-kernels` dependencies

## v1.1.0 -- 2026-03-12

### Native CUDA C++ Backend

- Added 9 CUDA C++ starter kernels with advanced GPU features:
  - **matmul** -- Tensor core GEMM via `wmma` API, 128x128 tiles, double-buffered shared memory
  - **softmax** -- Warp shuffle reductions, `half2` vectorized loads, grid-stride loop
  - **layernorm** -- Welford's single-pass algorithm, `float4` vectorized loads, warp shuffle stats
  - **rmsnorm** -- Warp shuffle cascade, `rsqrtf` fast inverse sqrt, `half2` vectorization
  - **flash_attention** -- Tiled online softmax, double-buffered shared memory, causal mask support
  - **fused_mlp** -- Fused SwiGLU (gate + up + SiLU + mul), shared memory tiling
  - **cross_entropy** -- Fused online log-sum-exp + NLL in single pass, warp reductions
  - **rotary_embedding** -- `__sincosf` intrinsic, `half2` read-modify-write
  - **reduce** -- Hierarchical warp shuffle + shared memory + grid-level atomic
- Added `kernels/cuda/_compile.py` -- shared compilation utility:
  - Hash-based caching (recompile only when source changes)
  - GPU architecture auto-detection via `torch.cuda.get_device_capability()`
  - Forward declaration extraction for cross-translation-unit linking
  - Thread-safe compilation with file locking
  - Detailed error diagnostics with source line numbers
- Added `--backend triton|cuda` flag to `extract.py`
- Added CUDA C++ optimization playbook to `program.md`
- Added `ninja` as optional dependency for faster compilation

### KernelBench Integration

- Added `kernelbench/bridge.py` -- problem loader supporting 3 sources:
  - HuggingFace datasets (`--source hf`)
  - Local KernelBench repo clone (`--source local`)
  - Individual Python files (`--source file`)
  - Automatic problem analysis (50+ operation patterns)
  - Starter `ModelNew` generation with CUDA/Triton templates
- Added `kernelbench/bench_kb.py` -- 4-stage evaluation pipeline:
  - Stage 1: Correctness (5 random input trials, atol/rtol=1e-2)
  - Stage 2: Stability (NaN/Inf detection)
  - Stage 3: Determinism (3 identical runs)
  - Stage 4: Performance (CUDA event timing, trimmed median)
  - Greppable output with `fast_p` at 7 thresholds
- Added `kernelbench/scorer.py` -- batch evaluation and metrics:
  - `fast_p` metric at thresholds: 1.0x, 1.1x, 1.25x, 1.5x, 2.0x, 3.0x, 5.0x
  - Incremental scoring with JSON persistence
  - Leaderboard-style reports with progress bars
- Added `kernelbench/program_kb.md` -- agent instructions for KernelBench mode:
  - Optimization playbook per difficulty level (L1-L4)
  - CUDA C++ and Triton strategy examples
  - Decision framework and anti-patterns

### Other

- Updated README with KernelBench section, dual-backend docs, and Discord link
- Added `datasets>=2.16.0` as optional `kernelbench` dependency

## v1.0.0 -- Initial Release

- Triton kernel optimization pipeline (profile, extract, bench, orchestrate, verify)
- 9 starter Triton kernels (matmul, softmax, layernorm, rmsnorm, flash_attention, fused_mlp, cross_entropy, rotary_embedding, reduce)
- 5-stage correctness harness + roofline analysis
- Amdahl's law orchestration for multi-kernel optimization
- Self-contained model definitions (GPT-2, LLaMA, BERT)
- TSV logging and experiment visualization
