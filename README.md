# AutoKernel

[![Discord](https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white)](https://discord.gg/UfEyc72t)

**Autoresearch for GPU kernels.** Give it any PyTorch model, go to sleep, wake up to optimized Triton or CUDA C++ kernels.

![AutoKernel Progress](progress.png)

Inspired by [@karpathy/autoresearch](https://github.com/karpathy/autoresearch) -- which demonstrated autonomous AI agents for LLM training research. AutoKernel applies the same philosophy to GPU kernel optimization: agent modifies one file, runs a fixed evaluation, keeps or reverts, repeats forever.

## How It Works

Give AutoKernel any PyTorch model. It will:

1. **Profile** the model to find which GPU kernels are bottlenecks
2. **Extract** each bottleneck as a standalone Triton or CUDA C++ kernel
3. **Optimize** each kernel autonomously (edit, benchmark, keep/revert -- forever)
4. **Verify** end-to-end correctness and report the total speedup

The agent reads `program.md` -- the "research org code" -- which contains comprehensive instructions for autonomous operation. It edits `kernel.py` one kernel at a time, runs `bench.py` (fixed benchmark with 5-stage correctness checks + roofline analysis), and either keeps or reverts the change. The orchestrator decides when to move to the next kernel using Amdahl's law.

Each experiment takes ~90 seconds. That's ~40 experiments/hour, ~320 overnight, across all kernels.

## Quick Start

**Requirements:** NVIDIA GPU (tested on H100/A100/RTX 4090), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/RightNow-AI/autokernel.git
cd autokernel
uv sync

# One-time setup: test data + baselines
uv run prepare.py

# Profile a model (ships with GPT-2, LLaMA, BERT -- no transformers needed)
uv run profile.py --model models/llama_7b.py --class-name LlamaModel \
 --input-shape 1,512 --dtype float16

# Extract top bottleneck kernels
uv run extract.py --top 5

# Verify benchmark works
uv run bench.py
```

## Running the Agent

Spin up Claude, Codex, or any coding agent in this directory:

```
Read program.md and let's kick off a new experiment. Start with setup.
```

The agent will:
1. Profile your model and present the optimization plan
2. Create a branch (e.g., `autokernel/mar10-llama7b`)
3. Optimize each bottleneck kernel in priority order
4. Verify end-to-end correctness and report total speedup

`program.md` is intentionally comprehensive so the agent can run 10+ hours without getting stuck. It includes a 6-tier optimization playbook, decision framework, crash handling, and Amdahl's law reasoning.

## The Pipeline

```
                 profile.py              extract.py           bench.py (loop)         verify.py
Any PyTorch  ──>  Rank kernels  ──>  Generate baseline  ──>  Optimize each  ──>  End-to-end
   model          by GPU time       Triton/CUDA kernels     kernel (agent)       verification
```

| Tool | What it does |
|------|-------------|
| `profile.py` | Profiles any PyTorch model with `torch.profiler`, ranks kernels by GPU time, classifies as compute/memory-bound |
| `extract.py` | Extracts top-N bottleneck kernels into standalone Triton or CUDA C++ kernel files (`--backend triton\|cuda`) |
| `orchestrate.py` | Multi-kernel scheduler: decides which kernel to optimize next using Amdahl's law, tracks aggregate progress |
| `bench.py` | Fixed benchmark: 5-stage correctness (smoke, shape sweep, numerical stability, determinism, edge cases) + performance + roofline |
| `verify.py` | Plugs optimized kernels back into the model, checks end-to-end correctness, reports total speedup |
| `calibrate.py` | Measures achieved HBM bandwidth and FP16/BF16 GEMM throughput on a new GPU and compares them to the roofline peaks in the GPU database |

## Supported Kernels

9 kernel types covering the core operations of modern deep learning:

| Kernel | Description | Key Metric |
|--------|-------------|------------|
| **matmul** | Dense matrix multiplication (M x K) @ (K x N) | TFLOPS |
| **softmax** | Row-parallel numerically stable softmax | GB/s |
| **layernorm** | Layer normalization with affine transform | GB/s |
| **rmsnorm** | RMS normalization (LLaMA-style) | GB/s |
| **flash_attention** | Scaled dot-product attention with causal masking | TFLOPS |
| **fused_mlp** | SwiGLU-style fused MLP (gate + up + down) | TFLOPS |
| **cross_entropy** | Fused cross entropy loss | GB/s |
| **rotary_embedding** | Rotary position embeddings (RoPE) | GB/s |
| **reduce** | Parallel reduction (sum) | GB/s |

Each has a PyTorch reference in `reference.py`, a starter Triton kernel in `kernels/`, and a starter CUDA C++ kernel in `kernels/cuda/`.

## Example Models

Self-contained model definitions ship with AutoKernel (no `transformers` library needed):

| Model | File | Params | Usage |
|-------|------|--------|-------|
| GPT-2 Small | `models/gpt2.py` | 124M | `--class-name GPT2 --input-shape 1,1024` |
| LLaMA (compact) | `models/llama_7b.py` | 160M | `--class-name LlamaModel --input-shape 1,512` |
| LLaMA 7B | `models/llama_7b.py` | 7B | `--class-name LlamaModel7B --input-shape 1,2048` |
| BERT-base | `models/bert_base.py` | 110M | `--class-name BertModel --input-shape 8,512` |
| Custom | `models/custom.py` | -- | Template for your own model |

For HuggingFace models (`uv sync --extra models`):

```bash
uv run profile.py --module transformers --class-name AutoModelForCausalLM \
 --pretrained meta-llama/Llama-2-7b-hf --input-shape 1,2048 --dtype float16
```

## KernelBench Integration

AutoKernel integrates with [KernelBench](https://github.com/ScalingIntelligence/KernelBench),
the standard benchmark for evaluating AI-generated GPU kernels (250+ problems across 4 difficulty
levels). While most KernelBench evaluations use one-shot LLM generation, AutoKernel runs
**50-300+ iterative refinement experiments per problem** -- systematically exploring the
optimization space instead of guessing.

```bash
# Install KernelBench dependencies
uv sync --extra kernelbench

# Fetch Level 1 problems from HuggingFace
uv run kernelbench/bridge.py fetch --source hf --level 1

# Set up a specific problem for optimization
uv run kernelbench/bridge.py setup --level 1 --problem 1 --source hf

# Evaluate (correctness + speedup vs PyTorch reference)
uv run kernelbench/bench_kb.py

# Batch score an entire level (computes fast_p metric)
uv run kernelbench/scorer.py --level 1
```

The agent reads `kernelbench/program_kb.md` for KernelBench-specific optimization instructions:
how to write `ModelNew` classes, when to use CUDA C++ vs Triton, fusion strategies per problem
level, and the edit-bench-keep/revert loop adapted for the KernelBench `fast_p` metric.

| Tool | What it does |
|------|-------------|
| `kernelbench/bridge.py` | Loads problems from HuggingFace or local repo, caches them, generates starter `kernel.py` |
| `kernelbench/bench_kb.py` | Evaluates `ModelNew` vs `Model`: 5-trial correctness + CUDA event timing + stability + determinism |
| `kernelbench/scorer.py` | Batch evaluation across a level, computes `fast_p` at thresholds (1.0x, 1.5x, 2.0x, 3.0x, 5.0x) |
| `kernelbench/program_kb.md` | Agent instructions for KernelBench mode |

## Engine Mode: Decode-Step Benchmark

Kernel speedups only matter if the serving engine's decode step gets faster. `engine/decode_step.py`
records **SSE event-gap latency** and usage-based throughput in a common decode window at requested
batch B. vLLM's output collector can merge outputs when a consumer falls behind: an SSE event
is not necessarily one engine step. `/metrics` supplies **sampled occupancy**, not scheduler-level
proof that exactly B requests ran at every instant.

Supply a local identity JSON for each arm. All fields below are required; values are
illustrative placeholders to replace with the actual measurement configuration.
`sm_count` is per GPU; `gpu_count` is the allocated GPU count. Use `method: "none"`,
`k: 0`, and `draft_revision: null` when speculation is disabled. Additional fields
are preserved and compared. The harness records declarations; it does not attest them.

```json
{
  "hardware": {"gpu_name": "NVIDIA H100", "gpu_count": 2, "sm_count": 132, "driver": "<driver version>"},
  "image": {"digest": "sha256:<actual image digest>"},
  "engine": {"version": "<engine version or commit>"},
  "model": {"id": "Qwen/Qwen3-8B", "revision": "<immutable revision>"},
  "parallelism": {"tp": 2, "dcp": 1, "ep": 1},
  "speculation": {"method": "none", "k": 0, "draft_revision": null},
  "kv_dtype": "auto",
  "arm": {"label": "stock"}
}
```

```bash
uv sync --extra engine

# Baseline twice (the A/A ratio is your noise floor), then the candidate
uv run engine/decode_step.py run --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B \
 --batch 1,2,4,8,16,32,64 --isl 1 --osl 1024 --label stock \
 --identity stock-identity.json --out results/stock.json
# Repeat the same workload for stock-repeat.json, then ours.json with its own identity.

uv run engine/decode_step.py compare results/stock.json results/stock-repeat.json
uv run engine/decode_step.py compare results/stock.json results/ours.json --arm-delta image.digest,arm.label
uv run engine/decode_step.py table results/stock.json results/ours.json --arm-delta image.digest,arm.label
```

- Prompts are random token IDs, so ISL is exact and the prefix cache cannot help.
- Tokens come from continuous `usage.completion_tokens`, including multi-token speculative chunks.
  Missing, non-increasing, or inconsistent usage invalidates the stream; chunk counts never substitute.
  A successful stream also requires `[DONE]`, a `stop` or `length` finish reason, the exact
  requested completion count, and observed prompt usage equal to the requested ISL. Truncated
  streams, missing terminal evidence and token-count mismatches invalidate the point with reasons.
- `/metrics` sums **all** matching `vllm:num_requests_running` series across engines, with a
  250 ms pause between polls. Every attempt is retained in `metric_polls`, including failures
  and request/response timestamps. The `occupancy` record states the coverage bounds and reasons:
  at least `max(2, ceil(window_s / 1.5) + 1)` distinct fresh samples; no sample-to-sample or
  window-boundary gap above **1.5 s**; a fresh sample's scrape completes within **1.0 s**; and the
  engine-series set remains present and stable. Failed or slow polls never count as fresh and are
  reported, but a lone one voids nothing while the gap bound holds, since `/metrics` competes with
  B streams on a busy API server. Missing,
  duplicate, negative or non-finite metric values are failed polls. Insufficient coverage makes
  `exact_b: null` / `UNVERIFIED`; adequate coverage with an observed count different from B gives
  `false`. Even `true` means only that this **sampled occupancy** check passed.
- `compare` and `table` require `exact_b_all: true`, coverage evidence, and usable event statistics,
  checking raw repetitions when present. Failures refuse with **exit 5**. `--allow-inexact`
  permits diagnostic output with affected rows and ratios explicitly marked.
- Compatibility keys `step_ms_p50`, `step_ms_p90` and `tokens_per_step` remain, with
  `step_basis: "sse-event-gap"`; tables say **event gap** and **tokens/event**. When identity
  declares `speculation.k == 0`, any event carrying more than one usage token (including the
  first event) is counted in `merged_events`. A positive count marks that point's `step_status`
  `UNVERIFIED` and blocks event-gap ratios unless `--allow-inexact` is passed. `EVENT_GAP` means
  only an event statistic, never a verified engine step. With speculation or unknown identity,
  merging cannot be inferred from token counts: `merged_events` is null and the limitation is
  recorded in `step_reason`. Cross-check latency with **graph-replay time from a profiler**.
- Both commands refuse different workload profiles, primary rate formulas, or undeclared identity
  differences with **exit 4**, naming the differing keys. `--arm-delta` lists exact dotted leaf
  keys, such as `image.digest,arm.label` or `speculation.method,speculation.k,speculation.draft_revision`.
  It grants no workload exceptions. Parent keys and unknown keys refuse; no differences are implicit.
- Without `--identity`, `run` warns and writes `identity: null`. Comparisons refuse missing identity
  with **exit 4** unless `--allow-missing-identity` is supplied; output then says `UNQUALIFIED`.
  An incomplete or malformed identity cannot use that override. `table` applies the same identity gate.
- `run --dry-run` prints the resolved measurement plan without HTTP calls or output writes.
  Each real warmup/timed run logs progress to `<out>.log` (or `decode-step.log` without `--out`);
  `--log` selects another log path. Keep credentials out of identity files, labels, URLs and notes.

New output uses schema `autokernel-decode-step.v2`. Each repetition includes `streams`, even
without `--keep-timelines`: `t_first_output_token`, `t_last_output_token` (client monotonic
seconds), usage token counts, and both `tok_s_n_over_dt` and `tok_s_n_minus_1_over_dt`.
Here **N is the full-stream usage count** and **dt is last output arrival minus first output
arrival**; usage-only trailers do not extend dt. Single-token or nonpositive intervals have null rates.
Each repetition's full-stream rates are the mean across its streams; summaries take the median
across valid repetitions. `rate_formula_primary` defaults to **`window`**: usage-token deltas inside
the common decode window divided by its duration (`window_decode_tok_s`), per user as
`tok_s_window_per_user`. The window starts at the latest first-token arrival and ends at the
earliest last-token arrival. For **each stream**, subtract the last cumulative usage **at or before
the start** from the last cumulative usage **at or before the end**; interpolate nothing. This
accounting is independent of the fully-contained-gap filter used for event latency and tokens/event.
`window_tokens` and `window_boundary_usage` retain the numerator and per-stream boundary counts;
`window_token_basis: "cumulative-at-boundaries"` identifies this corrected accounting.
Select `--rate-formula-primary '(N-1)/dt'` (production decode telemetry's
formula) or `'N/dt'` explicitly for the full-stream rates; these can include other requests'
prefill overlap. Tables name the selected formula; `tok_s_per_user` uses it and `decode_tok_s` is
B times it.

Old JSON still loads. Its missing identity needs the explicit override, and missing occupancy or
event-statistic evidence also requires `--allow-inexact`. Old rates retain the label
`legacy window delta/dt`; summaries cannot reconstruct first/last stream timestamps or missing
boundary tokens. Older files counted only tokens in fully-contained event gaps, so their window
rates **cannot compare with corrected boundary-based rates** (exit 4, `window_token_basis`).
Different primary formulas also refuse with exit 4: rerun both arms on the same accounting basis
before using their ratios. Exit 2 denotes bad run arguments or no valid runs;
exit 3 denotes an unreadable result or log. Overrides are diagnostic, not qualified evidence.

Offline regression gates (no server or GPU; includes refusal cases):

```bash
python -m pytest engine/test_decode_step.py
# If pytest is unavailable:
python -m unittest engine.test_decode_step -v
```

Run the client on the server's host, and cross-check event-gap measurements against nsys graph-replay time.

## HuggingFace Kernels Export

Export optimized kernels to the [HuggingFace Hub](https://huggingface.co/docs/kernels/en/index)
for easy distribution. Users can then load your kernels with a single line:

```python
from kernels import get_kernel
module = get_kernel("your-username/kernel-name")
```

```bash
# Export an optimized CUDA kernel
uv run export_hf.py --name my_matmul

# Upload to Hub (requires `pip install kernels` and `huggingface-cli login`)
cd workspace/hf_export/my_matmul
kernels upload . --repo_id your-username/my_matmul
```

## Project Structure

```
autokernel/
  kernel.py             the file the agent modifies (one kernel at a time)
  program.md            agent instructions -- the "research org code"

  bench.py              fixed benchmark + 5-stage correctness harness
  reference.py          PyTorch reference implementations (ground truth)
  prepare.py            one-time setup: test data, baselines

  profile.py            profile any PyTorch model, rank kernels by GPU time
  extract.py            extract bottleneck kernels into workspace/
  orchestrate.py        multi-kernel scheduler (Amdahl's law)
  verify.py             end-to-end model verification + speedup report
  export_hf.py          export optimized kernels to HuggingFace Kernels format
  analysis.py           experiment visualization (generates progress.png)

  kernels/              starter Triton kernels (9 types)
  kernels/cuda/         starter CUDA C++ kernels (9 types, tensor core accelerated)
  kernelbench/          KernelBench integration (bridge, eval harness, scorer)
  models/               self-contained model definitions (GPT-2, LLaMA, BERT)
  workspace/            runtime artifacts (gitignored)
```

## Design Choices

**Dual backend: Triton + CUDA C++.** Triton for fast iteration (Python-like syntax, compiles in seconds). CUDA C++ for maximum performance (direct access to tensor cores via `wmma`, PTX intrinsics, shared memory bank-conflict-free layouts). Triton regularly reaches 80-95% of cuBLAS; CUDA C++ can match or exceed it. Both backends share the same `kernel_fn()` interface -- `bench.py` runs identically on either.

**Correctness first.** The benchmark checks kernel output against PyTorch before measuring performance. A fast but wrong kernel is immediately reverted. This prevents the agent from "optimizing" by producing garbage.

**Amdahl's law orchestration.** The orchestrator prioritizes by impact. A 1.5x speedup on a 60% kernel (1.25x end-to-end) beats a 3x speedup on a 5% kernel (1.03x end-to-end). It moves on when diminishing returns set in.

**Single file to modify.** The agent only touches `kernel.py`. Scope stays manageable, diffs reviewable, reverts clean.

**TSV logging.** Results go to a plain `results.tsv` file. Human-readable, git-friendly, trivially parseable, no infrastructure.

## Results Format

Every experiment is logged to `results.tsv` (tab-separated):

| Column | Description |
|--------|-------------|
| `experiment` | Sequential experiment number (0 = baseline) |
| `tag` | Short identifier |
| `kernel_type` | Which kernel (e.g., `matmul`) |
| `throughput_tflops` | Measured throughput (higher is better) |
| `latency_us` | Execution time in microseconds |
| `pct_peak` | Percentage of GPU theoretical peak |
| `speedup_vs_pytorch` | Speedup vs PyTorch/cuBLAS |
| `correctness` | PASS, FAIL, TIMEOUT, or CRASH |
| `peak_vram_mb` | Peak GPU memory usage |
| `description` | What was tried |

## Credits

This project is **autoresearch for GPU kernels** -- directly inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch), the original experiment in autonomous AI research agents for LLM training. Karpathy showed that an AI agent can run hundreds of experiments overnight, methodically exploring a search space and logging every result. AutoKernel applies that same loop -- agent edits one file, runs a fixed evaluation, keeps or reverts -- to the domain of GPU kernel optimization with Triton and native CUDA C++.

**KernelBench** integration is based on the work of Simon Guo, Sean Resta, et al. at Stanford's Scaling Intelligence Lab. Their paper ["KernelBench: Can LLMs Write GPU Kernels?"](https://arxiv.org/abs/2502.10517) (2025) established the standard benchmark for evaluating AI-generated GPU kernels. AutoKernel extends this by applying iterative optimization (300+ experiments per problem) instead of one-shot generation. KernelBench dataset and evaluation protocol: [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench).

Built by [RightNow AI](https://www.rightnowai.co). For enterprise GPU optimization, check out [RightNow Enterprise](https://www.rightnowai.co/forge).

## Changelog

### v1.3.0
- AMD ROCm GPU support: MI300X, MI325X, MI350X, MI355X detection and specs (thanks [@andyluo7](https://github.com/andyluo7))
- Fixed `verify.py` SyntaxError on Python 3.13+
- Fixed CUDA flash_attention ignoring `sm_scale` parameter
- Fixed CUDA cross_entropy returning wrong dtype
- Fixed Triton rotary_embedding broadcasting truncation
- Fixed Triton reduce output shape for non-last-dim reductions

### v1.2.0
- Enhanced profiler: `--export-trace`, `--memory-snapshot`, `--torch-compile-log` flags
- HuggingFace Kernels export via `export_hf.py`

### v1.1.0
- Native CUDA C++ backend with 9 starter kernels (tensor cores, warp intrinsics, shared memory tiling)
- KernelBench integration (250+ standardized GPU kernel problems)
- `--backend triton|cuda` flag for `extract.py`

### v1.0.0
- Initial release: Triton kernel optimization pipeline with 5-stage correctness harness

See [CHANGELOG.md](CHANGELOG.md) for full details.

## License

MIT
