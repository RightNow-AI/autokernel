# MotionKernel Roadmap

MotionKernel is the video-kernel optimization layer, not a competing
video-generation runtime. It discovers and validates kernels that frameworks
such as FastVideo and Diffusers can consume.

Detailed execution instructions for the first two milestones are available in
[docs/WEEK_1_2_AGENT_BRIEF.md](docs/WEEK_1_2_AGENT_BRIEF.md).

## Milestone 0: standalone foundation

- Preserve upstream provenance and MIT attribution.
- Maintain separate `origin` and `upstream` remotes.
- Establish the MotionKernel identity while retaining a compatible
  `autokernel` import namespace.
- Establish safe contribution and experiment practices.
- Add lightweight CPU-only validation for every change.

## Milestone 1: custom-operation platform

- Introduce a `KernelSpec` registry.
- Load operation specifications without editing the core benchmark.
- Move existing built-in operations onto the same registry.
- Define stable interfaces for reference functions, inputs, cases, tolerances,
  output comparison, performance metrics, and integration hooks.

Exit criterion: a new single-output operation can be optimized from an external
specification without changes to `bench.py`, `extract.py`, or `reference.py`.

## Milestone 2: production verification

- Compare tensor, tuple, and nested outputs.
- Add optional backward and gradient verification.
- Add deterministic execution checks.
- Add `torch.compile` full-graph compatibility checks.
- Support model-specific replacement adapters.
- Record GPU, software, shape, and benchmark metadata with every result.

Exit criterion: a custom multi-output operation can be validated in isolation
and inside a model, including backward execution when requested.

## Milestone 3: video and diffusion transformer kernels

- Add modulated LayerNorm and RMSNorm.
- Add gated residual updates.
- Add combined gated-residual, normalization, and modulation.
- Cover affine and non-affine variants.
- Cover batch, frame, token, and spatial broadcast layouts.
- Tune FP16 and BF16 paths using FP32 accumulation.

Exit criterion: promoted kernels pass correctness and compile gates and show a
meaningful speedup on production shape distributions.

## Milestone 4: model adoption

- Integrate and benchmark Wan through FastVideo.
- Integrate and benchmark LTX-Video.
- Integrate and benchmark Cosmos.
- Integrate and benchmark Kandinsky.
- Validate single-GPU and sequence-parallel execution.

Runtime integrations will use exported kernels with native PyTorch fallbacks;
they will not require the optimization platform at inference time.

## Milestone 5: continuous kernel research

- Run parallel searches across GPU workers.
- Maintain architecture-specific tuning records.
- Track performance regressions between revisions.
- Promote candidates through experimental, validated, and production stages.
- Expand into attention, MLP, quantization, and communication-aware fusion.

## Long-term: VideoKernelBench and ecosystem adoption

- Publish reproducible operator and end-to-end video workload benchmarks.
- Track latency, throughput, memory, compile time, numerical accuracy, and
  output-quality regressions across hardware generations.
- Maintain FastVideo and Diffusers adapters with stable replacement APIs.
- Grow shared and model-specific kernel packs without hard-coding model logic
  into the optimizer core.
- Add training, lower-precision, multi-GPU, and additional hardware backends
  only after the inference path is reliable.
