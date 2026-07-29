# Roadmap

## Milestone 0: downstream foundation

- Preserve upstream provenance and MIT attribution.
- Maintain separate `origin` and `upstream` remotes.
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

- Integrate and benchmark Wan.
- Integrate and benchmark Kandinsky.
- Integrate and benchmark Cosmos.
- Integrate and benchmark LTX.
- Validate single-GPU and sequence-parallel execution.

Runtime integrations will use exported kernels with native PyTorch fallbacks;
they will not require the optimization platform at inference time.

## Milestone 5: continuous kernel research

- Run parallel searches across GPU workers.
- Maintain architecture-specific tuning records.
- Track performance regressions between revisions.
- Promote candidates through experimental, validated, and production stages.
- Expand into attention, MLP, quantization, and communication-aware fusion.
