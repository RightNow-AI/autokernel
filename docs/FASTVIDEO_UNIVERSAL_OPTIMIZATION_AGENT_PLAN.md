# Universal FastVideo optimization agent plan

## Mission

Turn MotionKernel into a model-independent optimization system for models that
already run in FastVideo. A user should be able to supply a FastVideo model,
representative workload, GPU budget, and output directory. MotionKernel should
profile the workload, discover worthwhile kernel regions, search for optimized
implementations, validate full-generation correctness and performance, and
produce compatible artifacts plus an honest morning report.

LTX is the first proof model. Completing this plan means LTX can be optimized
without adding LTX-specific fusion calls to its FastVideo implementation.

## Repositories

- MotionKernel: `<motionkernel-checkout>` (this repository)
- FastVideo: `<fastvideo-checkout>`
- Existing FastVideo guide:
  `<fastvideo-checkout>/docs/contributing/kernel_optimization.md`
- Existing Wan measurement script:
  `<fastvideo-checkout>/examples/inference/optimizations/wan_fusions_ab.py`
- Existing Wan results:
  `docs/WAN_KERNEL_RESULTS.md`

Before editing either repository, read its `AGENTS.md`, inspect the current
branches and open PRs, and synchronize with the repository's main branch
without discarding unrelated work.

## Current foundation

Do not rebuild these pieces:

- FastVideo metadata-only campaign capture and workload timing.
- MotionKernel campaign validation, ranking, preparation, resumable execution,
  terminal receipts, and morning reports.
- `KernelSpec` validation and production shape corpora.
- Opt-in loading of promoted MotionKernel artifacts in FastVideo.
- Three Wan fusion targets and their isolated GB200 measurements.
- A reproducible full-generation native-versus-fused Wan benchmark.

The Wan result is an important constraint: isolated kernels improved by
approximately 8.6-10.6x, but a 50-step generation remained approximately
36.67 seconds in both modes. The universal system must rank candidates by
expected end-to-end value and avoid spending an overnight budget on regions
whose theoretical model-level impact is negligible.

## Scope

Implement the first complete version for:

- inference only;
- forward kernels only;
- one GPU;
- CUDA tensor graphs;
- models already supported by FastVideo;
- LTX as the end-to-end acceptance model.

Keep the contracts extensible for training, backward kernels, sequence
parallelism, and multi-GPU execution, but do not block the first usable system
on those capabilities.

## Required user experience

The final interface should be equivalent to:

```bash
motionkernel optimize \
  --fastvideo-checkout /path/to/FastVideo \
  --model Lightricks/LTX-Video \
  --workload workloads/ltx_480p.yaml \
  --budget-hours 10 \
  --output workspace/ltx
```

One invocation must perform or resume:

1. native baseline generation;
2. profiling and graph capture;
3. candidate discovery and impact ranking;
4. safe reference-spec generation;
5. kernel search;
6. isolated correctness and performance validation;
7. full-generation native-versus-optimized validation;
8. artifact packaging and the morning report.

It is acceptable to return `no_worthwhile_candidate`. It is not acceptable to
claim success based only on isolated operator speedup.

## Workstream 1: Workload contract and launcher

Add a versioned workload manifest shared by the FastVideo adapter and
MotionKernel. It must describe:

- model identifier and optional immutable revision;
- task and prompt or prompt-file reference;
- width, height, frame count, inference steps, guidance, and seed;
- dtype and attention backend;
- device count and distributed settings;
- warmup and measured repetitions;
- output-parity policy and performance threshold.

Add initial manifests for the existing Wan benchmark and one canonical LTX
text-to-video workload. Do not encode model-specific Python callables in the
manifest.

Build a FastVideo launcher that:

- resolves the model through FastVideo's existing registries;
- runs baseline and candidate modes in separate processes;
- captures wall time, generation time, peak allocated memory, environment
  identity, and output frames;
- writes structured JSON with stable schemas;
- preserves logs and failure reasons;
- can be resumed without repeating completed stages.

Exit criteria:

- The launcher reproduces the existing Wan A/B measurement.
- The same launcher runs an unmodified LTX model from a workload manifest.

## Workstream 2: Universal profiling and graph capture

The current `optimization_target` API records known regions. Add automatic
discovery data without requiring model-specific annotations.

Use two complementary sources:

1. `torch.profiler` for end-to-end CUDA time, call frequency, launch behavior,
   and hotspot attribution.
2. Dynamo/FX graph capture for executable tensor subgraphs and dependency
   information.

Do not require capture of the entire generation pipeline as one graph. Begin
with repeated DiT/module calls and fall back to smaller graphable scopes when
graph breaks occur. Record graph breaks rather than hiding them.

The capture must contain metadata only by default. Never serialize prompts,
weights, activations, tensor values, credentials, or model outputs into a
campaign.

For each observed operator or region, capture:

- stable graph fingerprint;
- ordered operations and dependencies;
- input/output tensor signatures;
- constants that are safe and necessary for semantics;
- shape frequency;
- inclusive and self CUDA time;
- call count;
- parent module scope;
- hardware and software identity.

Exit criteria:

- Profiling unmodified Wan and LTX produces ranked operator data.
- Repeated runs produce stable fingerprints for equivalent regions.
- Graph breaks and unsupported operations are visible in the report.

## Workstream 3: Candidate discovery and value ranking

Implement a region discovery pass over captured graphs. Start with an
allowlist of pure tensor operations and reject regions with mutation, data
dependent Python control flow, collectives, unsupported custom operators, or
unknown aliasing.

Initial pattern families:

- elementwise chains;
- normalization plus modulation;
- residual, gate, and normalization chains;
- activation and projection epilogues;
- RoPE application and Q/K preparation;
- attention pre-processing and post-processing;
- layout, cast, and contiguous-copy chains;
- VAE elementwise and normalization chains.

Generate overlapping candidate regions, deduplicate them by fingerprint, and
rank them using measured production frequency rather than single-call latency.

Every candidate must report:

- observed total GPU time;
- percentage of end-to-end GPU time;
- estimated reducible fraction;
- estimated maximum end-to-end improvement;
- confidence and rejection reasons.

Enforce a configurable impact floor. By default, do not search a candidate
when its optimistic end-to-end improvement is below 0.5%. Prefer candidates
that can plausibly exceed the 1% promotion threshold.

Exit criteria:

- The Wan elementwise targets are discovered but correctly classified as
  low-value for the measured 50-step workload.
- LTX produces a short ranked list with evidence for each candidate.

## Workstream 4: Graph-derived KernelSpec generation

Create a `KernelSpec` adapter that turns a supported captured FX region into a
search problem. The generated spec must include:

- an executable PyTorch reference derived from the captured graph;
- ordered input and output contracts;
- a weighted corpus from observed production shapes;
- dtype, layout, device, and alignment constraints;
- numerical tolerances appropriate to output dtype and operation family;
- determinism and edge-case inputs;
- a stable operation and graph fingerprint.

Do not use arbitrary source from the campaign. Spec generation must operate on
a validated, allowlisted intermediate representation. Fail closed when
semantics cannot be represented safely.

Initially support the smallest useful ATen subset needed by the highest-value
LTX candidates. Add operations incrementally from actual profile evidence
instead of attempting to support all PyTorch operators upfront.

Exit criteria:

- At least one high-value LTX candidate becomes a valid generated
  `KernelSpec`.
- The generated eager reference agrees with the original captured region.
- Existing handwritten Wan specs remain supported.

## Workstream 5: Generic artifact and runtime dispatch

Replace model-specific enablement with a generic artifact directory or
registry:

```bash
FASTVIDEO_OPTIMIZED_KERNELS=/path/to/artifacts
```

Define a versioned artifact bundle containing:

- manifest;
- graph fingerprint and operation identity;
- optimized kernel;
- compatibility constraints;
- isolated benchmark results;
- full-generation validation results;
- source campaign identity;
- morning report.

Artifact selection must check:

- model and optional model revision;
- graph fingerprint;
- tensor signature;
- GPU architecture;
- PyTorch, CUDA, and Triton compatibility;
- inference/training mode;
- distributed configuration.

On a mismatch, load failure, or runtime failure, FastVideo must use the native
path and record the fallback reason. Preserve explicit trust boundaries:
FastVideo must never silently import code from an untrusted directory.

Provide a generic graph-region dispatch mechanism. Do not add
`FASTVIDEO_LTX_FUSIONS` or handwritten LTX conditionals. The LTX source should
remain unchanged except for reusable framework hooks that apply equally to
other models.

Exit criteria:

- A promoted artifact is selected by compatibility rather than model-specific
  code.
- An incompatible artifact reliably falls back to native execution.
- Disabling the artifact directory has zero behavioral effect.

## Workstream 6: Orchestration and validation gates

Extend the unattended runner into the top-level `optimize` workflow. Each
candidate moves through explicit states:

```text
discovered -> specified -> searching -> operator_validated
           -> end_to_end_validated -> promoted
```

Terminal alternatives include `unsupported`, `incorrect`, `plateaued`,
`regressed`, `below_impact_floor`, and `budget_exhausted`.

Before promotion, require:

- isolated correctness over the weighted shape corpus;
- numerical stability, determinism, and edge checks;
- isolated speedup;
- byte equality or declared full-output tolerance;
- no unacceptable peak-memory regression;
- repeated end-to-end measurements in separate processes;
- a default minimum 1% repeatable end-to-end improvement.

Use medians and retain individual samples. Treat results within expected timing
noise as neutral, not as speedups. The morning report must distinguish isolated
operator speedup from model-level improvement.

Exit criteria:

- Interrupting and resuming does not lose completed work.
- A neutral result such as the current Wan pack is not promoted by default.
- Every terminal run leaves enough structured evidence to reproduce its
  conclusion.

## Workstream 7: LTX proof

Run the complete workflow on a canonical LTX workload using the provided GPU
cluster.

The proof must:

1. use an LTX model already supported by FastVideo;
2. avoid LTX-specific optimization annotations and switches;
3. capture and rank real end-to-end hotspots;
4. generate at least one spec from the captured graph;
5. run the unattended kernel search;
6. perform full native-versus-optimized generation validation;
7. produce a complete artifact bundle and morning report.

Success is either:

- at least one promoted artifact with a repeatable end-to-end improvement of
  1% or more and acceptable parity; or
- an evidence-backed `no_worthwhile_candidate` result after the system
  correctly evaluates the available regions.

The framework behavior is the acceptance target; do not fabricate a promotion
to make LTX appear faster.

## Workstream 8: Generalization audit

After the LTX proof, run discovery-only campaigns on at least two structurally
different FastVideo models, preferably Cosmos and Kandinsky. Do not optimize
every candidate yet. Confirm:

- no model-specific source changes are required;
- workload manifests are sufficient;
- graph capture failures are reported clearly;
- candidate fingerprints and compatibility checks remain stable;
- unsupported operation families generate actionable backlog items.

Use the findings to create a prioritized ATen/graph-pattern support matrix.

## Required tests

Keep tests focused on contracts and failure modes:

- workload schema validation;
- metadata privacy;
- fingerprint stability;
- graph-region safety and rejection;
- impact ranking and Amdahl ceiling calculations;
- generated-reference parity;
- artifact compatibility and native fallback;
- resume state transitions;
- end-to-end result classification.

GPU validation should use representative production shapes and one canonical
full-generation workload. Do not multiply expensive tests without a distinct
risk they cover.

## Commit and review strategy

Keep MotionKernel and FastVideo changes in separate branches and PRs. Prefer
small commits by contract:

1. workload schema and launcher;
2. profiler and graph capture;
3. discovery and ranking;
4. generated specs;
5. artifact compatibility and generic dispatch;
6. orchestration and validation;
7. LTX evidence and documentation.

At the end of each workstream:

- run the relevant focused CPU tests;
- run GPU validation when the workstream first touches CUDA behavior;
- update the plan with deviations and newly discovered gaps;
- commit the coherent result;
- report exact commands, results, commit hashes, and blockers.

Do not wait for a full code review before starting the next independent
workstream, but do not build later contracts on unresolved correctness issues.

## Agent operating rules

- Profile before selecting targets.
- Optimize measured production shapes, not toy inputs.
- Calculate the theoretical end-to-end ceiling before starting a search.
- Preserve native fallbacks.
- Never claim a model improvement from an isolated microbenchmark.
- Never expose or commit SSH keys, kubeconfigs, model credentials, prompts,
  activations, weights, or generated user content.
- Do not run GPU workloads on the login node; use SLURM compute allocations.
- Use resumable batch jobs for overnight work.
- Evaluate gaps in this plan while implementing it. Record material gaps,
  decisions, and revised acceptance criteria in this document or an adjacent
  design document rather than silently working around them.

## Definition of done

This project is ready for any FastVideo-supported inference model when a new
model normally requires only:

1. selecting its existing FastVideo model identifier;
2. adding a declarative workload manifest;
3. running `motionkernel optimize`.

No model-specific fusion annotations, handwritten `KernelSpec`, environment
switch, or runtime branch should be required. The run must conclude with
reproducible evidence for a promoted speedup or a clear explanation that no
worthwhile compatible kernel was found.

## Implementation progress

### Workstream 1

MotionKernel PR: https://github.com/RightNow-AI/autokernel/pull/15

- `autokernel/workload/` — versioned workload schema (`schema_version: 1`),
  generation-result schema, end-to-end classification, and FastVideo launcher
  bridge with resume state.
- `workloads/wan_t2v_1.3b_480p.yaml` — reproduces the existing Wan A/B request.
- `workloads/ltx_480p.yaml` — canonical LTX-2 distilled T2V proof workload.
- `workload.py` CLI: `validate`, `show`, `validate-result`, `run-ab`.
- CPU tests in `tests/test_workload.py`.

FastVideo PR: https://github.com/hao-ai-lab/FastVideo/pull/1668

- `examples/inference/optimizations/generation_launcher.py` — model-agnostic
  launcher that loads a workload manifest, runs one mode per process, and
  writes structured result JSON.

### Workstream 2 (in progress)

MotionKernel:

- `autokernel/discovery/` — metadata-only discovery report schema, stable graph
  fingerprints, pure-tensor allowlist, collective/data-dependent rejection.
- CPU tests in `tests/test_discovery.py`.
- `ranking.py`, `fx_capture.py`, `profiler_parse.py` — impact floor ranking,
  CPU FX region capture, profiler table parse (CUDA times still need GPU).
- Next GPU wall: end-to-end torch.profiler on Wan/LTX generation.
