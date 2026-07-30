# Agent brief: Week 1 and Week 2

## Mission

Build the first two weeks of the downstream kernel platform:

1. Week 1: replace hard-coded operation configuration with a stable,
   externally loadable `KernelSpec` registry while preserving all existing
   behavior.
2. Week 2: generalize correctness verification for structured outputs, custom
   production shape corpora, optional backward checks, and
   `torch.compile(fullgraph=True)` compatibility.

This is an implementation assignment, not a design-only exercise. Finish each
week with tested code, documentation, and a reviewable pull request.

Do not wait for changes or approval from the upstream repository. This is an
independently maintained downstream fork. Preserve the upstream MIT license and
attribution in all derived source.

## Required operating rules

- Begin from this fork's current `main`, after the downstream-foundation PR has
  merged.
- Keep the Week 1 and Week 2 work in separate pull requests.
- Use branches named:
  - `agent/kernel-spec-registry`
  - `agent/generalized-verification`
- Never develop directly on `main`.
- Never push to the `upstream` remote.
- Do not use `git reset --hard` in the primary checkout.
- Run autonomous kernel experiments only in a disposable clone or Git
  worktree.
- Do not overwrite unrelated local changes.
- Do not silently weaken an existing correctness tolerance or remove an edge
  case to make a test pass.
- Preserve the existing no-argument CLI behavior.
- Prefer typed dataclasses and small pure functions over unstructured
  dictionaries and new global maps.
- Keep GPU-specific imports and allocation out of registry discovery so CPU
  tests can load and inspect every built-in specification.
- Do not claim GPU correctness or performance without recording the actual GPU
  run.
- If a requirement cannot be completed, leave the code in a safe state,
  document the exact blocker, and do not substitute a fake implementation.

## Baseline architecture

Before editing, understand the current ownership:

- `bench.py`
  - contains deterministic input generators;
  - contains the hard-coded `KERNEL_CONFIGS` dictionary;
  - performs five correctness stages;
  - assumes each candidate returns one tensor;
  - benchmarks `kernel_fn(**inputs)`.
- `reference.py`
  - contains the PyTorch reference functions.
- `extract.py`
  - duplicates shape keys, aliases, tolerances, FLOP formulas, byte formulas,
    speedup estimates, and starter-kernel lookup;
  - generates standalone candidate kernel files.
- `kernel.py`
  - declares `KERNEL_TYPE`;
  - exports `kernel_fn` with a signature matching the reference function.
- `profile.py`
  - classifies GPU kernel names with a hard-coded rule list;
  - discovers supported types by scanning `kernels/*.py`.
- `verify.py`
  - has model-level replacement strategies for only a subset of operations;
  - assumes a simple output comparison path.
- `orchestrate.py`
  - consumes extracted kernel metadata and benchmark results.

The first two weeks must reduce these hard-coded seams without breaking the
existing nine built-in kernel types.

## Non-goals for Weeks 1 and 2

Do not implement the following yet:

- modulated normalization kernels;
- gated-residual kernels;
- model-specific integration for Wan, Kandinsky, Cosmos, or LTX;
- a distributed GPU scheduler;
- an experiment database;
- package or repository renaming;
- a new web service or user interface;
- automatic graph rewriting for arbitrary model code;
- replacement of the existing profiler classification system;
- broad formatting or unrelated cleanup.

Week 3 will use the interfaces created here to implement the first fused
operation. Do not pull Week 3 into these changes.

---

# Week 1: custom-operation registry

## Week 1 outcome

At the end of Week 1, an external operation must be able to supply its
reference, inputs, sizes, tolerances, edge cases, performance accounting, and
starter kernel through one `KernelSpec`. `bench.py` and `extract.py` must
consume that specification without adding operation-specific branches.

All nine existing built-in operations must behave exactly as they did before
the refactor.

## Step 1: capture the compatibility baseline

Before refactoring:

1. Record the current built-in names from `bench.py::KERNEL_CONFIGS`.
2. Record the output of:

   ```bash
   uv run bench.py --help
   uv run extract.py --help
   uv run profile.py --help
   uv run verify.py --help
   ```

3. Add CPU tests that freeze:
   - the ordered set of built-in operation names;
   - each built-in's size labels;
   - supported dtype names;
   - tolerance values;
   - shape-key and alias metadata used by extraction;
   - the existing `--kernel` CLI option.
4. If a GPU is available, run a quick baseline for at least:
   - `matmul`;
   - `layernorm`;
   - `rmsnorm`.
5. Save the commands and environment metadata in the PR description. Do not
   commit generated benchmark artifacts.

The compatibility tests are required before moving configuration. They prevent
the refactor from accidentally changing benchmark coverage.

## Step 2: add a real Python package

Create the following structure:

```text
autokernel/
  __init__.py
  specs/
    __init__.py
    types.py
    registry.py
    loader.py
    builtins.py
    inputs.py
tests/
  test_spec_registry.py
  test_spec_loader.py
  test_builtin_specs.py
  fixtures/
    custom_add.py
```

Do not move the command-line scripts into the package during this milestone.
They should import the new package. This keeps the migration focused and
preserves current entry points.

Add a `dev` optional dependency group in `pyproject.toml` containing `pytest`.
Update CI to install the development dependencies and run CPU tests.

## Step 3: define `KernelSpec`

Define the public types in `autokernel/specs/types.py`. Use type aliases and
dataclasses rather than an untyped dictionary.

The initial interface should cover:

```python
@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


@dataclass(frozen=True)
class EdgeCase:
    name: str
    size: Mapping[str, int]
    dtype: str | None = None
    seed: int = 42
    input_transform: Callable[[InputMap], InputMap] | None = None


@dataclass(frozen=True)
class KernelSpec:
    name: str
    reference_fn: Callable[..., Any]
    input_generator: Callable[..., InputMap]
    sizes: Mapping[str, Mapping[str, int]]
    dtypes: tuple[str, ...]
    tolerances: Mapping[str, Tolerance]
    edge_cases: tuple[EdgeCase, ...]
    flops_fn: Callable[[Mapping[str, int]], int | float]
    bytes_fn: Callable[[Mapping[str, int], int], int | float]
    shape_keys: tuple[str, ...]
    shape_aliases: Mapping[str, str]
    starter_kernels: Mapping[str, Path]
    speedup_estimate: str | None = None
```

The exact spelling can change if tests demonstrate a cleaner API, but retain
these responsibilities.

### Type and validation requirements

`KernelSpec` construction or registration must reject:

- an empty or non-identifier-like operation name;
- duplicate size labels;
- a missing `small`, `medium`, or `large` size for built-ins;
- unknown dtype strings;
- missing tolerances for a declared dtype;
- negative tolerances;
- missing starter-kernel files;
- duplicate shape aliases that resolve inconsistently;
- a reference or input generator that is not callable.

Validation errors must identify the specification and invalid field.

Use canonical dtype strings at the specification boundary:

- `float16`
- `bfloat16`
- `float32`

Translate to `torch.dtype` only inside runtime code. This keeps discovery
serializable and CPU-friendly.

Do not put backward or structured-output fields into `KernelSpec` during the
first refactor unless they are optional and unused. Week 2 will add them with
tests.

## Step 4: implement the registry

In `autokernel/specs/registry.py`, implement a small registry with:

- `register(spec: KernelSpec) -> None`
- `get(name: str) -> KernelSpec`
- `list_names() -> tuple[str, ...]`
- `contains(name: str) -> bool`
- a duplicate-registration error;
- deterministic ordering;
- no GPU initialization at import time.

Provide a function that creates the default registry rather than relying only
on mutable module globals:

```python
def create_builtin_registry() -> KernelRegistry:
    ...
```

A fresh registry must be usable in tests without affecting another test.

## Step 5: migrate built-in metadata

Create one `KernelSpec` for every current built-in:

- `matmul`
- `softmax`
- `layernorm`
- `rmsnorm`
- `flash_attention`
- `fused_mlp`
- `cross_entropy`
- `rotary_embedding`
- `reduce`

Move deterministic input generators out of `bench.py` into
`autokernel/specs/inputs.py`.

The reference functions may remain in `reference.py` for compatibility.
`builtins.py` may import them. Do not duplicate reference math in multiple
locations.

Move the following duplicated `extract.py` metadata into the specs:

- `SHAPE_KEYS`
- `SHAPE_ALIAS_MAP`
- `TOLERANCES_MAP`
- `FLOPS_FN_SRC` responsibilities;
- `BYTES_FN_SRC` responsibilities;
- `SPEEDUP_ESTIMATES`;
- starter-kernel paths.

Use real callables for FLOP and byte accounting. Do not continue storing Python
function bodies as source strings in the central model.

If generated candidate files must remain standalone, update extraction to
serialize a safe, generated function representation from known metadata. Do
not call `eval` on external specification content. A preferable solution is to
emit simple numeric expressions through a constrained serializer or have the
generated file retain a reference to its spec.

Keep compatibility aliases temporarily if other modules still import an old
constant, but mark them as deprecated and derive them from the registry. There
must be only one source of truth.

## Step 6: add external spec loading

Implement `autokernel/specs/loader.py`.

Support:

```text
package.module:SPEC
/absolute/path/to/spec.py:SPEC
relative/path/to/spec.py:SPEC
```

The selected object may be:

- a `KernelSpec`; or
- a zero-argument callable returning a `KernelSpec`.

Reject:

- a missing file or module;
- a missing attribute;
- an attribute returning the wrong type;
- a loaded spec whose declared starter kernel does not exist;
- a spec name that collides with a built-in unless an explicit override flag
  is supplied.

Return actionable errors including the original locator.

Do not mutate `sys.path` permanently. If loading from a file, use
`importlib.util.spec_from_file_location` with a unique module name.

## Step 7: connect the CLI

Add `--spec` to `bench.py` and `extract.py`.

Required precedence:

1. explicit `--spec`;
2. explicit `--kernel`;
3. `kernel.py::KERNEL_TYPE`, preserving current behavior.

Examples:

```bash
uv run bench.py --spec tests/fixtures/custom_add.py:SPEC
uv run extract.py --spec tests/fixtures/custom_add.py:SPEC --top 1
```

When `--spec` is supplied:

- load and validate it;
- register it in an isolated registry for that command;
- select it by its declared name;
- do not require editing `KERNEL_CONFIGS`, `reference.py`, or the hard-coded
  extraction maps;
- preserve existing `--quick`, `--profile`, `--sizes`, and backend behavior.

An external spec must not be imported when the CLI only asks for `--help`.

## Step 8: provide a minimal external example

Add `examples/custom_ops/add.py` and document it.

The example should:

- define a simple PyTorch reference;
- generate deterministic inputs;
- define small, medium, and large sizes;
- declare tolerances and accounting functions;
- point to a starter Triton kernel;
- export `SPEC`;
- run through the same benchmark path as a built-in.

The example exists to prove extensibility, not performance. Keep it small.

Do not use modulated normalization as the Week 1 example; that would combine
the registry refactor with the Week 3 feature.

## Week 1 tests

At minimum, add tests for:

- registering and retrieving a valid spec;
- deterministic name ordering;
- duplicate registration;
- every validation error described above;
- loading a module locator;
- loading absolute and relative file locators;
- callable spec factories;
- bad module, path, attribute, and object errors;
- collision behavior;
- exact built-in metadata compatibility;
- deterministic inputs for a fixed seed;
- CLI precedence;
- external example discovery;
- `extract.py` consuming an external spec without an operation-specific map.

CPU CI must run:

```bash
uv sync --extra dev
uv run pytest -m "not gpu"
python -m compileall -q .
```

Mark GPU tests with `@pytest.mark.gpu` and register the marker in
`pyproject.toml`.

If a supported GPU is available, also run:

```bash
uv run bench.py --kernel matmul --quick
uv run bench.py --kernel layernorm --quick
uv run bench.py --kernel rmsnorm --quick
uv run bench.py --spec examples/custom_ops/add.py:SPEC --quick
```

## Week 1 acceptance criteria

Week 1 is complete only when:

- all nine built-in specs are registered;
- old built-in CLI commands behave unchanged;
- a new external operation runs without editing central operation maps;
- `bench.py` and `extract.py` use `KernelSpec` as their source of operation
  metadata;
- no duplicate authoritative metadata remains in `extract.py`;
- registry discovery works on a CPU-only machine;
- CPU CI passes;
- available GPU smoke tests pass;
- README documentation contains a custom-operation quick start;
- the PR documents compatibility evidence and any GPU coverage not run.

## Week 1 commit sequence

Prefer small commits in this order:

1. `Add kernel specification types and registry`
2. `Migrate built-in operation metadata`
3. `Load external kernel specifications`
4. `Use kernel specifications in benchmark and extraction`
5. `Document and test custom operations`

Do not mix formatting-only changes into these commits.

---

# Week 2: generalized verification

## Week 2 outcome

At the end of Week 2, the benchmark harness must correctly validate operations
that return multiple or nested outputs, accept production shape corpora without
editing Python source, optionally compare gradients, and optionally verify
`torch.compile(fullgraph=True)` compatibility.

Week 2 begins only after the Week 1 registry PR is merged.

## Step 1: extend the public specification

Add explicit verification types in `autokernel/specs/types.py`.

Recommended responsibilities:

```python
@dataclass(frozen=True)
class OutputSpec:
    # Optional paths to tensor leaves that participate in correctness.
    included_paths: tuple[str, ...] | None = None
    # Whether non-tensor leaves must match exactly.
    compare_non_tensors: bool = True


@dataclass(frozen=True)
class BackwardSpec:
    differentiable_inputs: tuple[str, ...]
    output_paths: tuple[str, ...] | None = None
    tolerances: Mapping[str, Tolerance] | None = None
    enabled_by_default: bool = False


@dataclass(frozen=True)
class CompileSpec:
    enabled: bool = False
    fullgraph: bool = True
    dynamic: bool = False
```

Add optional fields to `KernelSpec`:

- `output_spec`;
- `backward_spec`;
- `compile_spec`.

Existing built-ins must receive defaults that preserve their current behavior.

Keep integration/replacement hooks out of this PR unless required for an
isolated compile test. Model replacement is a later milestone.

## Step 2: implement output-tree handling

Create:

```text
autokernel/
  verification/
    __init__.py
    outputs.py
    backward.py
    compile.py
```

`outputs.py` must handle:

- a single tensor;
- tuples;
- lists;
- dictionaries with deterministic key order;
- named tuples if they appear in current model outputs;
- nested combinations of the above;
- exact comparison of supported non-tensor leaves when enabled.

Represent every leaf with a stable path such as:

```text
output
output[0]
output.updated_residual
output["aux"][1]
```

For tensor leaves:

- require the same tree structure;
- require the same shape;
- require compatible dtype expectations;
- detect NaN and infinity per path;
- compute maximum and mean absolute error per path;
- apply dtype-specific tolerances;
- report the failing path and a concise error summary.

Do not silently compare only the first tensor. Do not flatten dictionaries in
insertion-dependent ways. Do not coerce mismatched shapes.

Return structured comparison results suitable for JSON output, not only
printed text.

## Step 3: refactor benchmark correctness

Replace the single-tensor assumptions in `bench.py` with the output-tree
comparator.

Preserve the existing five forward correctness stages:

1. smoke;
2. shape sweep;
3. numerical stability;
4. determinism;
5. edge cases.

Structured outputs must pass all applicable stages.

Determinism means:

- identical output tree structure across runs;
- every tensor leaf compared;
- exact or configured deterministic tolerance;
- non-tensor leaves unchanged when comparison is enabled.

Keep the public benchmark summary compatible, but add leaf-level details to a
structured result artifact under `workspace/`.

Performance timing must call the full operation and must not introduce output
tree traversal inside the timed region.

## Step 4: add production shape corpora

Define a versioned JSON schema. The first version should resemble:

```json
{
  "schema_version": 1,
  "operation": "gated_residual_norm",
  "cases": [
    {
      "name": "example-production-shape",
      "size": {"batch": 2, "tokens": 4096, "dim": 3072},
      "dtype": "bfloat16",
      "weight": 37,
      "tags": ["production", "forward"]
    }
  ]
}
```

Add `--shape-corpus PATH` to `bench.py`.

Rules:

- the corpus operation must match the selected spec;
- schema versions must be validated;
- size keys must be accepted by the spec;
- dtypes must be declared by the spec;
- names must be unique;
- weights must be positive integers;
- unknown fields should produce a clear error unless the schema explicitly
  allows them;
- duplicate cases should be deterministically deduplicated or rejected;
- committed corpora must not include confidential model inputs or tensor data.

The corpus contains metadata only. It must never serialize model activations,
weights, prompts, or user data.

Add merge behavior:

- default: use the spec's built-in cases;
- `--shape-corpus`: append validated corpus cases;
- `--shape-corpus-only`: use only corpus cases.

Use `weight` for aggregate reporting, not to repeat allocations or benchmark
loops unnecessarily.

Report both:

- unweighted latency by case;
- weighted aggregate latency and speedup.

Do not mix results from different dtypes into a single unexplained aggregate.

## Step 5: add optional backward verification

Backward verification is opt-in during Week 2:

```bash
uv run bench.py --spec path/to/spec.py:SPEC --check-backward
```

Implementation requirements:

1. Generate one canonical input mapping.
2. Deep-clone tensor inputs for reference and candidate paths.
3. Set `requires_grad=True` only for names declared in
   `BackwardSpec.differentiable_inputs`.
4. Run reference and candidate independently.
5. Select tensor output leaves declared by `output_paths`, or all floating
   tensor leaves when paths are omitted.
6. Produce deterministic upstream gradients with a fixed seed. Do not use
   only `output.sum()` for every test because cancellation and symmetry can
   hide errors.
7. Call `torch.autograd.grad` with matching inputs and upstream gradients.
8. Compare every requested input gradient by name.
9. Report missing gradients, unexpected gradients, shape differences, NaN,
   infinity, maximum error, and mean error.
10. Do not accumulate gradients between cases.

If a candidate is intentionally forward-only and no `BackwardSpec` exists,
`--check-backward` must fail with an actionable unsupported message rather
than silently skipping.

Backward execution is correctness-only in Week 2. Do not add backward
performance claims yet.

## Step 6: add optional compile verification

Add:

```bash
uv run bench.py --spec path/to/spec.py:SPEC --check-compile
```

Compile verification must:

- wrap the candidate call in a stable Python function;
- call `torch.compile(..., fullgraph=True)` by default;
- compile outside all timed performance regions;
- run at least twice after compilation;
- compare compiled output against the eager reference through the same output
  tree comparator;
- report graph breaks and compilation exceptions without swallowing them;
- record PyTorch, Triton, CUDA, GPU, and compile-mode metadata.

When dynamic shape support is declared, test at least two compatible shapes
through the same compiled callable. Otherwise use static shapes.

Do not report the first-call compilation latency as kernel execution latency.

Direct Triton kernels may need to be exposed through a compile-aware wrapper.
Prefer PyTorch's supported Triton custom-operation integration rather than
adding graph-break exceptions. Keep the wrapper outside the core
`KernelSpec`; the spec should describe behavior, not own global registration
side effects.

If `torch.compile` is not available in the installed PyTorch version, emit a
clear unsupported result. Do not mark the check as passed.

## Step 7: add a structured-output fixture

Add an example operation under `examples/custom_ops/` that returns:

```python
{
    "output": tensor,
    "aux": (updated_tensor, metadata_value),
}
```

It should have a simple differentiable reference and candidate so CPU unit
tests can verify:

- nested output traversal;
- path generation;
- tensor comparison;
- exact metadata comparison;
- gradient comparison.

GPU integration can use a small Triton implementation, but the core output and
gradient tests must also run on CPU.

This fixture proves the framework. It is not the production modulated-norm
kernel.

## Step 8: preserve machine-readable results

Define versioned result records for:

- forward correctness;
- per-output errors;
- backward correctness;
- per-input gradient errors;
- compile correctness;
- shape-corpus identity;
- environment metadata.

Write JSON atomically under `workspace/`. Use a temporary file followed by
rename so an interrupted run does not leave valid-looking partial JSON.

Do not break the existing greppable console output that the autonomous agent
loop consumes. Add new stable lines such as:

```text
FORWARD_CORRECTNESS: PASS
BACKWARD_CORRECTNESS: PASS
COMPILE_CORRECTNESS: PASS
```

Only print `PASS` after the complete requested stage succeeds.

## Week 2 tests

Add CPU tests for:

- every supported output container;
- nested path stability;
- mismatched structures;
- mismatched shapes;
- per-leaf tolerance selection;
- NaN and infinity reporting;
- non-tensor metadata comparison;
- deterministic output verification;
- shape-corpus schema validation;
- operation mismatch;
- invalid dtype, keys, weights, and duplicates;
- corpus merging and corpus-only behavior;
- weighted aggregate calculations;
- gradient parity;
- missing and unexpected gradients;
- deterministic upstream gradients;
- unsupported backward behavior;
- compile option validation through mocks when a compiler/GPU is unavailable;
- atomic JSON result writing.

GPU tests must cover:

- a built-in single-output kernel;
- an external structured-output kernel;
- forward comparison;
- requested backward comparison when implemented;
- full-graph compile comparison.

Run:

```bash
uv sync --extra dev
uv run pytest -m "not gpu"
python -m compileall -q .
```

On the target GPU, also run the documented GPU suite and save the command,
environment, and result summary in the PR.

## Week 2 acceptance criteria

Week 2 is complete only when:

- existing built-in single-tensor behavior remains unchanged;
- tensor, tuple, list, dictionary, named-tuple, and nested outputs are compared
  correctly;
- all output leaves receive stable diagnostic paths;
- a valid external shape corpus is accepted without editing source;
- invalid corpora fail before GPU allocation;
- weighted and unweighted results are reported separately;
- optional backward checks compare every requested gradient;
- optional compile checks use full-graph mode and reference parity;
- compile time is excluded from runtime latency;
- requested checks cannot be silently skipped;
- machine-readable results are versioned and atomically written;
- CPU CI passes;
- available GPU correctness checks pass;
- documentation contains copy-paste examples for all new CLI options.

## Week 2 commit sequence

Prefer small commits in this order:

1. `Compare structured kernel outputs`
2. `Load production shape corpora`
3. `Verify optional kernel gradients`
4. `Check full-graph kernel compilation`
5. `Record generalized verification results`
6. `Document and test verification workflows`

---

# Pull request requirements

Each pull request must include:

## Summary

- what changed;
- why the chosen interface is stable enough for Week 3;
- compatibility impact;
- explicit non-goals.

## Evidence

- CPU test command and result;
- compile-all command and result;
- GPU model and software versions, if run;
- exact GPU commands and results;
- before/after CLI compatibility notes;
- sample external-spec invocation;
- sample machine-readable result.

## Risk assessment

Address:

- import-time GPU initialization;
- circular imports between scripts and the new package;
- external module-loading errors;
- arbitrary-code trust boundary for external specs;
- structured-output comparison gaps;
- accidental compile time in latency measurements;
- backward state or gradient accumulation;
- changes to autonomous-loop console parsing.

## Review boundaries

The Week 1 PR must not contain Week 2 verification features beyond harmless
optional type placeholders. The Week 2 PR must not contain production fused
kernels or model integrations.

If either PR becomes too large to review confidently, split it along the commit
sequence above while keeping the acceptance criteria intact.

# Final handoff

At the end of Week 2, produce:

- links to both merged pull requests;
- the final public `KernelSpec` example;
- a list of supported output structures;
- the shape-corpus schema and an example;
- forward, backward, and compile validation examples;
- CPU and GPU validation results;
- known limitations;
- a recommendation for the Week 3 modulated-norm specification.

Do not begin Week 3 until the handoff demonstrates that a new external,
multi-output operation can be introduced without editing central operation
maps.
