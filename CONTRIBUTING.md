# Contributing

## Development workflow

1. Start from an up-to-date `main`.
2. Create a focused branch.
3. Keep framework changes separate from generated kernel experiments.
4. Run CPU validation before pushing.
5. Run the relevant GPU correctness and performance suites before promoting a
   kernel.

Do not run autonomous experiments in a checkout containing unrelated or
uncommitted work. Use a disposable clone or Git worktree so an experiment can
be abandoned without affecting development state.

## Validation levels

### CPU baseline

The baseline check requires no GPU and verifies that tracked Python sources
compile:

```bash
python -m compileall -q .
```

### GPU correctness

GPU changes must run the relevant benchmark correctness stages across their
declared shapes, dtypes, layouts, and edge cases. Multi-output or training
operations must also validate every returned tensor and requested gradient.

### Performance

Performance claims must include:

- GPU model and compute capability;
- PyTorch, Triton, CUDA, and driver versions;
- input shapes, dtypes, and layouts;
- warmup and measurement methodology;
- median latency and variance; and
- the exact baseline being compared.

Generated candidates are experimental until their correctness and performance
results are reproducible.

## Git safety

- Never force-push shared branches.
- Never push changes to the `upstream` remote.
- Never use a destructive reset outside an isolated experiment branch or
  disposable worktree.
- Preserve the MIT license and upstream attribution.
