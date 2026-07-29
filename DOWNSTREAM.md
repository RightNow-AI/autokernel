# Downstream project

This repository is an independently maintained downstream fork of
[RightNow-AI/autokernel](https://github.com/RightNow-AI/autokernel).

## Provenance

- Upstream repository: `https://github.com/RightNow-AI/autokernel`
- Initial downstream base: `7843582` (`test hf kernels export`)
- License: MIT
- Original copyright: Copyright (c) 2026 RightNow AI

The upstream `LICENSE` file is preserved. Source files substantially derived
from upstream remain covered by that notice.

## Downstream direction

This fork is intended to become a general platform for discovering, testing,
tuning, and exporting production GPU kernels. Its first major additions will
focus on:

- external custom-operation specifications;
- multi-output, backward, determinism, and compile verification;
- production shape corpora captured from real models;
- modulated normalization and gated-residual fusion;
- architecture-aware tuning and reproducible experiment records; and
- clean export into runtime kernel packages.

The optimization platform and shipped runtime kernels are separate products:
the platform searches and validates candidates, while downstream applications
consume only promoted kernel implementations.

## Upstream relationship

Useful upstream changes can be incorporated without making upstream a release
dependency:

```bash
git fetch upstream
git switch main
git merge upstream/main
```

Downstream features do not require upstream approval. Contributions may still
be offered upstream when doing so benefits both projects.
