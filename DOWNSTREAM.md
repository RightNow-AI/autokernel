# MotionKernel provenance

MotionKernel is an independently maintained, MIT-licensed downstream fork of
[RightNow-AI/AutoKernel](https://github.com/RightNow-AI/autokernel). Its focus
is GPU kernel discovery, optimization, verification, and packaging for video
generation models.

## Provenance

- Upstream repository: `https://github.com/RightNow-AI/autokernel`
- Initial downstream base: `7843582` (`test hf kernels export`)
- License: MIT
- Original copyright: Copyright (c) 2026 RightNow AI

The upstream `LICENSE` file is preserved. Source files substantially derived
from upstream remain covered by that notice.

## MotionKernel direction

MotionKernel is intended to become a video-first, framework-agnostic platform
for discovering, testing, tuning, and exporting production GPU kernels. Its
initial work focuses on:

- external custom-operation specifications;
- multi-output, backward, determinism, and compile verification;
- production shape corpora captured from real models;
- modulated normalization, gated-residual, attention, and layout fusion;
- architecture-aware tuning and reproducible experiment records; and
- clean export into runtime kernel packages for FastVideo, Diffusers, and other
  PyTorch video runtimes.

The optimization platform and shipped runtime kernels are separate products:
the platform searches and validates candidates, while downstream applications
consume only promoted kernel implementations.

The initial model families are Wan, LTX-Video, Cosmos, and Kandinsky. Listing a
model as a target does not imply complete support: support is earned through a
published integration, representative workload corpus, correctness results,
and an end-to-end benchmark.

## Compatibility identity

The distribution is named `motionkernel`. The Python import namespace remains
`autokernel` temporarily so existing specifications, scripts, and downstream
users do not break during the project transition. A future namespace migration
will include a compatibility release and explicit upgrade instructions.

## Upstream relationship

Useful upstream changes can be incorporated without making upstream a release
dependency:

```bash
git fetch upstream
git switch main
git merge upstream/main
```

MotionKernel features do not require upstream approval. Improvements may still
be offered upstream when doing so benefits both projects.
