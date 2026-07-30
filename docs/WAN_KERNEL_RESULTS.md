# Wan kernel results

## Gated residual plus affine LayerNorm

The first Wan target fuses the post-self-attention gated residual update,
FP32 affine LayerNorm, two model-dtype casts, and both output writes into one
Triton launch.

Validation environment:

- GPU: NVIDIA GB200
- PyTorch: 2.11.0+cu128
- Triton: 3.6.0
- corpus: `models/wan_gated_residual_norm_corpus.json`
- commit: `2481c35`

All five forward gates passed: smoke, production-shape sweep, numerical
stability, determinism, and edge cases. The maximum absolute BF16 difference
was `0.03125`, with 100% of values inside the declared tolerance.

| Wan shape | Fused latency | Eager latency | Speedup |
|---|---:|---:|---:|
| 1.3B, 480p, 49 frames | 51.91 µs | 439.60 µs | 8.468× |
| 1.3B, 480p, 81 frames | 75.39 µs | 684.80 µs | 9.083× |
| 14B, 480p, 49 frames | 153.02 µs | 1344.22 µs | 8.785× |
| 14B, 480p, 49 frames, SP4 | 46.93 µs | 370.87 µs | 7.903× |
| 14B, 480p, 81 frames, SP4 | 67.82 µs | 572.97 µs | 8.449× |

The equally weighted corpus aggregate was 79.01 µs fused versus 682.49 µs
eager, an 8.638× operator speedup. These are isolated operator results; an
end-to-end Wan benchmark is still required to quantify generation-level
impact.
