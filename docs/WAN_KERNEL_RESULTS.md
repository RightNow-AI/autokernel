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

## Modulated pre-attention LayerNorm

The second target fuses FP32 non-affine LayerNorm with Wan's channel-wise
scale and shift. On the same GB200 class, commit `58e09c9` passed smoke,
the full 11-configuration shape sweep, numerical stability, determinism, and
edge cases. All five production corpus cases had 100% of values inside the
declared tolerance.

| Wan shape | Fused latency | Eager latency | Speedup |
|---|---:|---:|---:|
| 1.3B, 480p, 49 frames | 37.52 µs | 370.64 µs | 9.878× |
| 1.3B, 480p, 81 frames | 54.34 µs | 580.42 µs | 10.682× |
| 14B, 480p, 49 frames | 101.04 µs | 1117.76 µs | 11.062× |
| 14B, 480p, 49 frames, SP4 | 32.94 µs | 305.97 µs | 9.290× |
| 14B, 480p, 81 frames, SP4 | 46.87 µs | 474.74 µs | 10.129× |

The weighted corpus aggregate was 54.54 µs fused versus 569.91 µs eager,
a 10.449× isolated operator speedup.

## Post-MLP gated residual

The third target fuses Wan's FP32 gate multiplication and residual update into
one model-dtype output. Commit `58e09c9` passed the same five correctness
stages across all production and edge shapes on GB200.

| Wan shape | Fused latency | Eager latency | Speedup |
|---|---:|---:|---:|
| 1.3B, 480p, 49 frames | 36.90 µs | 373.02 µs | 10.108× |
| 1.3B, 480p, 81 frames | 54.06 µs | 587.57 µs | 10.868× |
| 14B, 480p, 49 frames | 103.57 µs | 1182.09 µs | 11.413× |
| 14B, 480p, 49 frames, SP4 | 33.87 µs | 316.75 µs | 9.351× |
| 14B, 480p, 81 frames, SP4 | 49.61 µs | 495.33 µs | 9.984× |

The weighted corpus aggregate was 55.60 µs fused versus 590.95 µs eager,
a 10.628× isolated operator speedup. All three results remain operator-level;
an end-to-end generation benchmark is the next measurement.
