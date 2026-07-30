"""Built-in kernel specifications.

This module is the single source of truth for the nine operations that used to
be described three times: in ``bench.py::KERNEL_CONFIGS``, in ``extract.py``'s
metadata maps, and in the extraction templates.

Sizes, dtypes, tolerances, edge cases and accounting formulas are carried over
unchanged from the pre-refactor harness; ``tests/test_builtin_specs.py`` freezes
them so a future edit cannot silently shrink benchmark coverage.

Importing this module does not import ``torch``: reference implementations are
resolved lazily on first call.
"""

from __future__ import annotations

from pathlib import Path

from .accounting import DT_BYTES, size
from .inputs import (
    gen_cross_entropy_inputs,
    gen_flash_attention_inputs,
    gen_fused_mlp_inputs,
    gen_layernorm_inputs,
    gen_matmul_inputs,
    gen_reduce_inputs,
    gen_rmsnorm_inputs,
    gen_rotary_embedding_inputs,
    gen_softmax_inputs,
)
from .lazy import lazy_callable
from .types import EdgeCase, KernelSpec, Tolerance

__all__ = ["REPO_ROOT", "builtin_specs", "starter_kernels_for"]

#: Repository root: ``<root>/autokernel/specs/builtins.py`` -> ``<root>``.
REPO_ROOT = Path(__file__).resolve().parents[2]

_KERNELS_DIR = REPO_ROOT / "kernels"

# Tolerance sets shared by several operations, spelled out per operation below
# so a change to one never silently changes another.
_TOL_LOOSE = {
    "float16": Tolerance(atol=1e-2, rtol=1e-2),
    "bfloat16": Tolerance(atol=2e-2, rtol=2e-2),
    "float32": Tolerance(atol=1e-4, rtol=1e-4),
}
_TOL_TIGHT = {
    "float16": Tolerance(atol=1e-3, rtol=1e-3),
    "bfloat16": Tolerance(atol=2e-3, rtol=2e-3),
    "float32": Tolerance(atol=1e-5, rtol=1e-5),
}
_TOL_REDUCTION = {
    "float16": Tolerance(atol=1e-2, rtol=1e-2),
    "bfloat16": Tolerance(atol=1e-1, rtol=5e-2),
}

_FP16_BF16_FP32 = ("float16", "bfloat16", "float32")
_FP16_BF16 = ("float16", "bfloat16")


def starter_kernels_for(name: str) -> dict[str, Path]:
    """Return the Triton and CUDA starter kernel paths for a built-in operation."""
    starters: dict[str, Path] = {}
    triton_path = _KERNELS_DIR / f"{name}.py"
    if triton_path.is_file():
        starters["triton"] = triton_path
    cuda_path = _KERNELS_DIR / "cuda" / f"{name}.py"
    if cuda_path.is_file():
        starters["cuda"] = cuda_path
    return starters


def _spec_matmul() -> KernelSpec:
    return KernelSpec(
        name="matmul",
        reference_fn=lazy_callable("reference", "matmul_ref"),
        input_generator=gen_matmul_inputs,
        sizes=[
            ("tiny", {"M": 128, "N": 128, "K": 128}),
            ("small", {"M": 512, "N": 512, "K": 512}),
            ("medium", {"M": 1024, "N": 1024, "K": 1024}),
            ("large", {"M": 2048, "N": 2048, "K": 2048}),
            ("xlarge", {"M": 4096, "N": 4096, "K": 4096}),
            ("tall", {"M": 8192, "N": 1024, "K": 1024}),
            ("wide", {"M": 1024, "N": 8192, "K": 1024}),
            ("deep_k", {"M": 1024, "N": 1024, "K": 8192}),
            ("llm_qkv", {"M": 4096, "N": 4096, "K": 512}),
            ("llm_mlp", {"M": 4096, "N": 11008, "K": 4096}),
        ],
        dtypes=_FP16_BF16_FP32,
        tolerances=_TOL_LOOSE,
        flops_fn=2 * size("M") * size("N") * size("K"),
        bytes_fn=(
            size("M") * size("K") + size("K") * size("N") + size("M") * size("N")
        ) * DT_BYTES,
        edge_cases=(
            EdgeCase(name="edge_1023", size={"M": 1023, "N": 1023, "K": 1023}),
            EdgeCase(name="edge_4097", size={"M": 4097, "N": 4097, "K": 512}),
            EdgeCase(name="edge_1537", size={"M": 1537, "N": 1537, "K": 1537}),
        ),
        shape_keys=("M", "N", "K"),
        shape_aliases={"M": "M", "N": "N", "K": "K"},
        starter_kernels=starter_kernels_for("matmul"),
        speedup_estimate="2-3x",
    )


def _spec_softmax() -> KernelSpec:
    return KernelSpec(
        name="softmax",
        reference_fn=lazy_callable("reference", "softmax_ref"),
        input_generator=gen_softmax_inputs,
        sizes=[
            ("tiny", {"rows": 32, "cols": 128}),
            ("small", {"rows": 256, "cols": 512}),
            ("medium", {"rows": 1024, "cols": 1024}),
            ("large", {"rows": 4096, "cols": 4096}),
            ("xlarge", {"rows": 8192, "cols": 8192}),
            ("wide", {"rows": 1024, "cols": 32768}),
            ("narrow", {"rows": 32768, "cols": 128}),
            ("vocab", {"rows": 4096, "cols": 50257}),
        ],
        dtypes=_FP16_BF16_FP32,
        tolerances=_TOL_TIGHT,
        # exp + sub + sum + div + max
        flops_fn=5 * size("rows") * size("cols"),
        # read + write
        bytes_fn=2 * size("rows") * size("cols") * DT_BYTES,
        edge_cases=(
            EdgeCase(name="edge_1023", size={"rows": 1023, "cols": 1023}),
            EdgeCase(name="edge_4097", size={"rows": 4097, "cols": 4097}),
            EdgeCase(name="edge_50257", size={"rows": 1024, "cols": 50257}),
        ),
        shape_keys=("rows", "cols"),
        shape_aliases={"M": "rows", "N": "cols", "rows": "rows", "cols": "cols"},
        starter_kernels=starter_kernels_for("softmax"),
        speedup_estimate="1.5-3x",
    )


def _spec_layernorm() -> KernelSpec:
    return KernelSpec(
        name="layernorm",
        reference_fn=lazy_callable("reference", "layernorm_ref"),
        input_generator=gen_layernorm_inputs,
        sizes=[
            ("tiny", {"batch": 32, "dim": 128}),
            ("small", {"batch": 256, "dim": 512}),
            ("medium", {"batch": 1024, "dim": 1024}),
            ("large", {"batch": 4096, "dim": 2048}),
            ("xlarge", {"batch": 8192, "dim": 4096}),
            ("wide", {"batch": 1024, "dim": 8192}),
            ("llm_7b", {"batch": 4096, "dim": 4096}),
            ("llm_13b", {"batch": 4096, "dim": 5120}),
        ],
        dtypes=_FP16_BF16_FP32,
        tolerances=_TOL_TIGHT,
        # mean, var, norm, scale, shift
        flops_fn=8 * size("batch") * size("dim"),
        bytes_fn=(2 * size("batch") * size("dim") + 2 * size("dim")) * DT_BYTES,
        edge_cases=(
            EdgeCase(name="edge_1023", size={"batch": 1023, "dim": 1023}),
            EdgeCase(name="edge_4097", size={"batch": 4097, "dim": 4097}),
        ),
        shape_keys=("batch", "dim"),
        shape_aliases={
            "M": "batch",
            "N": "dim",
            "rows": "batch",
            "cols": "dim",
            "batch": "batch",
            "dim": "dim",
        },
        starter_kernels=starter_kernels_for("layernorm"),
        speedup_estimate="1.5-3x",
    )


def _spec_flash_attention() -> KernelSpec:
    bhsd = size("batch") * size("heads") * size("seq_len") * size("head_dim")
    return KernelSpec(
        name="flash_attention",
        reference_fn=lazy_callable("reference", "flash_attention_ref"),
        input_generator=gen_flash_attention_inputs,
        sizes=[
            ("tiny", {"batch": 1, "heads": 4, "seq_len": 64, "head_dim": 64}),
            ("small", {"batch": 2, "heads": 8, "seq_len": 256, "head_dim": 64}),
            ("medium", {"batch": 2, "heads": 16, "seq_len": 512, "head_dim": 64}),
            ("large", {"batch": 2, "heads": 32, "seq_len": 1024, "head_dim": 64}),
            ("xlarge", {"batch": 2, "heads": 32, "seq_len": 2048, "head_dim": 64}),
            ("long", {"batch": 1, "heads": 32, "seq_len": 4096, "head_dim": 64}),
            ("gqa", {"batch": 2, "heads": 32, "seq_len": 1024, "head_dim": 128}),
            ("llm_7b", {"batch": 1, "heads": 32, "seq_len": 2048, "head_dim": 128}),
        ],
        dtypes=_FP16_BF16,
        tolerances=_TOL_LOOSE,
        # 4*B*H*S^2*D FLOPs (Q@K^T + softmax + attn@V)
        flops_fn=(
            4 * size("batch") * size("heads") * (size("seq_len") ** 2) * size("head_dim")
        ),
        # Q, K, V in and one output tensor out
        bytes_fn=4 * bhsd * DT_BYTES,
        edge_cases=(
            EdgeCase(name="edge_127", size={"batch": 1, "heads": 8, "seq_len": 127, "head_dim": 64}),
            EdgeCase(
                name="edge_1023", size={"batch": 1, "heads": 8, "seq_len": 1023, "head_dim": 64}
            ),
        ),
        shape_keys=("batch", "heads", "seq_len", "head_dim"),
        shape_aliases={
            "B": "batch",
            "H": "heads",
            "N": "seq_len",
            "S": "seq_len",
            "D": "head_dim",
            "batch": "batch",
            "heads": "heads",
            "seq_len": "seq_len",
            "head_dim": "head_dim",
        },
        starter_kernels=starter_kernels_for("flash_attention"),
        speedup_estimate="2-4x",
    )


def _spec_fused_mlp() -> KernelSpec:
    return KernelSpec(
        name="fused_mlp",
        reference_fn=lazy_callable("reference", "fused_mlp_ref"),
        input_generator=gen_fused_mlp_inputs,
        sizes=[
            ("tiny", {"batch": 32, "dim": 128, "hidden": 256}),
            ("small", {"batch": 256, "dim": 512, "hidden": 1024}),
            ("medium", {"batch": 1024, "dim": 1024, "hidden": 2048}),
            ("large", {"batch": 2048, "dim": 2048, "hidden": 5504}),
            ("xlarge", {"batch": 4096, "dim": 4096, "hidden": 11008}),
            ("llm_7b", {"batch": 2048, "dim": 4096, "hidden": 11008}),
            ("llm_13b", {"batch": 2048, "dim": 5120, "hidden": 13824}),
        ],
        dtypes=_FP16_BF16_FP32,
        tolerances=_TOL_LOOSE,
        # gate_proj + up_proj + down_proj
        flops_fn=2 * size("batch") * size("dim") * size("hidden") * 3,
        bytes_fn=(
            size("batch") * size("dim")
            + size("hidden") * size("dim") * 3
            + size("batch") * size("dim")
        ) * DT_BYTES,
        edge_cases=(
            EdgeCase(name="edge_1023", size={"batch": 1023, "dim": 1024, "hidden": 2048}),
            EdgeCase(name="edge_4097", size={"batch": 4097, "dim": 512, "hidden": 1024}),
        ),
        shape_keys=("batch", "dim", "hidden"),
        shape_aliases={
            "M": "batch",
            "N": "hidden",
            "K": "dim",
            "batch": "batch",
            "dim": "dim",
            "hidden": "hidden",
        },
        starter_kernels=starter_kernels_for("fused_mlp"),
        speedup_estimate="2-3x",
    )


def _spec_cross_entropy() -> KernelSpec:
    return KernelSpec(
        name="cross_entropy",
        reference_fn=lazy_callable("reference", "cross_entropy_ref"),
        input_generator=gen_cross_entropy_inputs,
        sizes=[
            ("tiny", {"batch": 32, "vocab": 256}),
            ("small", {"batch": 256, "vocab": 1024}),
            ("medium", {"batch": 1024, "vocab": 4096}),
            ("large", {"batch": 4096, "vocab": 32000}),
            ("xlarge", {"batch": 8192, "vocab": 50257}),
            ("llama", {"batch": 4096, "vocab": 32000}),
            ("gpt2", {"batch": 4096, "vocab": 50257}),
        ],
        dtypes=_FP16_BF16_FP32,
        tolerances={
            "float16": Tolerance(atol=1e-2, rtol=1e-2),
            "bfloat16": Tolerance(atol=2e-2, rtol=2e-2),
            "float32": Tolerance(atol=1e-5, rtol=1e-5),
        },
        # log_softmax + nll
        flops_fn=4 * size("batch") * size("vocab"),
        bytes_fn=(size("batch") * size("vocab") + size("batch")) * DT_BYTES,
        edge_cases=(
            EdgeCase(name="edge_1023", size={"batch": 1023, "vocab": 32000}),
            EdgeCase(name="edge_50257", size={"batch": 4096, "vocab": 50257}),
        ),
        shape_keys=("batch", "vocab"),
        shape_aliases={"batch": "batch", "vocab": "vocab"},
        starter_kernels=starter_kernels_for("cross_entropy"),
        speedup_estimate="1.5-2x",
    )


def _spec_rotary_embedding() -> KernelSpec:
    return KernelSpec(
        name="rotary_embedding",
        reference_fn=lazy_callable("reference", "rotary_embedding_ref"),
        input_generator=gen_rotary_embedding_inputs,
        sizes=[
            ("tiny", {"batch": 1, "heads": 4, "seq_len": 64, "head_dim": 64}),
            ("small", {"batch": 2, "heads": 8, "seq_len": 256, "head_dim": 64}),
            ("medium", {"batch": 2, "heads": 16, "seq_len": 512, "head_dim": 64}),
            ("large", {"batch": 2, "heads": 32, "seq_len": 1024, "head_dim": 128}),
            ("xlarge", {"batch": 2, "heads": 32, "seq_len": 2048, "head_dim": 128}),
            ("llm_7b", {"batch": 1, "heads": 32, "seq_len": 2048, "head_dim": 128}),
            ("llm_13b", {"batch": 1, "heads": 40, "seq_len": 2048, "head_dim": 128}),
        ],
        dtypes=_FP16_BF16_FP32,
        tolerances=_TOL_TIGHT,
        # mul + add per element, x2 (cos and sin parts)
        flops_fn=6 * size("batch") * size("heads") * size("seq_len") * size("head_dim"),
        bytes_fn=(
            size("batch") * size("heads") * size("seq_len") * size("head_dim") * 2
            + size("seq_len") * size("head_dim")
        ) * DT_BYTES,
        edge_cases=(
            EdgeCase(name="edge_127", size={"batch": 1, "heads": 8, "seq_len": 127, "head_dim": 64}),
            EdgeCase(
                name="edge_1023", size={"batch": 1, "heads": 8, "seq_len": 1023, "head_dim": 128}
            ),
        ),
        shape_keys=("batch", "heads", "seq_len", "head_dim"),
        shape_aliases={
            "B": "batch",
            "H": "heads",
            "N": "seq_len",
            "S": "seq_len",
            "D": "head_dim",
            "batch": "batch",
            "heads": "heads",
            "seq_len": "seq_len",
            "head_dim": "head_dim",
        },
        starter_kernels=starter_kernels_for("rotary_embedding"),
        speedup_estimate="1.5-2x",
    )


def _spec_rmsnorm() -> KernelSpec:
    return KernelSpec(
        name="rmsnorm",
        reference_fn=lazy_callable("reference", "rmsnorm_ref"),
        input_generator=gen_rmsnorm_inputs,
        sizes=[
            ("small", {"M": 1024, "N": 768}),
            ("medium", {"M": 4096, "N": 1024}),
            ("large", {"M": 4096, "N": 4096}),
            ("llama", {"M": 2048, "N": 4096}),
        ],
        dtypes=_FP16_BF16,
        tolerances=_TOL_REDUCTION,
        # square, mean, sqrt, div, mul
        flops_fn=6 * size("M") * size("N"),
        bytes_fn=(2 * size("M") * size("N") + size("N")) * DT_BYTES,
        edge_cases=(
            EdgeCase(name="edge_1023", size={"M": 1023, "N": 768}),
            EdgeCase(name="edge_4097", size={"M": 4097, "N": 1024}),
        ),
        shape_keys=("M", "N"),
        shape_aliases={"M": "M", "N": "N"},
        starter_kernels=starter_kernels_for("rmsnorm"),
        speedup_estimate="1.5-3x",
    )


def _spec_reduce() -> KernelSpec:
    return KernelSpec(
        name="reduce",
        reference_fn=lazy_callable("reference", "reduce_sum_ref"),
        input_generator=gen_reduce_inputs,
        sizes=[
            ("small", {"M": 1024, "N": 1024}),
            ("medium", {"M": 4096, "N": 4096}),
            ("large", {"M": 8192, "N": 8192}),
            ("wide", {"M": 1024, "N": 32768}),
        ],
        dtypes=_FP16_BF16,
        tolerances=_TOL_REDUCTION,
        # N additions per row
        flops_fn=size("M") * size("N"),
        bytes_fn=(size("M") * size("N") + size("M")) * DT_BYTES,
        edge_cases=(
            EdgeCase(name="edge_1023", size={"M": 1023, "N": 1024}),
            EdgeCase(name="edge_4097", size={"M": 4096, "N": 4097}),
        ),
        shape_keys=("M", "N"),
        shape_aliases={"M": "M", "N": "N"},
        starter_kernels=starter_kernels_for("reduce"),
        speedup_estimate="1.5-2x",
        # Extraction fell back to 4096x4096 (not the 8192x8192 'large' size)
        # before the registry existed; keep that fallback byte-for-byte.
        default_shape={"M": 4096, "N": 4096},
    )


#: Factories in the order the operations were declared by ``KERNEL_CONFIGS``.
_BUILTIN_FACTORIES = (
    _spec_matmul,
    _spec_softmax,
    _spec_layernorm,
    _spec_flash_attention,
    _spec_fused_mlp,
    _spec_cross_entropy,
    _spec_rotary_embedding,
    _spec_rmsnorm,
    _spec_reduce,
)


def builtin_specs() -> tuple[KernelSpec, ...]:
    """Build every built-in specification, in canonical order."""
    return tuple(factory() for factory in _BUILTIN_FACTORIES)
