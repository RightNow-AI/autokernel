"""External specification fixture used by the loader and CLI tests.

Kept deliberately small and CPU-friendly: the tests here care about discovery,
validation and CLI plumbing, not about kernel performance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from autokernel.specs import DT_BYTES, EdgeCase, KernelSpec, Tolerance, resolve_torch_dtype, size

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Reuse the example starter kernel so the fixture declares a real file.
STARTER_KERNEL = REPO_ROOT / "examples" / "custom_ops" / "add_kernel.py"


def add_ref(x: Any, y: Any) -> Any:
    return x + y


def gen_inputs(size_map: Mapping[str, int], dtype: Any, device: str, seed: int = 42) -> dict:
    import torch

    torch.manual_seed(seed)
    torch_dtype = resolve_torch_dtype(dtype)
    rows, cols = size_map["rows"], size_map["cols"]
    return {
        "x": torch.randn(rows, cols, device=device, dtype=torch_dtype),
        "y": torch.randn(rows, cols, device=device, dtype=torch_dtype),
    }


def _build(name: str = "fixture_add") -> KernelSpec:
    return KernelSpec(
        name=name,
        reference_fn=add_ref,
        input_generator=gen_inputs,
        sizes={
            "small": {"rows": 8, "cols": 16},
            "medium": {"rows": 32, "cols": 32},
            "large": {"rows": 64, "cols": 64},
        },
        dtypes=("float32",),
        tolerances={"float32": Tolerance(atol=1e-5, rtol=1e-5)},
        flops_fn=size("rows") * size("cols"),
        bytes_fn=3 * size("rows") * size("cols") * DT_BYTES,
        edge_cases=(EdgeCase(name="edge_7", size={"rows": 7, "cols": 7}),),
        shape_keys=("rows", "cols"),
        shape_aliases={"M": "rows", "N": "cols"},
        starter_kernels={"triton": STARTER_KERNEL},
        speedup_estimate="1.0x",
    )


#: A ready-made specification.
SPEC = _build()


def SPEC_FACTORY() -> KernelSpec:
    """Zero-argument factory returning a specification."""
    return _build()


#: A specification whose name collides with a built-in operation.
COLLIDING_SPEC = _build(name="matmul")

#: Not a specification at all.
NOT_A_SPEC = 42


def BAD_FACTORY() -> int:
    """A callable that returns the wrong type."""
    return 42


def RAISING_FACTORY() -> KernelSpec:
    """A callable that fails."""
    raise RuntimeError("factory exploded")
