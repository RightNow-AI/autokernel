"""Structured-output external operation: affine transform with an aux branch.

This example proves the Week 2 verification framework: an operation whose
output is a *tree* --

.. code-block:: python

    {
        "output": y,                    # tensor
        "aux": (residual, n_terms),     # (tensor, metadata value)
    }

-- flows through the same five correctness stages as a single-tensor
operation. The reference is differentiable, so the example also exercises
optional backward verification, and it is pure PyTorch so every check runs
on CPU:

.. code-block:: bash

    cp examples/custom_ops/affine_kernel.py kernel.py
    uv run bench.py --spec examples/custom_ops/affine.py:SPEC --quick
    uv run bench.py --spec examples/custom_ops/affine.py:SPEC --check-backward
    uv run bench.py --spec examples/custom_ops/affine.py:SPEC --check-compile

The fixture exists to prove the framework. It is not a production kernel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from autokernel.specs import (
    DT_BYTES,
    BackwardSpec,
    EdgeCase,
    KernelSpec,
    Tolerance,
    resolve_torch_dtype,
    size,
)

_HERE = Path(__file__).resolve().parent

#: Starter candidate the agent begins optimizing from.
STARTER_KERNEL = _HERE / "affine_kernel.py"


def affine_ref(x: Any, scale: Any, bias: Any) -> dict:
    """Reference: ``y = x * scale + bias`` with an aux residual branch.

    Returns a nested output tree: the main tensor, an aux tuple holding the
    residual tensor, and a non-tensor metadata value (the number of terms in
    the affine expression).
    """
    y = x * scale + bias
    # Compute the auxiliary branch in float32 and cast once at its boundary.
    # This gives eager and Inductor-fused FP16 execution the same numerical
    # contract without loosening the output tolerance.
    residual = (
        x.float() * scale.float() + bias.float() - x.float()
    ).to(x.dtype)
    return {"output": y, "aux": (residual, 3)}


def gen_affine_inputs(
    size_map: Mapping[str, int], dtype: Any, device: str, seed: int = 42
) -> dict:
    """Deterministic inputs for a fixed seed."""
    import torch

    torch.manual_seed(seed)
    torch_dtype = resolve_torch_dtype(dtype)
    rows, cols = size_map["rows"], size_map["cols"]
    x = torch.randn(rows, cols, device=device, dtype=torch_dtype)
    scale = torch.randn(cols, device=device, dtype=torch_dtype)
    bias = torch.randn(cols, device=device, dtype=torch_dtype)
    return {"x": x, "scale": scale, "bias": bias}


SPEC = KernelSpec(
    name="custom_affine",
    reference_fn=affine_ref,
    input_generator=gen_affine_inputs,
    sizes={
        "small": {"rows": 128, "cols": 256},
        "medium": {"rows": 512, "cols": 512},
        "large": {"rows": 2048, "cols": 2048},
    },
    dtypes=("float16", "float32"),
    tolerances={
        "float16": Tolerance(atol=1e-3, rtol=1e-3),
        "float32": Tolerance(atol=1e-5, rtol=1e-5),
    },
    # mul + add + residual sub per element
    flops_fn=3 * size("rows") * size("cols"),
    # read x, scale, bias; write output and residual (metadata is tiny)
    bytes_fn=(3 * size("rows") + 2) * size("cols") * DT_BYTES,
    edge_cases=(
        EdgeCase(name="edge_1023", size={"rows": 1023, "cols": 1023}),
        EdgeCase(name="edge_single_row", size={"rows": 1, "cols": 1025}),
    ),
    shape_keys=("rows", "cols"),
    shape_aliases={"M": "rows", "N": "cols", "rows": "rows", "cols": "cols"},
    starter_kernels={"pytorch": STARTER_KERNEL},
    speedup_estimate="1.0-1.3x",
    backward_spec=BackwardSpec(
        differentiable_inputs=("x", "scale", "bias"),
    ),
)
