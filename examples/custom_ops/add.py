"""Minimal external custom operation: elementwise add.

This example exists to prove extensibility, not performance. It shows the whole
contract an out-of-tree operation must satisfy:

* a PyTorch reference,
* a deterministic input generator,
* ``small`` / ``medium`` / ``large`` sizes,
* tolerances per dtype,
* FLOP and byte accounting,
* a starter kernel,
* an exported ``SPEC``.

Run it through the normal harness::

    cp examples/custom_ops/add_kernel.py kernel.py
    uv run bench.py --spec examples/custom_ops/add.py:SPEC --quick
    uv run extract.py --spec examples/custom_ops/add.py:SPEC --top 1

No change to ``bench.py``, ``extract.py``, ``reference.py`` or any central
operation map is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from autokernel.specs import (
    DT_BYTES,
    EdgeCase,
    KernelSpec,
    Tolerance,
    resolve_torch_dtype,
    size,
)

_HERE = Path(__file__).resolve().parent

#: Starter kernel the agent begins optimizing from.
STARTER_KERNEL = _HERE / "add_kernel.py"


def add_ref(x: Any, y: Any) -> Any:
    """Reference implementation: elementwise sum of two tensors."""
    return x + y


def gen_add_inputs(
    size_map: Mapping[str, int], dtype: Any, device: str, seed: int = 42
) -> dict:
    """Deterministic inputs for a fixed seed."""
    import torch

    torch.manual_seed(seed)
    torch_dtype = resolve_torch_dtype(dtype)
    rows, cols = size_map["rows"], size_map["cols"]
    x = torch.randn(rows, cols, device=device, dtype=torch_dtype)
    y = torch.randn(rows, cols, device=device, dtype=torch_dtype)
    return {"x": x, "y": y}


SPEC = KernelSpec(
    name="custom_add",
    reference_fn=add_ref,
    input_generator=gen_add_inputs,
    sizes={
        "small": {"rows": 256, "cols": 512},
        "medium": {"rows": 1024, "cols": 1024},
        "large": {"rows": 4096, "cols": 4096},
    },
    dtypes=("float16", "bfloat16", "float32"),
    tolerances={
        "float16": Tolerance(atol=1e-3, rtol=1e-3),
        "bfloat16": Tolerance(atol=2e-3, rtol=2e-3),
        "float32": Tolerance(atol=1e-5, rtol=1e-5),
    },
    # one add per element
    flops_fn=size("rows") * size("cols"),
    # two reads and one write per element
    bytes_fn=3 * size("rows") * size("cols") * DT_BYTES,
    edge_cases=(
        EdgeCase(name="edge_1023", size={"rows": 1023, "cols": 1023}),
        EdgeCase(name="edge_4097", size={"rows": 4097, "cols": 129}),
    ),
    shape_keys=("rows", "cols"),
    shape_aliases={"M": "rows", "N": "cols", "rows": "rows", "cols": "cols"},
    starter_kernels={"triton": STARTER_KERNEL},
    speedup_estimate="1.0-1.2x",
)
