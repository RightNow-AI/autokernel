"""Starter kernel for the external ``custom_affine`` example operation.

Copy this file to ``kernel.py`` and benchmark it against the external spec::

    cp examples/custom_ops/affine_kernel.py kernel.py
    uv run bench.py --spec examples/custom_ops/affine.py:SPEC --quick

The candidate is intentionally plain PyTorch: the fixture proves that a
structured, multi-output operation flows through the harness (including CPU
verification), not that an affine transform can be made faster. Its signature
and output tree must match the spec's reference exactly.
"""

KERNEL_TYPE = "custom_affine"

import torch


def kernel_fn(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> dict:
    """Entry point called by bench.py. Must match the reference signature."""
    y = x * scale + bias
    residual = (y.float() - x.float()).to(x.dtype)
    return {"output": y, "aux": (residual, 3)}
