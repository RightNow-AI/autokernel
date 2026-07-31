"""Starter kernel for the external ``custom_add`` example operation.

Copy this file to ``kernel.py`` and benchmark it against the external spec::

    cp examples/custom_ops/add_kernel.py kernel.py
    uv run bench.py --spec examples/custom_ops/add.py:SPEC --quick

It is intentionally simple: the example proves that an out-of-tree operation
flows through the same harness, not that elementwise add can be made faster.
"""

KERNEL_TYPE = "custom_add"

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def kernel_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match the spec's reference signature."""
    assert x.shape == y.shape, f"shape mismatch: {x.shape} vs {y.shape}"
    x_contig = x.contiguous()
    y_contig = y.contiguous()
    out = torch.empty_like(x_contig)

    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    add_kernel[grid](x_contig, y_contig, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
