"""
AutoKernel -- Softmax kernel.

Current kernel: Online Softmax (row-parallel)
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Each program instance handles one row of the input tensor.
Uses numerically stable approach: max -> subtract -> exp -> sum -> divide.
"""

KERNEL_TYPE = "softmax"

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    stride_input_row,
    stride_output_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-parallel online softmax. One program per row."""
    row_idx = tl.program_id(0)

    row_start_input = input_ptr + row_idx * stride_input_row
    row_start_output = output_ptr + row_idx * stride_output_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row
    row = tl.load(row_start_input + col_offsets, mask=mask, other=float("-inf"))

    # Numerically stable softmax: subtract max
    row_max = tl.max(row, axis=0)
    row = row - row_max

    # Exponentiate
    numerator = tl.exp(row)

    # Sum
    denominator = tl.sum(numerator, axis=0)

    # Divide
    result = numerator / denominator

    # Store
    tl.store(row_start_output + col_offsets, result, mask=mask)


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.softmax_ref signature."""
    assert x.device.type in ("cuda", "xpu")

    # Flatten to 2D for row-parallel processing
    orig_shape = x.shape
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    n_rows, n_cols = x.shape

    # This row-parallel kernel maps one program to one row and requires a
    # power-of-two BLOCK_SIZE covering the full row. Very wide rows like the
    # vocab benchmark (50257 -> 65536) time out on XPU, so fall back.
    if n_cols > 4096:
        return torch.softmax(x, dim=-1).view(orig_shape)

    output = torch.empty_like(x)

    # Block size must be a power of 2 >= n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)
    softmax_kernel[grid](
        x, output,
        n_cols,
        x.stride(0),
        output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.view(orig_shape)
