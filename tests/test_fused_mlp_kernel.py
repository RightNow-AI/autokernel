"""Tests for the fused_mlp starter kernel with both SiLU and GELU activations.

On a CUDA machine it runs natively; elsewhere it runs in Triton's CPU interpreter (TRITON_INTERPRET=1).
"""

import importlib
import os
import sys
import unittest

import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402
import reference  # noqa: E402

try:
    kernel = importlib.import_module("kernels.fused_mlp")
except ImportError:
    kernel = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@unittest.skipIf(kernel is None, "triton is not installed")
class FusedMLPKernelTests(unittest.TestCase):
    def check_activation(self, activation: str, dtype: torch.dtype, atol: float, rtol: float, **size):
        size = {"batch": 2, "dim": 64, "hidden": 128, **size}
        inputs = bench.gen_fused_mlp_inputs(size, dtype, DEVICE, seed=42)
        out = kernel.kernel_fn(**inputs, activation=activation)
        expected = reference.fused_mlp_ref(**inputs, activation=activation)
        torch.testing.assert_close(out.float(), expected.float(), atol=atol, rtol=rtol, msg=f"{activation} {size}")

    def test_silu_activation(self):
        self.check_activation("silu", torch.float16, atol=1e-2, rtol=1e-2)
        self.check_activation("silu", torch.bfloat16, atol=2e-2, rtol=2e-2)
        self.check_activation("silu", torch.float32, atol=1e-4, rtol=1e-4)

    def test_gelu_activation(self):
        # Covers issue #19: ensure tanh is correctly imported and compiled in fused_gate_up_kernel
        self.check_activation("gelu", torch.float16, atol=1e-2, rtol=1e-2)
        self.check_activation("gelu", torch.bfloat16, atol=2e-2, rtol=2e-2)
        self.check_activation("gelu", torch.float32, atol=1e-4, rtol=1e-4)

    def test_multidimensional_batch(self):
        # Tests [batch, seq_len, dim] handling
        torch.manual_seed(42)
        x = torch.randn(2, 8, 64, device=DEVICE, dtype=torch.float16)
        w_gate = torch.randn(128, 64, device=DEVICE, dtype=torch.float16) * 0.02
        w_up = torch.randn(128, 64, device=DEVICE, dtype=torch.float16) * 0.02
        w_down = torch.randn(64, 128, device=DEVICE, dtype=torch.float16) * 0.02

        out_gelu = kernel.kernel_fn(x, w_gate, w_up, w_down, activation="gelu")
        expected_gelu = reference.fused_mlp_ref(x, w_gate, w_up, w_down, activation="gelu")
        self.assertEqual(out_gelu.shape, (2, 8, 64))
        torch.testing.assert_close(out_gelu.float(), expected_gelu.float(), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
