import unittest
import torch

import bench
import reference
from kernels.matmul import kernel_fn, matmul_kernel


class BenchTF32CorrectnessTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU tests")
    def test_ref_matmul_preserves_ieee_fp32(self):
        A = torch.randn(128, 128, device="cuda", dtype=torch.float32)
        B = torch.randn(128, 128, device="cuda", dtype=torch.float32)

        # Baseline IEEE reference with allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32 = False
        expected_ieee = reference.matmul_ref(A, B)

        # Ensure _ref_matmul produces IEEE result even if ambient allow_tf32 is True
        torch.backends.cuda.matmul.allow_tf32 = True
        ref_out = bench._ref_matmul({"A": A, "B": B})

        max_diff = (ref_out - expected_ieee).abs().max().item()
        self.assertAlmostEqual(max_diff, 0.0, places=5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU tests")
    def test_kernel_fn_fp32_passes_tight_tolerance(self):
        M, N, K = 128, 128, 128
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)

        out = kernel_fn(A, B)
        expected = bench._ref_matmul({"A": A, "B": B})

        tol = bench.KERNEL_CONFIGS["matmul"]["tolerances"][torch.float32]
        cmp_res = bench._compare(out, expected, **tol)
        self.assertTrue(
            cmp_res["match"],
            f"FP32 matmul failed tolerance check: {cmp_res.get('reason')}",
        )


if __name__ == "__main__":
    unittest.main()
