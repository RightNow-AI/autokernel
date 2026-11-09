"""The gqa_prefill_attention starter kernel against the float32 oracle, with contiguous and paged KV.

On a CUDA machine it runs natively; elsewhere it runs in Triton's CPU interpreter (TRITON_INTERPRET=1),
which needs only CPU PyTorch, numpy and Triton.
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

try:
    kernel = importlib.import_module("kernels.gqa_prefill_attention")
except ImportError:  # Triton missing
    kernel = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@unittest.skipIf(kernel is None, "triton is not installed")
class StarterKernelTests(unittest.TestCase):
    def check(self, dtype, atol, **size):
        size = {"batch": 1, "heads_q": 4, "heads_kv": 2, "head_dim": 32, **size}
        x = bench.gen_gqa_prefill_attention_inputs(size, dtype, DEVICE, seed=3)
        out = kernel.kernel_fn(**x)
        expected = bench._ref_gqa_prefill_attention(x)
        torch.testing.assert_close(out.float(), expected.float(), atol=atol, rtol=atol, msg=str(size))

    def test_chunks_against_longer_caches(self):
        for q_len, kv_len in ((40, 100), (64, 64), (1, 77), (130, 300), (65, 65)):
            self.check(torch.float32, 1e-4, q_len=q_len, kv_len=kv_len)

    def test_groups_batches_and_head_dims(self):
        self.check(torch.float32, 1e-4, batch=2, heads_q=8, heads_kv=1, q_len=33, kv_len=97)
        self.check(torch.float32, 1e-4, heads_q=4, heads_kv=4, q_len=20, kv_len=50, head_dim=64)
        self.check(torch.float32, 1e-4, heads_q=4, heads_kv=1, q_len=70, kv_len=150, head_dim=256)

    def test_half_precision(self):
        self.check(torch.float16, 1e-2, q_len=70, kv_len=200)

    def test_paged_kv(self):
        # The last page partly filled, one page larger than kv_len, a page size that is not a power of two,
        # a key per page, and two block-table rows
        for q_len, kv_len, page_size in ((40, 100, 16), (1, 77, 16), (65, 65, 64), (130, 300, 48), (20, 50, 1)):
            self.check(torch.float32, 1e-4, q_len=q_len, kv_len=kv_len, page_size=page_size)
        self.check(torch.float32, 1e-4, batch=2, heads_q=8, heads_kv=1, q_len=33, kv_len=97, page_size=16)
        self.check(torch.float32, 1e-4, heads_q=4, heads_kv=1, q_len=70, kv_len=150, head_dim=256, page_size=32)
        self.check(torch.float16, 1e-2, q_len=70, kv_len=200, page_size=16)


if __name__ == "__main__":
    unittest.main()
