"""CPU checks for the gqa_prefill_attention op: the float32 oracle, the SDPA timing baseline and the FLOP count.

Run: python -m unittest tests/test_gqa_prefill_attention.py (CPU PyTorch is enough).
"""

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402
import reference  # noqa: E402

CONFIG = bench.KERNEL_CONFIGS["gqa_prefill_attention"]


def naive(Q, K, V):
    """Full-matrix float64 attention with expanded KV heads and the bottom-right causal mask."""
    group = Q.shape[1] // K.shape[1]
    q, k, v = Q.double(), K.double().repeat_interleave(group, 1), V.double().repeat_interleave(group, 1)
    q_len, kv_len = q.shape[2], k.shape[2]
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    limit = torch.arange(q_len)[:, None] + (kv_len - q_len)
    s = s.masked_fill(torch.arange(kv_len)[None, :] > limit, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def inputs(batch=1, heads_q=8, heads_kv=2, q_len=16, kv_len=40, head_dim=32, dtype=torch.float32, seed=0):
    size = dict(batch=batch, heads_q=heads_q, heads_kv=heads_kv, q_len=q_len, kv_len=kv_len, head_dim=head_dim)
    return size, bench.gen_gqa_prefill_attention_inputs(size, dtype, "cpu", seed=seed)


class OracleTests(unittest.TestCase):
    def test_matches_full_matrix_attention_across_tilings(self):
        for shape in (dict(q_len=16, kv_len=16), dict(q_len=16, kv_len=40), dict(q_len=1, kv_len=33),
                      dict(batch=2, heads_q=4, heads_kv=4, q_len=9, kv_len=50), dict(heads_q=8, heads_kv=1)):
            _, x = inputs(**shape)
            expected = naive(x["Q"], x["K"], x["V"])
            for q_tile, kv_tile in ((1024, 8192), (7, 13), (1, 1), (16, 5)):
                out = reference.gqa_prefill_attention_ref(x["Q"], x["K"], x["V"], q_tile=q_tile, kv_tile=kv_tile)
                self.assertEqual(out.dtype, x["Q"].dtype)
                torch.testing.assert_close(out.double(), expected, atol=1e-5, rtol=1e-5,
                                           msg=f"{shape} tiles {q_tile}/{kv_tile}")

    def test_query_never_reads_past_its_causal_edge(self):
        _, x = inputs(q_len=8, kv_len=24)
        base = reference.gqa_prefill_attention_ref(x["Q"], x["K"], x["V"], kv_tile=5)
        # Query 0 sits at position 16: key 17 is invisible to it, key 16 is its last visible key.
        hidden, visible = {k: t.clone() for k, t in x.items()}, {k: t.clone() for k, t in x.items()}
        hidden["K"][:, :, 17] += 50.0
        hidden["V"][:, :, 17] += 50.0
        visible["V"][:, :, 16] += 50.0
        out_hidden = reference.gqa_prefill_attention_ref(hidden["Q"], hidden["K"], hidden["V"], kv_tile=5)
        out_visible = reference.gqa_prefill_attention_ref(visible["Q"], visible["K"], visible["V"], kv_tile=5)
        torch.testing.assert_close(out_hidden[:, :, 0], base[:, :, 0], atol=0, rtol=0)
        self.assertFalse(torch.allclose(out_hidden[:, :, 1:], base[:, :, 1:]))
        self.assertFalse(torch.allclose(out_visible[:, :, 0], base[:, :, 0]))

    def test_refuses_inconsistent_shapes(self):
        _, x = inputs()
        _, y = inputs(heads_q=6, heads_kv=4)
        with self.assertRaises(ValueError):
            reference.gqa_prefill_attention_ref(y["Q"], y["K"], y["V"])  # 6 query heads over 4 KV heads
        with self.assertRaises(ValueError):
            reference.gqa_prefill_attention_ref(x["Q"], x["K"][:, :, :8], x["V"][:, :, :8])  # q_len > kv_len
        with self.assertRaises(ValueError):
            reference.gqa_prefill_attention_ref(x["Q"], x["K"], x["V"][:, :, :20])


class ConfigTests(unittest.TestCase):
    def test_flops_count_only_visible_pairs(self):
        for q_len, kv_len in ((1, 1), (1, 9), (5, 5), (7, 30), (64, 1000)):
            size = dict(batch=2, heads_q=8, heads_kv=2, q_len=q_len, kv_len=kv_len, head_dim=32)
            pairs = sum(kv_len - q_len + i + 1 for i in range(q_len))
            self.assertEqual(CONFIG["flops_fn"](size), 4 * 2 * 8 * 32 * pairs)
        size = dict(batch=1, heads_q=16, heads_kv=2, q_len=8192, kv_len=1048576, head_dim=256)
        self.assertEqual(CONFIG["bytes_fn"](size, torch.bfloat16), (2 * 16 * 8192 + 2 * 2 * 1048576) * 256 * 2)

    def test_inputs_give_peaked_attention_with_order_one_outputs(self):
        size, x = inputs(heads_q=8, heads_kv=1, q_len=64, kv_len=16384, head_dim=128, seed=1)
        scores = (x["Q"][0, 0] @ x["K"][0, 0].T) / math.sqrt(128)
        self.assertAlmostEqual(scores.std().item(), 4.0, delta=0.3)
        out = reference.gqa_prefill_attention_ref(x["Q"], x["K"], x["V"])
        self.assertGreater(out.float().pow(2).mean().sqrt().item(), 0.3)  # far above the 2e-2 tolerance

    def test_sdpa_baseline_matches_the_oracle(self):
        for shape in (dict(q_len=16, kv_len=40), dict(q_len=40, kv_len=40), dict(q_len=1, kv_len=17),
                      dict(heads_q=8, heads_kv=8)):
            _, x = inputs(**shape)
            torch.testing.assert_close(bench._sdpa_gqa_prefill_attention(x),
                                       bench._ref_gqa_prefill_attention(x), atol=1e-5, rtol=1e-5)

    def test_registered_sizes_are_valid_gqa_chunks(self):
        for label, size in CONFIG["test_sizes"] + CONFIG["edge_sizes"]:
            self.assertEqual(size["heads_q"] % size["heads_kv"], 0, label)
            self.assertTrue(1 <= size["q_len"] <= size["kv_len"], label)
        self.assertIn("large", [label for label, _ in CONFIG["test_sizes"]])
        self.assertIs(CONFIG["test_dtypes"][0], torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
