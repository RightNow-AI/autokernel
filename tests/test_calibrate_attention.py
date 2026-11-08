"""CPU checks for calibrate.py's attention peak: shape parsing, the oracle gate before timing, the FLOP count."""

import math
import os
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate  # noqa: E402
import reference  # noqa: E402

SMALL = {"batch": 1, "heads_q": 8, "heads_kv": 2, "q_len": 16, "kv_len": 48, "head_dim": 32}
CPU_BACKENDS = ("MATH", "FLASH_ATTENTION")  # CUDA-only backends are unsupported on CPU


class ShapeTests(unittest.TestCase):
    def test_parses_and_refuses_shapes(self):
        self.assertEqual(calibrate.parse_attention_shape("8192x1048576:16/2:256"),
                         {"batch": 1, "heads_q": 16, "heads_kv": 2, "q_len": 8192, "kv_len": 1048576, "head_dim": 256})
        for bad in ("8192:16/2:256", "8192x131072:16/3:256", "9x8:2/2:64", "0x8:2/2:64", "axb:1/1:1"):
            with self.assertRaises(ValueError, msg=bad):
                calibrate.parse_attention_shape(bad)

    def test_default_shapes_parse(self):
        for spec in calibrate.ATTENTION_DEFAULT:
            calibrate.parse_attention_shape(spec)


class MeasureTests(unittest.TestCase):
    def test_non_causal_oracle_matches_full_attention(self):
        x = calibrate.gen_gqa_prefill_attention_inputs(SMALL, torch.float32, "cpu")
        k = x["K"].repeat_interleave(4, 1)
        v = x["V"].repeat_interleave(4, 1)
        expected = torch.softmax(x["Q"] @ k.transpose(-2, -1) / math.sqrt(32), dim=-1) @ v
        torch.testing.assert_close(reference.gqa_prefill_attention_ref(x["Q"], x["K"], x["V"], causal=False,
                                                                       kv_tile=7), expected, atol=1e-5, rtol=1e-5)

    def test_times_only_backends_that_match_the_oracle(self):
        timed = []
        with patch.object(calibrate, "ATTENTION_BACKENDS", CPU_BACKENDS + ("CUDNN_ATTENTION",)):
            result = calibrate.measure_attention(SMALL, torch.bfloat16, "cpu",
                                                 bench_fn=lambda fn: timed.append(fn) or 2.0)
        backends = result["backends"]
        self.assertEqual(backends["cudnn_attention"], {"unsupported": True})
        self.assertIn("tflops", backends["math"])
        flops = 4 * 8 * 16 * 48 * 32
        self.assertEqual(result["flops"], flops)
        self.assertAlmostEqual(backends["math"]["tflops"], flops / 2e-3 / 1e12)
        self.assertEqual(len(timed), sum("tflops" in b for b in backends.values()))
        self.assertIn(result["best_backend"], [k for k, b in backends.items() if "tflops" in b])

    def test_a_backend_with_wrong_output_is_reported_not_timed(self):
        real = F.scaled_dot_product_attention

        def broken(q, k, v, **kwargs):  # drops the last key: a "fast" kernel that attends to too little
            return real(q, k[..., :-1, :], v[..., :-1, :], **kwargs)
        timed = []
        with patch.object(calibrate, "ATTENTION_BACKENDS", ("MATH",)), patch.object(calibrate.F, "scaled_dot_product_attention", broken):
            result = calibrate.measure_attention(SMALL, torch.float16, "cpu",
                                                 bench_fn=lambda fn: timed.append(fn) or 1.0)
        self.assertTrue(result["backends"]["math"]["wrong"])
        self.assertEqual(timed, [])
        self.assertIsNone(result["best_backend"])

    def test_grouped_heads_fall_back_to_expanded_heads(self):
        real = F.scaled_dot_product_attention

        def refuses_groups(q, k, v, enable_gqa=False, **kwargs):
            if enable_gqa:
                raise RuntimeError("no grouped heads here")
            return real(q, k, v, **kwargs)
        with patch.object(calibrate, "ATTENTION_BACKENDS", ("MATH",)), patch.object(calibrate.F, "scaled_dot_product_attention", refuses_groups):
            result = calibrate.measure_attention(SMALL, torch.float32, "cpu", bench_fn=lambda fn: fn() is not None and 1.0)
        self.assertTrue(result["backends"]["math"]["kv_expanded"])
        self.assertIn("tflops", result["backends"]["math"])


if __name__ == "__main__":
    unittest.main()
