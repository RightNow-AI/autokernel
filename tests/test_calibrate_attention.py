"""CPU checks for calibrate.py's attention peak: shape parsing, the oracle gate before timing, the FLOP count,
and the attention entry in the result next to the bandwidth and GEMM peaks."""

import io
import json
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402
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

    def test_flashinfer_is_measured_when_installed_and_reported_otherwise(self):
        with patch.object(calibrate, "ATTENTION_BACKENDS", ()), patch.dict(sys.modules, {"flashinfer": None}):
            result = calibrate.measure_attention(SMALL, torch.float32, "cpu", bench_fn=lambda fn: 1.0)
        self.assertEqual(result["backends"]["flashinfer"], {"unavailable": True})

        def single_prefill_with_kv_cache(q, k, v, causal=False):  # NHD layout, grouped KV heads
            group = q.shape[1] // k.shape[1]
            qh, kh, vh = q.transpose(0, 1), k.repeat_interleave(group, 1).transpose(0, 1), v.repeat_interleave(group, 1).transpose(0, 1)
            return torch.softmax(qh @ kh.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1).matmul(vh).transpose(0, 1)
        fake = type(sys)("flashinfer")
        fake.single_prefill_with_kv_cache = single_prefill_with_kv_cache
        with patch.object(calibrate, "ATTENTION_BACKENDS", ()), patch.dict(sys.modules, {"flashinfer": fake}):
            result = calibrate.measure_attention(SMALL, torch.float32, "cpu", bench_fn=lambda fn: fn() is not None and 4.0)
        self.assertIn("tflops", result["backends"]["flashinfer"])
        self.assertEqual(result["best_backend"], "flashinfer")


class ResultTests(unittest.TestCase):
    """main() on CPU: the GPU-only measurements are stubbed, the attention measurement runs (math backend)."""

    MS = 1e-7  # every timed backend "takes" this long, so these tiny shapes come out at a few TFLOP/s

    def run_main(self, *argv):
        real = calibrate.measure_attention
        gpu = bench.GPUSpec(name="Test GPU", sm_count=4, memory_gb=8.0, peak_tflops_fp16=100.0,
                            peak_tflops_bf16=100.0, peak_bandwidth_gb_s=1000.0, l2_cache_mb=4.0,
                            compute_capability=(10, 0))
        out = io.StringIO()
        with patch.object(calibrate.torch.cuda, "is_available", lambda: True), \
                patch.object(calibrate, "detect_gpu", lambda: gpu), \
                patch.object(calibrate, "measure_bandwidth_gb_s", lambda copy_gb: 900.0), \
                patch.object(calibrate, "measure_gemm_tflops", lambda size, dtype: 80.0), \
                patch.object(calibrate, "ATTENTION_BACKENDS", ("MATH",)), \
                patch.dict(sys.modules, {"flashinfer": None}), \
                patch.object(calibrate, "measure_attention",
                             lambda size: real(size, torch.bfloat16, "cpu", bench_fn=lambda fn: self.MS)), \
                redirect_stdout(out):
            code = calibrate.main(list(argv))
        return code, out.getvalue()

    def test_attention_peak_sits_next_to_the_bandwidth_and_gemm_peaks(self):
        path = os.path.join(tempfile.mkdtemp(), "calibrate.json")
        code, out = self.run_main("--attention", "16x48:8/2:32", "--attention", "8x40:8/2:64", "--json", "--out", path)
        self.assertEqual(code, 0)
        result = json.loads(out)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), result)  # --out writes the same object
        keys = list(result)
        self.assertEqual(keys[keys.index("bandwidth_gb_s"):],
                         ["bandwidth_gb_s", "fp16_tflops", "bf16_tflops", "attention_tflops", "attention"])
        self.assertEqual(result["bf16_tflops"], {"measured": 80.0, "database": 100.0})
        flops = 4 * 8 * 16 * 48 * 32  # the larger of the two shapes: the same time, so the higher TFLOP/s
        self.assertEqual(result["attention_tflops"],
                         {"measured": round(flops / (self.MS / 1e3) / 1e12, 1), "database": None, "backend": "math",
                          "shape": calibrate.parse_attention_shape("16x48:8/2:32")})
        self.assertGreater(result["attention_tflops"]["measured"], 1.0)
        self.assertEqual([e["shape"]["head_dim"] for e in result["attention"]], [32, 64])
        for entry in result["attention"]:
            self.assertEqual(entry["best_backend"], "math")
            self.assertEqual(entry["best_pct_of_bf16_gemm_measured"], round(100 * entry["best_tflops"] / 80.0, 1))
        # bench.py --calibration reads this file: the head_dim 64 entry is the peak for a head_dim 64 size
        peak = bench.calibrated_attention_peak(result, {"head_dim": 64}, torch.bfloat16)
        self.assertEqual(peak["tflops"], result["attention"][1]["best_tflops"])

    def test_table_output_and_no_attention(self):
        code, out = self.run_main("--attention", "16x48:8/2:32")
        self.assertEqual(code, 0)
        self.assertIn("attention_tflops", out)
        self.assertIn("attention 16x48 8/2 d32 bfloat16 dense: math", out)
        self.assertIn("flashinfer unavailable", out)
        code, out = self.run_main("--attention", "none", "--json")
        result = json.loads(out)
        self.assertEqual((result["attention"], result["attention_tflops"]["measured"]), ([], None))


if __name__ == "__main__":
    unittest.main()
