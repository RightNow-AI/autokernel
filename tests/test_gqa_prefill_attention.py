"""CPU checks for the gqa_prefill_attention op: the float32 oracle (contiguous and paged KV), the SDPA timing
baseline, the FLOP count, the --page-size override and the calibrated attention peak.

Run: python -m unittest tests/test_gqa_prefill_attention.py (CPU PyTorch is enough).
"""

import math
import os
import sys
import unittest
from unittest.mock import patch

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


def inputs(batch=1, heads_q=8, heads_kv=2, q_len=16, kv_len=40, head_dim=32, dtype=torch.float32, seed=0, **extra):
    size = dict(batch=batch, heads_q=heads_q, heads_kv=heads_kv, q_len=q_len, kv_len=kv_len, head_dim=head_dim, **extra)
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


class PagedTests(unittest.TestCase):
    """Paged KV: page pools [num_pages, page_size, heads_kv, head_dim] read through a block table."""

    SHAPES = (dict(page_size=16), dict(page_size=7), dict(page_size=1), dict(page_size=64),  # 64 > kv_len: one page
              dict(batch=2, q_len=9, kv_len=50, page_size=8), dict(q_len=1, kv_len=33, page_size=16),
              dict(heads_q=8, heads_kv=1, q_len=40, kv_len=40, page_size=5))

    def test_paged_oracle_equals_the_contiguous_one_and_naive_attention(self):
        for shape in self.SHAPES:
            paged_size, p = inputs(**shape)
            _, c = inputs(**{k: v for k, v in shape.items() if k != "page_size"})
            torch.testing.assert_close(p["Q"], c["Q"], atol=0, rtol=0)  # one generator: same logical values
            for q_tile, kv_tile in ((1024, 8192), (7, 13), (16, 5)):
                out = reference.gqa_prefill_attention_ref(p["Q"], p["K"], p["V"], q_tile=q_tile, kv_tile=kv_tile,
                                                          block_table=p["block_table"], kv_len=p["kv_len"])
                same = reference.gqa_prefill_attention_ref(c["Q"], c["K"], c["V"], q_tile=q_tile, kv_tile=kv_tile)
                self.assertTrue(torch.equal(out, same), msg=f"{shape} tiles {q_tile}/{kv_tile}")
                torch.testing.assert_close(out.double(), naive(c["Q"], c["K"], c["V"]), atol=1e-5, rtol=1e-5)

    def test_page_pool_layout(self):
        size, x = inputs(batch=2, kv_len=50, page_size=8)  # 7 pages per sequence, the last holds 2 of 8 keys
        table, pool = x["block_table"], x["K"]
        self.assertEqual(tuple(pool.shape), (1 + 2 * 7, 8, 2, 32))
        self.assertEqual((table.dtype, tuple(table.shape)), (torch.int32, (2, 8)))
        self.assertTrue(torch.equal(table[:, -1], torch.zeros(2, dtype=torch.int32)))  # spare column: page 0
        listed = table[:, :-1].flatten().sort().values
        self.assertTrue(torch.equal(listed, torch.arange(1, 15, dtype=torch.int32)))  # every page once, never 0
        self.assertFalse(torch.equal(table[:, :-1].flatten(), torch.arange(1, 15, dtype=torch.int32)))  # shuffled
        _, c = inputs(batch=2, kv_len=50)
        for name in ("K", "V"):  # the pool holds exactly the contiguous values
            self.assertTrue(torch.equal(bench.gather_paged_kv(x[name], table, x["kv_len"]), c[name]))
        tail = pool[table[:, 6].long(), 2:]  # unused slots of each last page
        self.assertGreater(tail.abs().mean().item(), 20 * c["K"].abs().mean().item())
        self.assertGreater(pool[0].abs().mean().item(), 20 * c["K"].abs().mean().item())

    def test_oracle_never_reads_the_reserved_page_or_unused_slots(self):
        _, x = inputs(kv_len=50, page_size=8)
        base = bench._ref_gqa_prefill_attention(x)
        changed = dict(x, K=x["K"].clone(), V=x["V"].clone())
        last = x["block_table"][0, 6].long()
        for name in ("K", "V"):
            changed[name][0] = float("nan")
            changed[name][last, 2:] = float("nan")
        self.assertTrue(torch.equal(bench._ref_gqa_prefill_attention(changed), base))

    def test_taking_the_padded_length_as_kv_len_misses_the_tolerance(self):
        # The causal mask hides keys past kv_len from every query, so the unused slots of the last page are
        # read only by a kernel that gets the length wrong, e.g. pages x page_size instead of kv_len.
        _, x = inputs(dtype=torch.bfloat16, q_len=64, kv_len=200, page_size=16, head_dim=64)
        padded = -(-x["kv_len"] // 16) * 16
        K = bench.gather_paged_kv(x["K"], x["block_table"], padded)
        V = bench.gather_paged_kv(x["V"], x["block_table"], padded)
        wrong = reference.gqa_prefill_attention_ref(x["Q"], K, V)
        tol = CONFIG["tolerances"][torch.bfloat16]
        error = (wrong.float() - bench._ref_gqa_prefill_attention(x).float()).abs().max().item()
        self.assertGreater(error, 100 * tol["atol"])

    def test_refuses_a_table_that_does_not_cover_kv_len(self):
        _, x = inputs(kv_len=50, page_size=8)
        with self.assertRaises(ValueError):
            reference.gqa_prefill_attention_ref(x["Q"], x["K"], x["V"], block_table=x["block_table"][:, :6],
                                                kv_len=x["kv_len"])
        with self.assertRaises(ValueError):
            reference.gqa_prefill_attention_ref(x["Q"], x["K"], x["V"], block_table=x["block_table"])  # no kv_len
        with self.assertRaises(ValueError):
            bench.paged_kv_cache(torch.zeros(1, 2, 8, 32), torch.zeros(1, 2, 8, 32), 0)

    def test_sdpa_baseline_reads_paged_inputs_through_a_contiguous_copy(self):
        for shape in (dict(kv_len=50, page_size=8), dict(batch=2, q_len=50, kv_len=50, page_size=16)):
            _, x = inputs(**shape)
            contiguous = bench._contiguous_gqa_inputs(x)
            self.assertEqual(sorted(contiguous), ["K", "Q", "V"])
            torch.testing.assert_close(bench._sdpa_gqa_prefill_attention(contiguous),
                                       bench._ref_gqa_prefill_attention(x), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(bench._sdpa_gqa_prefill_attention(x), bench._ref_gqa_prefill_attention(x),
                                       atol=1e-5, rtol=1e-5)
        _, c = inputs()
        self.assertIs(bench._contiguous_gqa_inputs(c), c)


class ConfigTests(unittest.TestCase):
    def test_flops_count_only_visible_pairs(self):
        for q_len, kv_len in ((1, 1), (1, 9), (5, 5), (7, 30), (64, 1000)):
            size = dict(batch=2, heads_q=8, heads_kv=2, q_len=q_len, kv_len=kv_len, head_dim=32)
            pairs = sum(kv_len - q_len + i + 1 for i in range(q_len))
            self.assertEqual(CONFIG["flops_fn"](size), 4 * 2 * 8 * 32 * pairs)
            self.assertEqual(CONFIG["flops_fn"](dict(size, page_size=16)), CONFIG["flops_fn"](size))
        size = dict(batch=1, heads_q=16, heads_kv=2, q_len=8192, kv_len=1048576, head_dim=256)
        self.assertEqual(CONFIG["bytes_fn"](size, torch.bfloat16), (2 * 16 * 8192 + 2 * 2 * 1048576) * 256 * 2)
        # A whole causal prefill (q_len == kv_len) counts about half of the dense product...
        n = 4096
        full = dict(size, q_len=n, kv_len=n)
        self.assertAlmostEqual(CONFIG["flops_fn"](full) / (4 * 16 * 256 * n * n), (n + 1) / (2 * n), places=12)
        # ...while a 32K chunk at the end of 1M keys masks only the triangle at its end (about 1.6%)
        chunk = dict(size, q_len=32768)
        dense = 4 * 16 * 256 * 32768 * 1048576
        self.assertAlmostEqual(CONFIG["flops_fn"](chunk) / dense, 1 - 32767 / 2 / 1048576, places=12)

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
            self.assertTrue(type(size.get("page_size", 0)) is int and size.get("page_size", 0) >= 0, label)
        self.assertIn("large", [label for label, _ in CONFIG["test_sizes"]])
        self.assertIs(CONFIG["test_dtypes"][0], torch.bfloat16)
        paged = {label: size["page_size"] for label, size in CONFIG["test_sizes"] + CONFIG["edge_sizes"]
                 if size.get("page_size")}
        self.assertGreaterEqual(len(set(paged.values())), 2)  # more than one page size
        self.assertTrue(any(p & (p - 1) for p in paged.values()))  # one that is not a power of two
        self.assertTrue(any(s["kv_len"] == 1048576 and s.get("page_size") for _, s in CONFIG["test_sizes"]))

    def test_page_size_override(self):
        config = bench.with_page_size(CONFIG, 32)
        sizes = config["test_sizes"] + config["edge_sizes"]
        self.assertTrue(all(size["page_size"] == 32 for _, size in sizes))
        labels = [label for label, _ in config["test_sizes"]]
        self.assertIn("large", labels)
        self.assertNotIn("paged_8k_128k", labels)  # the same shape as large once both use 32-token pages
        self.assertEqual(len({tuple(sorted(s.items())) for _, s in sizes}), len(sizes))
        contiguous = bench.with_page_size(CONFIG, 0)
        self.assertFalse(any("page_size" in size for _, size in contiguous["test_sizes"] + contiguous["edge_sizes"]))
        self.assertIs(contiguous["input_generator"], CONFIG["input_generator"])
        for bad in ((CONFIG, -1), (bench.KERNEL_CONFIGS["flash_attention"], 16)):
            with self.assertRaises(ValueError):
                bench.with_page_size(*bad)

    def test_calibrated_attention_peak_takes_the_best_at_the_same_head_dim(self):
        def entry(head_dim, tflops, dtype="bfloat16", backend="flash_attention"):
            return {"shape": {"q_len": 8192, "kv_len": 131072, "heads_q": 16, "heads_kv": 2, "head_dim": head_dim},
                    "dtype": dtype, "best_tflops": tflops, "best_backend": backend}
        calibration = {"attention": [entry(256, 900.0), entry(256, 1100.0, backend="flashinfer"), entry(128, 1500.0),
                                     entry(256, 2000.0, dtype="float16"), entry(256, None), "junk", {"shape": 1}]}
        size = {"head_dim": 256}
        peak = bench.calibrated_attention_peak(calibration, size, torch.bfloat16)
        self.assertEqual((peak["tflops"], peak["backend"]), (1100.0, "flashinfer"))
        self.assertEqual(bench.calibrated_attention_peak(calibration, {"head_dim": 128}, torch.bfloat16)["tflops"], 1500.0)
        self.assertIsNone(bench.calibrated_attention_peak(calibration, {"head_dim": 64}, torch.bfloat16))
        self.assertIsNone(bench.calibrated_attention_peak({}, size, torch.bfloat16))

    def test_performance_reports_the_calibrated_peak_and_times_the_baseline_on_contiguous_inputs(self):
        size = dict(batch=1, heads_q=8, heads_kv=2, q_len=16, kv_len=40, head_dim=32, page_size=8)
        kernel_keys, baseline_keys = [], []

        def kernel_fn(**kw):
            kernel_keys.append(sorted(kw))

        def baseline(x):
            baseline_keys.append(sorted(x))
            return bench._sdpa_gqa_prefill_attention(x)
        config = {**CONFIG, "test_sizes": [("large", size)], "baseline_fn": baseline,
                  "input_generator": lambda s, dt, device, seed=42: bench.gen_gqa_prefill_attention_inputs(s, dt, "cpu", seed)}
        calibration = {"attention": [{"shape": {"head_dim": 32}, "dtype": "bfloat16", "best_tflops": 1e-3,
                                      "best_backend": "math"}]}
        gpu = bench.GPUSpec(name="test", peak_tflops_fp16=1.0, peak_bandwidth_gb_s=1.0)
        with patch.object(bench, "_do_bench", lambda fn, warmup=0, rep=0: (fn(), 2.0)[1]):
            result = bench.run_performance(kernel_fn, config, gpu, calibration=calibration)["primary"]
        self.assertEqual(kernel_keys, [["K", "Q", "V", "block_table", "kv_len"]])
        self.assertEqual(baseline_keys, [["K", "Q", "V"]])  # gathered once, outside the timed call
        tflops = CONFIG["flops_fn"](size) / 2e-3 / 1e12
        self.assertAlmostEqual(result["throughput_tflops"], tflops)
        self.assertAlmostEqual(result["pct_calibrated_peak"], 100 * tflops / 1e-3)
        self.assertEqual(result["calibrated_peak"]["backend"], "math")
        with patch.object(bench, "_do_bench", lambda fn, warmup=0, rep=0: 2.0):
            plain = bench.run_performance(kernel_fn, config, gpu)["primary"]
        self.assertIsNone(plain["pct_calibrated_peak"])


if __name__ == "__main__":
    unittest.main()
