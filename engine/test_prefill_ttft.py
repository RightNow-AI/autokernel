"""Offline gates for engine/prefill_ttft.py. Every HTTP stream here is synthetic."""

import asyncio
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from engine import prefill_ttft as pt
from engine.test_decode_step import identity


class Clock:
    """perf_counter stand-in: every reading advances 0.25 s, so TTFTs are exact and positive."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.25
        return self.t


class FakeResponse:
    def __init__(self, status, lines):
        self.status_code, self.lines = status, lines

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class FakeStream:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *exc):
        return False


class FakeServer:
    """Answers one token per request; optionally emulates a 16-token-block prefix cache."""

    def __init__(self, cache=True, break_with=None):
        self.cache, self.break_with, self.prompts, self.seen = cache, break_with, [], set()

    def stream(self, method, url, json=None, headers=None):
        prompt = json["prompt"]
        self.prompts.append(prompt)
        cached = 0
        if self.cache:
            for n in range(16, len(prompt), 16):
                if tuple(prompt[:n]) in self.seen:
                    cached = n
            for n in range(16, len(prompt) + 1, 16):
                self.seen.add(tuple(prompt[:n]))
        usage = {"prompt_tokens": len(prompt), "completion_tokens": 1,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        lines = [f"data: {_json({'choices': [{'text': 'x', 'finish_reason': None}], 'usage': usage})}",
                 f"data: {_json({'choices': [{'text': '', 'finish_reason': 'length'}], 'usage': usage})}",
                 f"data: {_json({'choices': [], 'usage': usage})}", "data: [DONE]"]
        if self.break_with == "no_done":
            lines = lines[:-1]
        if self.break_with == "short_prompt":
            lines = [line.replace(f'"prompt_tokens": {len(prompt)}', '"prompt_tokens": 7') for line in lines]
        return FakeStream(FakeResponse(200, lines))


def _json(value):
    return json.dumps(value)


def run_args(**overrides):
    args = dict(base_url="http://127.0.0.1:8000", model="m", isls=[64, 128], mode="cold", suffix_tokens=0,
                runs=2, warmup=1, seed=0, token_id_range="1000,20000", api_key_env="")
    args.update(overrides)
    return SimpleNamespace(**args)


def measure(server, **overrides):
    with patch.object(pt.time, "perf_counter", Clock()), redirect_stdout(io.StringIO()):
        return asyncio.run(pt.measure(run_args(**overrides), io.StringIO(), client=server))


class MeasureTests(unittest.TestCase):
    def test_cold_runs_send_fresh_prompts_of_exact_length(self):
        server = FakeServer()
        results = measure(server)
        self.assertEqual([len(p) for p in server.prompts], [64] * 3 + [128] * 3)
        self.assertEqual(len({tuple(p) for p in server.prompts}), 6)
        for isl in (64, 128):
            summary = results[str(isl)]["summary"]
            self.assertEqual(summary["valid_runs"], 2)
            self.assertEqual(summary["ttft_s_p50"], 0.25)  # request start and first output: one clock tick apart
            self.assertEqual(summary["prefill_tok_s_p50"], isl / 0.25)
            self.assertEqual(summary["cached_tokens"], [0, 0])

    def test_warm_runs_time_a_fresh_suffix_after_a_primed_prefix(self):
        server = FakeServer()
        results = measure(server, isls=[256], mode="warm", suffix_tokens=32)
        prime, timed = server.prompts[2], server.prompts[3]  # second warmup-free run: requests 3 and 4
        self.assertEqual(len(prime), 224)
        self.assertEqual(timed[:224], prime)
        self.assertEqual(len(timed), 256)
        runs = results["256"]["runs"]
        self.assertEqual([r["cached_tokens"] for r in runs], [224, 224])
        self.assertEqual(results["256"]["summary"]["no_cache_hit_runs"], 0)

    def test_warm_runs_without_a_cache_hit_are_marked(self):
        results = measure(FakeServer(cache=False), isls=[256], mode="warm", suffix_tokens=32)
        self.assertEqual(results["256"]["summary"]["no_cache_hit_runs"], 2)

    def test_incomplete_or_mismatched_streams_are_invalid(self):
        for brokenness, message in (("no_done", "missing [DONE]"), ("short_prompt", "prompt_tokens: observed 7")):
            results = measure(FakeServer(break_with=brokenness), isls=[64])
            self.assertEqual(results["64"]["summary"]["valid_runs"], 0)
            self.assertIn(message, results["64"]["runs"][0]["error"])

    def test_percentile_is_nearest_rank(self):
        self.assertEqual(pt.percentile([3.0, 1.0, 2.0], 50), 2.0)
        self.assertEqual(pt.percentile([1.0, 2.0, 3.0, 4.0], 90), 4.0)
        self.assertEqual(pt.percentile([5.0], 90), 5.0)


def doc(label, p50, ident, **profile):
    base = {"schema": pt.SCHEMA, "identity": ident, "label": label, "notes": "", "model": "m", "isls": [64],
            "mode": "cold", "suffix_tokens": 0, "runs": 2, "seed": 0, "token_id_range": "1000,20000",
            "results": {"64": {"runs": [], "summary": {"valid_runs": 2, "ttft_s_p50": p50, "ttft_s_p90": p50,
                                                         "ttft_s_spread": [p50, p50], "prefill_tok_s_p50": 64 / p50,
                                                         "cached_tokens": [0, 0], "no_cache_hit_runs": 0}}}}
    base.update(profile)
    return base


class CommandTests(unittest.TestCase):
    def files(self, *docs):
        folder = tempfile.mkdtemp()
        paths = []
        for i, d in enumerate(docs):
            paths.append(os.path.join(folder, f"{i}.json"))
            with open(paths[-1], "w", encoding="utf-8") as f:
                json.dump(d, f)
        return paths

    def call(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = pt.main(list(argv))
        return code, out.getvalue()

    def test_compare_reports_the_speedup(self):
        a, b = identity(), identity()
        b["arm"]["label"] = "ours"
        code, out = self.call("compare", *self.files(doc("stock", 2.0, a), doc("ours", 0.5, b)), "--arm-delta", "arm.label")
        self.assertEqual(code, 0)
        self.assertIn("4.000x", out)

    def test_compare_refuses_profile_and_identity_mismatches(self):
        a = identity()
        code, out = self.call("compare", *self.files(doc("a", 1.0, a), doc("b", 1.0, a, mode="warm")))
        self.assertEqual((code, out.split(":")[0]), (4, "PROFILE_MISMATCH"))
        changed = identity()
        changed["kv_dtype"] = "fp8"
        code, out = self.call("compare", *self.files(doc("a", 1.0, a), doc("b", 1.0, changed)))
        self.assertEqual((code, out.strip()), (4, "IDENTITY_MISMATCH: kv_dtype"))
        code, out = self.call("compare", *self.files(doc("a", 1.0, None), doc("b", 1.0, a)))
        self.assertEqual(code, 4)
        self.assertIn("IDENTITY_MISSING: baseline", out)

    def test_table_and_missing_files(self):
        a = identity()
        code, out = self.call("table", *self.files(doc("stock", 2.0, a), doc("exp1", 1.0, a)))
        self.assertEqual(code, 0)
        self.assertIn("| 1 | exp1 | 1.000 (2.00x) |", out)
        code, _ = self.call("compare", os.path.join(tempfile.mkdtemp(), "missing.json"), "x.json")
        self.assertEqual(code, 3)

    def test_run_validates_before_any_network_call(self):
        code, out = self.call("run", "--base-url", "http://x", "--model", "m", "--label", "a", "--isl", "64",
                              "--mode", "warm", "--suffix-tokens", "64")
        self.assertEqual((code, out.strip()), (2, "ERROR: warm mode needs 0 < --suffix-tokens < every ISL"))
        code, out = self.call("run", "--base-url", "http://x", "--model", "m", "--label", "a",
                              "--isl", "131072,1048576", "--dry-run")
        self.assertEqual(code, 0)
        plan = json.loads(out)
        self.assertEqual((plan["isls"], plan["suffix_tokens"], plan["requests_per_run"]), ([131072, 1048576], 0, 1))

    def test_run_writes_a_loadable_result(self):
        folder = tempfile.mkdtemp()
        out_path = os.path.join(folder, "r.json")
        fake = SimpleNamespace(AsyncClient=lambda **kw: _Closing(FakeServer()), Timeout=lambda s: s)
        with patch.object(pt, "httpx", fake), patch.object(pt.time, "perf_counter", Clock()):
            code, _ = self.call("run", "--base-url", "http://x", "--model", "m", "--label", "stock", "--isl", "64",
                                "--runs", "2", "--out", out_path)
        self.assertEqual(code, 0)
        loaded = pt.load(out_path)
        self.assertEqual((loaded["label"], loaded["results"]["64"]["summary"]["valid_runs"]), ("stock", 2))


class _Closing:
    def __init__(self, server):
        self.server = server

    def stream(self, *a, **kw):
        return self.server.stream(*a, **kw)

    async def aclose(self):
        return None


if __name__ == "__main__":
    unittest.main()
