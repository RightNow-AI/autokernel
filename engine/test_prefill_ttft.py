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

MS_PER_TOKEN = 0.001  # the fake server's prefill time per uncached prompt token


class Clock:
    """perf_counter stand-in: every reading advances 0.25 s; the fake server adds its prefill time."""

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
    """Answers one token per request and emulates a 16-token-block prefix cache.

    Opening a stream advances the shared clock by MS_PER_TOKEN per uncached prompt token, so TTFT is
    0.25 s plus that prefill time. fail: request indexes answered with HTTP 500. cached_override: the cached
    token count to report instead. report_cache=False: usage carries no prompt_tokens_details.
    """

    def __init__(self, cache=True, break_with=None, report_cache=True, cached_override=None, fail=()):
        self.cache, self.break_with, self.report_cache = cache, break_with, report_cache
        self.cached_override, self.fail = cached_override, set(fail)
        self.prompts, self.seen, self.clock = [], set(), Clock()

    def stream(self, method, url, json=None, headers=None):
        prompt = json["prompt"]
        self.prompts.append(prompt)
        if len(self.prompts) - 1 in self.fail:
            return FakeStream(FakeResponse(500, []))
        cached = 0
        if self.cache:
            for n in range(16, len(prompt), 16):
                if tuple(prompt[:n]) in self.seen:
                    cached = n
            for n in range(16, len(prompt) + 1, 16):
                self.seen.add(tuple(prompt[:n]))
        self.clock.t += (len(prompt) - cached) * MS_PER_TOKEN
        if self.cached_override is not None:
            cached = self.cached_override
        usage = {"prompt_tokens": len(prompt), "completion_tokens": 1}
        if self.report_cache:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
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
    with patch.object(pt.time, "perf_counter", server.clock), redirect_stdout(io.StringIO()):
        return asyncio.run(pt.measure(run_args(**overrides), io.StringIO(), client=server))


def ttft(uncached_tokens):
    return 0.25 + uncached_tokens * MS_PER_TOKEN


class MeasureTests(unittest.TestCase):
    def test_cold_runs_send_fresh_prompts_of_exact_length(self):
        server = FakeServer()
        results = measure(server)
        self.assertEqual([len(p) for p in server.prompts], [64] * 3 + [128] * 3)
        self.assertEqual(len({tuple(p) for p in server.prompts}), 6)
        for isl in (64, 128):
            summary = results[str(isl)]["summary"]
            self.assertEqual(sorted(summary), ["cold"])
            self.assertEqual(summary["cold"]["valid_runs"], 2)
            self.assertAlmostEqual(summary["cold"]["ttft_s_p50"], ttft(isl))
            self.assertAlmostEqual(summary["cold"]["prefill_tok_s_p50"], isl / ttft(isl))
            self.assertEqual(summary["cold"]["cached_tokens"], [0, 0])

    def test_warm_runs_send_one_prefix_twice_and_time_both(self):
        server = FakeServer()
        results = measure(server, isls=[256], mode="warm", suffix_tokens=32)
        self.assertEqual(len(server.prompts), 6)  # one warmup and two timed runs, two requests each
        cold, warm = server.prompts[2], server.prompts[3]  # the first timed run
        self.assertEqual((len(cold), len(warm)), (256, 256))  # both exactly ISL
        self.assertEqual(warm[:224], cold[:224])
        self.assertNotEqual(warm[224:], cold[224:])
        self.assertNotEqual(server.prompts[4][:224], cold[:224])  # every run reads a fresh prefix
        runs, summary = results["256"]["runs"], results["256"]["summary"]
        self.assertEqual([r["run"] for r in runs], [1, 2])
        self.assertEqual([r["cold"]["cached_tokens"] for r in runs], [0, 0])
        self.assertEqual([r["warm"]["cached_tokens"] for r in runs], [224, 224])
        self.assertEqual((summary["cold"]["valid_runs"], summary["warm"]["valid_runs"]), (2, 2))
        self.assertAlmostEqual(summary["cold"]["ttft_s_p50"], ttft(256))
        self.assertAlmostEqual(summary["warm"]["ttft_s_p50"], ttft(32))
        self.assertAlmostEqual(summary["warm_speedup_p50"], ttft(256) / ttft(32))
        self.assertEqual(summary["warm"]["unreported_cache_runs"], 0)

    def test_warm_with_no_suffix_sends_the_same_prompt_twice(self):
        server = FakeServer()
        results = measure(server, isls=[256], mode="warm", suffix_tokens=0)
        self.assertEqual(server.prompts[2], server.prompts[3])
        self.assertEqual([r["warm"]["cached_tokens"] for r in results["256"]["runs"]], [240, 240])
        self.assertAlmostEqual(results["256"]["summary"]["warm_speedup_p50"], ttft(256) / ttft(16))

    def test_a_warm_request_without_a_cache_hit_is_invalid(self):
        results = measure(FakeServer(cache=False), isls=[256], mode="warm", suffix_tokens=32)
        summary = results["256"]["summary"]
        self.assertEqual((summary["cold"]["valid_runs"], summary["warm"]["valid_runs"]), (2, 0))
        self.assertIn("warm request reported no cached tokens", results["256"]["runs"][0]["warm"]["error"])
        self.assertNotIn("warm_speedup_p50", summary)

    def test_a_cold_request_that_hits_the_cache_is_invalid(self):
        results = measure(FakeServer(cached_override=16), isls=[64])
        self.assertEqual(results["64"]["summary"]["cold"]["valid_runs"], 0)
        self.assertIn("cold request reported 16 cached tokens", results["64"]["runs"][0]["cold"]["error"])

    def test_an_unreported_cache_hit_is_counted_not_refused(self):
        results = measure(FakeServer(report_cache=False), isls=[256], mode="warm", suffix_tokens=32)
        summary = results["256"]["summary"]
        self.assertEqual((summary["cold"]["valid_runs"], summary["warm"]["valid_runs"]), (2, 2))
        self.assertEqual(summary["warm"]["unreported_cache_runs"], 2)
        self.assertEqual(summary["warm"]["cached_tokens"], [None, None])

    def test_a_failed_cold_request_invalidates_its_warm_request(self):
        results = measure(FakeServer(fail={2}), isls=[256], mode="warm", suffix_tokens=32)
        first, second = results["256"]["runs"]
        self.assertEqual(first["cold"]["error"].split(";")[0], "HTTP 500")
        self.assertIn("its cold request failed", first["warm"]["error"])
        self.assertIsNone(second["cold"]["error"])
        self.assertIsNone(second["warm"]["error"])
        summary = results["256"]["summary"]
        self.assertEqual((summary["cold"]["valid_runs"], summary["warm"]["valid_runs"]), (1, 1))

    def test_incomplete_or_mismatched_streams_are_invalid(self):
        for brokenness, message in (("no_done", "missing [DONE]"), ("short_prompt", "prompt_tokens: observed 7")):
            results = measure(FakeServer(break_with=brokenness), isls=[64])
            self.assertEqual(results["64"]["summary"]["cold"]["valid_runs"], 0)
            self.assertIn(message, results["64"]["runs"][0]["cold"]["error"])

    def test_percentile_is_nearest_rank(self):
        self.assertEqual(pt.percentile([3.0, 1.0, 2.0], 50), 2.0)
        self.assertEqual(pt.percentile([1.0, 2.0, 3.0, 4.0], 90), 4.0)
        self.assertEqual(pt.percentile([5.0], 90), 5.0)


def leg(p50, isl=64, unreported=0):
    return {"valid_runs": 2, "invalid_runs": 0, "ttft_s_p50": p50, "ttft_s_p90": p50, "ttft_s_spread": [p50, p50],
            "prefill_tok_s_p50": isl / p50, "cached_tokens": [0, 0], "unreported_cache_runs": unreported}


def doc(label, p50, ident, warm=None, unreported=0, **profile):
    """A loadable v2 result: cold mode, or warm mode when a warm p50 is given."""
    mode = "cold" if warm is None else "warm"
    summary = {"cold": leg(p50)}
    if warm is not None:
        summary.update(warm=leg(warm, unreported=unreported), warm_speedup_p50=p50 / warm)
    base = {"schema": pt.SCHEMA, "identity": ident, "label": label, "notes": "", "model": "m", "isls": [64],
            "mode": mode, "suffix_tokens": 0 if warm is None else 16, "runs": 2, "seed": 0,
            "token_id_range": "1000,20000", "legs": list(pt.LEGS[mode]),
            "results": {"64": {"runs": [], "summary": summary}}}
    base.update(profile)
    return base


class _Closing:
    def __init__(self, server):
        self.server = server

    def stream(self, *a, **kw):
        return self.server.stream(*a, **kw)

    async def aclose(self):
        return None


def write_files(*docs):
    folder = tempfile.mkdtemp()
    paths = []
    for i, d in enumerate(docs):
        paths.append(os.path.join(folder, f"{i}.json"))
        with open(paths[-1], "w", encoding="utf-8") as f:
            json.dump(d, f)
    return paths


def call(*argv):
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        code = pt.main(list(argv))
    return code, out.getvalue()


def run_server(server, *argv):
    """`run` against a fake server, with httpx and the clock replaced."""
    fake = SimpleNamespace(AsyncClient=lambda **kw: _Closing(server), Timeout=lambda s: s)
    with patch.object(pt, "httpx", fake), patch.object(pt.time, "perf_counter", server.clock):
        return call("run", "--base-url", "http://x", "--model", "m", *argv)


class CommandTests(unittest.TestCase):
    def files(self, *docs):
        return write_files(*docs)

    def call(self, *argv):
        return call(*argv)

    def test_compare_reports_the_speedup(self):
        a, b = identity(), identity()
        b["arm"]["label"] = "ours"
        code, out = self.call("compare", *self.files(doc("stock", 2.0, a), doc("ours", 0.5, b)), "--arm-delta", "arm.label")
        self.assertEqual(code, 0)
        self.assertIn("4.000x", out)

    def test_compare_reports_both_legs_and_the_prefix_cache_speedup(self):
        a = identity()
        code, out = self.call("compare", *self.files(doc("stock", 2.0, a, warm=0.5), doc("ours", 1.0, a, warm=0.4)))
        self.assertEqual(code, 0)
        rows = [line.split() for line in out.splitlines() if line.strip().startswith("64 ")]
        self.assertEqual([(row[1], row[-2]) for row in rows], [("cold", "2.000x"), ("warm", "1.250x")])
        self.assertIn("ISL 64: A 4.00x, B 2.50x", out)

    def test_compare_refuses_profile_and_identity_mismatches(self):
        a = identity()
        code, out = self.call("compare", *self.files(doc("a", 1.0, a), doc("b", 1.0, a, warm=0.5)))
        self.assertEqual((code, out.strip()), (4, "PROFILE_MISMATCH: mode, suffix_tokens"))
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
        self.assertIn("| 1 | exp1 | cold | 1.000 (2.00x) |", out)
        code, _ = self.call("compare", os.path.join(tempfile.mkdtemp(), "missing.json"), "x.json")
        self.assertEqual(code, 3)

    def test_table_of_one_warm_run_reports_cold_vs_warm(self):
        code, out = self.call("table", *self.files(doc("stock", 2.0, identity(), warm=0.5)))
        self.assertEqual(code, 0)
        self.assertIn("| 0 | stock | cold | 2.000 (1.00x) |", out)
        self.assertIn("| 0 | stock | warm | 0.500 (1.00x; 4.00x vs cold) |", out)

    def test_an_unreported_cache_hit_is_marked(self):
        a = identity()
        code, out = self.call("compare", *self.files(doc("a", 2.0, a, warm=0.5, unreported=2), doc("b", 2.0, a, warm=0.5)))
        self.assertEqual(code, 0)
        self.assertIn("[cache hit unreported]", out)
        self.assertEqual(out.count("[cache hit unreported]"), 1)  # the warm row only

    def test_run_validates_before_any_network_call(self):
        code, out = self.call("run", "--base-url", "http://x", "--model", "m", "--label", "a", "--isl", "64",
                              "--mode", "warm", "--suffix-tokens", "64")
        self.assertEqual((code, out.strip()), (2, "ERROR: warm mode needs 0 <= --suffix-tokens < every ISL"))
        code, out = self.call("run", "--base-url", "http://x", "--model", "m", "--label", "a",
                              "--isl", "131072,1048576", "--dry-run")
        self.assertEqual(code, 0)
        plan = json.loads(out)
        self.assertEqual((plan["isls"], plan["suffix_tokens"], plan["legs"], plan["requests_per_run"]),
                         ([131072, 1048576], 0, ["cold"], 1))
        code, out = self.call("run", "--base-url", "http://x", "--model", "m", "--label", "a",
                              "--isl", "131072", "--mode", "warm", "--suffix-tokens", "0", "--dry-run")
        plan = json.loads(out)
        self.assertEqual((code, plan["legs"], plan["requests_per_run"]), (0, ["cold", "warm"], 2))

    def test_run_writes_a_loadable_result(self):
        out_path = os.path.join(tempfile.mkdtemp(), "r.json")
        code, _ = run_server(FakeServer(), "--label", "stock", "--isl", "64", "--runs", "2", "--out", out_path)
        self.assertEqual(code, 0)
        loaded = pt.load(out_path)
        self.assertEqual((loaded["label"], loaded["results"]["64"]["summary"]["cold"]["valid_runs"]), ("stock", 2))


class SchemaTests(unittest.TestCase):
    """The raw run JSON is the evidence file: every documented field is present and typed."""

    RECORD_KEYS = {"requested_prompt_tokens", "prompt_tokens", "completion_tokens", "cached_tokens", "done_received",
                   "finish_reason", "ttft_s", "total_s", "error"}
    LEG_KEYS = {"valid_runs", "invalid_runs", "ttft_s_p50", "ttft_s_p90", "ttft_s_spread", "prefill_tok_s_p50",
                "cached_tokens", "unreported_cache_runs"}

    def test_a_warm_run_writes_the_documented_schema(self):
        folder = tempfile.mkdtemp()
        out_path, id_path = os.path.join(folder, "r.json"), os.path.join(folder, "id.json")
        with open(id_path, "w", encoding="utf-8") as f:
            json.dump(identity(), f)
        code, _ = run_server(FakeServer(), "--label", "stock", "--isl", "64,128", "--mode", "warm",
                             "--suffix-tokens", "16", "--runs", "2", "--warmup", "1", "--notes", "n",
                             "--identity", id_path, "--out", out_path)
        self.assertEqual(code, 0)
        with open(out_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.assertEqual(set(raw), {"schema", "identity", "label", "notes", "started_utc", "finished_utc", "model",
                                    "isls", "mode", "suffix_tokens", "runs", "seed", "token_id_range", "warmup",
                                    "legs", "results"})
        self.assertEqual((raw["schema"], raw["mode"], raw["legs"], raw["isls"], raw["suffix_tokens"]),
                         ("autokernel-prefill-ttft.v2", "warm", ["cold", "warm"], [64, 128], 16))
        self.assertEqual(raw["identity"], identity())
        self.assertEqual(set(raw["results"]), {"64", "128"})
        for isl in (64, 128):
            entry = raw["results"][str(isl)]
            self.assertEqual(set(entry), {"runs", "summary"})
            self.assertEqual([run["run"] for run in entry["runs"]], [1, 2])  # the warmup run is not kept
            for run in entry["runs"]:
                self.assertEqual(set(run), {"run", "timed", "cold", "warm"})
                self.assertIs(run["timed"], True)
                for name in ("cold", "warm"):
                    rec = run[name]
                    self.assertEqual(set(rec), self.RECORD_KEYS)
                    self.assertEqual((rec["requested_prompt_tokens"], rec["prompt_tokens"], rec["completion_tokens"],
                                      rec["done_received"], rec["finish_reason"], rec["error"]),
                                     (isl, isl, 1, True, "length", None))
                    self.assertGreater(rec["ttft_s"], 0)
                    self.assertGreaterEqual(rec["total_s"], rec["ttft_s"])
                self.assertEqual((run["cold"]["cached_tokens"], run["warm"]["cached_tokens"]), (0, isl - 16))
            summary = entry["summary"]
            self.assertEqual(set(summary), {"cold", "warm", "warm_speedup_p50"})
            for name in ("cold", "warm"):
                self.assertEqual(set(summary[name]), self.LEG_KEYS)
                self.assertEqual(summary[name]["valid_runs"], 2)
            self.assertAlmostEqual(summary["warm_speedup_p50"], ttft(isl) / ttft(16))
        with open(out_path + ".log", encoding="utf-8") as f:
            log = [json.loads(line) for line in f]
        self.assertEqual(len(log), 2 * 3 * 2)  # two ISLs x (one warmup + two timed runs) x two requests
        self.assertEqual({(r["leg"], r["timed"]) for r in log}, {("cold", False), ("warm", False),
                                                                 ("cold", True), ("warm", True)})
        self.assertIsNotNone(pt.load(out_path))

    def test_load_refuses_other_schemas_and_missing_legs(self):
        good = doc("a", 2.0, identity(), warm=0.5)
        old = dict(good, schema="autokernel-prefill-ttft.v1")
        no_warm = json.loads(json.dumps(good))
        del no_warm["results"]["64"]["summary"]["warm"]
        bad_p50 = json.loads(json.dumps(good))
        bad_p50["results"]["64"]["summary"]["warm"]["ttft_s_p50"] = 0
        paths = write_files(good, old, no_warm, bad_p50)
        self.assertIsNotNone(pt.load(paths[0]))
        self.assertEqual([pt.load(p) for p in paths[1:]], [None, None, None])


if __name__ == "__main__":
    unittest.main()
