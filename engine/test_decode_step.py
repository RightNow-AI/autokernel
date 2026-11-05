"""Offline regression gates. All result files and HTTP streams are synthetic."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import AsyncMock, Mock, patch

from engine import decode_step as ds
from engine.identity import flatten_identity, validate_identity


def identity():
    return {
        "hardware": {"gpu_name": "NVIDIA H100", "gpu_count": 2, "sm_count": 132, "driver": "test-driver"},
        "image": {"digest": "sha256:" + "1" * 64}, "engine": {"version": "test-version"},
        "model": {"id": "Qwen/Qwen3-8B", "revision": "a" * 40},
        "parallelism": {"tp": 2, "dcp": 1, "ep": 1},
        "speculation": {"method": "none", "k": 0, "draft_revision": None},
        "kv_dtype": "auto", "arm": {"label": "stock"},
    }


def result():
    return {
        "schema": "autokernel-decode-step.v2", "identity": identity(),
        "label": "stock", "model": "Qwen/Qwen3-8B", "isl": 1, "osl": 100,
        "batches": [1], "runs": 1, "seed": 0, "token_id_range": [1000, 20000],
        "rate_formula_primary": "(N-1)/dt",
        "results": {"1": {"summary": {
            "valid_runs": 1, "step_ms_p50": 10, "step_ms_p90": 11,
            "decode_tok_s": 99, "tok_s_per_user": 99, "tokens_per_step": 1,
            "tok_s_n_over_dt": 100, "tok_s_n_minus_1_over_dt": 99, "tok_s_window_per_user": 98,
            "window_decode_tok_s": 98, "step_ms_spread": [9, 11], "exact_b_all": True,
        }}},
    }


def record(start=10, tokens_per_chunk=1):
    return {
        "t_start": start - 1, "chunks": [(start + i, (i + 1) * tokens_per_chunk) for i in range(16)],
        "completion_tokens": 16 * tokens_per_chunk, "prompt_tokens": 1,
        "token_source": "usage", "error": None,
    }


def finish_without_io(coroutine):
    """Fake streams/AsyncMocks complete inline; no event-loop sockets are needed."""
    try:
        coroutine.send(None)
    except StopIteration as done:
        return done.value
    finally:
        coroutine.close()
    raise AssertionError("synthetic coroutine unexpectedly waited for I/O")


class OfflineCase(unittest.TestCase):
    def setUp(self):
        for name in ("socket.socket.connect", "socket.create_connection"):
            guard = patch(name, side_effect=AssertionError("network is forbidden in these tests"))
            guard.start()
            self.addCleanup(guard.stop)

    def invoke(self, argv, files):
        def read_file(path, *args, **kwargs):
            if str(path) not in files:
                raise FileNotFoundError(str(path))
            return io.StringIO(json.dumps(files[str(path)]))

        out, err = io.StringIO(), io.StringIO()
        with patch("builtins.open", side_effect=read_file), redirect_stdout(out), redirect_stderr(err):
            code = ds.main(argv)
        return code, out.getvalue() + err.getvalue()

    def compare(self, a, b, flags=(), command="compare"):
        return self.invoke([command, "a.json", "b.json", *flags], {"a.json": a, "b.json": b})


class ComparisonTests(OfflineCase):
    def test_identical_identities_pass_both_commands(self):
        for command in ("compare", "table"):
            with self.subTest(command=command):
                code, output = self.compare(result(), result(), command=command)
                self.assertEqual(code, 0, output)
                self.assertIn("(N-1)/dt", output)
                self.assertNotIn("UNQUALIFIED", output)

    def test_every_identity_leaf_difference_refuses(self):
        for key in flatten_identity(identity()):
            a, b = result(), result()
            parts = key.split(".")
            parent = b["identity"]
            for part in parts[:-1]:
                parent = parent[part]
            old = parent[parts[-1]]
            parent[parts[-1]] = old + 1 if type(old) is int else "changed"
            with self.subTest(key=key):
                code, output = self.compare(a, b)
                self.assertEqual(code, 4, output)
                self.assertIn(key, output)
                code, output = self.compare(a, b, ["--arm-delta", key])
                self.assertEqual(code, 0, output)
                self.assertIn("DECLARED_ARM_DELTA", output)

    def test_declared_delta_does_not_allow_other_changes(self):
        b = result()
        b["identity"]["image"]["digest"] = "sha256:" + "2" * 64
        b["identity"]["speculation"]["k"] = 3
        for command in ("compare", "table"):
            with self.subTest(command=command):
                code, output = self.compare(result(), b, ["--arm-delta", "image.digest"], command)
                self.assertEqual(code, 4, output)
                self.assertIn("speculation.k", output)
                code, output = self.compare(result(), b, ["--arm-delta", "image.digest,speculation.k"], command)
                self.assertEqual(code, 0, output)

    def test_unverified_delta_key_refuses(self):
        for key in ("hardware", "hardware.gpu_typo", "*"):
            code, output = self.compare(result(), result(), ["--arm-delta", key])
            self.assertEqual(code, 4, output)
            self.assertIn("ARM_DELTA_UNKNOWN", output)

    def test_extra_identity_fields_are_compared(self):
        b = result()
        b["identity"]["hardware"]["fabric"] = "test-fabric"
        code, output = self.compare(result(), b)
        self.assertEqual(code, 4, output)
        self.assertIn("hardware.fabric", output)

    def test_nested_extension_value_types_are_compared(self):
        a, b = result(), result()
        a["identity"]["extra"] = [1]
        b["identity"]["extra"] = [True]
        code, output = self.compare(a, b)
        self.assertEqual(code, 4, output)
        self.assertIn("extra", output)

    def test_missing_identity_on_either_side_requires_marked_override(self):
        for command in ("compare", "table"):
            for side in (0, 1):
                for remove in (False, True):
                    docs = [result(), result()]
                    if remove:
                        del docs[side]["identity"]
                    else:
                        docs[side]["identity"] = None
                    with self.subTest(command=command, side=side, remove=remove):
                        code, output = self.compare(*docs, command=command)
                        self.assertEqual(code, 4, output)
                        self.assertIn("IDENTITY_MISSING", output)
                        code, output = self.compare(*docs, ["--allow-missing-identity"], command)
                        self.assertEqual(code, 0, output)
                        self.assertIn("UNQUALIFIED", output)

    def test_malformed_identity_cannot_use_missing_override(self):
        for invalid in ({}, [], "unverified id"):
            b = result()
            b["identity"] = invalid
            code, output = self.compare(result(), b, ["--allow-missing-identity"])
            self.assertEqual(code, 4, output)
            self.assertIn("IDENTITY_INVALID", output)

    def test_workload_mismatch_cannot_be_declared_as_identity_delta(self):
        b = result()
        b["isl"] = 8192
        code, output = self.compare(result(), b, ["--arm-delta", "isl"])
        self.assertEqual(code, 4, output)
        self.assertIn("PROFILE_MISMATCH: isl", output)

    def test_inexact_and_unverified_b_require_marked_override(self):
        for command in ("compare", "table"):
            for side in (0, 1):
                for state in (False, None):
                    docs = [result(), result()]
                    docs[side]["results"]["1"]["summary"]["exact_b_all"] = state
                    with self.subTest(command=command, side=side, state=state):
                        code, output = self.compare(*docs, command=command)
                        self.assertEqual(code, 5, output)
                        self.assertIn("EXACT_B_REFUSED", output)
                        code, output = self.compare(*docs, ["--allow-inexact"], command)
                        self.assertEqual(code, 0, output)
                        self.assertIn("UNQUALIFIED", output)
                        self.assertIn("INEXACT" if state is False else "UNVERIFIED", output)

    def test_raw_inexact_repetition_cannot_hide_in_summary(self):
        b = result()
        b["results"]["1"]["runs"] = [{"valid": True, "exact_b": False}]
        code, output = self.compare(result(), b)
        self.assertEqual(code, 5, output)

    def test_table_checks_later_candidates_before_printing_table(self):
        bad = result()
        bad["results"]["1"]["summary"]["exact_b_all"] = False
        code, output = self.invoke(["table", "base", "first", "later"],
                                   {"base": result(), "first": result(), "later": bad})
        self.assertEqual(code, 5, output)
        self.assertNotIn("| # |", output)

    def test_legacy_json_loads_and_keeps_legacy_formula(self):
        old = result()
        old["schema"] = "autokernel-decode-step.v1"
        del old["identity"], old["rate_formula_primary"]
        for field in (*ds.RATE_FIELDS.values(), "window_decode_tok_s"):
            del old["results"]["1"]["summary"][field]
        for command in ("compare", "table"):
            code, output = self.compare(old, old, command=command)
            self.assertEqual(code, 4, output)
            self.assertIn("IDENTITY_MISSING", output)
            code, output = self.compare(old, old, ["--allow-missing-identity"], command)
            self.assertEqual(code, 0, output)
            self.assertIn("legacy window delta/dt", output)
            self.assertIn("UNQUALIFIED", output)
        code, output = self.compare(old, result(), ["--allow-missing-identity"])
        self.assertEqual(code, 4, output)
        self.assertIn("rate_formula_primary", output)
        # v1 files used the exact-B window delta, so they compare with the default "window" formula.
        windowed = result()
        windowed["rate_formula_primary"] = "window"
        code, output = self.compare(old, windowed, ["--allow-missing-identity"])
        self.assertEqual(code, 0, output)
        self.assertIn("UNQUALIFIED", output)

    def test_different_primary_formula_refuses(self):
        b = result()
        b["rate_formula_primary"] = "N/dt"
        code, output = self.compare(result(), b)
        self.assertEqual(code, 4, output)
        self.assertIn("rate_formula_primary", output)

    def test_table_uses_selected_rate_not_stale_alias(self):
        a = result()
        a["rate_formula_primary"] = "N/dt"
        a["results"]["1"]["summary"]["tok_s_per_user"] = 123456
        code, output = self.compare(a, a, command="table")
        self.assertEqual(code, 0, output)
        self.assertIn("[N/dt]", output)
        self.assertIn("100.0", output)
        self.assertNotIn("123456", output)

    def test_missing_rates_and_chunk_count_rates_refuse(self):
        for change in ("missing", "chunks", "nan"):
            b = result()
            summary = b["results"]["1"]["summary"]
            if change == "missing":
                del summary["tok_s_n_over_dt"]
            elif change == "chunks":
                summary["token_source"] = "chunks"
            else:
                summary["tok_s_n_over_dt"] = float("nan")
            code, output = self.compare(result(), b)
            self.assertEqual(code, 4, output)
            self.assertIn("RATE_INVALID", output)

    def test_missing_and_non_result_json_refuse(self):
        for files in ({}, {"a": []}, {"a": {"batches": [1]}}):
            code, output = self.invoke(["compare", "a", "b"], files)
            self.assertEqual(code, 3, output)

    def test_invalid_step_or_raw_record_refuses_cleanly(self):
        for change in ("zero_step", "bad_spread", "bad_record"):
            bad = result()
            entry = bad["results"]["1"]
            if change == "zero_step":
                entry["summary"]["step_ms_p50"] = 0
            elif change == "bad_spread":
                entry["summary"]["step_ms_spread"] = []
            else:
                entry["runs"] = [None]
            code, output = self.compare(result(), bad)
            self.assertEqual(code, 3, output)

    def test_process_exit_codes(self):
        # Exercise the direct-script import and sys.exit, still using only in-memory JSON.
        program = (
            "import io,json,runpy,sys\nfrom unittest.mock import patch\n"
            "files=json.load(sys.stdin)\nsys.path.insert(0,'engine')\n"
            "sys.argv=['engine/decode_step.py','compare','a','b']\n"
            "with patch('builtins.open',side_effect=lambda p,*a,**k: io.StringIO(json.dumps(files[p]))):\n"
            "    runpy.run_path('engine/decode_step.py',run_name='__main__')\n"
        )
        for expected in (0, 4, 5):
            a, b = result(), result()
            if expected == 4:
                b["identity"]["hardware"]["driver"] = "changed"
            if expected == 5:
                b["results"]["1"]["summary"]["exact_b_all"] = False
            with self.subTest(expected=expected):
                completed = subprocess.run([sys.executable, "-B", "-c", program],
                                           input=json.dumps({"a": a, "b": b}), text=True,
                                           capture_output=True, timeout=20, cwd=Path(__file__).resolve().parents[1])
                self.assertEqual(completed.returncode, expected, completed.stdout + completed.stderr)


class RateTests(OfflineCase):
    def test_both_formulas(self):
        self.assertEqual(ds.token_rates(5, 10, 12), {"tok_s_n_over_dt": 2.5, "tok_s_n_minus_1_over_dt": 2})

    def test_undefined_intervals_are_null(self):
        for n, first, last in ((1, 1, 1), (0, 1, 2), (None, 1, 2), (5, None, 2), (5, 2, 1), (5, 1, float("nan"))):
            self.assertTrue(all(value is None for value in ds.token_rates(n, first, last).values()))

    def test_speculative_usage_and_full_stream_rates_are_not_chunk_counts(self):
        measured = ds.analyze([record(tokens_per_chunk=3)], [(15, 1)], 1)
        self.assertTrue(measured["valid"])
        self.assertEqual(measured["rate_formula_primary"], "window")
        self.assertAlmostEqual(measured["tok_s_n_over_dt"], 48 / 15)
        self.assertAlmostEqual(measured["tok_s_n_minus_1_over_dt"], 47 / 15)
        # Default primary is the exact-B window rate, not the full-stream rate.
        self.assertEqual(measured["tok_s_window_per_user"], 3)
        self.assertEqual(measured["tok_s_per_user"], 3)
        self.assertEqual(measured["tokens_per_step"], 3)
        self.assertEqual(measured["window_decode_tok_s"], 3)
        full_stream = ds.analyze([record(tokens_per_chunk=3)], [(15, 1)], 1, "(N-1)/dt")
        self.assertAlmostEqual(full_stream["tok_s_per_user"], 47 / 15)
        stream = measured["streams"][0]
        self.assertEqual((stream["t_first_output_token"], stream["t_last_output_token"]), (10, 25))
        self.assertEqual(stream["completion_tokens"], 48)

    def test_primary_formula_and_multistream_aggregation(self):
        records = [record(tokens_per_chunk=3), record(start=11, tokens_per_chunk=3)]
        measured = ds.analyze(records, [(15, 2)], 2, "N/dt")
        self.assertAlmostEqual(measured["tok_s_per_user"], 48 / 15)
        self.assertAlmostEqual(measured["decode_tok_s"], 2 * 48 / 15)
        self.assertEqual(measured["window_decode_tok_s"], 6)
        self.assertEqual(measured["window_s"], 14)
        summary = ds.summarize([measured])
        self.assertEqual(summary["tok_s_n_over_dt"], 3.2)
        self.assertTrue(summary["exact_b_all"])

    def test_absent_metrics_remain_unverified(self):
        measured = ds.analyze([record()], [], 1)
        self.assertIsNone(measured["exact_b"])
        self.assertIsNone(ds.summarize([measured])["exact_b_all"])

    def test_invalid_or_inexact_repetition_is_not_discarded(self):
        good = ds.analyze([record()], [(15, 1)], 1)
        for bad, expected in (({"valid": False}, None), ({"valid": False, "exact_b": False}, False)):
            self.assertIs(ds.summarize([good, bad])["exact_b_all"], expected)

    def test_missing_usage_does_not_fall_back_to_chunks(self):
        bad = record()
        bad["completion_tokens"] = None
        self.assertFalse(ds.analyze([bad], [(15, 1)], 1)["valid"])
        bad = record()
        bad["token_source"] = "chunks"
        self.assertFalse(ds.analyze([bad], [(15, 1)], 1)["valid"])


class StreamingTests(OfflineCase):
    def stream(self, events):
        class Response:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def aiter_lines(self):
                for event in events:
                    yield "data: " + json.dumps(event)
                yield "data: [DONE]"

        client = Mock()
        client.stream.return_value = Response()
        with patch.object(ds.time, "perf_counter", side_effect=range(100)):
            return finish_without_io(ds.stream_one(client, "http://localhost/v1/completions", {}, "Qwen/Qwen3-8B", [1], 16))

    def test_first_last_output_timestamps_exclude_usage_only_trailer(self):
        measured = self.stream([
            {"choices": [{"text": "ab"}], "usage": {"completion_tokens": 2}},
            {"choices": [{"text": "cde"}], "usage": {"completion_tokens": 5}},
            {"choices": [], "usage": {"completion_tokens": 5}},
        ])
        self.assertIsNone(measured["error"])
        self.assertEqual(measured["t_first_output_token"], 1)
        self.assertEqual(measured["t_last_output_token"], 2)
        self.assertEqual(measured["tok_s_n_over_dt"], 5)
        self.assertEqual(measured["tok_s_n_minus_1_over_dt"], 4)

    def test_no_continuous_usage_refuses_instead_of_counting_chunks(self):
        measured = self.stream([
            {"choices": [{"text": "a"}]},
            {"choices": [{"text": "b"}]},
            {"choices": [], "usage": {"completion_tokens": 5}},
        ])
        self.assertIn("completion_tokens usage", measured["error"])
        self.assertEqual(measured["chunks"], [])
        self.assertEqual(measured["token_source"], "usage")

    def test_decreasing_usage_refuses(self):
        measured = self.stream([
            {"choices": [{"text": "a"}], "usage": {"completion_tokens": 5}},
            {"choices": [{"text": "b"}], "usage": {"completion_tokens": 3}},
        ])
        self.assertIsNotNone(measured["error"])

    def test_empty_text_token_with_usage_still_has_timestamp(self):
        measured = self.stream([
            {"choices": [{"text": ""}], "usage": {"completion_tokens": 1}},
            {"choices": [{"text": "a"}], "usage": {"completion_tokens": 2}},
            {"choices": [{"text": ""}], "usage": {"completion_tokens": 3}},
            {"choices": [], "usage": {"completion_tokens": 3}},
        ])
        self.assertIsNone(measured["error"])
        self.assertEqual(measured["t_first_output_token"], 1)
        self.assertEqual(measured["t_last_output_token"], 3)
        self.assertEqual(measured["tok_s_n_minus_1_over_dt"], 1)


class RunTests(OfflineCase):
    argv = ["run", "--base-url", "http://localhost:8000", "--model", "Qwen/Qwen3-8B", "--batch", "1", "--label", "stock"]

    def test_dry_run_is_offline_and_records_default_formula_and_null_identity(self):
        with patch.object(ds, "run_all", new_callable=AsyncMock) as run:
            code, output = self.invoke([*self.argv, "--dry-run"], {})
            self.assertEqual(code, 0, output)
            self.assertIn('"identity": null', output)
            self.assertIn('"rate_formula_primary": "window"', output)
            self.assertIn("WARNING", output)
            run.assert_not_called()

    def test_identity_file_is_embedded_without_network_in_dry_run(self):
        code, output = self.invoke([*self.argv, "--identity", "identity.json", "--dry-run"], {"identity.json": identity()})
        self.assertEqual(code, 0, output)
        plan = json.loads(output)
        self.assertEqual(plan["identity"], identity())
        self.assertNotIn("--dry-run", plan["argv"])
        self.assertIn("--token-id-range", plan["argv"])
        self.assertIn("1000,20000", plan["argv"])

    def test_invalid_identity_refuses_before_run(self):
        with patch.object(ds, "run_all", new_callable=AsyncMock) as run:
            code, output = self.invoke([*self.argv, "--identity", "identity.json"], {"identity.json": {}})
            self.assertEqual(code, 4, output)
            run.assert_not_called()

    def test_invalid_run_arguments_and_overlapping_paths_refuse(self):
        for extra in (["--runs", "0"], ["--warmup", "-1"], ["--batch", "0"],
                      ["--batch", "1,1"], ["--out", "same.json", "--log", "same.json"]):
            code, output = self.invoke([*self.argv, *extra, "--dry-run"], {})
            self.assertEqual(code, 2, output)

    def test_progress_is_logged_for_warmup_and_measurement(self):
        args = SimpleNamespace(api_key_env="", batches=[1], seed=0, warmup=1, runs=1)
        measured = ds.analyze([record()], [(15, 1)], 1)
        client = Mock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        transport = Mock()
        transport.AsyncClient.return_value = client
        log = io.StringIO()
        with patch.object(ds, "httpx", transport), patch.object(ds, "measure", new_callable=AsyncMock, return_value=measured), \
                redirect_stdout(io.StringIO()):
            doc = finish_without_io(ds.run_all(args, log))
        self.assertEqual(doc["1"]["summary"]["valid_runs"], 1)
        self.assertIn("warmup 0: started", log.getvalue())
        self.assertIn("run 0: started", log.getvalue())
        self.assertIn("tok/s/user", log.getvalue())

    def test_unwritable_log_refuses_before_network(self):
        with patch.object(ds, "httpx", Mock()), patch.object(ds.os, "makedirs"), \
                patch.object(ds, "run_all", new_callable=AsyncMock) as run:
            code, output = self.invoke(self.argv, {})
            self.assertEqual(code, 3, output)
            self.assertIn("log", output)
            run.assert_not_called()

    def test_actual_run_document_embeds_identity_and_new_schema(self):
        for supplied in (False, True):
            captured = {}
            out, err = io.StringIO(), io.StringIO()
            args = [*self.argv, "--out", "result.json"]
            if supplied:
                args += ["--identity", "identity.json"]
            def open_file(path, *a, **k):
                return io.StringIO(json.dumps(identity()) if path == "identity.json" else "")
            def capture(doc, *a, **k):
                captured.update(copy.deepcopy(doc))
            with patch.object(ds, "httpx", Mock()), patch.object(ds, "run_all", new_callable=AsyncMock, return_value=result()["results"]), \
                    patch("builtins.open", side_effect=open_file), patch.object(ds.os, "makedirs"), \
                    patch.object(ds.json, "dump", side_effect=capture), patch.object(ds.asyncio, "run", side_effect=finish_without_io), \
                    redirect_stdout(out), redirect_stderr(err):
                self.assertEqual(ds.main(args), 0)
            self.assertEqual(captured["identity"], identity() if supplied else None)
            self.assertEqual(captured["schema"], "autokernel-decode-step.v2")
            self.assertEqual(captured["rate_formula_primary"], "window")
            self.assertEqual("WARNING" in err.getvalue(), not supplied)

    def test_required_identity_types(self):
        good = identity()
        self.assertEqual(validate_identity(good), good)
        for field in ("gpu_count", "sm_count"):
            bad = identity()
            bad["hardware"][field] = True
            with self.assertRaisesRegex(ValueError, field):
                validate_identity(bad)
        bad = identity()
        bad["extra"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_identity(bad)


if __name__ == "__main__":
    unittest.main()
