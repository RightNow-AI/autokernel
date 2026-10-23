"""
AutoKernel engine mode -- decode-step benchmark for a running vLLM server.

Measures what a serving-engine decode optimization changes: the time of one
decode step at exactly B running requests, with prefill excluded. This is the
methodology of vLLM's specialized-decode PRs (#61867, #61868): median ITL at
exactly B running requests, fixed ISL, OSL 1024, greedy.

How one measurement works:
  1. Launch B streaming requests at once against /v1/completions. Prompts are
     random token IDs (exact ISL, distinct per request so the prefix cache
     cannot help), temperature 0, ignore_eos, max_tokens = min_tokens = OSL.
  2. Every chunk's arrival time and cumulative completion_tokens are recorded
     (stream_options.continuous_usage_stats), so token counts are exact even
     under speculative decoding, where one chunk carries several tokens.
  3. The decode window opens when the LAST request receives its first token
     (every prefill is done) and closes when the FIRST request finishes (all B
     still running). Only chunk gaps fully inside the window count.
  4. While it runs, /metrics is polled for vllm:num_requests_running, so the
     result records whether the engine really had exactly B running.

Reported per B: step_ms (median chunk gap = one engine step), p90, decode
tok/s (all requests), tok/s/user, tokens per step (> 1 under spec decode),
and the running-request range seen in the window.

Run the client on the same host as the server: network jitter lands directly
in the step time. Cross-check the step time with nsys graph-replay time.

Usage:
    uv run --extra engine engine/decode_step.py run --base-url http://127.0.0.1:8000 \\
        --model <served-name> --batch 1,2,4,8 --isl 1 --osl 1024 --label stock \\
        --out results/stock.json
    uv run --extra engine engine/decode_step.py compare results/stock.json results/ours.json
    uv run --extra engine engine/decode_step.py table results/stock.json results/exp01.json results/exp02.json

Exit codes: 0 ok, 2 usage error or every request failed, 3 missing/unreadable
file, 4 the two files measured different profiles (compare refuses).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None

# Fields that must match between two runs for a comparison to mean anything
PROFILE_FIELDS = ("model", "isl", "osl", "batches", "runs", "seed", "token_id_range")

METRICS_POLL_S = 0.25
REQUEST_TIMEOUT_S = 3600.0


# ---------------------------------------------------------------------------
# One request
# ---------------------------------------------------------------------------

async def stream_one(client, url: str, headers: dict, model: str,
                     prompt_ids: List[int], osl: int) -> dict:
    """Stream one completion; return chunk timeline [(t, cumulative_tokens)]."""
    body = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": osl,
        "min_tokens": osl,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    rec = {"t_start": time.perf_counter(), "chunks": [], "prompt_tokens": None,
           "completion_tokens": None, "token_source": "usage", "error": None}
    try:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                rec["error"] = f"HTTP {resp.status_code}: {(await resp.aread())[:300]!r}"
                return rec
            n_chunks = 0
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload.strip() == "[DONE]":
                    break
                t = time.perf_counter()
                msg = json.loads(payload)
                usage = msg.get("usage") or {}
                has_text = any(c.get("text") for c in msg.get("choices", []))
                if usage.get("prompt_tokens") is not None:
                    rec["prompt_tokens"] = usage["prompt_tokens"]
                if usage.get("completion_tokens") is not None:
                    rec["completion_tokens"] = usage["completion_tokens"]
                if has_text:
                    n_chunks += 1
                    if usage.get("completion_tokens") is not None:
                        tokens = usage["completion_tokens"]
                    else:
                        # Server ignored continuous_usage_stats: chunk count
                        # undercounts under speculative decoding.
                        rec["token_source"] = "chunks"
                        tokens = n_chunks
                    rec["chunks"].append((t, tokens))
    except Exception as exc:  # noqa: BLE001 -- recorded, not raised
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["t_end"] = time.perf_counter()
    return rec


async def poll_running(client, metrics_url: str, stop: asyncio.Event, samples: list):
    """Sample vllm:num_requests_running until stopped."""
    while not stop.is_set():
        try:
            resp = await client.get(metrics_url, timeout=5.0)
            t = time.perf_counter()
            for line in resp.text.splitlines():
                if line.startswith("vllm:num_requests_running"):
                    samples.append((t, float(line.rsplit(" ", 1)[1])))
                    break
        except Exception:  # noqa: BLE001 -- metrics are optional
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=METRICS_POLL_S)
        except asyncio.TimeoutError:
            pass


# ---------------------------------------------------------------------------
# One measurement at batch B
# ---------------------------------------------------------------------------

def analyze(records: List[dict], running: list, batch: int) -> dict:
    ok = [r for r in records if not r["error"] and len(r["chunks"]) >= 2]
    out = {"batch": batch, "ok_requests": len(ok), "failed_requests": len(records) - len(ok),
           "errors": [r["error"] for r in records if r["error"]][:3]}
    if len(ok) != batch:
        out["valid"] = False
        return out

    w_start = max(r["chunks"][0][0] for r in ok)   # last prefill done
    w_end = min(r["chunks"][-1][0] for r in ok)    # first request done
    gaps_ms, tokens_in_window, steps_in_window = [], 0, 0
    for r in ok:
        for (t0, n0), (t1, n1) in zip(r["chunks"], r["chunks"][1:]):
            if t0 >= w_start and t1 <= w_end:
                gaps_ms.append((t1 - t0) * 1e3)
                tokens_in_window += n1 - n0
                steps_in_window += 1

    in_window = [v for t, v in running if w_start <= t <= w_end]
    out.update({
        "window_s": round(w_end - w_start, 4),
        "step_samples": len(gaps_ms),
        "token_source": "chunks" if any(r["token_source"] == "chunks" for r in ok) else "usage",
        "prompt_tokens": sorted({r["prompt_tokens"] for r in ok if r["prompt_tokens"] is not None}),
        "running_min": min(in_window) if in_window else None,
        "running_max": max(in_window) if in_window else None,
    })
    if len(gaps_ms) < 10 or w_end <= w_start:
        out["valid"] = False
        return out

    gaps_ms.sort()
    decode_tok_s = tokens_in_window / (w_end - w_start)
    out.update({
        "valid": True,
        "exact_b": (out["running_min"] == batch and out["running_max"] == batch) if in_window else None,
        "step_ms_p50": round(statistics.median(gaps_ms), 4),
        "step_ms_p90": round(gaps_ms[int(0.9 * (len(gaps_ms) - 1))], 4),
        "tokens_per_step": round(tokens_in_window / steps_in_window, 3),
        "decode_tok_s": round(decode_tok_s, 2),
        "tok_s_per_user": round(decode_tok_s / batch, 2),
    })
    return out


async def measure(client, args, headers, batch: int, run_idx: int, rng: random.Random) -> dict:
    lo, hi = args.token_id_range
    prompts = [[rng.randrange(lo, hi) for _ in range(args.isl)] for _ in range(batch)]
    stop, running = asyncio.Event(), []
    poller = asyncio.create_task(poll_running(client, args.base_url.rstrip("/") + "/metrics", stop, running))
    url = args.base_url.rstrip("/") + "/v1/completions"
    records = await asyncio.gather(*[
        stream_one(client, url, headers, args.model, p, args.osl) for p in prompts
    ])
    stop.set()
    await poller
    result = analyze(list(records), running, batch)
    result["run"] = run_idx
    if args.keep_timelines:
        t0 = min(r["t_start"] for r in records)
        result["timelines"] = [[(round(t - t0, 6), n) for t, n in r["chunks"]] for r in records]
    return result


def summarize(points: List[dict]) -> dict:
    valid = [p for p in points if p.get("valid")]
    if not valid:
        return {"valid_runs": 0}
    med = lambda key: round(statistics.median(p[key] for p in valid), 4)  # noqa: E731
    return {
        "valid_runs": len(valid),
        "step_ms_p50": med("step_ms_p50"),
        "step_ms_p90": med("step_ms_p90"),
        "decode_tok_s": med("decode_tok_s"),
        "tok_s_per_user": med("tok_s_per_user"),
        "tokens_per_step": med("tokens_per_step"),
        # None = /metrics unreachable, so "exactly B running" is unverified
        "exact_b_all": (None if any(p.get("exact_b") is None for p in valid)
                        else all(p["exact_b"] for p in valid)),
        "step_ms_spread": [min(p["step_ms_p50"] for p in valid), max(p["step_ms_p50"] for p in valid)],
    }


async def run_all(args) -> dict:
    headers = {}
    if args.api_key_env and os.environ.get(args.api_key_env):
        headers["Authorization"] = f"Bearer {os.environ[args.api_key_env]}"
    limits = httpx.Limits(max_connections=max(args.batches) + 4)
    timeout = httpx.Timeout(REQUEST_TIMEOUT_S, connect=30.0)
    rng = random.Random(args.seed)
    results = {}
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        for batch in args.batches:
            for w in range(args.warmup):
                await measure(client, args, headers, batch, -1 - w, rng)
            points = []
            for run_idx in range(args.runs):
                p = await measure(client, args, headers, batch, run_idx, rng)
                points.append(p)
                state = (f"step {p['step_ms_p50']:.3f} ms, {p['tok_s_per_user']:.1f} tok/s/user"
                         if p.get("valid") else f"INVALID ({p['ok_requests']}/{batch} ok)")
                print(f"  B={batch} run {run_idx}: {state}", flush=True)
            results[str(batch)] = {"summary": summarize(points), "runs": points}
    return results


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def print_table(doc: dict):
    print(f"\n{doc['label']}: {doc['model']}  ISL {doc['isl']}  OSL {doc['osl']}")
    print(f"{'B':>4}{'step_ms':>10}{'p90_ms':>10}{'tok/s':>11}{'tok/s/user':>12}{'tok/step':>10}{'exact_B':>9}")
    for b in doc["batches"]:
        s = doc["results"][str(b)]["summary"]
        if not s.get("valid_runs"):
            print(f"{b:>4}{'invalid':>10}")
            continue
        print(f"{b:>4}{s['step_ms_p50']:>10.3f}{s['step_ms_p90']:>10.3f}{s['decode_tok_s']:>11.1f}"
              f"{s['tok_s_per_user']:>12.1f}{s['tokens_per_step']:>10.2f}{str(s['exact_b_all']):>9}")


def cmd_run(args) -> int:
    if httpx is None:
        print("ERROR: httpx is not installed (uv sync --extra engine)")
        return 2
    started = datetime.now(timezone.utc).isoformat()
    results = asyncio.run(run_all(args))
    doc = {
        "schema": "autokernel-decode-step.v1",
        "label": args.label,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "notes": args.notes,
        **{k: getattr(args, k) for k in PROFILE_FIELDS},
        "warmup": args.warmup,
        "results": results,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"\nwrote {args.out}")
    print_table(doc)
    return 0 if any(r["summary"].get("valid_runs") for r in results.values()) else 2


def _load(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else None
    except (OSError, ValueError):
        return None


def cmd_compare(args) -> int:
    a, b = _load(args.baseline), _load(args.candidate)
    if a is None or b is None:
        print(f"MISSING_ARM: could not read {'baseline' if a is None else 'candidate'}")
        return 3
    diff = [k for k in PROFILE_FIELDS if a.get(k) != b.get(k)]
    if diff:
        print(f"PROFILE_MISMATCH: {', '.join(f'{k} ({a.get(k)} vs {b.get(k)})' for k in diff)}")
        return 4
    print(f"\n{a['model']}  ISL {a['isl']}  OSL {a['osl']}   A = {a['label']}   B = {b['label']}")
    print(f"{'B':>4}{'A step_ms':>11}{'B step_ms':>11}{'speedup':>9}{'A tok/s/u':>11}{'B tok/s/u':>11}"
          f"{'tok/s/u':>9}{'A spread':>18}{'exact_B':>9}")
    for batch in a["batches"]:
        sa, sb = a["results"][str(batch)]["summary"], b["results"][str(batch)]["summary"]
        if not (sa.get("valid_runs") and sb.get("valid_runs")):
            print(f"{batch:>4}{'absent':>11}")
            continue
        spread = f"{sa['step_ms_spread'][0]:.3f}-{sa['step_ms_spread'][1]:.3f}"
        print(f"{batch:>4}{sa['step_ms_p50']:>11.3f}{sb['step_ms_p50']:>11.3f}"
              f"{sa['step_ms_p50'] / sb['step_ms_p50']:>8.3f}x{sa['tok_s_per_user']:>11.1f}"
              f"{sb['tok_s_per_user']:>11.1f}{sb['tok_s_per_user'] / sa['tok_s_per_user']:>8.3f}x"
              f"{spread:>18}{str(sa['exact_b_all'] and sb['exact_b_all']):>9}")
    print("\nspeedup = A step / B step; tok/s/u = B tok/s/user / A tok/s/user (differs when tokens per step differ,"
          " e.g. spec decode).\nRun the baseline twice first: its A/A ratio is the noise floor.")
    return 0


def cmd_table(args) -> int:
    """Markdown progress table: every candidate run against one baseline, in order."""
    base = _load(args.baseline)
    if base is None:
        print(f"MISSING_ARM: could not read baseline {args.baseline}")
        return 3
    rows = []
    for path in args.candidates:
        doc = _load(path)
        if doc is None:
            print(f"MISSING_ARM: could not read {path}")
            return 3
        diff = [k for k in PROFILE_FIELDS if base.get(k) != doc.get(k)]
        if diff:
            print(f"PROFILE_MISMATCH in {path}: {', '.join(diff)}")
            return 4
        rows.append(doc)

    batches = base["batches"]
    print(f"**{base['model']}**, ISL {base['isl']}, OSL {base['osl']}; baseline = `{base['label']}`. "
          f"Cells: step ms (speedup vs baseline). Last column: tok/s/user at B={batches[0]}.\n")
    print("| # | arm | " + " | ".join(f"B={b}" for b in batches) + f" | tok/s/user B={batches[0]} | exact_B | notes |")
    print("|---|---|" + "---|" * len(batches) + "---|---|---|")
    for i, doc in enumerate([base] + rows):
        cells, exact = [], True
        for b in batches:
            s, sb = doc["results"][str(b)]["summary"], base["results"][str(b)]["summary"]
            if not (s.get("valid_runs") and sb.get("valid_runs")):
                cells.append("absent")
                continue
            cells.append(f"{s['step_ms_p50']:.3f} ({sb['step_ms_p50'] / s['step_ms_p50']:.2f}x)")
            exact = exact and bool(s["exact_b_all"])
        s0 = doc["results"][str(batches[0])]["summary"]
        tps = f"{s0['tok_s_per_user']:.1f}" if s0.get("valid_runs") else "absent"
        print(f"| {i} | {doc['label']} | " + " | ".join(cells) + f" | {tps} | {exact} | {doc.get('notes', '')} |")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode-step benchmark at exactly B running requests")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Measure a running server")
    r.add_argument("--base-url", required=True, help="Server root, e.g. http://127.0.0.1:8000")
    r.add_argument("--model", required=True, help="Served model name")
    r.add_argument("--batch", default="1,2,4,8", help="Comma-separated B values")
    r.add_argument("--isl", type=int, default=1, help="Prompt length in tokens")
    r.add_argument("--osl", type=int, default=1024, help="Output length in tokens")
    r.add_argument("--runs", type=int, default=3, help="Timed runs per B")
    r.add_argument("--warmup", type=int, default=1, help="Discarded runs per B")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--token-id-range", default="1000,20000",
                   help="Random prompt token IDs are drawn from [lo, hi)")
    r.add_argument("--api-key-env", default="", help="Env var holding a bearer token, if the server needs one")
    r.add_argument("--label", required=True, help="Arm name, e.g. stock or ours-exp12")
    r.add_argument("--notes", default="", help="Free text stored in the output (image digest, flags)")
    r.add_argument("--out", default="", help="Raw JSON output path")
    r.add_argument("--keep-timelines", action="store_true", help="Store every chunk timestamp")

    c = sub.add_parser("compare", help="Compare two run files (A = baseline, B = candidate)")
    c.add_argument("baseline")
    c.add_argument("candidate")

    t = sub.add_parser("table", help="Markdown progress table of candidate runs vs one baseline")
    t.add_argument("baseline")
    t.add_argument("candidates", nargs="+", help="Run files in experiment order")

    args = parser.parse_args()
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "table":
        return cmd_table(args)

    try:
        args.batches = [int(x) for x in args.batch.split(",") if x.strip()]
        args.token_id_range = [int(x) for x in args.token_id_range.split(",")]
        assert args.batches and min(args.batches) >= 1 and args.isl >= 1 and args.osl >= 16
        assert len(args.token_id_range) == 2 and 0 <= args.token_id_range[0] < args.token_id_range[1]
    except (ValueError, AssertionError):
        print("ERROR: bad --batch, --isl, --osl or --token-id-range")
        return 2
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
