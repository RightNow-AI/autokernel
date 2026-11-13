"""
AutoKernel engine mode -- decode-step benchmark for a running vLLM or SGLang server.

Records SSE event-gap latency and usage-based decode throughput in a common
decode window at requested batch B, fixed ISL, OSL 1024, greedy. SSE events
can merge multiple engine outputs, so these are not engine-step timestamps.
Occupancy is sampled through /metrics, not proven at every scheduler step.

How one measurement works:
  1. Launch B streaming requests at once against /v1/completions. Prompts are
     random token IDs (exact ISL, distinct per request so the prefix cache
     cannot help), temperature 0, ignore_eos, max_tokens = min_tokens = OSL.
  2. Every chunk's arrival time and cumulative completion_tokens are recorded
     (stream_options.continuous_usage_stats), so token counts are exact even
     under speculative decoding, where one chunk carries several tokens.
  3. Require [DONE], a successful finish reason, requested OSL and observed ISL.
     The window opens at the LAST first-token arrival and closes at the FIRST
     last-token arrival. Latency uses only event gaps fully inside the window;
     token counts use cumulative usage at or before each boundary separately.
  4. While it runs, /metrics is polled for the server's own running-request
     gauge (vllm:num_requests_running, or sglang:num_running_reqs with
     --engine sglang), summed over its series, so the result records sampled
     occupancy across all engines with coverage bounds.

SGLang (--engine sglang): start the server with --enable-metrics, or /metrics
has no gauge and every B stays unverified (exact_b_all false). Its gauge is
summed over all series it publishes; the result keeps the series names, so
check on the first real run that they are one per data-parallel group. Token
counts still come only from continuous usage in the stream: a server that does
not send it is refused, never counted from chunks.

Reported per B: event gap ms (legacy step_ms keys), p90, tokens per event
(legacy tokens_per_step key), and sampled occupancy in the window.
The primary rate ("window", the default) is the usage-token delta inside that
exact-B window divided by its duration, so it excludes prefill like step_ms.
Full-stream rates use usage N and dt = last output arrival - first output
arrival, with both N/dt and (N-1)/dt retained as alternatives; they can
include other requests' prefill overlap.

Run the client on the same host as the server: network jitter lands directly
in the event gaps. Cross-check with graph-replay time from a profiler such as nsys.

Usage:
    uv run --extra engine engine/decode_step.py run --base-url http://127.0.0.1:8000 \\
        --model <served-name> --batch 1,2,4,8 --isl 1 --osl 1024 --label stock \\
        --out results/stock.json
    uv run --extra engine engine/decode_step.py compare results/stock.json results/ours.json
    uv run --extra engine engine/decode_step.py table results/stock.json results/exp01.json results/exp02.json

Exit codes: 0 ok, 2 usage error or every request failed, 3 missing/unreadable
file, 4 incompatible profile/identity/rate basis, 5 inexact/unverified occupancy
or an unverified event-gap statistic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional

if __package__:
    from .identity import flatten_identity, identity_differences, validate_identity
else:
    from identity import flatten_identity, identity_differences, validate_identity

try:
    import httpx
except ImportError:
    httpx = None

# Fields that must match between two runs for a comparison to mean anything
PROFILE_FIELDS = ("model", "isl", "osl", "batches", "runs", "seed", "token_id_range")
FULL_STREAM_FIELDS = {"N/dt": "tok_s_n_over_dt", "(N-1)/dt": "tok_s_n_minus_1_over_dt"}
# "window": usage-token delta inside the exact-B common window / window duration, per user.
# It is the default because it excludes prefill, matching the event-gap window.
RATE_FIELDS = {"window": "tok_s_window_per_user", **FULL_STREAM_FIELDS}
DEFAULT_RATE_FORMULA = "window"
LEGACY_RATE_FORMULA = "legacy window delta/dt"

# The running-request gauge each engine publishes on /metrics. Exact B is proven
# only from this gauge; a run where it cannot be read stays unverified.
RUNNING_METRICS = {"vllm": "vllm:num_requests_running", "sglang": "sglang:num_running_reqs"}
DEFAULT_ENGINE = "vllm"

METRICS_POLL_S = 0.25
# /metrics competes with B streams on a busy API server, so one slow or failed poll
# only stops counting as fresh; coverage fails when the fresh gap bound breaks.
METRICS_MAX_GAP_S = 1.5
METRICS_MAX_POLL_S = 1.0
STEP_BASIS = "sse-event-gap"
WINDOW_TOKEN_BASIS = "cumulative-at-boundaries"
REQUEST_TIMEOUT_S = 3600.0


def token_rates(n: Optional[int], first: Optional[float], last: Optional[float]) -> dict:
    """Both full-stream formulas; undefined intervals stay null, never infinity."""
    rates = {key: None for key in FULL_STREAM_FIELDS.values()}
    if (type(n) is not int or n < 2 or first is None or last is None
            or not math.isfinite(first) or not math.isfinite(last) or last <= first):
        return rates
    dt = last - first
    return {"tok_s_n_over_dt": n / dt, "tok_s_n_minus_1_over_dt": (n - 1) / dt}


def stream_stats(record: dict) -> dict:
    """Persist per-stream evidence even when full chunk timelines are omitted."""
    chunks = record["chunks"]
    first = record.get("t_first_output_token", chunks[0][0] if chunks else None)
    last = record.get("t_last_output_token", chunks[-1][0] if chunks else None)
    n = record.get("completion_tokens")
    return {"t_first_output_token": first, "t_last_output_token": last,
            "completion_tokens": n, "prompt_tokens": record.get("prompt_tokens"),
            **{key: record.get(key) for key in ("done_received", "finish_reason",
                "requested_prompt_tokens", "requested_completion_tokens")},
            "token_source": record.get("token_source"), "error": record.get("error"),
            **token_rates(n, first, last)}


def stream_error(record: dict) -> Optional[str]:
    """Require successful terminal evidence and the requested token counts."""
    errors = [record["error"]] if record.get("error") else []
    if record.get("done_received") is not True:
        errors.append("missing [DONE]")
    if record.get("finish_reason") not in ("stop", "length"):
        errors.append("missing or unsuccessful finish reason")
    for key in ("completion_tokens", "prompt_tokens"):
        observed, requested = record.get(key), record.get("requested_" + key)
        if type(requested) is not int or requested < 1:
            errors.append(f"missing requested {key}")
        elif type(observed) is not int or observed != requested:
            value = observed if type(observed) is int else "missing/invalid"
            errors.append(f"{key}: observed {value}, requested {requested}")
    existing = record.get("error") or ""
    return "; ".join(dict.fromkeys(e for e in errors if e == existing or e not in existing)) or None


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
           "requested_prompt_tokens": len(prompt_ids), "requested_completion_tokens": osl,
           "done_received": False, "finish_reason": None,
           "t_first_output_token": None, "t_last_output_token": None,
           "completion_tokens": None, "token_source": "usage", "error": None}
    try:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                rec["error"] = f"HTTP {resp.status_code}"
                return rec
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload.strip() == "[DONE]":
                    rec["done_received"] = True
                    break
                t = time.perf_counter()
                msg = json.loads(payload)
                usage = msg.get("usage") or {}
                has_text = any(c.get("text") for c in msg.get("choices", []))
                tokens = usage.get("completion_tokens")
                # A token may decode to empty text. Only a choices-bearing event
                # with increasing usage is an output event; trailers are not.
                has_output = has_text or (bool(msg.get("choices")) and type(tokens) is int
                                          and tokens > (rec["completion_tokens"] or 0))
                if usage.get("prompt_tokens") is not None:
                    rec["prompt_tokens"] = usage["prompt_tokens"]
                    if type(rec["prompt_tokens"]) is not int or rec["prompt_tokens"] != len(prompt_ids):
                        rec["error"] = "observed prompt_tokens differs from requested ISL"
                if usage.get("completion_tokens") is not None:
                    rec["completion_tokens"] = usage["completion_tokens"]
                if has_output:
                    if rec["finish_reason"] is not None:
                        rec["error"] = "output received after finish reason"
                    if rec["t_first_output_token"] is None:
                        rec["t_first_output_token"] = t
                    rec["t_last_output_token"] = t
                    if type(tokens) is not int or tokens < 1:
                        rec["error"] = "missing or invalid continuous completion_tokens usage"
                        continue
                    if rec["chunks"] and tokens <= rec["chunks"][-1][1]:
                        rec["error"] = "completion_tokens usage did not increase"
                        continue
                    rec["chunks"].append((t, tokens))
                for choice in msg.get("choices", []):
                    if choice.get("finish_reason") is not None:
                        rec["finish_reason"] = choice["finish_reason"]
            if rec["chunks"] and rec["completion_tokens"] != rec["chunks"][-1][1]:
                rec["error"] = "final usage differs from last output-token usage"
    except Exception as exc:  # noqa: BLE001 -- recorded, not raised
        # Exception text may contain request credentials or response content.
        rec["error"] = f"stream failed: {type(exc).__name__}"
    rec["t_end"] = time.perf_counter()
    rec["error"] = stream_error(rec)
    rec.update(token_rates(rec["completion_tokens"], rec["t_first_output_token"], rec["t_last_output_token"]))
    return rec


def parse_running_metrics(body: str, metric: str = RUNNING_METRICS[DEFAULT_ENGINE]) -> tuple[int, list[str]]:
    """Sum every engine's gauge; reject missing, duplicate or invalid samples."""
    series, total = set(), 0
    name_re = re.escape(metric)
    for line in body.splitlines():
        if not re.match(rf"^{name_re}(?:\{{|\s|$)", line):
            continue
        match = re.fullmatch(rf"({name_re}(?:\{{.*\}})?)\s+(\S+)(?:\s+\S+)?", line)
        if not match:
            raise ValueError("malformed running-request metric")
        name, raw = match.groups()
        value = float(raw)
        if name in series or not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError("duplicate or invalid running-request metric")
        series.add(name)
        total += int(value)
    if not series:
        raise ValueError("missing running-request metric")
    return total, sorted(series)


async def poll_running(client, metrics_url: str, stop: asyncio.Event, samples: list,
                       metric: str = RUNNING_METRICS[DEFAULT_ENGINE]):
    """Retain every poll attempt, including failures and request/response times."""
    while not stop.is_set():
        poll = {"t_start": time.perf_counter(), "running": None, "series": [], "error": None}
        try:
            resp = await client.get(metrics_url, timeout=5.0)
            if resp.status_code != 200:
                raise ValueError("metrics HTTP failure")
            poll["running"], poll["series"] = parse_running_metrics(resp.text, metric)
        except Exception as exc:  # noqa: BLE001 -- absence must remain evidence
            poll["error"] = f"metrics poll failed: {type(exc).__name__}"
        poll["t_end"] = time.perf_counter()
        samples.append(poll)
        try:
            await asyncio.wait_for(stop.wait(), timeout=METRICS_POLL_S)
        except asyncio.TimeoutError:
            pass


def occupancy_evidence(polls: list, start: float, end: float, batch: int) -> dict:
    """Bounded sampled occupancy, not scheduler-level proof of every instant."""
    # Historical (timestamp, count) tuples lack poll-duration/failure evidence.
    polls = [p if isinstance(p, dict) else {"t_start": p[0], "t_end": p[0],
             "running": p[1], "series": [], "error": "legacy sample lacks poll evidence"} for p in polls]
    relevant = [p for p in polls if p["t_start"] <= end and p["t_end"] >= start]
    failed = sum(bool(p.get("error")) for p in relevant)
    stale = sum(not 0 <= p["t_end"] - p["t_start"] <= METRICS_MAX_POLL_S for p in relevant)
    fresh = sorted((p for p in relevant if not p.get("error")
                    and 0 <= p["t_end"] - p["t_start"] <= METRICS_MAX_POLL_S
                    and start <= p["t_end"] <= end), key=lambda p: p["t_end"])
    times = sorted({p["t_end"] for p in fresh})
    gaps = [b - a for a, b in zip([start, *times], [*times, end])]
    required = max(2, math.ceil(max(0, end - start) / METRICS_MAX_GAP_S) + 1)
    reasons = []
    if len(times) < required:
        reasons.append(f"insufficient fresh samples: {len(times)} < {required}")
    if max(gaps, default=0) > METRICS_MAX_GAP_S + 1e-9:
        reasons.append("sample or boundary gap exceeds bound")
    series_sets = {tuple(p.get("series", [])) for p in fresh}
    if not series_sets or () in series_sets or len(series_sets) != 1:
        reasons.append("missing or changing engine series")
    values = [p.get("running") for p in fresh]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 or int(v) != v for v in values):
        reasons.append("invalid running-request counts")
    coverage = not reasons and end > start
    return {"basis": "sampled-occupancy", "poll_interval_s": METRICS_POLL_S,
            "max_gap_s": METRICS_MAX_GAP_S, "max_poll_duration_s": METRICS_MAX_POLL_S,
            "poll_attempts": len(relevant), "failed_polls": failed, "stale_polls": stale,
            "fresh_samples": len(times), "required_samples": required,
            "largest_gap_s": max(gaps, default=None), "coverage_ok": coverage, "reasons": reasons,
            "running_min": min(values) if values and not reasons else None,
            "running_max": max(values) if values and not reasons else None,
            "exact_b": all(v == batch for v in values) if coverage else None}


def cumulative_at(chunks: list, boundary: float) -> int:
    """Right-continuous cumulative usage; never interpolate between events."""
    return next(n for t, n in reversed(chunks) if t <= boundary)


def event_gap_evidence(records: list[dict], identity: Optional[dict]) -> dict:
    k = (identity or {}).get("speculation", {}).get("k")
    merged = None
    if k == 0:
        merged = 0
        for record in records:
            previous = 0
            for _, tokens in record["chunks"]:
                merged += tokens - previous > 1
                previous = tokens
    return {"step_basis": STEP_BASIS, "merged_events": merged,
            "step_status": "UNVERIFIED" if merged else "EVENT_GAP",
            "step_reason": ("multi-token events without speculation" if merged else
                            "merging not detectable with speculation or unknown identity" if k != 0 else
                            "no merged events detected; event timing is not engine-step timing")}


# ---------------------------------------------------------------------------
# One measurement at batch B
# ---------------------------------------------------------------------------

def analyze(records: List[dict], running: list, batch: int,
            rate_formula_primary: str = DEFAULT_RATE_FORMULA, identity: Optional[dict] = None) -> dict:
    records = [{**r, "error": stream_error(r)} for r in records]
    streams = [stream_stats(r) for r in records]
    ok = [r for r, s in zip(records, streams) if not r["error"] and len(r["chunks"]) >= 2
          and r.get("token_source") == "usage" and s["tok_s_n_over_dt"] is not None]
    out = {"batch": batch, "ok_requests": len(ok), "failed_requests": len(records) - len(ok),
           "errors": [r["error"] for r in records if r["error"]][:3],
           "streams": streams, "rate_formula_primary": rate_formula_primary, "exact_b": None,
           "metric_polls": running, "window_token_basis": WINDOW_TOKEN_BASIS,
           **event_gap_evidence(records, identity)}
    if len(ok) != batch:
        out["valid"] = False
        out["step_status"] = "UNVERIFIED"
        return out

    w_start = max(r["chunks"][0][0] for r in ok)   # last prefill done
    w_end = min(r["chunks"][-1][0] for r in ok)    # first request done
    gaps_ms, gap_tokens = [], 0
    for r in ok:
        for (t0, n0), (t1, n1) in zip(r["chunks"], r["chunks"][1:]):
            if t0 >= w_start and t1 <= w_end:
                gaps_ms.append((t1 - t0) * 1e3)
                gap_tokens += n1 - n0

    occupancy = occupancy_evidence(running, w_start, w_end, batch)
    out.update({
        "window_start": w_start, "window_end": w_end,
        "window_s": round(w_end - w_start, 4),
        "step_samples": len(gaps_ms),
        "token_source": "usage",
        "prompt_tokens": sorted({r["prompt_tokens"] for r in ok if r["prompt_tokens"] is not None}),
        "occupancy": occupancy,
        **{key: occupancy[key] for key in ("running_min", "running_max", "exact_b")},
    })
    if len(gaps_ms) < 10 or w_end <= w_start:
        out["valid"] = False
        out["step_status"] = "UNVERIFIED"
        return out

    boundaries = [{"start": cumulative_at(r["chunks"], w_start),
                   "end": cumulative_at(r["chunks"], w_end)} for r in ok]
    tokens_in_window = sum(b["end"] - b["start"] for b in boundaries)
    gaps_ms.sort()
    rates = {key: statistics.mean(s[key] for s in streams) for key in FULL_STREAM_FIELDS.values()}
    window_decode_tok_s = tokens_in_window / (w_end - w_start)
    rates["tok_s_window_per_user"] = window_decode_tok_s / batch
    primary = rates[RATE_FIELDS[rate_formula_primary]]
    out.update({
        "valid": True,
        "step_ms_p50": round(statistics.median(gaps_ms), 4),
        "step_ms_p90": round(gaps_ms[int(0.9 * (len(gaps_ms) - 1))], 4),
        "tokens_per_step": round(gap_tokens / len(gaps_ms), 3),
        "window_tokens": tokens_in_window, "window_boundary_usage": boundaries,
        "window_decode_tok_s": window_decode_tok_s,
        **rates,
        "decode_tok_s": primary * batch,
        "tok_s_per_user": primary,
    })
    return out


async def measure(client, args, headers, batch: int, run_idx: int, rng: random.Random) -> dict:
    lo, hi = args.token_id_range
    prompts = [[rng.randrange(lo, hi) for _ in range(args.isl)] for _ in range(batch)]
    stop, running = asyncio.Event(), []
    metric = RUNNING_METRICS[getattr(args, "engine", DEFAULT_ENGINE)]
    poller = asyncio.create_task(poll_running(client, args.base_url.rstrip("/") + "/metrics", stop, running, metric))
    url = args.base_url.rstrip("/") + "/v1/completions"
    records = await asyncio.gather(*[
        stream_one(client, url, headers, args.model, p, args.osl) for p in prompts
    ])
    stop.set()
    await poller
    result = analyze(list(records), running, batch, args.rate_formula_primary, args.run_identity)
    result["run"] = run_idx
    if args.keep_timelines:
        t0 = min(r["t_start"] for r in records)
        result["timelines"] = [[(round(t - t0, 6), n) for t, n in r["chunks"]] for r in records]
    return result


def summarize(points: List[dict]) -> dict:
    valid = [p for p in points if p.get("valid")]
    # A failed or unobserved repetition cannot disappear behind valid ones.
    states = [p.get("exact_b") if p.get("valid") or p.get("exact_b") is False else None for p in points]
    exact = False if any(s is False for s in states) else (True if states and all(s is True for s in states) else None)
    evidence = {
        "step_basis": STEP_BASIS,
        "step_status": "EVENT_GAP" if points and all(p.get("valid") and p.get("step_status") == "EVENT_GAP"
                                                       for p in points) else "UNVERIFIED",
        "merged_events": sum(p["merged_events"] for p in points)
                         if points and all(type(p.get("merged_events")) is int for p in points) else None,
        "occupancy_basis": "sampled-occupancy",
        "occupancy_coverage_all": bool(points) and all(p.get("occupancy", {}).get("coverage_ok") is True for p in points),
    }
    if not valid:
        return {"valid_runs": 0, "exact_b_all": exact, **evidence}
    med = lambda key: round(statistics.median(p[key] for p in valid), 4)  # noqa: E731
    return {
        "valid_runs": len(valid),
        **evidence,
        "step_ms_p50": med("step_ms_p50"),
        "step_ms_p90": med("step_ms_p90"),
        "decode_tok_s": med("decode_tok_s"),
        "tok_s_per_user": med("tok_s_per_user"),
        "tokens_per_step": med("tokens_per_step"),
        **{key: med(key) for key in (*RATE_FIELDS.values(), "window_decode_tok_s")},
        # None = sampled occupancy lacks complete, fresh coverage.
        "exact_b_all": exact,
        "step_ms_spread": [min(p["step_ms_p50"] for p in valid), max(p["step_ms_p50"] for p in valid)],
    }


async def run_all(args, log) -> dict:
    def progress(message: str):
        print(message, flush=True)
        print(message, file=log, flush=True)

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
                progress(f"  B={batch} warmup {w}: started")
                await measure(client, args, headers, batch, -1 - w, rng)
            points = []
            for run_idx in range(args.runs):
                progress(f"  B={batch} run {run_idx}: started")
                p = await measure(client, args, headers, batch, run_idx, rng)
                points.append(p)
                state = (f"event gap {p['step_ms_p50']:.3f} ms [{p['step_status']}], {p['tok_s_per_user']:.1f} tok/s/user"
                         if p.get("valid") else f"INVALID ({p['ok_requests']}/{batch} ok): {'; '.join(p['errors'])}")
                progress(f"  B={batch} run {run_idx}: {state}")
            results[str(batch)] = {"summary": summarize(points), "runs": points}
    return results


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def rate_formula(doc: dict) -> str:
    # Old window-delta rates cannot be relabelled or reconstructed from summaries.
    return doc.get("rate_formula_primary", LEGACY_RATE_FORMULA)


def comparable_formula(doc: dict) -> str:
    # The formula name alone is insufficient: check window_token_basis as well.
    formula = rate_formula(doc)
    return "window" if formula == LEGACY_RATE_FORMULA else formula


def primary_rates(doc: dict, summary: dict, batch: int) -> tuple[float, float]:
    formula = rate_formula(doc)
    if formula == LEGACY_RATE_FORMULA:
        return summary["decode_tok_s"], summary["tok_s_per_user"]
    per_user = summary[RATE_FIELDS[formula]]
    return per_user * batch, per_user


def exact_b_status(doc: dict, batch: int) -> str:
    entry = doc["results"][str(batch)]
    states = [entry["summary"].get("exact_b_all")]
    states += [p.get("exact_b") for p in entry.get("runs", [])]
    if any(s is False for s in states):
        return "INEXACT"
    if any(s is not True for s in states):
        return "UNVERIFIED"
    if entry["summary"].get("occupancy_coverage_all") is not True:
        return "UNVERIFIED"
    if any(p.get("occupancy", {}).get("coverage_ok") is not True for p in entry.get("runs", [])):
        return "UNVERIFIED"
    if not entry["summary"].get("valid_runs"):
        return "UNVERIFIED"
    if any(p.get("valid") is False for p in entry.get("runs", [])):
        return "UNVERIFIED"
    return "True"


def check_exact_b(docs: list[dict], allow_inexact: bool) -> int:
    issues = [f"{doc['label']} B={batch}: {exact_b_status(doc, batch)}"
              for doc in docs for batch in doc["batches"] if exact_b_status(doc, batch) != "True"]
    if not issues:
        return 0
    print(("UNQUALIFIED exact_B override: " if allow_inexact else "EXACT_B_REFUSED: ") + "; ".join(issues))
    return 0 if allow_inexact else 5


def step_status(doc: dict, batch: int) -> str:
    entry = doc["results"][str(batch)]
    for point in [entry["summary"], *entry.get("runs", [])]:
        if (point.get("step_basis") != STEP_BASIS or point.get("step_status") != "EVENT_GAP"
                or point.get("merged_events") or point.get("valid") is False):
            return "UNVERIFIED"
    return "EVENT_GAP"


def check_step_statistics(docs: list[dict], allow_inexact: bool) -> int:
    issues = [f"{doc['label']} B={batch}: UNVERIFIED event gap"
              for doc in docs for batch in doc["batches"] if step_status(doc, batch) == "UNVERIFIED"]
    if not issues:
        return 0
    print(("UNQUALIFIED step statistic override: " if allow_inexact else "STEP_STATISTIC_REFUSED: ") + "; ".join(issues))
    return 0 if allow_inexact else 5


def check_comparison(a: dict, b: dict, args) -> int:
    diff = [k for k in PROFILE_FIELDS if a.get(k) != b.get(k)]
    # Files written before --engine existed measured vLLM.
    if a.get("engine", DEFAULT_ENGINE) != b.get("engine", DEFAULT_ENGINE):
        diff.append("engine")
    if comparable_formula(a) != comparable_formula(b):
        diff.append("rate_formula_primary")
    elif comparable_formula(a) == "window" and (a.get("window_token_basis", "contained-event-gaps")
                                               != b.get("window_token_basis", "contained-event-gaps")):
        diff.append("window_token_basis")
    if diff:
        print("PROFILE_MISMATCH: " + ", ".join(diff))
        return 4
    identities = [a.get("identity"), b.get("identity")]
    for side, identity in zip(("baseline", "candidate"), identities):
        if identity is not None:
            try:
                validate_identity(identity)
            except ValueError as exc:
                print(f"IDENTITY_INVALID ({side}): {exc}")
                return 4
    if any(identity is None for identity in identities):
        if not args.allow_missing_identity:
            sides = [side for side, identity in zip(("baseline", "candidate"), identities) if identity is None]
            print("IDENTITY_MISSING: " + ", ".join(sides) + "; use --allow-missing-identity for UNQUALIFIED output")
            return 4
    else:
        allowed = {key.strip() for key in args.arm_delta.split(",") if key.strip()}
        known = flatten_identity(identities[0]).keys() | flatten_identity(identities[1]).keys()
        unknown = allowed - known
        if unknown:
            print("ARM_DELTA_UNKNOWN: " + ", ".join(sorted(unknown)))
            return 4
        changed = identity_differences(*identities)
        undeclared = [key for key in changed if key not in allowed]
        if undeclared:
            print("IDENTITY_MISMATCH: " + ", ".join(undeclared))
            return 4
        if changed:
            print("DECLARED_ARM_DELTA: " + ", ".join(changed))
    for doc in (a, b):
        formula = rate_formula(doc)
        if formula not in (*RATE_FIELDS, LEGACY_RATE_FORMULA):
            print("RATE_INVALID: rate_formula_primary")
            return 4
        for batch in doc["batches"]:
            entry = doc["results"][str(batch)]
            if any(p.get("token_source") == "chunks" for p in [entry["summary"], *entry.get("runs", [])]):
                print(f"RATE_INVALID: {doc['label']} B={batch} used chunk counts")
                return 4
            if entry["summary"].get("valid_runs"):
                keys = ("decode_tok_s", "tok_s_per_user") if formula == LEGACY_RATE_FORMULA else tuple(RATE_FIELDS.values())
                if any(type(entry["summary"].get(key)) not in (int, float)
                       or not math.isfinite(entry["summary"][key]) or entry["summary"][key] <= 0 for key in keys):
                    print(f"RATE_INVALID: {doc['label']} B={batch} missing or invalid rate fields")
                    return 4
    return 0


def print_table(doc: dict):
    print(f"\n{doc['label']}: {doc['model']}  ISL {doc['isl']}  OSL {doc['osl']}")
    formula = rate_formula(doc)
    basis = ("exact-B window usage delta per user" if formula in ("window", LEGACY_RATE_FORMULA)
             else "full-stream mean per user, may include prefill overlap")
    print(f"Rate formula: {formula} ({basis})")
    print("Latency basis: SSE event gap; exact_B is sampled occupancy")
    print(f"{'B':>4}{'event gap ms':>14}{'p90_ms':>10}{'tok/s':>11}{'tok/s/user':>12}{'tok/event':>10}{'exact_B':>12}  event statistic")
    for b in doc["batches"]:
        s = doc["results"][str(b)]["summary"]
        if not s.get("valid_runs"):
            print(f"{b:>4}{'invalid':>10}")
            continue
        total, per_user = primary_rates(doc, s, b)
        print(f"{b:>4}{s['step_ms_p50']:>14.3f}{s['step_ms_p90']:>10.3f}{total:>11.1f}"
              f"{per_user:>12.1f}{s['tokens_per_step']:>10.2f}{exact_b_status(doc, b):>12}  {step_status(doc, b)}")


def cmd_run(args) -> int:
    identity = None
    if args.identity:
        try:
            with open(args.identity, encoding="utf-8-sig") as f:
                identity = validate_identity(json.load(f))
        except (OSError, ValueError) as exc:
            print(f"IDENTITY_INVALID: {exc}")
            return 4
    else:
        print("WARNING: no --identity supplied; recording identity: null. Comparisons will refuse it by default.", file=sys.stderr)
    args.run_identity = identity
    log_path = args.log or (args.out + ".log" if args.out else "decode-step.log")
    paths = [os.path.normcase(os.path.realpath(path)) for path in (args.identity, args.out, log_path) if path]
    if len(paths) != len(set(paths)):
        print("ERROR: identity, result and log paths must be distinct")
        return 2
    if args.dry_run:
        invocation = [sys.executable, "engine/decode_step.py", "run"]
        for option in ("engine", "base_url", "model", "batch", "isl", "osl", "runs", "warmup", "seed",
                       "token_id_range", "api_key_env", "label", "notes", "out", "log", "identity",
                       "rate_formula_primary"):
            value = getattr(args, option)
            if value is not None and value != "":
                rendered = ",".join(map(str, value)) if isinstance(value, list) else str(value)
                invocation.extend(["--" + option.replace("_", "-"), rendered])
        if args.keep_timelines:
            invocation.append("--keep-timelines")
        print(json.dumps({"dry_run": True, "argv": invocation, "base_url": args.base_url,
                          **{k: getattr(args, k) for k in PROFILE_FIELDS},
                          "warmup": args.warmup, "label": args.label, "out": args.out, "log": log_path,
                          "identity": identity, "rate_formula_primary": args.rate_formula_primary,
                          "requests_per_batch": "B concurrent POST /v1/completions per warmup and timed run",
                          "request_url": args.base_url.rstrip("/") + "/v1/completions",
                          "metrics_url": args.base_url.rstrip("/") + "/metrics",
                          "engine": args.engine, "running_metric": RUNNING_METRICS[args.engine],
                          "prompt_generation": "seeded random token IDs; warmups advance the same RNG",
                          "stream_options": {"include_usage": True, "continuous_usage_stats": True},
                          "temperature": 0.0, "ignore_eos": True, "max_tokens": args.osl,
                          "min_tokens": args.osl, "metrics_poll_s": METRICS_POLL_S,
                          "metrics_max_gap_s": METRICS_MAX_GAP_S, "metrics_max_poll_s": METRICS_MAX_POLL_S,
                          "step_basis": STEP_BASIS, "window_token_basis": WINDOW_TOKEN_BASIS}, indent=2))
        return 0
    if httpx is None:
        print("ERROR: httpx is not installed (uv sync --extra engine)")
        return 2
    started = datetime.now(timezone.utc).isoformat()
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log:
            print(f"{started} rate_formula_primary={args.rate_formula_primary}", file=log, flush=True)
            results = asyncio.run(run_all(args, log))
    except OSError:
        print("ERROR: could not open/write benchmark log or complete network I/O")
        return 3
    doc = {
        "schema": "autokernel-decode-step.v2",
        "identity": identity,
        "step_basis": STEP_BASIS, "window_token_basis": WINDOW_TOKEN_BASIS,
        "rate_formula_primary": args.rate_formula_primary,
        "engine": args.engine, "running_metric": RUNNING_METRICS[args.engine],
        "label": args.label,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "notes": args.notes,
        **{k: getattr(args, k) for k in PROFILE_FIELDS},
        "warmup": args.warmup,
        "results": results,
    }
    if args.out:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=1)
        except OSError:
            print("ERROR: could not write result JSON")
            return 3
        print(f"\nwrote {args.out}")
    print_table(doc)
    return 0 if any(r["summary"].get("valid_runs") for r in results.values()) else 2


def _load(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8-sig") as f:
            doc = json.load(f)
        if (not isinstance(doc, dict) or not isinstance(doc.get("batches"), list)
                or not doc["batches"] or any(type(b) is not int or b < 1 for b in doc["batches"])):
            return None
        # These fields are also present in v1. Do not require new metadata here.
        for key in ("label", *PROFILE_FIELDS):
            doc[key]
        for batch in doc["batches"]:
            entry = doc["results"][str(batch)]
            if not isinstance(entry["summary"], dict) or not isinstance(entry.get("runs", []), list):
                return None
            if any(not isinstance(point, dict) for point in entry.get("runs", [])):
                return None
            summary = entry["summary"]
            if type(summary.get("valid_runs")) is not int or summary["valid_runs"] < 0:
                return None
            if summary["valid_runs"]:
                values = [summary[key] for key in ("step_ms_p50", "step_ms_p90", "tokens_per_step")]
                spread = summary["step_ms_spread"]
                if not isinstance(spread, list) or len(spread) != 2:
                    return None
                if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in values + spread):
                    return None
        return doc
    except (OSError, ValueError, KeyError, TypeError):
        return None


def cmd_compare(args) -> int:
    a, b = _load(args.baseline), _load(args.candidate)
    if a is None or b is None:
        print(f"MISSING_ARM: could not read {'baseline' if a is None else 'candidate'}")
        return 3
    code = check_comparison(a, b, args)
    if code:
        return code
    code = check_exact_b([a, b], args.allow_inexact)
    if code:
        return code
    code = check_step_statistics([a, b], args.allow_inexact)
    if code:
        return code
    if args.allow_missing_identity:
        print("UNQUALIFIED: --allow-missing-identity enabled")
    print(f"\n{a['model']}  ISL {a['isl']}  OSL {a['osl']}   A = {a['label']}   B = {b['label']}")
    print(f"Rate formula: {rate_formula(a)}")
    print("Latency basis: SSE event gap; exact_B is sampled occupancy")
    print(f"{'B':>4}{'A gap_ms':>11}{'B gap_ms':>11}{'gap ratio':>10}{'A tok/s/u':>11}{'B tok/s/u':>11}"
          f"{'tok/s/u':>9}{'A spread':>18}{'exact_B':>9}")
    for batch in a["batches"]:
        sa, sb = a["results"][str(batch)]["summary"], b["results"][str(batch)]["summary"]
        if not (sa.get("valid_runs") and sb.get("valid_runs")):
            print(f"{batch:>4}{'absent':>11}")
            continue
        spread = f"{sa['step_ms_spread'][0]:.3f}-{sa['step_ms_spread'][1]:.3f}"
        _, ra = primary_rates(a, sa, batch)
        _, rb = primary_rates(b, sb, batch)
        exact = f"A:{exact_b_status(a, batch)} B:{exact_b_status(b, batch)}"
        statistic = " UNVERIFIED event gap" if "UNVERIFIED" in (step_status(a, batch), step_status(b, batch)) else ""
        print(f"{batch:>4}{sa['step_ms_p50']:>11.3f}{sb['step_ms_p50']:>11.3f}"
              f"{sa['step_ms_p50'] / sb['step_ms_p50']:>8.3f}x{ra:>11.1f}"
              f"{rb:>11.1f}{rb / ra:>8.3f}x{spread:>18} {exact}{statistic}")
    print("\ngap ratio = A event gap / B event gap; tok/s/u = B tok/s/user / A tok/s/user (differs when tokens per event differ,"
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
        code = check_comparison(base, doc, args)
        if code:
            return code
        rows.append(doc)

    code = check_exact_b([base, *rows], args.allow_inexact)
    if code:
        return code
    code = check_step_statistics([base, *rows], args.allow_inexact)
    if code:
        return code
    if args.allow_missing_identity:
        print("UNQUALIFIED: --allow-missing-identity enabled")
    batches = base["batches"]
    print(f"**{base['model']}**, ISL {base['isl']}, OSL {base['osl']}; baseline = `{base['label']}`. "
          f"Cells: SSE event gap ms (gap ratio vs baseline). exact_B is sampled occupancy. Last rate column: tok/s/user at B={batches[0]}.\n")
    print("| # | arm | " + " | ".join(f"B={b}" for b in batches)
          + f" | tok/s/user B={batches[0]} [{rate_formula(base)}] | exact_B | notes |")
    print("|---|---|" + "---|" * len(batches) + "---|---|---|")
    for i, doc in enumerate([base] + rows):
        cells, markers = [], []
        for b in batches:
            s, sb = doc["results"][str(b)]["summary"], base["results"][str(b)]["summary"]
            state = exact_b_status(doc, b)
            if state != "True":
                markers.append(f"B={b} {state}")
            if not (s.get("valid_runs") and sb.get("valid_runs")):
                cells.append("absent")
                continue
            base_state = exact_b_status(base, b)
            mark = " [UNQUALIFIED exact_B]" if state != "True" or base_state != "True" else ""
            if "UNVERIFIED" in (step_status(doc, b), step_status(base, b)):
                mark += " [UNVERIFIED event gap]"
            cells.append(f"{s['step_ms_p50']:.3f} ({sb['step_ms_p50'] / s['step_ms_p50']:.2f}x){mark}")
        s0 = doc["results"][str(batches[0])]["summary"]
        tps = f"{primary_rates(doc, s0, batches[0])[1]:.1f}" if s0.get("valid_runs") else "absent"
        exact = "; ".join(markers) or "True"
        note = "UNQUALIFIED identity; " if args.allow_missing_identity else ""
        print(f"| {i} | {doc['label']} | " + " | ".join(cells) + f" | {tps} | {exact} | {note}{doc.get('notes', '')} |")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Decode-window SSE event gaps and sampled occupancy at batch B")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Measure a running server")
    r.add_argument("--engine", choices=tuple(RUNNING_METRICS), default=DEFAULT_ENGINE,
                   help="Serving engine; picks the running-request gauge read from /metrics")
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
    r.add_argument("--log", default="", help="Progress log (default: <out>.log, or decode-step.log)")
    r.add_argument("--keep-timelines", action="store_true", help="Store every chunk timestamp")
    r.add_argument("--identity", help="Local JSON identity file (see README for required fields)")
    r.add_argument("--rate-formula-primary", choices=tuple(RATE_FIELDS), default=DEFAULT_RATE_FORMULA)
    r.add_argument("--dry-run", action="store_true", help="Print the resolved measurement plan without network calls or writes")

    c = sub.add_parser("compare", help="Compare two run files (A = baseline, B = candidate)")
    c.add_argument("baseline")
    c.add_argument("candidate")

    t = sub.add_parser("table", help="Markdown progress table of candidate runs vs one baseline")
    t.add_argument("baseline")
    t.add_argument("candidates", nargs="+", help="Run files in experiment order")

    for command in (c, t):
        command.add_argument("--arm-delta", default="", help="Comma-separated identity leaf keys allowed to differ, e.g. image.digest,arm.label")
        command.add_argument("--allow-missing-identity", action="store_true", help="Allow missing identity; mark output UNQUALIFIED")
        command.add_argument("--allow-inexact", action="store_true", help="Allow inexact/unverified occupancy or event statistics; mark affected rows")

    args = parser.parse_args(argv)
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "table":
        return cmd_table(args)

    try:
        args.batches = [int(x) for x in args.batch.split(",") if x.strip()]
        args.token_id_range = [int(x) for x in args.token_id_range.split(",")]
        if (not args.batches or min(args.batches) < 1 or len(set(args.batches)) != len(args.batches)
                or args.isl < 1 or args.osl < 16 or args.runs < 1 or args.warmup < 0
                or len(args.token_id_range) != 2 or not 0 <= args.token_id_range[0] < args.token_id_range[1]):
            raise ValueError
    except ValueError:
        print("ERROR: bad --batch, --isl, --osl, --runs, --warmup or --token-id-range")
        return 2
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
