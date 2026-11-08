"""
AutoKernel engine mode -- time to first token (TTFT) for long prompts on a running vLLM server.

Prefill as the client sees it. Each run is one streaming /v1/completions request with a prompt of exactly
ISL random token IDs, max_tokens = min_tokens = 1 (no decode tail), temperature 0, so the TTFT is one prefill
alone on the engine. Runs go one at a time: nothing else shares the engine while a run is timed.

Modes:
  cold  every run sends a fresh random prompt, so the prefix cache cannot help (the default).
  warm  every run first sends an untimed priming request whose prompt is a fresh prefix of ISL - SUFFIX
        tokens, then times a request with that same prefix and a fresh SUFFIX-token tail: the follow-up
        question over an already-read document. It needs prefix caching on the server. The server's
        usage.prompt_tokens_details.cached_tokens is recorded when reported (vLLM:
        --enable-prompt-tokens-details); a warm run that reports no cached tokens is marked no_cache_hit.

TTFT = arrival of the first output event - request start (client clock, perf_counter). Every run must end
with [DONE], one completion token and prompt_tokens equal to ISL, or it is invalid. Prompts are token IDs,
so tokenization is not measured. Run the client on the server's host, and cross-check against the server's
histogram (vllm:time_to_first_token_seconds).

Usage:
    uv run --extra engine engine/prefill_ttft.py run --base-url http://127.0.0.1:8000 --model <served-name> \\
        --isl 131072,262144,524288,1048576 --runs 3 --label stock --out results/ttft-stock.json
    uv run --extra engine engine/prefill_ttft.py run ... --mode warm --suffix-tokens 512 --label stock-warm
    uv run --extra engine engine/prefill_ttft.py compare results/ttft-stock.json results/ttft-exp01.json
    uv run --extra engine engine/prefill_ttft.py table results/ttft-stock.json results/ttft-exp*.json

Exit codes: 0 ok, 2 usage error or no valid run at any ISL, 3 missing/unreadable file, 4 incompatible
profile or identity.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
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

SCHEMA = "autokernel-prefill-ttft.v1"
# Fields that must match between two runs for a comparison to mean anything
PROFILE_FIELDS = ("model", "isls", "mode", "suffix_tokens", "runs", "seed", "token_id_range")
REQUEST_TIMEOUT_S = 3600.0


def percentile(values: List[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100) of a nonempty list."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


# ---------------------------------------------------------------------------
# One request
# ---------------------------------------------------------------------------

async def stream_first_token(client, url: str, headers: dict, model: str, prompt_ids: List[int]) -> dict:
    """Stream one single-token completion and time its first output event."""
    body = {
        "model": model, "prompt": prompt_ids, "max_tokens": 1, "min_tokens": 1, "ignore_eos": True,
        "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    rec = {"requested_prompt_tokens": len(prompt_ids), "prompt_tokens": None, "completion_tokens": None,
           "cached_tokens": None, "done_received": False, "finish_reason": None, "ttft_s": None,
           "total_s": None, "error": None}
    t_start = time.perf_counter()
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
                if type(usage.get("prompt_tokens")) is int:
                    rec["prompt_tokens"] = usage["prompt_tokens"]
                if type(usage.get("completion_tokens")) is int:
                    rec["completion_tokens"] = usage["completion_tokens"]
                details = usage.get("prompt_tokens_details") or {}
                if type(details.get("cached_tokens")) is int:
                    rec["cached_tokens"] = details["cached_tokens"]
                choices = msg.get("choices") or []
                if rec["ttft_s"] is None and choices and (any(c.get("text") for c in choices)
                                                          or (rec["completion_tokens"] or 0) >= 1):
                    rec["ttft_s"] = t - t_start
                for choice in choices:
                    if choice.get("finish_reason") is not None:
                        rec["finish_reason"] = choice["finish_reason"]
    except Exception as exc:  # noqa: BLE001 -- recorded, not raised
        # Exception text may contain request credentials or response content.
        rec["error"] = f"stream failed: {type(exc).__name__}"
    rec["total_s"] = time.perf_counter() - t_start
    errors = [rec["error"]] if rec["error"] else []
    if not rec["done_received"]:
        errors.append("missing [DONE]")
    if rec["finish_reason"] not in ("stop", "length"):
        errors.append("missing or unsuccessful finish reason")
    if rec["ttft_s"] is None:
        errors.append("no output event")
    if rec["prompt_tokens"] != len(prompt_ids):
        errors.append(f"prompt_tokens: observed {rec['prompt_tokens']}, requested {len(prompt_ids)}")
    if rec["completion_tokens"] != 1:
        errors.append(f"completion_tokens: observed {rec['completion_tokens']}, requested 1")
    rec["error"] = "; ".join(dict.fromkeys(errors)) or None
    return rec


# ---------------------------------------------------------------------------
# A whole run
# ---------------------------------------------------------------------------

def make_prompt(rng: random.Random, n: int, token_range: tuple) -> List[int]:
    return [rng.randint(*token_range) for _ in range(n)]


async def measure(args, log, client=None) -> dict:
    url = args.base_url.rstrip("/") + "/v1/completions"
    headers = {"Authorization": f"Bearer {os.environ[args.api_key_env]}"} if args.api_key_env else {}
    token_range = tuple(int(v) for v in args.token_id_range.split(","))
    rng = random.Random(args.seed)
    own = client is None
    if own:
        client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT_S))
    results = {}
    try:
        for isl in args.isls:
            runs = []
            for i in range(args.warmup + args.runs):
                timed = i >= args.warmup
                if args.mode == "warm":
                    prefix = make_prompt(rng, isl - args.suffix_tokens, token_range)
                    prime = await stream_first_token(client, url, headers, args.model, prefix)
                    prompt = prefix + make_prompt(rng, args.suffix_tokens, token_range)
                else:
                    prime, prompt = None, make_prompt(rng, isl, token_range)
                rec = await stream_first_token(client, url, headers, args.model, prompt)
                if prime is not None:
                    rec["prime_error"] = prime["error"]
                    rec["no_cache_hit"] = None if rec["cached_tokens"] is None else rec["cached_tokens"] == 0
                    if prime["error"] and not rec["error"]:
                        rec["error"] = "priming request failed"
                rec["timed"] = timed
                print(json.dumps({"isl": isl, "run": i, **rec}), file=log, flush=True)
                if timed:
                    runs.append(rec)
            results[str(isl)] = {"runs": runs, "summary": summarize(isl, runs)}
            s = results[str(isl)]["summary"]
            state = (f"p50 {s['ttft_s_p50']:.3f} s, {s['prefill_tok_s_p50']:.0f} tok/s" if s["valid_runs"]
                     else "no valid run")
            print(f"  ISL {isl}: {s['valid_runs']}/{len(runs)} valid, {state}", flush=True)
    finally:
        if own:
            await client.aclose()
    return results


def summarize(isl: int, runs: List[dict]) -> dict:
    valid = [r for r in runs if not r["error"]]
    summary = {"valid_runs": len(valid), "invalid_runs": len(runs) - len(valid)}
    if valid:
        ttft = [r["ttft_s"] for r in valid]
        p50 = percentile(ttft, 50)
        summary.update({"ttft_s_p50": p50, "ttft_s_p90": percentile(ttft, 90), "ttft_s_spread": [min(ttft), max(ttft)],
                        "prefill_tok_s_p50": isl / p50, "cached_tokens": [r["cached_tokens"] for r in valid],
                        "no_cache_hit_runs": sum(1 for r in valid if r.get("no_cache_hit"))})
    return summary


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def parse_isls(text: str) -> List[int]:
    values = [int(v) for v in text.split(",") if v.strip()]
    if not values or any(v < 1 for v in values) or len(set(values)) != len(values):
        raise ValueError("--isl takes distinct positive integers")
    return values


def cmd_run(args) -> int:
    try:
        args.isls = parse_isls(args.isl)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    if args.mode == "warm" and not all(0 < args.suffix_tokens < isl for isl in args.isls):
        print("ERROR: warm mode needs 0 < --suffix-tokens < every ISL")
        return 2
    if args.mode == "cold":
        args.suffix_tokens = 0
    if args.runs < 1 or args.warmup < 0:
        print("ERROR: --runs must be at least 1 and --warmup at least 0")
        return 2
    identity = None
    if args.identity:
        try:
            with open(args.identity, encoding="utf-8-sig") as f:
                identity = validate_identity(json.load(f))
        except (OSError, ValueError) as exc:
            print(f"IDENTITY_INVALID: {exc}")
            return 4
    else:
        print("WARNING: no --identity supplied; recording identity: null. Comparisons will refuse it by default.",
              file=sys.stderr)
    log_path = args.log or (args.out + ".log" if args.out else "prefill-ttft.log")
    plan = {"base_url": args.base_url, **{k: getattr(args, k) for k in PROFILE_FIELDS}, "warmup": args.warmup,
            "label": args.label, "out": args.out, "log": log_path, "identity": identity,
            "requests_per_run": 2 if args.mode == "warm" else 1,
            "request_url": args.base_url.rstrip("/") + "/v1/completions", "max_tokens": 1}
    if args.dry_run:
        print(json.dumps({"dry_run": True, **plan}, indent=2))
        return 0
    if httpx is None:
        print("ERROR: httpx is not installed (uv sync --extra engine)")
        return 2
    started = datetime.now(timezone.utc).isoformat()
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log:
            results = asyncio.run(measure(args, log))
    except OSError:
        print("ERROR: could not open/write the log or complete network I/O")
        return 3
    doc = {"schema": SCHEMA, "identity": identity, "label": args.label, "notes": args.notes,
           "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
           **{k: getattr(args, k) for k in PROFILE_FIELDS}, "warmup": args.warmup, "results": results}
    if args.out:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=1)
        except OSError:
            print("ERROR: could not write result JSON")
            return 3
        print(f"wrote {args.out}")
    return 0 if any(r["summary"]["valid_runs"] for r in results.values()) else 2


def load(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8-sig") as f:
            doc = json.load(f)
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            return None
        for key in ("label", *PROFILE_FIELDS):
            doc[key]
        for isl in doc["isls"]:
            summary = doc["results"][str(isl)]["summary"]
            if type(summary.get("valid_runs")) is not int or summary["valid_runs"] < 0:
                return None
            if summary["valid_runs"]:
                value = summary["ttft_s_p50"]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    return None
        return doc
    except (OSError, ValueError, KeyError, TypeError):
        return None


def check_pair(a: dict, b: dict, args) -> int:
    """Same gates as decode_step: the profile must match; identities must exist and differ only as declared."""
    diff = [k for k in PROFILE_FIELDS if a.get(k) != b.get(k)]
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
        return 0
    allowed = {key.strip() for key in args.arm_delta.split(",") if key.strip()}
    known = flatten_identity(identities[0]).keys() | flatten_identity(identities[1]).keys()
    if allowed - known:
        print("ARM_DELTA_UNKNOWN: " + ", ".join(sorted(allowed - known)))
        return 4
    changed = identity_differences(*identities)
    undeclared = [key for key in changed if key not in allowed]
    if undeclared:
        print("IDENTITY_MISMATCH: " + ", ".join(undeclared))
        return 4
    if changed:
        print("DECLARED_ARM_DELTA: " + ", ".join(changed))
    return 0


def cell(summary: dict) -> str:
    if not summary["valid_runs"]:
        return "absent"
    lo, hi = summary["ttft_s_spread"]
    return f"{summary['ttft_s_p50']:.3f} s [{lo:.3f}-{hi:.3f}]"


def cmd_compare(args) -> int:
    a, b = load(args.baseline), load(args.candidate)
    if a is None or b is None:
        print(f"MISSING_ARM: could not read {'baseline' if a is None else 'candidate'}")
        return 3
    code = check_pair(a, b, args)
    if code:
        return code
    if args.allow_missing_identity:
        print("UNQUALIFIED: --allow-missing-identity enabled")
    print(f"\n{a['model']}  mode {a['mode']}   A = {a['label']}   B = {b['label']}")
    print(f"{'ISL':>9}{'A TTFT p50 [min-max]':>28}{'B TTFT p50 [min-max]':>28}{'speedup':>9}{'B tok/s':>12}")
    for isl in a["isls"]:
        sa, sb = a["results"][str(isl)]["summary"], b["results"][str(isl)]["summary"]
        ratio = (f"{sa['ttft_s_p50'] / sb['ttft_s_p50']:.3f}x" if sa["valid_runs"] and sb["valid_runs"] else "absent")
        rate = f"{sb['prefill_tok_s_p50']:.0f}" if sb["valid_runs"] else "absent"
        print(f"{isl:>9}{cell(sa):>28}{cell(sb):>28}{ratio:>9}{rate:>12}")
    print("\nspeedup = A TTFT p50 / B TTFT p50. Run the baseline twice first: its A/A ratio is the noise floor.")
    return 0


def cmd_table(args) -> int:
    base = load(args.baseline)
    if base is None:
        print(f"MISSING_ARM: could not read baseline {args.baseline}")
        return 3
    rows = []
    for path in args.candidates:
        doc = load(path)
        if doc is None:
            print(f"MISSING_ARM: could not read {path}")
            return 3
        code = check_pair(base, doc, args)
        if code:
            return code
        rows.append(doc)
    if args.allow_missing_identity:
        print("UNQUALIFIED: --allow-missing-identity enabled")
    isls = base["isls"]
    print(f"**{base['model']}**, TTFT ({base['mode']}), baseline = `{base['label']}`. "
          "Cells: TTFT p50 s (speedup vs baseline).\n")
    print("| # | arm | " + " | ".join(f"ISL={i}" for i in isls) + " | notes |")
    print("|---|---|" + "---|" * len(isls) + "---|")
    for i, doc in enumerate([base] + rows):
        cells = []
        for isl in isls:
            s, sb = doc["results"][str(isl)]["summary"], base["results"][str(isl)]["summary"]
            cells.append(f"{s['ttft_s_p50']:.3f} ({sb['ttft_s_p50'] / s['ttft_s_p50']:.2f}x)"
                         if s["valid_runs"] and sb["valid_runs"] else "absent")
        note = "UNQUALIFIED identity; " if args.allow_missing_identity else ""
        print(f"| {i} | {doc['label']} | " + " | ".join(cells) + f" | {note}{doc.get('notes', '')} |")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Time to first token for long prompts, cold or with a cached prefix")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Measure a running server")
    r.add_argument("--base-url", required=True, help="Server root, e.g. http://127.0.0.1:8000")
    r.add_argument("--model", required=True, help="Served model name")
    r.add_argument("--isl", default="131072", help="Comma-separated prompt lengths in tokens")
    r.add_argument("--mode", choices=("cold", "warm"), default="cold")
    r.add_argument("--suffix-tokens", type=int, default=512, help="warm mode: fresh tokens after the cached prefix")
    r.add_argument("--runs", type=int, default=3, help="Timed runs per ISL")
    r.add_argument("--warmup", type=int, default=1, help="Discarded runs per ISL")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--token-id-range", default="1000,20000", help="Inclusive range for random prompt token IDs")
    r.add_argument("--api-key-env", default="", help="Env var holding a bearer token, if the server needs one")
    r.add_argument("--label", required=True, help="Arm name, e.g. stock or ours-exp12")
    r.add_argument("--notes", default="", help="Free text stored in the output (image digest, flags)")
    r.add_argument("--out", default="", help="Raw JSON output path")
    r.add_argument("--log", default="", help="Per-request log (default: <out>.log, or prefill-ttft.log)")
    r.add_argument("--identity", help="Local JSON identity file (same fields as decode_step.py)")
    r.add_argument("--dry-run", action="store_true", help="Print the resolved plan without network calls or writes")
    c = sub.add_parser("compare", help="Compare two run files (A = baseline, B = candidate)")
    c.add_argument("baseline")
    c.add_argument("candidate")
    t = sub.add_parser("table", help="Markdown progress table of candidate runs vs one baseline")
    t.add_argument("baseline")
    t.add_argument("candidates", nargs="+", help="Run files in experiment order")
    for command in (c, t):
        command.add_argument("--arm-delta", default="", help="Comma-separated identity leaf keys allowed to differ")
        command.add_argument("--allow-missing-identity", action="store_true", help="Allow missing identity; mark output UNQUALIFIED")
    args = parser.parse_args(argv)
    return {"run": cmd_run, "compare": cmd_compare, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
