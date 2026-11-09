"""
AutoKernel engine mode -- time to first token (TTFT) for long prompts on a running vLLM server.

Prefill as the client sees it. Each request is one streaming /v1/completions request with a prompt of exactly
ISL random token IDs, max_tokens = min_tokens = 1 (no decode tail), temperature 0, so its TTFT is one prefill
alone on the engine. Requests go one at a time: nothing else shares the engine while one is timed.

Modes:
  cold  every run sends one fresh random prompt, so the prefix cache cannot help (the default).
  warm  every run sends the same prefix twice, one request right after the other, and times both. The cold
        request is a fresh prefix of ISL - SUFFIX tokens plus a fresh SUFFIX-token tail; the warm request is
        the same prefix plus another fresh tail (--suffix-tokens 0: the same prompt again). Both are exactly
        ISL tokens, so one run reports cold and warm TTFT at the same length: the first question over a
        document, then a follow-up over the already-read document. It needs prefix caching on the server.
        When the server reports usage.prompt_tokens_details.cached_tokens (vLLM: --enable-prompt-tokens-details)
        it is checked: a cold request that reports cached tokens, or a warm one that reports none, is invalid.
        A warm request whose cold request failed is invalid too.

TTFT = arrival of the first output event - request start (client clock, perf_counter). Every request must end
with [DONE], one completion token and prompt_tokens equal to ISL (both from usage), or it is invalid. Prompts
are token IDs, so tokenization is not measured. Run the client on the server's host, and cross-check against
the server's histogram (vllm:time_to_first_token_seconds). The server's max_model_len must be at least ISL + 1.

Usage:
    uv run --extra engine engine/prefill_ttft.py run --base-url http://127.0.0.1:8000 --model <served-name> \\
        --isl 131072,262144,524288,1048575 --runs 3 --label stock --out results/ttft-stock.json
    uv run --extra engine engine/prefill_ttft.py run ... --mode warm --suffix-tokens 512 --label stock-warm
    uv run --extra engine engine/prefill_ttft.py compare results/ttft-stock.json results/ttft-exp01.json
    uv run --extra engine engine/prefill_ttft.py table results/ttft-stock-warm.json [results/ttft-exp*.json]

Exit codes: 0 ok, 2 usage error or no valid request at any ISL, 3 missing/unreadable file, 4 incompatible
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

SCHEMA = "autokernel-prefill-ttft.v2"
# Fields that must match between two runs for a comparison to mean anything
PROFILE_FIELDS = ("model", "isls", "mode", "suffix_tokens", "runs", "seed", "token_id_range")
# The timed requests of one run, in the order they are sent
LEGS = {"cold": ("cold",), "warm": ("cold", "warm")}
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


def judge_leg(leg: str, rec: dict, run: dict) -> dict:
    """Cache evidence on top of the stream checks. A cold request must not hit the prefix cache; a warm one
    needs its cold request to have succeeded and, when the server reports cached tokens, a hit."""
    errors = [rec["error"]] if rec["error"] else []
    cached = rec["cached_tokens"]
    if leg == "cold" and cached:
        errors.append(f"cold request reported {cached} cached tokens")
    if leg == "warm":
        if run["cold"]["error"]:
            errors.append("its cold request failed")
        if cached == 0:
            errors.append("warm request reported no cached tokens")
    rec["error"] = "; ".join(errors) or None
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
    legs = LEGS[args.mode]
    own = client is None
    if own:
        client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT_S))
    results = {}
    try:
        for isl in args.isls:
            runs = []
            for i in range(args.warmup + args.runs):
                timed = i >= args.warmup
                # Cold mode: suffix_tokens is 0, so the prefix is the whole fresh prompt.
                prefix = make_prompt(rng, isl - args.suffix_tokens, token_range)
                run = {"run": i, "timed": timed}
                for leg in legs:
                    prompt = prefix + make_prompt(rng, args.suffix_tokens, token_range)
                    rec = await stream_first_token(client, url, headers, args.model, prompt)
                    run[leg] = judge_leg(leg, rec, run)
                    print(json.dumps({"isl": isl, "run": i, "leg": leg, "timed": timed, **rec}), file=log, flush=True)
                if timed:
                    runs.append(run)
            results[str(isl)] = {"runs": runs, "summary": summarize(isl, runs, legs)}
            print(f"  ISL {isl}: {progress(results[str(isl)]['summary'], legs, len(runs))}", flush=True)
    finally:
        if own:
            await client.aclose()
    return results


def summarize_leg(isl: int, recs: List[dict]) -> dict:
    valid = [r for r in recs if not r["error"]]
    summary = {"valid_runs": len(valid), "invalid_runs": len(recs) - len(valid)}
    if valid:
        ttft = [r["ttft_s"] for r in valid]
        p50 = percentile(ttft, 50)
        summary.update({"ttft_s_p50": p50, "ttft_s_p90": percentile(ttft, 90), "ttft_s_spread": [min(ttft), max(ttft)],
                        "prefill_tok_s_p50": isl / p50, "cached_tokens": [r["cached_tokens"] for r in valid],
                        # Valid runs whose server reported no cached_tokens: their cache state is unverified
                        "unreported_cache_runs": sum(r["cached_tokens"] is None for r in valid)})
    return summary


def summarize(isl: int, runs: List[dict], legs: tuple) -> dict:
    summary = {leg: summarize_leg(isl, [run[leg] for run in runs]) for leg in legs}
    if "warm" in summary and summary["cold"]["valid_runs"] and summary["warm"]["valid_runs"]:
        summary["warm_speedup_p50"] = summary["cold"]["ttft_s_p50"] / summary["warm"]["ttft_s_p50"]
    return summary


def progress(summary: dict, legs: tuple, runs: int) -> str:
    parts = []
    for leg in legs:
        s = summary[leg]
        state = (f"p50 {s['ttft_s_p50']:.3f} s, {s['prefill_tok_s_p50']:.0f} tok/s" if s["valid_runs"]
                 else "no valid run")
        parts.append(f"{leg} {s['valid_runs']}/{runs} valid, {state}")
    if "warm_speedup_p50" in summary:
        parts.append(f"cold/warm {summary['warm_speedup_p50']:.2f}x")
    return "; ".join(parts)


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
    if args.mode == "warm" and not all(0 <= args.suffix_tokens < isl for isl in args.isls):
        print("ERROR: warm mode needs 0 <= --suffix-tokens < every ISL")
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
    legs = LEGS[args.mode]
    plan = {"base_url": args.base_url, **{k: getattr(args, k) for k in PROFILE_FIELDS}, "warmup": args.warmup,
            "label": args.label, "out": args.out, "log": log_path, "identity": identity,
            "legs": list(legs), "requests_per_run": len(legs),
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
           **{k: getattr(args, k) for k in PROFILE_FIELDS}, "warmup": args.warmup, "legs": list(legs),
           "results": results}
    if args.out:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=1)
        except OSError:
            print("ERROR: could not write result JSON")
            return 3
        print(f"wrote {args.out}")
    return 0 if any(r["summary"][leg]["valid_runs"] for r in results.values() for leg in legs) else 2


def _positive(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def load(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8-sig") as f:
            doc = json.load(f)
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or doc.get("mode") not in LEGS:
            return None
        for key in ("label", *PROFILE_FIELDS):
            doc[key]
        if not isinstance(doc["isls"], list) or not doc["isls"]:
            return None
        for isl in doc["isls"]:
            summary = doc["results"][str(isl)]["summary"]
            for leg in LEGS[doc["mode"]]:
                s = summary[leg]
                if type(s.get("valid_runs")) is not int or s["valid_runs"] < 0:
                    return None
                if s["valid_runs"]:
                    spread = s["ttft_s_spread"]
                    if (not _positive(s["ttft_s_p50"]) or not isinstance(spread, list) or len(spread) != 2
                            or not all(_positive(v) for v in spread)):
                        return None
        return doc
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
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


def leg_summary(doc: dict, isl: int, leg: str) -> dict:
    return doc["results"][str(isl)]["summary"][leg]


def cache_mark(leg: str, *summaries: dict) -> str:
    """Warm cells whose cache hit the server never reported are marked, not hidden."""
    if leg == "warm" and any(s.get("valid_runs") and s.get("unreported_cache_runs") for s in summaries):
        return " [cache hit unreported]"
    return ""


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
    legs = LEGS[a["mode"]]
    print(f"\n{a['model']}  mode {a['mode']}   A = {a['label']}   B = {b['label']}")
    print(f"{'ISL':>9}{'leg':>6}{'A TTFT p50 [min-max]':>28}{'B TTFT p50 [min-max]':>28}{'speedup':>9}{'B tok/s':>12}")
    for isl in a["isls"]:
        for leg in legs:
            sa, sb = leg_summary(a, isl, leg), leg_summary(b, isl, leg)
            ratio = (f"{sa['ttft_s_p50'] / sb['ttft_s_p50']:.3f}x" if sa["valid_runs"] and sb["valid_runs"] else "absent")
            rate = f"{sb['prefill_tok_s_p50']:.0f}" if sb["valid_runs"] else "absent"
            print(f"{isl:>9}{leg:>6}{cell(sa):>28}{cell(sb):>28}{ratio:>9}{rate:>12}{cache_mark(leg, sa, sb)}")
    if "warm" in legs:
        print("\ncold/warm TTFT p50 (prefix-cache speedup):")
        for isl in a["isls"]:
            sides = [f"{side} " + (f"{doc['results'][str(isl)]['summary']['warm_speedup_p50']:.2f}x"
                                   if "warm_speedup_p50" in doc["results"][str(isl)]["summary"] else "absent")
                     for side, doc in (("A", a), ("B", b))]
            print(f"  ISL {isl}: " + ", ".join(sides))
    print("\nspeedup = A TTFT p50 / B TTFT p50, per leg. Run the baseline twice first: its A/A ratio is the noise floor.")
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
    isls, legs = base["isls"], LEGS[base["mode"]]
    print(f"**{base['model']}**, TTFT ({base['mode']}), baseline = `{base['label']}`. "
          "Cells: TTFT p50 s (speedup vs baseline" + ("; warm rows: and vs that arm's cold TTFT" if "warm" in legs else "")
          + ").\n")
    print("| # | arm | leg | " + " | ".join(f"ISL={i}" for i in isls) + " | notes |")
    print("|---|---|---|" + "---|" * len(isls) + "---|")
    for i, doc in enumerate([base] + rows):
        for leg in legs:
            cells = []
            for isl in isls:
                s, sb = leg_summary(doc, isl, leg), leg_summary(base, isl, leg)
                if not (s["valid_runs"] and sb["valid_runs"]):
                    cells.append("absent")
                    continue
                vs_cold = doc["results"][str(isl)]["summary"].get("warm_speedup_p50") if leg == "warm" else None
                cells.append(f"{s['ttft_s_p50']:.3f} ({sb['ttft_s_p50'] / s['ttft_s_p50']:.2f}x"
                             + (f"; {vs_cold:.2f}x vs cold" if vs_cold else "") + ")" + cache_mark(leg, s))
            note = "UNQUALIFIED identity; " if args.allow_missing_identity else ""
            print(f"| {i} | {doc['label']} | {leg} | " + " | ".join(cells) + f" | {note}{doc.get('notes', '')} |")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Time to first token for long prompts, cold or with a cached prefix")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Measure a running server")
    r.add_argument("--base-url", required=True, help="Server root, e.g. http://127.0.0.1:8000")
    r.add_argument("--model", required=True, help="Served model name")
    r.add_argument("--isl", default="131072", help="Comma-separated prompt lengths in tokens")
    r.add_argument("--mode", choices=tuple(LEGS), default="cold")
    r.add_argument("--suffix-tokens", type=int, default=512,
                   help="warm mode: fresh tokens after the shared prefix in each request (0: the same prompt twice)")
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
    t = sub.add_parser("table", help="Markdown table: one run file's legs, or candidate runs vs one baseline")
    t.add_argument("baseline")
    t.add_argument("candidates", nargs="*", help="Run files in experiment order")
    for command in (c, t):
        command.add_argument("--arm-delta", default="", help="Comma-separated identity leaf keys allowed to differ")
        command.add_argument("--allow-missing-identity", action="store_true", help="Allow missing identity; mark output UNQUALIFIED")
    args = parser.parse_args(argv)
    return {"run": cmd_run, "compare": cmd_compare, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
