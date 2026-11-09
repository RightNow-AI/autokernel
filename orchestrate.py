#!/usr/bin/env python3
"""
AutoKernel Multi-Kernel Orchestrator -- Schedule and track optimization across kernels.

Usage:
    uv run orchestrate.py status                    # show current optimization state
    uv run orchestrate.py next                      # print which kernel to optimize next
    uv run orchestrate.py record <kernel_file> <throughput_tflops> <status> <description>
    uv run orchestrate.py report                    # generate aggregate report
    uv run orchestrate.py plan                      # show the full optimization plan with estimated impact

Reads: workspace/optimization_plan.json, workspace/results/*.tsv
Writes: workspace/orchestration_state.json, workspace/aggregate_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = SCRIPT_DIR / "workspace"
PLAN_PATH = WORKSPACE / "optimization_plan.json"
STATE_PATH = WORKSPACE / "orchestration_state.json"
RESULTS_DIR = WORKSPACE / "results"
REPORT_PATH = WORKSPACE / "aggregate_report.md"

# ---------------------------------------------------------------------------
# Move-on criteria
# ---------------------------------------------------------------------------

MOVE_ON_CRITERIA = {
    "consecutive_reverts": 5,       # last N experiments all reverted
    "pct_peak_threshold": 90.0,     # achieved N% of theoretical GPU peak
    "max_minutes_per_kernel": 120,  # 2 hours max per kernel
    "speedup_threshold": 2.0,       # already 2x vs baseline
}

# ---------------------------------------------------------------------------
# Status labels
# ---------------------------------------------------------------------------

STATUS_PENDING = "pending"
STATUS_OPTIMIZING = "optimizing"
STATUS_DONE = "done"
STATUS_SKIPPED = "skipped"

_STATUS_DISPLAY = {
    STATUS_PENDING: "PENDING",
    STATUS_OPTIMIZING: "OPTIMIZING",
    STATUS_DONE: "DONE",
    STATUS_SKIPPED: "SKIPPED",
}

# ---------------------------------------------------------------------------
# Result TSV columns (matches analysis.py convention)
# ---------------------------------------------------------------------------

RESULT_TSV_COLUMNS = [
    "experiment", "tag", "kernel_type", "throughput_tflops", "latency_us",
    "pct_peak", "speedup_vs_pytorch", "correctness", "peak_vram_mb", "description",
]

RESULT_TSV_HEADER = "\t".join(RESULT_TSV_COLUMNS)

# ---------------------------------------------------------------------------
# Helpers -- filesystem
# ---------------------------------------------------------------------------


def _ensure_workspace() -> None:
    """Create workspace directories if they do not exist."""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Helpers -- plan loading
# ---------------------------------------------------------------------------


def load_plan() -> dict | None:
    """Load workspace/optimization_plan.json. Returns None if missing."""
    if not PLAN_PATH.exists():
        return None
    try:
        with open(PLAN_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: Failed to read {PLAN_PATH}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Helpers -- state management
# ---------------------------------------------------------------------------


def _default_kernel_entry(
    rank: int,
    file: str,
    op_type: str,
    pct_total: float = 0.0,
) -> dict:
    """Return a default kernel entry for the orchestration state."""
    return {
        "rank": rank,
        "file": file,
        "op_type": op_type,
        "pct_total": pct_total,
        "status": STATUS_PENDING,
        "baseline_tflops": None,
        "best_tflops": None,
        "speedup": None,
        "pct_peak": None,
        "experiments_run": 0,
        "experiments_kept": 0,
        "consecutive_reverts": 0,
        "time_spent_minutes": 0,
        "kernel_started_at": None,
    }


def _initialize_state_from_plan(plan: dict) -> dict:
    """Build a fresh orchestration state from an optimization plan."""
    kernels_raw = plan.get("kernels_to_optimize", plan.get("kernels", []))
    kernels = []
    for i, kp in enumerate(kernels_raw):
        kernels.append(
            _default_kernel_entry(
                rank=kp.get("rank", i + 1),
                file=kp.get("file", f"workspace/kernel_{kp.get('op_type', 'unknown')}_{i+1}.py"),
                op_type=kp.get("op_type", "unknown"),
                pct_total=kp.get("pct_total", 0.0),
            )
        )

    first_file = kernels[0]["file"] if kernels else None
    if kernels:
        kernels[0]["status"] = STATUS_OPTIMIZING
        kernels[0]["kernel_started_at"] = _now_iso()

    return {
        "current_kernel_idx": 0,
        "current_kernel_file": first_file,
        "started_at": _now_iso(),
        "kernels": kernels,
    }


def load_state() -> dict | None:
    """Load the orchestration state, or None if it does not exist."""
    if not STATE_PATH.exists():
        return None
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Minimal validation
        if not isinstance(data, dict) or "kernels" not in data:
            raise ValueError("State file missing 'kernels' key")
        return data
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        print(f"WARNING: Orchestration state corrupted ({exc}). Re-initializing.")
        return None


def save_state(state: dict) -> None:
    """Persist the orchestration state to disk."""
    _ensure_workspace()
    tmp = STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_PATH)


def get_or_create_state() -> dict:
    """Load existing state, or create one from the optimization plan."""
    state = load_state()
    if state is not None:
        return state

    plan = load_plan()
    if plan is None:
        print("ERROR: No optimization_plan.json found. Run extract.py first.")
        sys.exit(1)

    print("Initializing orchestration state from optimization_plan.json ...")
    state = _initialize_state_from_plan(plan)
    save_state(state)
    return state


# ---------------------------------------------------------------------------
# Helpers -- result TSV I/O
# ---------------------------------------------------------------------------


def _kernel_results_path(kernel_file: str) -> Path:
    """Derive the per-kernel results TSV path from a kernel file name."""
    name = Path(kernel_file).stem  # e.g. kernel_matmul_1
    return RESULTS_DIR / f"{name}_results.tsv"


def _append_result_row(kernel_file: str, row: dict) -> None:
    """Append a single result row to the per-kernel TSV."""
    _ensure_workspace()
    path = _kernel_results_path(kernel_file)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(RESULT_TSV_HEADER + "\n")
        values = [str(row.get(c, "")) for c in RESULT_TSV_COLUMNS]
        f.write("\t".join(values) + "\n")


def _load_result_rows(kernel_file: str) -> list[dict]:
    """Load all result rows for a kernel. Returns empty list if no file."""
    path = _kernel_results_path(kernel_file)
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
        if not header_line:
            return []
        cols = header_line.split("\t")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            row = {}
            for i, col in enumerate(cols):
                row[col] = parts[i] if i < len(parts) else ""
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Amdahl's law
# ---------------------------------------------------------------------------


def estimate_aggregate_speedup(kernels: list[dict]) -> float:
    """
    Amdahl's law: S = 1 / ((1 - p) + p / s)
    where p = fraction of total GPU time in optimized kernels,
          s = speedup of those kernels.

    We compute incrementally: the unoptimized fraction shrinks by the
    time saved in each kernel.

    Equivalent closed-form:
        remaining_frac = 1 - sum over optimized kernels of (frac_i * (1 - 1/s_i))
        S = 1 / remaining_frac
    """
    remaining_frac = 1.0
    for k in kernels:
        speedup = k.get("speedup")
        pct = k.get("pct_total", 0.0)
        if speedup is not None and speedup > 1.0 and pct > 0:
            frac = pct / 100.0
            remaining_frac -= frac * (1.0 - 1.0 / speedup)
    # Guard against degenerate cases
    if remaining_frac <= 0:
        return float("inf")
    return 1.0 / remaining_frac


def _hypothetical_speedup(kernels: list[dict], assumed_speedup: float, top_n: int) -> float:
    """What-if analysis: if we achieve *assumed_speedup* on the top-N kernels by pct_total."""
    sorted_k = sorted(kernels, key=lambda k: k.get("pct_total", 0), reverse=True)
    remaining_frac = 1.0
    for k in sorted_k[:top_n]:
        pct = k.get("pct_total", 0.0)
        # Use actual speedup if already achieved and better, else assumed
        actual = k.get("speedup")
        s = max(actual, assumed_speedup) if actual and actual > 1.0 else assumed_speedup
        frac = pct / 100.0
        remaining_frac -= frac * (1.0 - 1.0 / s)
    if remaining_frac <= 0:
        return float("inf")
    return 1.0 / remaining_frac


# ---------------------------------------------------------------------------
# Move-on logic
# ---------------------------------------------------------------------------


def _should_move_on(kernel: dict) -> tuple[bool, str]:
    """
    Evaluate move-on criteria for the current kernel.
    Returns (should_move, reason).
    """
    consec = kernel.get("consecutive_reverts", 0)
    if consec >= MOVE_ON_CRITERIA["consecutive_reverts"]:
        return True, (
            f"Plateau detected: {consec} consecutive reverts "
            f"(threshold: {MOVE_ON_CRITERIA['consecutive_reverts']})"
        )

    pct_peak = kernel.get("pct_peak")
    if pct_peak is not None and pct_peak >= MOVE_ON_CRITERIA["pct_peak_threshold"]:
        return True, (
            f"Near theoretical peak: {pct_peak:.1f}% of peak "
            f"(threshold: {MOVE_ON_CRITERIA['pct_peak_threshold']:.0f}%)"
        )

    minutes = kernel.get("time_spent_minutes", 0)
    if minutes >= MOVE_ON_CRITERIA["max_minutes_per_kernel"]:
        return True, (
            f"Time budget exhausted: {minutes:.0f} min "
            f"(max: {MOVE_ON_CRITERIA['max_minutes_per_kernel']} min)"
        )

    speedup = kernel.get("speedup")
    if speedup is not None and speedup >= MOVE_ON_CRITERIA["speedup_threshold"]:
        return True, (
            f"Strong speedup achieved: {speedup:.2f}x "
            f"(threshold: {MOVE_ON_CRITERIA['speedup_threshold']:.1f}x)"
        )

    return False, "Current kernel still has optimization headroom"


def _find_next_pending(kernels: list[dict], current_idx: int) -> int | None:
    """Return the index of the next pending kernel after current_idx, or None."""
    for i in range(current_idx + 1, len(kernels)):
        if kernels[i]["status"] == STATUS_PENDING:
            return i
    return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_status(state: dict) -> None:
    """Print the current orchestration status."""
    kernels = state["kernels"]
    idx = state.get("current_kernel_idx", 0)
    current = kernels[idx] if idx < len(kernels) else None

    print()
    print("=" * 55)
    print("  AutoKernel Orchestration Status")
    print("=" * 55)
    print()

    # Check if all done
    all_done = all(k["status"] in (STATUS_DONE, STATUS_SKIPPED) for k in kernels)
    if all_done:
        print("  All kernels optimized. Run verify.py for end-to-end check.")
        print()
    elif current:
        kname = Path(current["file"]).name
        print(f"  Currently optimizing: {kname} (rank {current['rank']}, {current['op_type']})")
        exp_run = current["experiments_run"]
        exp_kept = current["experiments_kept"]
        minutes = current["time_spent_minutes"]
        print(f"  Progress: {exp_run} experiments ({exp_kept} kept), {minutes} min elapsed")

        baseline = current["baseline_tflops"]
        best = current["best_tflops"]
        speedup = current["speedup"]
        if baseline is not None and best is not None and speedup is not None:
            print(f"  Baseline: {baseline:.1f} TFLOPS -> Current best: {best:.1f} TFLOPS ({speedup:.1f}x speedup)")
        elif baseline is not None:
            print(f"  Baseline: {baseline:.1f} TFLOPS (no improvement yet)")
        print()

    # Kernel table
    print("  Kernel Status:")
    max_op_len = max((len(k["op_type"]) for k in kernels), default=8)
    for k in kernels:
        tag = _STATUS_DISPLAY.get(k["status"], k["status"].upper())
        op = k["op_type"]
        rank = k["rank"]
        if k["status"] in (STATUS_DONE, STATUS_OPTIMIZING) and k["speedup"] is not None:
            detail = f"{k['speedup']:.1f}x speedup, {k['experiments_run']} experiments"
        elif k["status"] == STATUS_SKIPPED:
            detail = "skipped"
        else:
            detail = ""
        pad_op = op.ljust(max_op_len)
        if detail:
            print(f"    [{tag:<10}] {pad_op}  (rank {rank}) -> {detail}")
        else:
            print(f"    [{tag:<10}] {pad_op}  (rank {rank})")
    print()

    # Aggregate speedup
    agg = estimate_aggregate_speedup(kernels)
    if agg > 1.0:
        print(f"  Estimated aggregate model speedup: {agg:.2f}x")
    else:
        print("  Estimated aggregate model speedup: (no improvements yet)")
    print()


def cmd_next(state: dict) -> None:
    """Determine which kernel to optimize next and print the decision."""
    kernels = state["kernels"]
    idx = state.get("current_kernel_idx", 0)

    # All done?
    all_done = all(k["status"] in (STATUS_DONE, STATUS_SKIPPED) for k in kernels)
    if all_done:
        print("All kernels optimized. Run verify.py for end-to-end check.")
        return

    current = kernels[idx] if idx < len(kernels) else None
    if current is None or current["status"] in (STATUS_DONE, STATUS_SKIPPED):
        # Current is already finished; find next pending
        next_idx = _find_next_pending(kernels, -1)
        if next_idx is None:
            print("All kernels optimized. Run verify.py for end-to-end check.")
            return
        _transition_to(state, next_idx)
        save_state(state)
        _print_next_decision(state, kernels[next_idx], "Previous kernel already finished")
        return

    # Evaluate move-on criteria for the current kernel
    should_move, reason = _should_move_on(current)
    if should_move:
        current["status"] = STATUS_DONE
        next_idx = _find_next_pending(kernels, idx)
        if next_idx is None:
            print(f"Kernel {Path(current['file']).name} done ({reason}).")
            print("All kernels optimized. Run verify.py for end-to-end check.")
            save_state(state)
            return
        _transition_to(state, next_idx)
        save_state(state)
        _print_next_decision(state, kernels[next_idx], reason)
    else:
        # Continue current
        kname = Path(current["file"]).name
        print(f"DECISION: Continue optimizing {kname}")
        print(f"  Reason: {reason}")
        print(f"  Rank {current['rank']} | {current['op_type']} | "
              f"{current['experiments_run']} experiments | "
              f"speedup {current['speedup'] or 'N/A'}")


def _transition_to(state: dict, next_idx: int) -> None:
    """Move the orchestrator to a new kernel index."""
    kernels = state["kernels"]
    kernels[next_idx]["status"] = STATUS_OPTIMIZING
    kernels[next_idx]["kernel_started_at"] = _now_iso()
    state["current_kernel_idx"] = next_idx
    state["current_kernel_file"] = kernels[next_idx]["file"]


def _print_next_decision(state: dict, kernel: dict, reason: str) -> None:
    """Print the move-on decision."""
    kname = Path(kernel["file"]).name
    print(f"DECISION: Move to {kname} (rank {kernel['rank']}, {kernel['op_type']})")
    print(f"  Reason: {reason}")
    print(f"  File: {kernel['file']}")


def cmd_record(state: dict, kernel_file: str, throughput_tflops: float, status: str, description: str) -> None:
    """
    Record an experiment result for a kernel.

    *status* is one of: kept, revert, failed, crash, timeout
    """
    kernels = state["kernels"]

    # Find the kernel entry
    target = None
    for k in kernels:
        if k["file"] == kernel_file or Path(k["file"]).name == Path(kernel_file).name:
            target = k
            break

    if target is None:
        print(f"ERROR: Kernel '{kernel_file}' not found in orchestration state.")
        print("Known kernels:")
        for k in kernels:
            print(f"  {k['file']}")
        sys.exit(1)

    # Normalize status
    status_lower = status.strip().lower()
    is_kept = status_lower in ("kept", "keep", "improved")
    is_revert = status_lower in ("revert", "reverted", "slower", "same")
    is_failure = status_lower in ("failed", "fail", "crash", "error", "timeout")

    # Update experiment counts
    target["experiments_run"] += 1

    if is_kept:
        target["experiments_kept"] += 1
        target["consecutive_reverts"] = 0
        # Update best if improved
        if target["best_tflops"] is None or throughput_tflops > target["best_tflops"]:
            target["best_tflops"] = throughput_tflops
        # Set baseline on first kept result if not already set
        if target["baseline_tflops"] is None:
            target["baseline_tflops"] = throughput_tflops
    elif is_revert:
        target["consecutive_reverts"] += 1
        # First experiment sets the baseline even on revert
        if target["baseline_tflops"] is None:
            target["baseline_tflops"] = throughput_tflops
        if target["best_tflops"] is None:
            target["best_tflops"] = throughput_tflops
    elif is_failure:
        target["consecutive_reverts"] += 1
    else:
        # Unknown status -- treat as revert
        print(f"WARNING: Unrecognized status '{status}', treating as revert.")
        target["consecutive_reverts"] += 1

    # Compute speedup
    if target["baseline_tflops"] and target["best_tflops"] and target["baseline_tflops"] > 0:
        target["speedup"] = round(target["best_tflops"] / target["baseline_tflops"], 3)

    # Update time_spent_minutes from kernel_started_at (per-kernel, not global)
    k_started = target.get("kernel_started_at")
    if k_started:
        try:
            # Accept both timezone-aware and naive timestamps
            start_dt = datetime.fromisoformat(k_started)
            now_dt = datetime.now(timezone.utc)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            delta = now_dt - start_dt
            target["time_spent_minutes"] = round(delta.total_seconds() / 60.0)
        except (ValueError, TypeError):
            pass

    # Append to per-kernel results TSV
    tag_label = "kept" if is_kept else ("revert" if is_revert else "failed")
    correctness = "PASS" if not is_failure else "FAIL"
    row = {
        "experiment": target["experiments_run"],
        "tag": tag_label,
        "kernel_type": target["op_type"],
        "throughput_tflops": f"{throughput_tflops:.4f}" if throughput_tflops else "0",
        "latency_us": "",
        "pct_peak": "",
        "speedup_vs_pytorch": f"{target['speedup']:.3f}" if target["speedup"] else "",
        "correctness": correctness,
        "peak_vram_mb": "",
        "description": description,
    }
    _append_result_row(kernel_file, row)

    save_state(state)

    # Summary
    kname = Path(target["file"]).name
    print(f"Recorded: {kname} exp #{target['experiments_run']} -> {tag_label} ({throughput_tflops:.2f} TFLOPS)")
    if target["speedup"]:
        print(f"  Speedup: {target['speedup']:.2f}x | Best: {target['best_tflops']:.2f} TFLOPS")
    if target["consecutive_reverts"] > 0 and not is_kept:
        print(f"  Consecutive reverts: {target['consecutive_reverts']}"
              f" / {MOVE_ON_CRITERIA['consecutive_reverts']} until move-on")


def cmd_report(state: dict) -> None:
    """Generate the aggregate report at workspace/aggregate_report.md."""
    _ensure_workspace()
    kernels = state["kernels"]
    plan = load_plan()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []
    lines.append("# AutoKernel -- Aggregate Optimization Report")
    lines.append("")
    lines.append(f"Generated: {timestamp}")
    lines.append("")

    # ---- Per-kernel summary table ----
    lines.append("## Per-Kernel Summary")
    lines.append("")
    lines.append("| Rank | Kernel | Op Type | Status | Baseline (TFLOPS) | Best (TFLOPS) | Speedup | Experiments | Kept | Keep Rate | Time (min) |")
    lines.append("|------|--------|---------|--------|-------------------|---------------|---------|-------------|------|-----------|------------|")

    for k in kernels:
        kname = Path(k["file"]).name
        baseline_str = f"{k['baseline_tflops']:.2f}" if k["baseline_tflops"] is not None else "--"
        best_str = f"{k['best_tflops']:.2f}" if k["best_tflops"] is not None else "--"
        speedup_str = f"{k['speedup']:.2f}x" if k["speedup"] is not None else "--"
        exp_run = k["experiments_run"]
        exp_kept = k["experiments_kept"]
        keep_rate = f"{exp_kept / exp_run * 100:.0f}%" if exp_run > 0 else "--"
        minutes = k["time_spent_minutes"]
        status = _STATUS_DISPLAY.get(k["status"], k["status"])
        lines.append(
            f"| {k['rank']} | {kname} | {k['op_type']} | {status} | {baseline_str} | "
            f"{best_str} | {speedup_str} | {exp_run} | {exp_kept} | {keep_rate} | {minutes} |"
        )
    lines.append("")

    # ---- Aggregate speedup ----
    agg = estimate_aggregate_speedup(kernels)
    lines.append("## Aggregate Model Speedup (Amdahl's Law)")
    lines.append("")
    if agg > 1.0:
        lines.append(f"**Estimated end-to-end model speedup: {agg:.2f}x**")
    else:
        lines.append("No measurable aggregate speedup yet.")
    lines.append("")

    # Breakdown
    lines.append("Breakdown by kernel (fraction of total GPU time):")
    lines.append("")
    for k in kernels:
        pct = k.get("pct_total", 0)
        speedup = k.get("speedup")
        kname = Path(k["file"]).name
        if pct > 0:
            speedup_str = f"{speedup:.2f}x" if speedup and speedup > 1.0 else "1.00x"
            saved = pct * (1 - 1 / speedup) if speedup and speedup > 1.0 else 0.0
            lines.append(f"- **{kname}**: {pct:.1f}% of GPU time, {speedup_str} speedup ({saved:.1f}% time saved)")
    lines.append("")

    # ---- Time allocation ----
    lines.append("## Time Allocation")
    lines.append("")
    total_minutes = sum(k["time_spent_minutes"] for k in kernels)
    if total_minutes > 0:
        lines.append(f"Total optimization time: {total_minutes} minutes ({total_minutes / 60:.1f} hours)")
        lines.append("")
        for k in kernels:
            m = k["time_spent_minutes"]
            pct = m / total_minutes * 100 if total_minutes > 0 else 0
            kname = Path(k["file"]).name
            lines.append(f"- {kname}: {m} min ({pct:.0f}%)")
    else:
        lines.append("No time tracked yet.")
    lines.append("")

    # ---- Keep rates ----
    lines.append("## Keep Rates")
    lines.append("")
    for k in kernels:
        if k["experiments_run"] > 0:
            rate = k["experiments_kept"] / k["experiments_run"] * 100
            kname = Path(k["file"]).name
            lines.append(f"- {kname}: {k['experiments_kept']}/{k['experiments_run']} ({rate:.0f}%)")
    lines.append("")

    # ---- Headroom analysis ----
    lines.append("## Headroom Analysis")
    lines.append("")
    lines.append("Kernels that may still have optimization potential:")
    lines.append("")
    has_headroom = False
    for k in kernels:
        reasons: list[str] = []
        speedup = k.get("speedup")
        pct_peak = k.get("pct_peak")
        pct_total = k.get("pct_total", 0)

        if k["status"] == STATUS_PENDING:
            reasons.append("not yet optimized")
        elif k["status"] in (STATUS_OPTIMIZING, STATUS_DONE):
            if speedup is not None and speedup < MOVE_ON_CRITERIA["speedup_threshold"]:
                reasons.append(f"speedup only {speedup:.2f}x (target: {MOVE_ON_CRITERIA['speedup_threshold']:.1f}x)")
            if pct_peak is not None and pct_peak < MOVE_ON_CRITERIA["pct_peak_threshold"]:
                reasons.append(f"only {pct_peak:.1f}% of peak (headroom to {MOVE_ON_CRITERIA['pct_peak_threshold']:.0f}%)")
            if pct_total >= 10 and (speedup is None or speedup < 1.5):
                reasons.append(f"high impact ({pct_total:.1f}% of GPU time) with low speedup")

        if reasons:
            has_headroom = True
            kname = Path(k["file"]).name
            lines.append(f"- **{kname}** (rank {k['rank']}): {'; '.join(reasons)}")

    if not has_headroom:
        lines.append("- All kernels appear well-optimized or have been addressed.")
    lines.append("")

    # Write
    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Aggregate report written to {REPORT_PATH}")
    print()

    # Also print a terminal summary
    print("=" * 55)
    print("  Aggregate Report Summary")
    print("=" * 55)
    print()
    total_exp = sum(k["experiments_run"] for k in kernels)
    total_kept = sum(k["experiments_kept"] for k in kernels)
    done_count = sum(1 for k in kernels if k["status"] in (STATUS_DONE, STATUS_SKIPPED))
    print(f"  Kernels: {done_count}/{len(kernels)} completed")
    print(f"  Total experiments: {total_exp} ({total_kept} kept)")
    print(f"  Total time: {total_minutes} min")
    if agg > 1.0:
        print(f"  Aggregate speedup: {agg:.2f}x")
    print()


def cmd_plan(state: dict) -> None:
    """Show the full optimization plan with Amdahl's law analysis."""
    plan = load_plan()
    if plan is None:
        print("ERROR: No optimization_plan.json found. Run extract.py first.")
        sys.exit(1)

    kernels_plan = plan.get("kernels_to_optimize", plan.get("kernels", []))
    kernels_state = state["kernels"]

    # Build a lookup from file -> state entry
    state_by_file: dict[str, dict] = {}
    for k in kernels_state:
        state_by_file[k["file"]] = k
        state_by_file[Path(k["file"]).name] = k

    print()
    print("=" * 65)
    print("  AutoKernel -- Optimization Plan")
    print("=" * 65)
    print()

    # Plan table
    total_gpu_time = plan.get("total_gpu_time_ms", 0)
    if total_gpu_time > 0:
        print(f"  Total profiled GPU time: {total_gpu_time:.1f} ms")
        print()

    print(f"  {'Rank':<5} {'Op Type':<20} {'Shape':<30} {'GPU Time (ms)':<15} {'% Total':<10} {'Status':<12} {'Speedup':<10}")
    print(f"  {'-'*5} {'-'*20} {'-'*30} {'-'*15} {'-'*10} {'-'*12} {'-'*10}")

    for kp in kernels_plan:
        rank = kp.get("rank", "?")
        op_type = kp.get("op_type", "unknown")
        shape = kp.get("shape", "")
        if isinstance(shape, dict):
            shape = ", ".join(f"{k}={v}" for k, v in shape.items())
        elif isinstance(shape, list):
            shape = str(shape)
        gpu_time = kp.get("gpu_time_ms", 0)
        pct_total = kp.get("pct_total", 0)

        # Match to state
        file_key = kp.get("file", "")
        sk = state_by_file.get(file_key) or state_by_file.get(Path(file_key).name) if file_key else None

        status = sk["status"].upper() if sk else "UNKNOWN"
        speedup_str = f"{sk['speedup']:.2f}x" if sk and sk.get("speedup") else "--"

        # Truncate shape for display
        shape_disp = shape[:28] + ".." if len(str(shape)) > 30 else str(shape)

        print(f"  {rank:<5} {op_type:<20} {shape_disp:<30} {gpu_time:<15.2f} {pct_total:<10.1f} {status:<12} {speedup_str:<10}")

    print()

    # Amdahl's law what-if analysis
    print("  Amdahl's Law What-If Analysis:")
    print("  " + "-" * 50)

    for s in [1.5, 2.0, 3.0, 5.0]:
        for n in [1, 3, 5, min(len(kernels_state), 10)]:
            if n > len(kernels_state):
                continue
            projected = _hypothetical_speedup(kernels_state, s, n)
            print(f"    If top-{n} kernels achieve {s:.1f}x -> model speedup: {projected:.2f}x")
        print()

    # Current actual
    actual = estimate_aggregate_speedup(kernels_state)
    if actual > 1.0:
        print(f"  Current actual aggregate speedup: {actual:.2f}x")
    else:
        print("  Current actual aggregate speedup: (none yet)")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchestrate",
        description="AutoKernel Multi-Kernel Orchestrator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show current optimization state")
    sub.add_parser("next", help="Determine which kernel to optimize next")

    rec = sub.add_parser("record", help="Record an experiment result")
    rec.add_argument("kernel_file", help="Kernel file path (e.g. workspace/kernel_matmul_1.py)")
    rec.add_argument("throughput_tflops", type=float, help="Throughput in TFLOPS")
    rec.add_argument("status", help="Experiment status: kept | revert | failed | crash | timeout")
    rec.add_argument("description", help="Brief description of the experiment")

    sub.add_parser("report", help="Generate aggregate optimization report")
    sub.add_parser("plan", help="Show optimization plan with Amdahl's law analysis")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    state = get_or_create_state()

    if args.command == "status":
        cmd_status(state)
    elif args.command == "next":
        cmd_next(state)
    elif args.command == "record":
        cmd_record(state, args.kernel_file, args.throughput_tflops, args.status, args.description)
    elif args.command == "report":
        cmd_report(state)
    elif args.command == "plan":
        cmd_plan(state)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

