"""Resumable unattended execution for prepared optimization campaigns."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import (
    CampaignError,
    OptimizationCampaign,
    prepare_campaign,
    rank_targets,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _corpus_for(operation: str, repo_root: Path) -> str | None:
    candidate = repo_root / "models" / f"{operation}_corpus.json"
    if candidate.is_file():
        return str(candidate.relative_to(repo_root))
    return None


def _reset_runtime_files(workspace: Path) -> None:
    for name in (
        "orchestration_state.json",
        "aggregate_report.md",
        "overnight_agent.log",
        "agent_last_message.md",
        "morning_report.md",
    ):
        path = workspace / name
        if path.is_file():
            path.unlink()


def build_overnight_prompt(
    campaign: OptimizationCampaign,
    *,
    repo_root: Path,
    budget_hours: float,
) -> str:
    """Build the campaign-specific instructions layered over ``program.md``."""
    rows = []
    for rank, target in enumerate(rank_targets(campaign), start=1):
        corpus = _corpus_for(target.operation, repo_root)
        bench = f"uv run bench.py --spec {target.spec_locator}"
        if corpus is not None:
            bench += f" --shape-corpus {corpus}"
        rows.append(
            "\n".join(
                [
                    f"{rank}. {target.name}",
                    f"   operation: {target.operation}",
                    f"   candidate: workspace/kernel_{target.operation}_{rank}.py",
                    f"   spec: {target.spec_locator}",
                    f"   benchmark: {bench}",
                    (
                        f"   estimated model impact: "
                        f"{target.impact_pct(campaign.total_profiled_device_time_us):.2f}%"
                    ),
                ]
            )
        )
    targets = "\n\n".join(rows)
    return f"""\
You are running a prepared MotionKernel optimization campaign unattended.
Read program.md for the optimization playbook, correctness rules, experiment
discipline, crash recovery, and move-on criteria. This prompt overrides its
interactive Phase A: profiling, target selection, and approval are complete.
Do not ask the user questions.

Workload: {campaign.workload['workload_id']}
Model: {campaign.workload['model_id']}
Budget: {budget_hours:.2f} hours

Ranked targets:

{targets}

For each target in rank order:
1. Use `uv run orchestrate.py next` to confirm the active target.
2. Copy its prepared candidate to kernel.py.
3. Run the exact target-specific benchmark command shown above. Always retain
   its `--spec` and `--shape-corpus` arguments for every experiment.
4. Record the baseline and every kept/reverted experiment with orchestrate.py.
5. Never weaken correctness tolerances or edit references, specs, corpora,
   bench.py, verification code, or orchestration code.
6. Save the best passing implementation as
   `workspace/kernel_<operation>_<rank>_optimized.py` before moving on.
7. Continue until every target is done/plateaued or the budget is nearly
   exhausted. Leave enough time to run `uv run orchestrate.py report`.

Write a concise final summary to the agent output requested by the runner.
"""


def _default_codex_command(
    repo_root: Path,
    prompt: str,
    last_message: Path,
) -> list[str]:
    executable = shutil.which("codex")
    if executable is None:
        raise CampaignError(
            "Codex CLI was not found; install it or pass --agent-command"
        )
    return [
        executable,
        "exec",
        "-C",
        str(repo_root),
        "-s",
        "workspace-write",
        "--output-last-message",
        str(last_message),
        prompt,
    ]


def parse_agent_command(
    value: str,
    *,
    repo_root: Path,
    prompt_path: Path,
) -> list[str]:
    """Parse a shell-like command without invoking a shell."""
    command = shlex.split(value)
    if not command:
        raise CampaignError("--agent-command must not be empty")
    replacements = {
        "{repo}": str(repo_root),
        "{prompt_file}": str(prompt_path),
    }
    return [
        argument.replace("{repo}", replacements["{repo}"]).replace(
            "{prompt_file}", replacements["{prompt_file}"]
        )
        for argument in command
    ]


def _write_morning_report(
    workspace: Path,
    receipt: dict[str, Any],
    *,
    report_stdout: str,
) -> Path:
    aggregate = workspace / "aggregate_report.md"
    agent_summary = workspace / "agent_last_message.md"
    lines = [
        "# MotionKernel Overnight Campaign",
        "",
        f"Status: **{receipt['status']}**",
        f"Workload: `{receipt['workload_id']}`",
        f"Started: {receipt['started_at']}",
        f"Finished: {receipt['finished_at']}",
        f"Budget: {receipt['budget_hours']:.2f} hours",
        "",
        "## Targets",
        "",
    ]
    for target in receipt["targets"]:
        lines.append(
            f"- {target['rank']}. `{target['operation']}` "
            f"({target['name']})"
        )
    if aggregate.is_file():
        lines.extend(["", aggregate.read_text(encoding="utf-8")])
    elif report_stdout.strip():
        lines.extend(["", "## Orchestrator output", "", "```text"])
        lines.extend([report_stdout.strip(), "```"])
    if agent_summary.is_file() and agent_summary.stat().st_size:
        lines.extend(
            [
                "",
                "## Agent summary",
                "",
                agent_summary.read_text(encoding="utf-8").strip(),
            ]
        )
    path = workspace / "morning_report.md"
    _write_text_atomic(path, "\n".join(lines).rstrip() + "\n")
    return path


def _campaign_progress(workspace: Path) -> dict[str, Any]:
    state_path = workspace / "orchestration_state.json"
    if not state_path.is_file():
        return {
            "complete": False,
            "completed_targets": 0,
            "total_targets": 0,
        }
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        kernels = state["kernels"]
        complete = [
            kernel
            for kernel in kernels
            if kernel.get("status") in ("done", "skipped")
        ]
    except (json.JSONDecodeError, KeyError, TypeError):
        return {
            "complete": False,
            "completed_targets": 0,
            "total_targets": 0,
        }
    return {
        "complete": bool(kernels) and len(complete) == len(kernels),
        "completed_targets": len(complete),
        "total_targets": len(kernels),
    }


def run_campaign(
    campaign: OptimizationCampaign,
    *,
    repo_root: str | Path,
    budget_hours: float = 10.0,
    resume: bool = False,
    dry_run: bool = False,
    trust_specs: bool = False,
    agent_command: Sequence[str] | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Prepare and run a campaign, preserving logs and a terminal receipt."""
    root = Path(repo_root).resolve()
    workspace = root / "workspace"
    if not root.joinpath("program.md").is_file():
        raise CampaignError(f"MotionKernel repository not found at {root}")
    if budget_hours <= 0:
        raise CampaignError("budget_hours must be greater than zero")

    prepared_receipt = workspace / "campaign_receipt.json"
    fresh_run = not resume or not prepared_receipt.is_file()
    if fresh_run:
        _reset_runtime_files(workspace)
        prepare_campaign(
            campaign,
            workspace,
            trust_specs=trust_specs,
            spec_root=root,
        )

    prompt = build_overnight_prompt(
        campaign, repo_root=root, budget_hours=budget_hours
    )
    prompt_path = workspace / "overnight_prompt.md"
    _write_text_atomic(prompt_path, prompt)
    receipt_path = workspace / "overnight_receipt.json"
    last_message = workspace / "agent_last_message.md"
    log_path = workspace / "overnight_agent.log"

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "campaign_source": campaign.source,
        "workload_id": campaign.workload["workload_id"],
        "started_at": _utc_now(),
        "budget_hours": budget_hours,
        "resume": resume,
        "status": "prepared" if dry_run else "running",
        "prompt": str(prompt_path),
        "log": str(log_path),
        "targets": [
            {
                "rank": rank,
                "name": target.name,
                "operation": target.operation,
            }
            for rank, target in enumerate(rank_targets(campaign), start=1)
        ],
    }
    _write_json_atomic(receipt_path, receipt)
    if dry_run:
        receipt["finished_at"] = _utc_now()
        _write_json_atomic(receipt_path, receipt)
        return receipt

    command = (
        list(agent_command)
        if agent_command is not None
        else _default_codex_command(root, prompt, last_message)
    )
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else budget_hours * 60 * 60
    )
    env = os.environ.copy()
    env["AUTOKERNEL_CAMPAIGN"] = campaign.source
    env["AUTOKERNEL_BUDGET_HOURS"] = str(budget_hours)
    timed_out = False
    returncode: int | None = None
    launch_error: str | None = None
    with log_path.open("a", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            launch_error = str(exc)
            log.write(f"agent launch failed: {exc}\n")
        else:
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.terminate()
                try:
                    returncode = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = process.wait()

    if launch_error is not None:
        receipt.update(
            {
                "finished_at": _utc_now(),
                "status": "agent_launch_failed",
                "error": launch_error,
                "agent_returncode": None,
                "timed_out": False,
            }
        )
        morning_report = _write_morning_report(
            workspace, receipt, report_stdout=""
        )
        receipt["morning_report"] = str(morning_report)
        _write_json_atomic(receipt_path, receipt)
        return receipt

    report = subprocess.run(
        [sys.executable, "orchestrate.py", "report"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    progress = _campaign_progress(workspace)
    if timed_out:
        status = "budget_exhausted"
    elif returncode != 0:
        status = "agent_failed"
    elif progress["complete"]:
        status = "completed"
    else:
        status = "incomplete"
    receipt.update(
        {
            "finished_at": _utc_now(),
            "status": status,
            "agent_returncode": returncode,
            "timed_out": timed_out,
            "report_returncode": report.returncode,
            "aggregate_report": str(workspace / "aggregate_report.md"),
            "progress": progress,
        }
    )
    morning_report = _write_morning_report(
        workspace,
        receipt,
        report_stdout=report.stdout,
    )
    receipt["morning_report"] = str(morning_report)
    _write_json_atomic(receipt_path, receipt)
    return receipt
