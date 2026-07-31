"""MotionKernel-side bridge that drives a FastVideo generation launcher.

The FastVideo checkout owns the GPU process. This module validates workloads,
invokes the shared launcher script in separate processes per mode, and records
resume-friendly stage state under an output directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .result import (
    GenerationRunResult,
    classify_end_to_end,
    compare_frame_outputs,
    load_generation_result,
)
from .types import WorkloadError, WorkloadManifest, load_workload

DEFAULT_LAUNCHER_RELATIVE = Path(
    "examples/inference/optimizations/generation_launcher.py"
)
STATE_NAME = "launcher_state.json"
STAGES = ("native", "optimized", "compare")


@dataclass(frozen=True)
class LauncherPaths:
    output_dir: Path
    state_path: Path
    native_result: Path
    optimized_result: Path
    comparison_path: Path


def _paths(output_dir: str | Path) -> LauncherPaths:
    root = Path(output_dir).expanduser().resolve()
    return LauncherPaths(
        output_dir=root,
        state_path=root / STATE_NAME,
        native_result=root / "native_result.json",
        optimized_result=root / "optimized_result.json",
        comparison_path=root / "comparison.json",
    )


def _read_state(path: Path) -> dict[str, Any]:
    fresh: dict[str, Any] = {
        "schema_version": 1,
        "completed_stages": [],
        "failed_stages": {},
    }
    if not path.is_file():
        return fresh
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkloadError(
            f"launcher state {path!s}: invalid JSON: {exc}"
        ) from exc
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise WorkloadError(
            f"launcher state {path!s}: unsupported or malformed state; "
            "delete the file or run with resume disabled"
        )
    merged = {**fresh, **state}
    if not isinstance(merged.get("completed_stages"), list):
        raise WorkloadError(
            f"launcher state {path!s}: completed_stages must be a list"
        )
    if not isinstance(merged.get("failed_stages"), dict):
        raise WorkloadError(
            f"launcher state {path!s}: failed_stages must be an object"
        )
    return merged


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_launcher(
    fastvideo_checkout: str | Path,
    launcher_script: str | Path | None = None,
) -> Path:
    """Locate the FastVideo generation launcher script."""
    root = Path(fastvideo_checkout)
    if not root.is_dir():
        raise WorkloadError(
            f"FastVideo checkout not found: {root}"
        )
    script = (
        Path(launcher_script)
        if launcher_script is not None
        else root / DEFAULT_LAUNCHER_RELATIVE
    )
    if not script.is_file():
        raise WorkloadError(
            f"generation launcher not found: {script}. "
            "Install/update the FastVideo branch that provides "
            "examples/inference/optimizations/generation_launcher.py"
        )
    return script


def build_launcher_command(
    *,
    python: str,
    launcher: Path,
    workload: Path,
    mode: str,
    output_dir: Path,
    model_override: str | None = None,
) -> list[str]:
    """Construct an argv list for one launcher process (no shell)."""
    command = [
        python,
        str(launcher),
        "--workload",
        str(workload),
        "--mode",
        mode,
        "--output-dir",
        str(output_dir),
    ]
    if model_override:
        command.extend(["--model", model_override])
    return command


def run_mode(
    *,
    fastvideo_checkout: str | Path,
    workload: str | Path | WorkloadManifest,
    mode: str,
    output_dir: str | Path,
    python: str | None = None,
    launcher_script: str | Path | None = None,
    model_override: str | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one baseline or optimized generation mode in a subprocess."""
    if isinstance(workload, WorkloadManifest):
        raise WorkloadError(
            "run_mode requires a workload file path so the child process "
            "can load the same manifest"
        )
    workload_path = Path(workload)
    if not workload_path.is_file():
        raise WorkloadError(f"workload file not found: {workload_path}")

    # Validate early on the parent side.
    load_workload(workload_path)

    launcher = resolve_launcher(fastvideo_checkout, launcher_script)
    command = build_launcher_command(
        python=python or sys.executable,
        launcher=launcher,
        workload=workload_path,
        mode=mode,
        output_dir=Path(output_dir),
        model_override=model_override,
    )
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    # Ensure the FastVideo checkout is importable even when the user has not
    # installed it into the active virtualenv.
    checkout = str(Path(fastvideo_checkout).resolve())
    existing = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (
        checkout if not existing else f"{checkout}{os.pathsep}{existing}"
    )

    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env=child_env,
        cwd=checkout,
    )
    if check and completed.returncode != 0:
        raise WorkloadError(
            f"launcher mode {mode!r} failed with exit {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def run_ab(
    *,
    fastvideo_checkout: str | Path,
    workload: str | Path,
    output_dir: str | Path,
    python: str | None = None,
    launcher_script: str | Path | None = None,
    model_override: str | None = None,
    modes: Sequence[str] = ("native", "optimized"),
    resume: bool = True,
) -> dict[str, Any]:
    """Run native and optimized modes with resume-friendly stage tracking."""
    paths = _paths(output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    allowed_modes = {"native", "optimized"}
    unknown = [mode for mode in modes if mode not in allowed_modes]
    if unknown:
        raise WorkloadError(
            f"unsupported launcher mode(s) {sorted(unknown)}; "
            "expected 'native' and/or 'optimized'"
        )
    manifest = load_workload(workload)
    state = _read_state(paths.state_path) if resume else {
        "schema_version": 1,
        "completed_stages": [],
        "failed_stages": {},
    }
    completed = set(state.get("completed_stages", []))

    results: dict[str, GenerationRunResult] = {}
    for mode in modes:
        result_path = (
            paths.native_result if mode == "native" else paths.optimized_result
        )
        # Accept legacy fused naming for Wan parity.
        if mode == "optimized" and not result_path.is_file():
            legacy = paths.output_dir / "fused_result.json"
            if legacy.is_file():
                result_path = legacy

        if resume and mode in completed and result_path.is_file():
            results[mode] = load_generation_result(result_path)
            continue

        mode_env = {}
        if manifest.mode_env is not None:
            mode_env = manifest.mode_env.for_mode(mode)
        try:
            run_mode(
                fastvideo_checkout=fastvideo_checkout,
                workload=workload,
                mode=mode,
                output_dir=paths.output_dir,
                python=python,
                launcher_script=launcher_script,
                model_override=model_override,
                env=mode_env or None,
                check=True,
            )
            # Launcher may write mode-specific names.
            written = paths.output_dir / f"{mode}_result.json"
            if not written.is_file() and mode == "optimized":
                written = paths.output_dir / "fused_result.json"
            if not written.is_file():
                raise WorkloadError(
                    f"launcher did not write expected result: {written}"
                )
            results[mode] = load_generation_result(written)
            completed.add(mode)
            state["completed_stages"] = sorted(completed)
            state.get("failed_stages", {}).pop(mode, None)
            _write_state(paths.state_path, state)
        except Exception as exc:  # noqa: BLE001 - record and re-raise
            state.setdefault("failed_stages", {})[mode] = str(exc)
            _write_state(paths.state_path, state)
            raise

    comparison = None
    if "native" in results and "optimized" in results:
        if not (resume and "compare" in completed and paths.comparison_path.is_file()):
            comparison = classify_end_to_end(
                results["native"],
                results["optimized"],
                min_speedup=manifest.performance.min_end_to_end_speedup
                if manifest.performance
                else 1.01,
                max_peak_memory_regression=(
                    manifest.performance.max_peak_memory_regression
                    if manifest.performance
                    else 0.05
                ),
            )
            parity_policy = manifest.parity
            parity = compare_frame_outputs(
                results["native"].frames_path,
                results["optimized"].frames_path,
                policy=parity_policy.policy if parity_policy else "byte_equal",
                atol=(
                    parity_policy.atol
                    if parity_policy and parity_policy.atol is not None
                    else 0.0
                ),
                rtol=(
                    parity_policy.rtol
                    if parity_policy and parity_policy.rtol is not None
                    else 0.0
                ),
            )
            comparison["parity"] = parity
            if not parity["passed"]:
                comparison["classification"] = "failed"
                comparison["reason"] = (
                    f"output parity failed: {parity['reason']}"
                )
            temporary = paths.comparison_path.with_suffix(
                paths.comparison_path.suffix + ".tmp"
            )
            temporary.write_text(
                json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
            )
            temporary.replace(paths.comparison_path)
            completed.add("compare")
            state["completed_stages"] = sorted(completed)
            _write_state(paths.state_path, state)
        else:
            comparison = json.loads(
                paths.comparison_path.read_text(encoding="utf-8")
            )

    return {
        "workload_id": manifest.workload_id,
        "output_dir": str(paths.output_dir),
        "results": {mode: result.as_dict() for mode, result in results.items()},
        "comparison": comparison,
        "state": state,
    }
