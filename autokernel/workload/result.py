"""Structured generation-run result schema written by FastVideo launchers.

MotionKernel validates these artifacts before ranking native-versus-optimized
end-to-end outcomes. Values are metadata and numeric measurements only; frame
tensors live in separate ``.npy`` files referenced by path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .types import WorkloadError, _finite_number, _mapping, _text

RESULT_SCHEMA_VERSION = 1

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "mode",
    "status",
    "workload_id",
    "model_id",
    "request",
    "warmups",
    "runs",
    "wall_seconds",
    "median_wall_seconds",
    "generation_seconds",
    "peak_memory_mb",
    "environment",
    "frames_path",
    "log_path",
    "failure_reason",
    "stage",
}

_STATUSES = {"ok", "failed", "skipped"}
_MODES = {"native", "optimized", "fused", "candidate"}


def _optional_text(
    value: Any, source: object, location: str
) -> str | None:
    if value is None:
        return None
    return _text(value, source, location)


def _non_negative_int(value: Any, source: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkloadError(
            f"generation result {source!r}: {location}: "
            "must be a non-negative integer"
        )
    return value


def _positive_int(value: Any, source: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkloadError(
            f"generation result {source!r}: {location}: "
            "must be a positive integer"
        )
    return value


def _number_list(
    value: Any, source: object, location: str
) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise WorkloadError(
            f"generation result {source!r}: {location}: must be a list"
        )
    numbers: list[float] = []
    for index, item in enumerate(value):
        if item is None:
            continue
        numbers.append(
            _finite_number(
                item,
                source,
                f"{location}[{index}]",
                minimum=0.0,
            )
        )
    return numbers


def _optional_number_list(
    value: Any, source: object, location: str
) -> list[float | None]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise WorkloadError(
            f"generation result {source!r}: {location}: must be a list"
        )
    result: list[float | None] = []
    for index, item in enumerate(value):
        if item is None:
            result.append(None)
            continue
        result.append(
            _finite_number(
                item,
                source,
                f"{location}[{index}]",
                minimum=0.0,
            )
        )
    return result


@dataclass(frozen=True)
class GenerationRunResult:
    """One mode's measured generation run (native or optimized)."""

    mode: str
    status: str
    workload_id: str
    model_id: str
    request: Mapping[str, Any]
    warmups: int
    runs: int
    wall_seconds: tuple[float, ...]
    median_wall_seconds: float | None
    generation_seconds: tuple[float | None, ...]
    peak_memory_mb: tuple[float | None, ...]
    environment: Mapping[str, Any]
    frames_path: str | None = None
    log_path: str | None = None
    failure_reason: str | None = None
    stage: str = "generate"
    schema_version: int = RESULT_SCHEMA_VERSION
    source: str = "<memory>"

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object = "<memory>"
    ) -> "GenerationRunResult":
        raw = _mapping(raw_value, source, "top level", non_empty=True)
        unknown = sorted(set(raw) - _TOP_LEVEL_FIELDS)
        if unknown:
            raise WorkloadError(
                f"generation result {source!r}: top level: "
                f"unknown field(s) {unknown}"
            )

        version = raw.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise WorkloadError(
                f"generation result {source!r}: schema_version: "
                "must be an integer"
            )
        if version != RESULT_SCHEMA_VERSION:
            raise WorkloadError(
                f"generation result {source!r}: schema_version: "
                f"unsupported version {version}; expected {RESULT_SCHEMA_VERSION}"
            )

        mode = _text(raw.get("mode"), source, "mode")
        if mode not in _MODES:
            raise WorkloadError(
                f"generation result {source!r}: mode: "
                f"must be one of {sorted(_MODES)}"
            )
        status = _text(raw.get("status", "ok"), source, "status")
        if status not in _STATUSES:
            raise WorkloadError(
                f"generation result {source!r}: status: "
                f"must be one of {sorted(_STATUSES)}"
            )

        wall = tuple(_number_list(raw.get("wall_seconds", []), source, "wall_seconds"))
        median = raw.get("median_wall_seconds")
        median_value = (
            None
            if median is None
            else _finite_number(
                median, source, "median_wall_seconds", minimum=0.0
            )
        )
        if status == "ok" and not wall:
            raise WorkloadError(
                f"generation result {source!r}: wall_seconds: "
                "required for successful runs"
            )

        return cls(
            mode=mode,
            status=status,
            workload_id=_text(raw.get("workload_id"), source, "workload_id"),
            model_id=_text(raw.get("model_id"), source, "model_id"),
            request=dict(
                _mapping(raw.get("request", {}), source, "request")
            ),
            warmups=_non_negative_int(raw.get("warmups", 0), source, "warmups"),
            runs=_positive_int(raw.get("runs", 1), source, "runs")
            if status == "ok"
            else _non_negative_int(raw.get("runs", 0), source, "runs"),
            wall_seconds=wall,
            median_wall_seconds=median_value,
            generation_seconds=tuple(
                _optional_number_list(
                    raw.get("generation_seconds", []),
                    source,
                    "generation_seconds",
                )
            ),
            peak_memory_mb=tuple(
                _optional_number_list(
                    raw.get("peak_memory_mb", []),
                    source,
                    "peak_memory_mb",
                )
            ),
            environment=dict(
                _mapping(raw.get("environment", {}), source, "environment")
            ),
            frames_path=_optional_text(
                raw.get("frames_path"), source, "frames_path"
            ),
            log_path=_optional_text(raw.get("log_path"), source, "log_path"),
            failure_reason=_optional_text(
                raw.get("failure_reason"), source, "failure_reason"
            ),
            stage=_text(raw.get("stage", "generate"), source, "stage"),
            schema_version=version,
            source=str(source),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "status": self.status,
            "workload_id": self.workload_id,
            "model_id": self.model_id,
            "request": dict(self.request),
            "warmups": self.warmups,
            "runs": self.runs,
            "wall_seconds": list(self.wall_seconds),
            "median_wall_seconds": self.median_wall_seconds,
            "generation_seconds": list(self.generation_seconds),
            "peak_memory_mb": list(self.peak_memory_mb),
            "environment": dict(self.environment),
            "stage": self.stage,
        }
        if self.frames_path is not None:
            payload["frames_path"] = self.frames_path
        if self.log_path is not None:
            payload["log_path"] = self.log_path
        if self.failure_reason is not None:
            payload["failure_reason"] = self.failure_reason
        return payload


def load_generation_result(path: str | Path) -> GenerationRunResult:
    """Load and validate a launcher result JSON file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise WorkloadError(f"generation result {file_path!s}: file: not found")
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkloadError(
            f"generation result {file_path!s}: JSON: invalid JSON: {exc}"
        ) from exc
    return GenerationRunResult.from_dict(raw, source=str(file_path))


def write_generation_result(
    result: GenerationRunResult, path: str | Path
) -> None:
    """Atomically write a generation result JSON file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)


def classify_end_to_end(
    native: GenerationRunResult,
    optimized: GenerationRunResult,
    *,
    min_speedup: float = 1.01,
    max_peak_memory_regression: float = 0.05,
) -> dict[str, Any]:
    """Classify a native-versus-optimized pair without claiming microbench wins."""
    if native.status != "ok" or optimized.status != "ok":
        return {
            "classification": "failed",
            "reason": "one or both modes failed",
            "native_status": native.status,
            "optimized_status": optimized.status,
        }
    if (
        native.median_wall_seconds is None
        or optimized.median_wall_seconds is None
        or native.median_wall_seconds <= 0
    ):
        return {
            "classification": "failed",
            "reason": "missing median wall times",
        }

    speedup = native.median_wall_seconds / optimized.median_wall_seconds
    native_mem = [m for m in native.peak_memory_mb if m is not None]
    opt_mem = [m for m in optimized.peak_memory_mb if m is not None]
    memory_regression = None
    if native_mem and opt_mem:
        native_peak = max(native_mem)
        opt_peak = max(opt_mem)
        if native_peak > 0:
            memory_regression = (opt_peak - native_peak) / native_peak

    if memory_regression is not None and memory_regression > max_peak_memory_regression:
        classification = "regressed"
        reason = "peak memory regression exceeds threshold"
    elif speedup >= min_speedup:
        classification = "improved"
        reason = "repeatable end-to-end wall-time improvement"
    elif speedup <= (1.0 / min_speedup):
        classification = "regressed"
        reason = "end-to-end wall time regressed"
    else:
        classification = "neutral"
        reason = "change within timing noise / below promotion threshold"

    return {
        "classification": classification,
        "reason": reason,
        "native_median_wall_seconds": native.median_wall_seconds,
        "optimized_median_wall_seconds": optimized.median_wall_seconds,
        "end_to_end_speedup": speedup,
        "min_end_to_end_speedup": min_speedup,
        "peak_memory_regression": memory_regression,
        "max_peak_memory_regression": max_peak_memory_regression,
    }
