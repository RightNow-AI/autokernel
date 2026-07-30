"""Versioned, atomic machine-readable benchmark results."""

from __future__ import annotations

import json
import math
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "RESULT_SCHEMA_VERSION",
    "collect_environment_metadata",
    "result_envelope",
    "write_result_atomic",
]

RESULT_SCHEMA_VERSION = 1


def collect_environment_metadata(device: str) -> dict[str, Any]:
    """Collect runtime versions without requiring Triton or a CUDA device."""
    import torch

    try:
        import triton

        triton_version = getattr(triton, "__version__", "unknown")
    except Exception:
        triton_version = None

    gpu_name = None
    if device.startswith("cuda"):
        try:
            gpu_name = torch.cuda.get_device_name(device)
        except Exception:
            gpu_name = None
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "triton_version": triton_version,
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "device": device,
        "gpu_name": gpu_name,
    }


def result_envelope(operation: str, **sections: Any) -> dict[str, Any]:
    """Build the stable top-level result record."""
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        **sections,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "as_dict"):
        return _json_safe(value.as_dict())
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_result_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically replace ``path`` with a complete JSON document."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                _json_safe(payload),
                handle,
                indent=2,
                sort_keys=True,
                default=_json_default,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination
