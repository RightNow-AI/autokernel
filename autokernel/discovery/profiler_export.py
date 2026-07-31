"""Load FastVideo metadata-only torch.profiler exports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .profiler_parse import parse_key_averages_rows
from .types import (
    DISCOVERY_SCHEMA_VERSION,
    DiscoveryError,
    DiscoveryReport,
)

_EXPORT_FIELDS = {
    "schema_version",
    "producer",
    "workload",
    "environment",
    "total_cuda_time_us",
    "rows",
    # Optional FX capture block, present only when the producer ran graph
    # capture alongside the timing profile. Older exports omit all four.
    "capture",
    "regions",
    "graph_breaks",
    "unsupported",
}

# Capture-block format the loader understands. Independent of the export's
# schema_version so a capture change does not invalidate timing-only readers.
SUPPORTED_CAPTURE_SCHEMA_VERSION = 1


def _fail(source: object, location: str, message: str) -> DiscoveryError:
    return DiscoveryError(f"profiler export {source!r}: {location}: {message}")


def _mapping(
    value: Any,
    *,
    source: object,
    location: str,
    non_empty: bool = False,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or (non_empty and not value):
        qualifier = "non-empty " if non_empty else ""
        raise _fail(source, location, f"must be a {qualifier}object")
    return value


def profiler_export_to_report(
    raw_value: Any,
    *,
    source: object = "<memory>",
) -> DiscoveryReport:
    """Validate a portable profiler export and create a discovery report."""
    raw = _mapping(
        raw_value,
        source=source,
        location="top level",
        non_empty=True,
    )
    unknown = sorted(set(raw) - _EXPORT_FIELDS)
    if unknown:
        raise _fail(source, "top level", f"unknown field(s) {unknown}")
    if raw.get("schema_version") != DISCOVERY_SCHEMA_VERSION:
        raise _fail(
            source,
            "schema_version",
            f"expected {DISCOVERY_SCHEMA_VERSION}",
        )
    rows = raw.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise _fail(source, "rows", "must be a list")

    try:
        operators = parse_key_averages_rows(rows)
    except (TypeError, ValueError) as exc:
        raise _fail(source, "rows", str(exc)) from exc

    total = raw.get("total_cuda_time_us")
    if total is None:
        total = sum(max(item.self_cuda_time_us, 0.0) for item in operators)

    capture = raw.get("capture")
    if capture is not None:
        capture = _mapping(
            capture,
            source=source,
            location="capture",
            non_empty=True,
        )
        version = capture.get("capture_schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise _fail(source, "capture.capture_schema_version", "must be an integer")
        if version != SUPPORTED_CAPTURE_SCHEMA_VERSION:
            raise _fail(
                source,
                "capture.capture_schema_version",
                f"unsupported version {version}; "
                f"expected {SUPPORTED_CAPTURE_SCHEMA_VERSION}",
            )
    else:
        # Captured payloads are versioned through the capture mapping; refuse
        # capture data that arrives without its version declaration.
        for name in ("regions", "graph_breaks", "unsupported"):
            if raw.get(name):
                raise _fail(
                    source,
                    "capture",
                    f"required when {name!r} is present",
                )

    def _list(name: str) -> list[Any]:
        value = raw.get(name, [])
        if value is None:
            return []
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise _fail(source, name, "must be a list")
        return list(value)

    payload = {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "producer": dict(
            _mapping(
                raw.get("producer"),
                source=source,
                location="producer",
                non_empty=True,
            )
        ),
        "workload": dict(
            _mapping(
                raw.get("workload"),
                source=source,
                location="workload",
                non_empty=True,
            )
        ),
        "environment": dict(
            _mapping(
                raw.get("environment"),
                source=source,
                location="environment",
                non_empty=True,
            )
        ),
        "total_cuda_time_us": total,
        "operators": [
            {
                **item.as_dict(),
                    "op_key": item.op_key,
            }
            for item in operators
        ],
        "regions": _list("regions"),
        "graph_breaks": _list("graph_breaks"),
        "unsupported": _list("unsupported"),
    }
    return DiscoveryReport.from_dict(payload, source=source)


def load_profiler_export(path: str | Path) -> DiscoveryReport:
    """Load a FastVideo profiler JSON artifact."""
    input_path = Path(path)
    if not input_path.is_file():
        raise _fail(str(input_path), "file", "not found")
    try:
        raw = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _fail(str(input_path), "JSON", f"invalid JSON: {exc}") from exc
    return profiler_export_to_report(raw, source=str(input_path))
