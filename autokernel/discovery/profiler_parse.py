"""Parse torch.profiler key-average style tables into OperatorHotspot rows.

Accepts already-exported JSON lists (no live profiler required). Real GPU
collection is a separate step; this module only normalizes metadata.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .types import OperatorHotspot


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_key_averages_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str = "torch_profiler",
) -> tuple[OperatorHotspot, ...]:
    """Convert list-of-dicts profiler exports into OperatorHotspot tuples.

    Recognized keys (flexible aliases):
    - name / key / op_name
    - cuda_time_total / cuda_time_us / device_time_total
    - self_cuda_time_total / self_cuda_time_us
    - cpu_time_total / cpu_time_us
    - count / calls
    """
    hotspots: list[OperatorHotspot] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"rows[{index}] must be a mapping")
        name = (
            row.get("name")
            or row.get("key")
            or row.get("op_name")
            or row.get("operator")
        )
        if not name:
            raise ValueError(f"rows[{index}] missing operator name")
        name = str(name)
        cuda = _num(
            row.get("cuda_time_us",
                    row.get("cuda_time_total",
                            row.get("device_time_total",
                                    row.get("cuda_time", 0)))),
        )
        # Profiler often reports times in microseconds already; if a *_ms key
        # is present, convert.
        if "cuda_time_ms" in row:
            cuda = _num(row["cuda_time_ms"]) * 1000.0
        self_cuda = _num(
            row.get(
                "self_cuda_time_us",
                row.get("self_cuda_time_total", row.get("self_cuda_time", cuda)),
            )
        )
        if "self_cuda_time_ms" in row:
            self_cuda = _num(row["self_cuda_time_ms"]) * 1000.0
        cpu = _num(row.get("cpu_time_us", row.get("cpu_time_total", 0)))
        if "cpu_time_ms" in row:
            cpu = _num(row["cpu_time_ms"]) * 1000.0
        calls = row.get("calls", row.get("count", 1))
        try:
            calls_i = int(calls)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rows[{index}].calls invalid") from exc
        if calls_i <= 0:
            continue
        shapes_raw = row.get("input_shapes") or row.get("shapes") or []
        shapes: list[tuple[int, ...]] = []
        if isinstance(shapes_raw, Sequence) and not isinstance(
            shapes_raw, (str, bytes)
        ):
            for shape in shapes_raw:
                if isinstance(shape, Sequence) and not isinstance(
                    shape, (str, bytes)
                ):
                    shapes.append(tuple(int(x) for x in shape))
        hotspots.append(
            OperatorHotspot(
                name=name,
                op_key=name,
                calls=calls_i,
                cuda_time_us=cuda,
                self_cuda_time_us=min(self_cuda, cuda) if cuda else self_cuda,
                cpu_time_us=cpu,
                input_shapes=tuple(shapes),
                parent_module=(
                    str(row["parent_module"])
                    if row.get("parent_module") is not None
                    else None
                ),
                source=source,
            )
        )
    return tuple(hotspots)
