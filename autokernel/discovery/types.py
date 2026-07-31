"""Metadata-only schemas for universal profiling and graph capture.

These records are the Workstream 2 foundation: torch.profiler hotspots and
Dynamo/FX graph regions without requiring model-specific annotations.
Content, secrets, weights, activations, and prompts are forbidden.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fingerprint import graph_fingerprint

DISCOVERY_SCHEMA_VERSION = 1

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "producer",
    "workload",
    "environment",
    "total_cuda_time_us",
    "operators",
    "regions",
    "graph_breaks",
    "unsupported",
}
_OPERATOR_FIELDS = {
    "name",
    "op_key",
    "calls",
    "cuda_time_us",
    "self_cuda_time_us",
    "cpu_time_us",
    "input_shapes",
    "parent_module",
    "source",
    "attributes",
}
_REGION_FIELDS = {
    "name",
    "fingerprint",
    "operations",
    "dependencies",
    "inputs",
    "outputs",
    "safe_constants",
    "shape_frequency",
    "cuda_time_us",
    "self_cuda_time_us",
    "calls",
    "parent_module",
    "pattern_family",
    "rejection_reasons",
    "attributes",
}
_TENSOR_FIELDS = {
    "name",
    "shape",
    "stride",
    "dtype",
    "device_type",
    "requires_grad",
}
_GRAPH_BREAK_FIELDS = {
    "scope",
    "reason",
    "op_name",
    "count",
}
_UNSUPPORTED_FIELDS = {
    "op_name",
    "reason",
    "count",
    "scope",
}

_FORBIDDEN_KEYS = {
    "credential",
    "credentials",
    "data",
    "password",
    "prompt",
    "secret",
    "secrets",
    "tensor_values",
    "token",
    "values",
    "weights",
    "activations",
}
_OP_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_./:-]{1,256}$")
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class DiscoveryError(ValueError):
    """Raised when a discovery/profile capture is malformed or unsafe."""


def _fail(source: object, location: str, message: str) -> DiscoveryError:
    return DiscoveryError(f"discovery report {source!r}: {location}: {message}")


def _mapping(
    value: Any,
    source: object,
    location: str,
    *,
    non_empty: bool = False,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or (non_empty and not value):
        qualifier = "non-empty " if non_empty else ""
        raise _fail(source, location, f"must be a {qualifier}object")
    for key in value:
        if not isinstance(key, str) or not key:
            raise _fail(source, location, "keys must be non-empty strings")
        if key.lower() in _FORBIDDEN_KEYS:
            raise _fail(
                source,
                f"{location}.{key}",
                "content or secret fields are forbidden",
            )
    return value


def _unknown_fields(
    raw: Mapping[str, Any],
    allowed: set[str],
    source: object,
    location: str,
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise _fail(source, location, f"unknown field(s) {unknown}")


def _text(value: Any, source: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(source, location, "must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, source: object, location: str) -> str | None:
    if value is None:
        return None
    return _text(value, source, location)


def _positive_int(value: Any, source: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _fail(source, location, "must be a positive integer")
    return value


def _non_negative_int(value: Any, source: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _fail(source, location, "must be a non-negative integer")
    return value


def _finite_non_negative(value: Any, source: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(source, location, "must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise _fail(source, location, "must be a finite non-negative number")
    return number


def _metadata_value(value: Any, source: object, location: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise _fail(source, location, "numbers must be finite")
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _metadata_value(item, source, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result = {}
        for key, item in _mapping(value, source, location).items():
            result[key] = _metadata_value(item, source, f"{location}.{key}")
        return result
    raise _fail(source, location, "must contain JSON metadata only")


@dataclass(frozen=True)
class TensorMeta:
    """Layout metadata for one tensor without any values."""

    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device_type: str
    requires_grad: bool = False

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object, location: str
    ) -> "TensorMeta":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _TENSOR_FIELDS, source, location)
        shape_raw = raw.get("shape")
        stride_raw = raw.get("stride")
        if not isinstance(shape_raw, Sequence) or isinstance(shape_raw, (str, bytes)):
            raise _fail(source, f"{location}.shape", "must be a list")
        if not isinstance(stride_raw, Sequence) or isinstance(
            stride_raw, (str, bytes)
        ):
            raise _fail(source, f"{location}.stride", "must be a list")
        shape = tuple(
            _non_negative_int(dim, source, f"{location}.shape[{i}]")
            for i, dim in enumerate(shape_raw)
        )
        stride = []
        for i, value in enumerate(stride_raw):
            if isinstance(value, bool) or not isinstance(value, int):
                raise _fail(source, f"{location}.stride[{i}]", "must be an integer")
            stride.append(value)
        if len(stride) != len(shape):
            raise _fail(
                source,
                f"{location}.stride",
                "must have the same length as shape",
            )
        requires_grad = raw.get("requires_grad", False)
        if not isinstance(requires_grad, bool):
            raise _fail(source, f"{location}.requires_grad", "must be a bool")
        return cls(
            name=_text(raw.get("name"), source, f"{location}.name"),
            shape=shape,
            stride=tuple(stride),
            dtype=_text(raw.get("dtype"), source, f"{location}.dtype"),
            device_type=_text(
                raw.get("device_type"), source, f"{location}.device_type"
            ),
            requires_grad=requires_grad,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "device_type": self.device_type,
            "requires_grad": self.requires_grad,
        }

    def signature_dict(self) -> dict[str, Any]:
        """Shape/dtype/layout only — used for fingerprinting."""
        return {
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "device_type": self.device_type,
            "requires_grad": self.requires_grad,
        }


@dataclass(frozen=True)
class OperatorHotspot:
    """One torch.profiler-attributed operator or ATen op aggregate."""

    name: str
    op_key: str
    calls: int
    cuda_time_us: float
    self_cuda_time_us: float
    cpu_time_us: float = 0.0
    input_shapes: tuple[tuple[int, ...], ...] = ()
    parent_module: str | None = None
    source: str = "torch_profiler"
    attributes: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object, location: str
    ) -> "OperatorHotspot":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _OPERATOR_FIELDS, source, location)
        op_key = _text(raw.get("op_key"), source, f"{location}.op_key")
        if not _OP_KEY_PATTERN.fullmatch(op_key):
            raise _fail(source, f"{location}.op_key", "invalid op key")
        shapes_raw = raw.get("input_shapes", [])
        if not isinstance(shapes_raw, Sequence) or isinstance(
            shapes_raw, (str, bytes)
        ):
            raise _fail(source, f"{location}.input_shapes", "must be a list")
        shapes: list[tuple[int, ...]] = []
        for index, shape in enumerate(shapes_raw):
            if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
                raise _fail(
                    source,
                    f"{location}.input_shapes[{index}]",
                    "must be a list of dimensions",
                )
            shapes.append(
                tuple(
                    _non_negative_int(
                        dim,
                        source,
                        f"{location}.input_shapes[{index}][{dim_i}]",
                    )
                    for dim_i, dim in enumerate(shape)
                )
            )
        attributes = _metadata_value(
            raw.get("attributes", {}),
            source,
            f"{location}.attributes",
        )
        return cls(
            name=_text(raw.get("name"), source, f"{location}.name"),
            op_key=op_key,
            calls=_positive_int(raw.get("calls"), source, f"{location}.calls"),
            cuda_time_us=_finite_non_negative(
                raw.get("cuda_time_us"), source, f"{location}.cuda_time_us"
            ),
            self_cuda_time_us=_finite_non_negative(
                raw.get("self_cuda_time_us"),
                source,
                f"{location}.self_cuda_time_us",
            ),
            cpu_time_us=_finite_non_negative(
                raw.get("cpu_time_us", 0), source, f"{location}.cpu_time_us"
            ),
            input_shapes=tuple(shapes),
            parent_module=_optional_text(
                raw.get("parent_module"), source, f"{location}.parent_module"
            ),
            source=_text(
                raw.get("source", "torch_profiler"),
                source,
                f"{location}.source",
            ),
            attributes=attributes if attributes else None,
        )

    def impact_pct(self, total_cuda_time_us: float) -> float:
        """Return this operator's exclusive share of measured CUDA time."""
        if total_cuda_time_us <= 0:
            return 0.0
        return 100.0 * self.self_cuda_time_us / total_cuda_time_us

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "op_key": self.op_key,
            "calls": self.calls,
            "cuda_time_us": self.cuda_time_us,
            "self_cuda_time_us": self.self_cuda_time_us,
            "cpu_time_us": self.cpu_time_us,
            "input_shapes": [list(shape) for shape in self.input_shapes],
            "source": self.source,
        }
        if self.parent_module is not None:
            payload["parent_module"] = self.parent_module
        if self.attributes:
            payload["attributes"] = dict(self.attributes)
        return payload


@dataclass(frozen=True)
class GraphBreakRecord:
    """A recorded Dynamo/FX graph break or capture limitation."""

    scope: str
    reason: str
    op_name: str | None = None
    count: int = 1

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object, location: str
    ) -> "GraphBreakRecord":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _GRAPH_BREAK_FIELDS, source, location)
        return cls(
            scope=_text(raw.get("scope"), source, f"{location}.scope"),
            reason=_text(raw.get("reason"), source, f"{location}.reason"),
            op_name=_optional_text(
                raw.get("op_name"), source, f"{location}.op_name"
            ),
            count=_positive_int(
                raw.get("count", 1), source, f"{location}.count"
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "scope": self.scope,
            "reason": self.reason,
            "count": self.count,
        }
        if self.op_name is not None:
            payload["op_name"] = self.op_name
        return payload


@dataclass(frozen=True)
class UnsupportedOpRecord:
    """An observed op that cannot yet enter the search pipeline."""

    op_name: str
    reason: str
    count: int = 1
    scope: str | None = None

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object, location: str
    ) -> "UnsupportedOpRecord":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _UNSUPPORTED_FIELDS, source, location)
        return cls(
            op_name=_text(raw.get("op_name"), source, f"{location}.op_name"),
            reason=_text(raw.get("reason"), source, f"{location}.reason"),
            count=_positive_int(
                raw.get("count", 1), source, f"{location}.count"
            ),
            scope=_optional_text(
                raw.get("scope"), source, f"{location}.scope"
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "op_name": self.op_name,
            "reason": self.reason,
            "count": self.count,
        }
        if self.scope is not None:
            payload["scope"] = self.scope
        return payload


@dataclass(frozen=True)
class GraphRegion:
    """One captured pure-tensor subgraph candidate (metadata only)."""

    name: str
    fingerprint: str
    operations: tuple[str, ...]
    inputs: tuple[TensorMeta, ...]
    outputs: tuple[TensorMeta, ...] = ()
    dependencies: tuple[str, ...] = ()
    safe_constants: Mapping[str, Any] | None = None
    shape_frequency: Mapping[str, int] | None = None
    cuda_time_us: float = 0.0
    self_cuda_time_us: float = 0.0
    calls: int = 1
    parent_module: str | None = None
    pattern_family: str | None = None
    rejection_reasons: tuple[str, ...] = ()
    attributes: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object, location: str
    ) -> "GraphRegion":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _REGION_FIELDS, source, location)
        name = _text(raw.get("name"), source, f"{location}.name")
        if not _NAME_PATTERN.fullmatch(name):
            raise _fail(source, f"{location}.name", "invalid region name")

        ops_raw = raw.get("operations")
        if (
            not isinstance(ops_raw, Sequence)
            or isinstance(ops_raw, (str, bytes))
            or not ops_raw
        ):
            raise _fail(
                source, f"{location}.operations", "must be a non-empty list"
            )
        operations = tuple(
            _text(op, source, f"{location}.operations[{i}]")
            for i, op in enumerate(ops_raw)
        )

        inputs_raw = raw.get("inputs")
        if (
            not isinstance(inputs_raw, Sequence)
            or isinstance(inputs_raw, (str, bytes))
            or not inputs_raw
        ):
            raise _fail(source, f"{location}.inputs", "must be a non-empty list")
        inputs = tuple(
            TensorMeta.from_dict(
                item, source=source, location=f"{location}.inputs[{i}]"
            )
            for i, item in enumerate(inputs_raw)
        )
        outputs_raw = raw.get("outputs", [])
        if not isinstance(outputs_raw, Sequence) or isinstance(
            outputs_raw, (str, bytes)
        ):
            raise _fail(source, f"{location}.outputs", "must be a list")
        outputs = tuple(
            TensorMeta.from_dict(
                item, source=source, location=f"{location}.outputs[{i}]"
            )
            for i, item in enumerate(outputs_raw)
        )

        deps_raw = raw.get("dependencies", [])
        if not isinstance(deps_raw, Sequence) or isinstance(
            deps_raw, (str, bytes)
        ):
            raise _fail(source, f"{location}.dependencies", "must be a list")
        dependencies = tuple(
            _text(dep, source, f"{location}.dependencies[{i}]")
            for i, dep in enumerate(deps_raw)
        )

        rejection_raw = raw.get("rejection_reasons", [])
        if not isinstance(rejection_raw, Sequence) or isinstance(
            rejection_raw, (str, bytes)
        ):
            raise _fail(
                source, f"{location}.rejection_reasons", "must be a list"
            )
        rejection_reasons = tuple(
            _text(reason, source, f"{location}.rejection_reasons[{i}]")
            for i, reason in enumerate(rejection_raw)
        )

        fingerprint = _text(
            raw.get("fingerprint"), source, f"{location}.fingerprint"
        )
        parent_module = _optional_text(
            raw.get("parent_module"), source, f"{location}.parent_module"
        )
        expected = graph_fingerprint(
            operations=operations,
            input_signatures=[item.signature_dict() for item in inputs],
            output_signatures=[item.signature_dict() for item in outputs],
            safe_constants=raw.get("safe_constants") or {},
            parent_module=parent_module,
        )
        # Recomputed fingerprint is the source of truth for equivalent regions.
        if fingerprint != expected:
            raise _fail(
                source,
                f"{location}.fingerprint",
                "does not match canonical graph fingerprint",
            )

        shape_frequency = raw.get("shape_frequency", {})
        if shape_frequency is None:
            shape_frequency = {}
        shape_frequency = _mapping(
            shape_frequency, source, f"{location}.shape_frequency"
        )
        freq: dict[str, int] = {}
        for key, count in shape_frequency.items():
            freq[key] = _positive_int(
                count, source, f"{location}.shape_frequency.{key}"
            )

        attributes = _metadata_value(
            raw.get("attributes", {}),
            source,
            f"{location}.attributes",
        )
        return cls(
            name=name,
            fingerprint=fingerprint,
            operations=operations,
            inputs=inputs,
            outputs=outputs,
            dependencies=dependencies,
            safe_constants=_metadata_value(
                raw.get("safe_constants", {}),
                source,
                f"{location}.safe_constants",
            )
            or None,
            shape_frequency=freq or None,
            cuda_time_us=_finite_non_negative(
                raw.get("cuda_time_us", 0),
                source,
                f"{location}.cuda_time_us",
            ),
            self_cuda_time_us=_finite_non_negative(
                raw.get("self_cuda_time_us", 0),
                source,
                f"{location}.self_cuda_time_us",
            ),
            calls=_positive_int(
                raw.get("calls", 1), source, f"{location}.calls"
            ),
            parent_module=parent_module,
            pattern_family=_optional_text(
                raw.get("pattern_family"), source, f"{location}.pattern_family"
            ),
            rejection_reasons=rejection_reasons,
            attributes=attributes if attributes else None,
        )

    @classmethod
    def build(
        cls,
        *,
        name: str,
        operations: Sequence[str],
        inputs: Sequence[TensorMeta],
        outputs: Sequence[TensorMeta] = (),
        parent_module: str | None = None,
        safe_constants: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "GraphRegion":
        """Construct a region with a recomputed stable fingerprint."""
        if not _NAME_PATTERN.fullmatch(name):
            raise ValueError(f"invalid graph region name: {name!r}")
        fingerprint = graph_fingerprint(
            operations=operations,
            input_signatures=[item.signature_dict() for item in inputs],
            output_signatures=[item.signature_dict() for item in outputs],
            safe_constants=safe_constants,
            parent_module=parent_module,
        )
        return cls(
            name=name,
            fingerprint=fingerprint,
            operations=tuple(operations),
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            parent_module=parent_module,
            safe_constants=dict(safe_constants) if safe_constants else None,
            **kwargs,
        )

    def impact_pct(self, total_cuda_time_us: float) -> float:
        if total_cuda_time_us <= 0:
            return 0.0
        return 100.0 * self.cuda_time_us / total_cuda_time_us

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "fingerprint": self.fingerprint,
            "operations": list(self.operations),
            "dependencies": list(self.dependencies),
            "inputs": [item.as_dict() for item in self.inputs],
            "outputs": [item.as_dict() for item in self.outputs],
            "cuda_time_us": self.cuda_time_us,
            "self_cuda_time_us": self.self_cuda_time_us,
            "calls": self.calls,
            "rejection_reasons": list(self.rejection_reasons),
        }
        if self.safe_constants:
            payload["safe_constants"] = dict(self.safe_constants)
        if self.shape_frequency:
            payload["shape_frequency"] = dict(self.shape_frequency)
        if self.parent_module is not None:
            payload["parent_module"] = self.parent_module
        if self.pattern_family is not None:
            payload["pattern_family"] = self.pattern_family
        if self.attributes:
            payload["attributes"] = dict(self.attributes)
        return payload


@dataclass(frozen=True)
class DiscoveryReport:
    """Combined profiler + graph-capture report for one workload run."""

    producer: Mapping[str, Any]
    workload: Mapping[str, Any]
    environment: Mapping[str, Any]
    total_cuda_time_us: float
    operators: tuple[OperatorHotspot, ...]
    regions: tuple[GraphRegion, ...] = ()
    graph_breaks: tuple[GraphBreakRecord, ...] = ()
    unsupported: tuple[UnsupportedOpRecord, ...] = ()
    source: str = "<memory>"
    schema_version: int = DISCOVERY_SCHEMA_VERSION

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object = "<memory>"
    ) -> "DiscoveryReport":
        raw = _mapping(raw_value, source, "top level", non_empty=True)
        _unknown_fields(raw, _TOP_LEVEL_FIELDS, source, "top level")
        version = raw.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise _fail(source, "schema_version", "must be an integer")
        if version != DISCOVERY_SCHEMA_VERSION:
            raise _fail(
                source,
                "schema_version",
                f"unsupported version {version}; expected {DISCOVERY_SCHEMA_VERSION}",
            )
        producer = dict(
            _metadata_value(
                _mapping(raw.get("producer"), source, "producer", non_empty=True),
                source,
                "producer",
            )
        )
        workload = dict(
            _metadata_value(
                _mapping(raw.get("workload"), source, "workload", non_empty=True),
                source,
                "workload",
            )
        )
        environment = dict(
            _metadata_value(
                _mapping(
                    raw.get("environment"),
                    source,
                    "environment",
                    non_empty=True,
                ),
                source,
                "environment",
            )
        )
        for field in ("name", "version"):
            _text(producer.get(field), source, f"producer.{field}")
        for field in ("workload_id", "model_id"):
            _text(workload.get(field), source, f"workload.{field}")

        operators_raw = raw.get("operators", [])
        if not isinstance(operators_raw, Sequence) or isinstance(
            operators_raw, (str, bytes)
        ):
            raise _fail(source, "operators", "must be a list")
        operators = tuple(
            OperatorHotspot.from_dict(
                item, source=source, location=f"operators[{i}]"
            )
            for i, item in enumerate(operators_raw)
        )
        regions_raw = raw.get("regions", [])
        if not isinstance(regions_raw, Sequence) or isinstance(
            regions_raw, (str, bytes)
        ):
            raise _fail(source, "regions", "must be a list")
        regions = tuple(
            GraphRegion.from_dict(
                item, source=source, location=f"regions[{i}]"
            )
            for i, item in enumerate(regions_raw)
        )
        breaks_raw = raw.get("graph_breaks", [])
        if not isinstance(breaks_raw, Sequence) or isinstance(
            breaks_raw, (str, bytes)
        ):
            raise _fail(source, "graph_breaks", "must be a list")
        graph_breaks = tuple(
            GraphBreakRecord.from_dict(
                item, source=source, location=f"graph_breaks[{i}]"
            )
            for i, item in enumerate(breaks_raw)
        )
        unsupported_raw = raw.get("unsupported", [])
        if not isinstance(unsupported_raw, Sequence) or isinstance(
            unsupported_raw, (str, bytes)
        ):
            raise _fail(source, "unsupported", "must be a list")
        unsupported = tuple(
            UnsupportedOpRecord.from_dict(
                item, source=source, location=f"unsupported[{i}]"
            )
            for i, item in enumerate(unsupported_raw)
        )
        return cls(
            producer=producer,
            workload=workload,
            environment=environment,
            total_cuda_time_us=_finite_non_negative(
                raw.get("total_cuda_time_us"),
                source,
                "total_cuda_time_us",
            ),
            operators=operators,
            regions=regions,
            graph_breaks=graph_breaks,
            unsupported=unsupported,
            source=str(source),
            schema_version=version,
        )

    def ranked_operators(self) -> tuple[OperatorHotspot, ...]:
        return tuple(
            sorted(
                self.operators,
                key=lambda op: (-op.self_cuda_time_us, -op.calls, op.name),
            )
        )

    def ranked_regions(self) -> tuple[GraphRegion, ...]:
        return tuple(
            sorted(
                self.regions,
                key=lambda region: (
                    -region.cuda_time_us,
                    -region.calls,
                    region.name,
                ),
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "producer": dict(self.producer),
            "workload": dict(self.workload),
            "environment": dict(self.environment),
            "total_cuda_time_us": self.total_cuda_time_us,
            "operators": [item.as_dict() for item in self.operators],
            "regions": [item.as_dict() for item in self.regions],
            "graph_breaks": [item.as_dict() for item in self.graph_breaks],
            "unsupported": [item.as_dict() for item in self.unsupported],
        }


def load_discovery_report(path: str | Path) -> DiscoveryReport:
    """Load and validate a discovery report without importing torch."""
    file_path = Path(path)
    if not file_path.is_file():
        raise DiscoveryError(f"discovery report {file_path!s}: file: not found")
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DiscoveryError(
            f"discovery report {file_path!s}: JSON: invalid JSON: {exc}"
        ) from exc
    return DiscoveryReport.from_dict(raw, source=str(file_path))


def write_discovery_report(report: DiscoveryReport, path: str | Path) -> None:
    """Atomically write a discovery report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
