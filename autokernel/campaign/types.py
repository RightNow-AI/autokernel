"""Portable FastVideo-to-AutoKernel optimization campaign contract.

The campaign contains metadata only: operation identities, tensor
shape/layout signatures, call counts, timing aggregates, and workload/runtime
identity. It must never contain tensor values, prompts, model weights, or
credentials.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

CAMPAIGN_SCHEMA_VERSION = 1

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "producer",
    "workload",
    "environment",
    "total_profiled_device_time_us",
    "targets",
}
_TARGET_FIELDS = {
    "name",
    "operation",
    "kind",
    "spec_locator",
    "total_device_time_us",
    "self_device_time_us",
    "calls",
    "requires_backward",
    "observations",
    "attributes",
}
_OBSERVATION_FIELDS = {
    "name",
    "count",
    "total_device_time_us",
    "inputs",
    "tags",
}
_TENSOR_FIELDS = {
    "name",
    "shape",
    "stride",
    "dtype",
    "device_type",
    "requires_grad",
}
_TARGET_KINDS = {"operator", "fusion", "graph_fragment"}
_OPERATION_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORBIDDEN_METADATA_KEYS = {
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
}


class CampaignError(ValueError):
    """Raised when a campaign is malformed or unsafe to execute."""


def _fail(source: object, location: str, message: str) -> CampaignError:
    return CampaignError(f"optimization campaign {source!r}: {location}: {message}")


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
    return value


def _finite_non_negative(value: Any, source: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(source, location, "must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise _fail(source, location, "must be a finite non-negative number")
    return normalized


def _positive_int(value: Any, source: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _fail(source, location, "must be a positive integer")
    return value


def _metadata_value(value: Any, source: object, location: str) -> Any:
    """Validate JSON metadata while excluding common content/secret fields."""
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
            if key.lower() in _FORBIDDEN_METADATA_KEYS:
                raise _fail(
                    source,
                    f"{location}.{key}",
                    "content or secret fields are forbidden",
                )
            result[key] = _metadata_value(item, source, f"{location}.{key}")
        return result
    raise _fail(source, location, "must contain JSON metadata only")


@dataclass(frozen=True)
class TensorSignature:
    """One tensor's layout metadata without any tensor contents."""

    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device_type: str
    requires_grad: bool = False

    @classmethod
    def from_dict(
        cls,
        raw_value: Any,
        *,
        source: object,
        location: str,
    ) -> "TensorSignature":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _TENSOR_FIELDS, source, location)
        name = _text(raw.get("name"), source, f"{location}.name")
        dtype = _text(raw.get("dtype"), source, f"{location}.dtype")
        device_type = _text(
            raw.get("device_type"), source, f"{location}.device_type"
        )

        shape_raw = raw.get("shape")
        if not isinstance(shape_raw, Sequence) or isinstance(
            shape_raw, (str, bytes)
        ):
            raise _fail(source, f"{location}.shape", "must be a list of dimensions")
        shape: list[int] = []
        for index, dimension in enumerate(shape_raw):
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
            ):
                raise _fail(
                    source,
                    f"{location}.shape[{index}]",
                    "must be a non-negative integer",
                )
            shape.append(dimension)

        stride_raw = raw.get("stride")
        if not isinstance(stride_raw, Sequence) or isinstance(
            stride_raw, (str, bytes)
        ):
            raise _fail(source, f"{location}.stride", "must be a list of strides")
        stride: list[int] = []
        for index, value in enumerate(stride_raw):
            if isinstance(value, bool) or not isinstance(value, int):
                raise _fail(
                    source,
                    f"{location}.stride[{index}]",
                    "must be an integer",
                )
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
            name=name,
            shape=tuple(shape),
            stride=tuple(stride),
            dtype=dtype,
            device_type=device_type,
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


@dataclass(frozen=True)
class ShapeObservation:
    """One repeated input signature observed during a model run."""

    name: str
    count: int
    total_device_time_us: float
    inputs: tuple[TensorSignature, ...]
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(
        cls,
        raw_value: Any,
        *,
        source: object,
        location: str,
    ) -> "ShapeObservation":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _OBSERVATION_FIELDS, source, location)
        name = _text(raw.get("name"), source, f"{location}.name")
        count = _positive_int(raw.get("count"), source, f"{location}.count")
        total = _finite_non_negative(
            raw.get("total_device_time_us", 0),
            source,
            f"{location}.total_device_time_us",
        )
        inputs_raw = raw.get("inputs")
        if (
            not isinstance(inputs_raw, Sequence)
            or isinstance(inputs_raw, (str, bytes))
            or not inputs_raw
        ):
            raise _fail(source, f"{location}.inputs", "must be a non-empty list")
        inputs = tuple(
            TensorSignature.from_dict(
                item,
                source=source,
                location=f"{location}.inputs[{index}]",
            )
            for index, item in enumerate(inputs_raw)
        )
        names = [item.name for item in inputs]
        if len(names) != len(set(names)):
            raise _fail(source, f"{location}.inputs", "contains duplicate names")

        tags_raw = raw.get("tags", [])
        if not isinstance(tags_raw, Sequence) or isinstance(
            tags_raw, (str, bytes)
        ):
            raise _fail(source, f"{location}.tags", "must be a list of strings")
        tags = tuple(
            _text(tag, source, f"{location}.tags[{index}]")
            for index, tag in enumerate(tags_raw)
        )
        if len(tags) != len(set(tags)):
            raise _fail(source, f"{location}.tags", "contains duplicates")
        return cls(
            name=name,
            count=count,
            total_device_time_us=total,
            inputs=inputs,
            tags=tags,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "total_device_time_us": self.total_device_time_us,
            "inputs": [item.as_dict() for item in self.inputs],
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class CampaignTarget:
    """One optimization opportunity ranked by model-level device time."""

    name: str
    operation: str
    kind: str
    total_device_time_us: float
    self_device_time_us: float
    calls: int
    requires_backward: bool
    observations: tuple[ShapeObservation, ...]
    spec_locator: str | None = None
    attributes: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(
        cls,
        raw_value: Any,
        *,
        source: object,
        location: str,
    ) -> "CampaignTarget":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _TARGET_FIELDS, source, location)
        name = _text(raw.get("name"), source, f"{location}.name")
        operation = _text(
            raw.get("operation"), source, f"{location}.operation"
        )
        if not _OPERATION_PATTERN.fullmatch(operation):
            raise _fail(
                source,
                f"{location}.operation",
                "must be a safe identifier containing letters, digits, and underscores",
            )
        kind = _text(raw.get("kind"), source, f"{location}.kind")
        if kind not in _TARGET_KINDS:
            raise _fail(
                source,
                f"{location}.kind",
                f"must be one of {sorted(_TARGET_KINDS)}",
            )
        total = _finite_non_negative(
            raw.get("total_device_time_us"),
            source,
            f"{location}.total_device_time_us",
        )
        self_time = _finite_non_negative(
            raw.get("self_device_time_us"),
            source,
            f"{location}.self_device_time_us",
        )
        if self_time > total:
            raise _fail(
                source,
                f"{location}.self_device_time_us",
                "cannot exceed total_device_time_us",
            )
        calls = _positive_int(raw.get("calls"), source, f"{location}.calls")
        requires_backward = raw.get("requires_backward", False)
        if not isinstance(requires_backward, bool):
            raise _fail(
                source, f"{location}.requires_backward", "must be a bool"
            )
        spec_locator_raw = raw.get("spec_locator")
        spec_locator = (
            None
            if spec_locator_raw is None
            else _text(
                spec_locator_raw, source, f"{location}.spec_locator"
            )
        )
        observations_raw = raw.get("observations")
        if (
            not isinstance(observations_raw, Sequence)
            or isinstance(observations_raw, (str, bytes))
            or not observations_raw
        ):
            raise _fail(
                source, f"{location}.observations", "must be a non-empty list"
            )
        observations = tuple(
            ShapeObservation.from_dict(
                item,
                source=source,
                location=f"{location}.observations[{index}]",
            )
            for index, item in enumerate(observations_raw)
        )
        observation_calls = sum(item.count for item in observations)
        if observation_calls != calls:
            raise _fail(
                source,
                f"{location}.observations",
                f"counts sum to {observation_calls}, expected calls={calls}",
            )
        attributes_raw = raw.get("attributes", {})
        attributes = _metadata_value(
            _mapping(attributes_raw, source, f"{location}.attributes"),
            source,
            f"{location}.attributes",
        )
        return cls(
            name=name,
            operation=operation,
            kind=kind,
            total_device_time_us=total,
            self_device_time_us=self_time,
            calls=calls,
            requires_backward=requires_backward,
            observations=observations,
            spec_locator=spec_locator,
            attributes=attributes,
        )

    def impact_pct(self, total_profiled_device_time_us: float) -> float:
        if total_profiled_device_time_us <= 0:
            return 0.0
        return 100.0 * self.total_device_time_us / total_profiled_device_time_us

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "operation": self.operation,
            "kind": self.kind,
            "spec_locator": self.spec_locator,
            "total_device_time_us": self.total_device_time_us,
            "self_device_time_us": self.self_device_time_us,
            "calls": self.calls,
            "requires_backward": self.requires_backward,
            "observations": [item.as_dict() for item in self.observations],
            "attributes": dict(self.attributes or {}),
        }


@dataclass(frozen=True)
class OptimizationCampaign:
    """A validated, versioned request to optimize one model workload."""

    producer: Mapping[str, Any]
    workload: Mapping[str, Any]
    environment: Mapping[str, Any]
    total_profiled_device_time_us: float
    targets: tuple[CampaignTarget, ...]
    source: str
    schema_version: int = CAMPAIGN_SCHEMA_VERSION

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object = "<memory>"
    ) -> "OptimizationCampaign":
        raw = _mapping(raw_value, source, "top level", non_empty=True)
        _unknown_fields(raw, _TOP_LEVEL_FIELDS, source, "top level")
        version = raw.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise _fail(source, "schema_version", "must be an integer")
        if version != CAMPAIGN_SCHEMA_VERSION:
            raise _fail(
                source,
                "schema_version",
                f"unsupported version {version}; expected {CAMPAIGN_SCHEMA_VERSION}",
            )
        producer = dict(
            _mapping(raw.get("producer"), source, "producer", non_empty=True)
        )
        workload = dict(
            _mapping(raw.get("workload"), source, "workload", non_empty=True)
        )
        environment = dict(
            _mapping(
                raw.get("environment"),
                source,
                "environment",
                non_empty=True,
            )
        )
        for location, mapping, required in (
            ("producer", producer, ("name", "version")),
            (
                "workload",
                workload,
                ("workload_id", "model_id", "task", "variant_id"),
            ),
            (
                "environment",
                environment,
                ("hardware_profile_id", "software_profile_id"),
            ),
        ):
            for field in required:
                _text(mapping.get(field), source, f"{location}.{field}")

        total = _finite_non_negative(
            raw.get("total_profiled_device_time_us"),
            source,
            "total_profiled_device_time_us",
        )
        if total <= 0:
            raise _fail(
                source,
                "total_profiled_device_time_us",
                "must be greater than zero",
            )
        targets_raw = raw.get("targets")
        if (
            not isinstance(targets_raw, Sequence)
            or isinstance(targets_raw, (str, bytes))
            or not targets_raw
        ):
            raise _fail(source, "targets", "must be a non-empty list")
        targets = tuple(
            CampaignTarget.from_dict(
                item, source=source, location=f"targets[{index}]"
            )
            for index, item in enumerate(targets_raw)
        )
        names = [target.name for target in targets]
        if len(names) != len(set(names)):
            raise _fail(source, "targets", "contains duplicate target names")
        return cls(
            producer=producer,
            workload=workload,
            environment=environment,
            total_profiled_device_time_us=total,
            targets=targets,
            source=str(source),
            schema_version=version,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "producer": dict(self.producer),
            "workload": dict(self.workload),
            "environment": dict(self.environment),
            "total_profiled_device_time_us": self.total_profiled_device_time_us,
            "targets": [target.as_dict() for target in self.targets],
        }


def load_campaign(path: str | Path) -> OptimizationCampaign:
    """Load and validate a campaign without importing torch or touching a GPU."""
    source = str(path)
    file_path = Path(path)
    if not file_path.is_file():
        raise _fail(source, "file", "not found")
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _fail(source, "JSON", f"invalid JSON: {exc}") from exc
    return OptimizationCampaign.from_dict(raw, source=source)


def rank_targets(campaign: OptimizationCampaign) -> tuple[CampaignTarget, ...]:
    """Rank by end-to-end device-time impact with stable name tie-breaking."""
    return tuple(
        sorted(
            campaign.targets,
            key=lambda target: (
                -target.total_device_time_us,
                -target.self_device_time_us,
                target.name,
            ),
        )
    )


def write_optimization_plan(
    campaign: OptimizationCampaign,
    path: str | Path,
    *,
    workspace_dir: str | Path = "workspace",
) -> dict[str, Any]:
    """Write the legacy orchestrator plan derived from a campaign."""
    kernels = []
    workspace = Path(workspace_dir)
    for rank, target in enumerate(rank_targets(campaign), start=1):
        kernels.append(
            {
                "rank": rank,
                "file": str(
                    workspace / f"kernel_{target.operation}_{rank}.py"
                ),
                "op_type": target.operation,
                "target_name": target.name,
                "spec_locator": target.spec_locator,
                "pct_total": target.impact_pct(
                    campaign.total_profiled_device_time_us
                ),
                "calls": target.calls,
                "requires_backward": target.requires_backward,
            }
        )
    plan = {
        "schema_version": 1,
        "campaign_source": campaign.source,
        "workload": dict(campaign.workload),
        "environment": dict(campaign.environment),
        "kernels_to_optimize": kernels,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return plan


def prepare_campaign(
    campaign: OptimizationCampaign,
    output_dir: str | Path = "workspace",
    *,
    trust_specs: bool = False,
) -> dict[str, Any]:
    """Materialize trusted starter kernels and an orchestration receipt."""
    if not trust_specs:
        raise CampaignError(
            "campaign preparation loads Python spec locators; "
            "pass trust_specs=True only for campaigns and specs you trust"
        )

    from autokernel.specs import SpecLoadError, load_spec

    workspace = Path(output_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    plan = write_optimization_plan(
        campaign,
        workspace / "optimization_plan.json",
        workspace_dir=workspace,
    )
    prepared = []
    ranked = rank_targets(campaign)
    for target, entry in zip(ranked, plan["kernels_to_optimize"]):
        if target.spec_locator is None:
            raise _fail(
                campaign.source,
                f"target {target.name!r}.spec_locator",
                "is required for campaign preparation",
            )
        try:
            spec = load_spec(target.spec_locator)
        except SpecLoadError as exc:
            raise _fail(
                campaign.source,
                f"target {target.name!r}.spec_locator",
                str(exc),
            ) from exc
        if spec.name != target.operation:
            raise _fail(
                campaign.source,
                f"target {target.name!r}.operation",
                f"{target.operation!r} does not match spec name {spec.name!r}",
            )
        starter = spec.starter_kernel("triton")
        if starter is None:
            raise _fail(
                campaign.source,
                f"target {target.name!r}",
                "spec does not declare a Triton starter kernel",
            )
        destination = Path(entry["file"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            starter.read_text(encoding="utf-8"), encoding="utf-8"
        )
        temporary.replace(destination)
        prepared.append(
            {
                "rank": entry["rank"],
                "target_name": target.name,
                "operation": target.operation,
                "candidate": str(destination),
                "spec_locator": target.spec_locator,
                "status": "prepared",
            }
        )

    receipt = {
        "schema_version": 1,
        "status": "prepared",
        "campaign_source": campaign.source,
        "optimization_plan": str(workspace / "optimization_plan.json"),
        "targets": prepared,
        "next_command": "Follow program.md with orchestrate.py next and bench.py",
    }
    receipt_path = workspace / "campaign_receipt.json"
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(receipt_path)
    return receipt
