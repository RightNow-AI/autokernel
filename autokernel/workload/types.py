"""Versioned FastVideo workload manifest contract.

A workload describes a representative generation job that MotionKernel and a
FastVideo launcher can execute without model-specific Python callables. It is
the shared input for baseline measurement, profiling, candidate validation,
and end-to-end native-versus-optimized comparisons.

Manifests must never embed tensor values, credentials, weights, or generated
user content. Prompt text is allowed only as an explicit generation input; use
``prompt_file`` when the prompt should stay out of the manifest body.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

WORKLOAD_SCHEMA_VERSION = 1

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "workload_id",
    "description",
    "model",
    "task",
    "prompt",
    "prompt_file",
    "sampling",
    "runtime",
    "measurement",
    "parity",
    "performance",
    "mode_env",
    "tags",
}
_MODEL_FIELDS = {
    "model_id",
    "revision",
    "trust_remote_code",
}
_SAMPLING_FIELDS = {
    "height",
    "width",
    "num_frames",
    "num_inference_steps",
    "guidance_scale",
    "seed",
    "fps",
    "dtype",
    "attention_backend",
}
_RUNTIME_FIELDS = {
    "num_gpus",
    "use_fsdp_inference",
    "dit_cpu_offload",
    "vae_cpu_offload",
    "text_encoder_cpu_offload",
    "image_encoder_cpu_offload",
    "pin_cpu_memory",
    "distributed_executor_backend",
    "tp_size",
    "sp_size",
}
_MEASUREMENT_FIELDS = {
    "warmups",
    "runs",
    "save_frames",
    "save_video",
}
_PARITY_FIELDS = {
    "policy",
    "atol",
    "rtol",
}
_PERFORMANCE_FIELDS = {
    "min_end_to_end_speedup",
    "max_peak_memory_regression",
}
_MODE_ENV_FIELDS = {
    "native",
    "optimized",
}

_TASKS = {"t2v", "i2v", "t2i", "i2i"}
_PARITY_POLICIES = {"byte_equal", "tolerance", "frames_only"}
_DTYPES = {
    "float16",
    "fp16",
    "bfloat16",
    "bf16",
    "float32",
    "fp32",
}
_WORKLOAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FORBIDDEN_KEYS = {
    "credential",
    "credentials",
    "data",
    "password",
    "secret",
    "secrets",
    "tensor_values",
    "token",
    "values",
    "weights",
    "activations",
}
FORBIDDEN_METADATA_KEYS = frozenset(_FORBIDDEN_KEYS)


class WorkloadError(ValueError):
    """Raised when a workload manifest is malformed or unsafe."""


def _fail(source: object, location: str, message: str) -> WorkloadError:
    return WorkloadError(f"workload {source!r}: {location}: {message}")


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


def _optional_text(
    value: Any, source: object, location: str
) -> str | None:
    if value is None:
        return None
    return _text(value, source, location)


def _bool(value: Any, source: object, location: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise _fail(source, location, "must be a bool")
    return value


def _positive_int(value: Any, source: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _fail(source, location, "must be a positive integer")
    return value


def _non_negative_int(value: Any, source: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _fail(source, location, "must be a non-negative integer")
    return value


def _finite_number(
    value: Any,
    source: object,
    location: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(source, location, "must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise _fail(source, location, "must be a finite number")
    if minimum is not None and number < minimum:
        raise _fail(source, location, f"must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise _fail(source, location, f"must be <= {maximum}")
    return number


def _optional_positive_int(
    value: Any, source: object, location: str
) -> int | None:
    if value is None:
        return None
    return _positive_int(value, source, location)


def _string_map(
    value: Any, source: object, location: str
) -> dict[str, str]:
    raw = _mapping(value, source, location)
    result: dict[str, str] = {}
    for key, item in raw.items():
        result[key] = _text(item, source, f"{location}.{key}")
    return result


@dataclass(frozen=True)
class ModelRef:
    """FastVideo-resolvable model identity."""

    model_id: str
    revision: str | None = None
    trust_remote_code: bool = False

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object, location: str
    ) -> "ModelRef":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _MODEL_FIELDS, source, location)
        return cls(
            model_id=_text(raw.get("model_id"), source, f"{location}.model_id"),
            revision=_optional_text(
                raw.get("revision"), source, f"{location}.revision"
            ),
            trust_remote_code=_bool(
                raw.get("trust_remote_code"),
                source,
                f"{location}.trust_remote_code",
                False,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"model_id": self.model_id}
        if self.revision is not None:
            payload["revision"] = self.revision
        if self.trust_remote_code:
            payload["trust_remote_code"] = True
        return payload


@dataclass(frozen=True)
class SamplingSpec:
    """Declarative generation parameters for a representative workload."""

    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    seed: int
    fps: int | None = None
    dtype: str | None = None
    attention_backend: str | None = None

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object, location: str
    ) -> "SamplingSpec":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _SAMPLING_FIELDS, source, location)
        dtype = _optional_text(raw.get("dtype"), source, f"{location}.dtype")
        if dtype is not None and dtype.lower() not in _DTYPES:
            raise _fail(
                source,
                f"{location}.dtype",
                f"must be one of {sorted(_DTYPES)}",
            )
        return cls(
            height=_positive_int(raw.get("height"), source, f"{location}.height"),
            width=_positive_int(raw.get("width"), source, f"{location}.width"),
            num_frames=_positive_int(
                raw.get("num_frames"), source, f"{location}.num_frames"
            ),
            num_inference_steps=_positive_int(
                raw.get("num_inference_steps"),
                source,
                f"{location}.num_inference_steps",
            ),
            guidance_scale=_finite_number(
                raw.get("guidance_scale"),
                source,
                f"{location}.guidance_scale",
                minimum=0.0,
            ),
            seed=_non_negative_int(raw.get("seed"), source, f"{location}.seed"),
            fps=_optional_positive_int(raw.get("fps"), source, f"{location}.fps"),
            dtype=dtype.lower() if dtype is not None else None,
            attention_backend=_optional_text(
                raw.get("attention_backend"),
                source,
                f"{location}.attention_backend",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
        }
        if self.fps is not None:
            payload["fps"] = self.fps
        if self.dtype is not None:
            payload["dtype"] = self.dtype
        if self.attention_backend is not None:
            payload["attention_backend"] = self.attention_backend
        return payload


@dataclass(frozen=True)
class RuntimeSpec:
    """Device and offload settings for the FastVideo generator process."""

    num_gpus: int = 1
    use_fsdp_inference: bool = False
    dit_cpu_offload: bool = False
    vae_cpu_offload: bool = False
    text_encoder_cpu_offload: bool = True
    image_encoder_cpu_offload: bool = True
    pin_cpu_memory: bool = False
    distributed_executor_backend: str | None = None
    tp_size: int | None = None
    sp_size: int | None = None

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object, location: str
    ) -> "RuntimeSpec":
        if raw_value is None:
            return cls()
        raw = _mapping(raw_value, source, location)
        _unknown_fields(raw, _RUNTIME_FIELDS, source, location)
        return cls(
            num_gpus=_positive_int(
                raw.get("num_gpus", 1), source, f"{location}.num_gpus"
            ),
            use_fsdp_inference=_bool(
                raw.get("use_fsdp_inference"),
                source,
                f"{location}.use_fsdp_inference",
                False,
            ),
            dit_cpu_offload=_bool(
                raw.get("dit_cpu_offload"),
                source,
                f"{location}.dit_cpu_offload",
                False,
            ),
            vae_cpu_offload=_bool(
                raw.get("vae_cpu_offload"),
                source,
                f"{location}.vae_cpu_offload",
                False,
            ),
            text_encoder_cpu_offload=_bool(
                raw.get("text_encoder_cpu_offload"),
                source,
                f"{location}.text_encoder_cpu_offload",
                True,
            ),
            image_encoder_cpu_offload=_bool(
                raw.get("image_encoder_cpu_offload"),
                source,
                f"{location}.image_encoder_cpu_offload",
                True,
            ),
            pin_cpu_memory=_bool(
                raw.get("pin_cpu_memory"),
                source,
                f"{location}.pin_cpu_memory",
                False,
            ),
            distributed_executor_backend=_optional_text(
                raw.get("distributed_executor_backend"),
                source,
                f"{location}.distributed_executor_backend",
            ),
            tp_size=_optional_positive_int(
                raw.get("tp_size"), source, f"{location}.tp_size"
            ),
            sp_size=_optional_positive_int(
                raw.get("sp_size"), source, f"{location}.sp_size"
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "num_gpus": self.num_gpus,
            "use_fsdp_inference": self.use_fsdp_inference,
            "dit_cpu_offload": self.dit_cpu_offload,
            "vae_cpu_offload": self.vae_cpu_offload,
            "text_encoder_cpu_offload": self.text_encoder_cpu_offload,
            "image_encoder_cpu_offload": self.image_encoder_cpu_offload,
            "pin_cpu_memory": self.pin_cpu_memory,
        }
        if self.distributed_executor_backend is not None:
            payload["distributed_executor_backend"] = (
                self.distributed_executor_backend
            )
        if self.tp_size is not None:
            payload["tp_size"] = self.tp_size
        if self.sp_size is not None:
            payload["sp_size"] = self.sp_size
        return payload


@dataclass(frozen=True)
class MeasurementSpec:
    """Warmup/run counts and artifact capture preferences."""

    warmups: int = 1
    runs: int = 2
    save_frames: bool = True
    save_video: bool = False

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object, location: str
    ) -> "MeasurementSpec":
        if raw_value is None:
            return cls()
        raw = _mapping(raw_value, source, location)
        _unknown_fields(raw, _MEASUREMENT_FIELDS, source, location)
        return cls(
            warmups=_non_negative_int(
                raw.get("warmups", 1), source, f"{location}.warmups"
            ),
            runs=_positive_int(raw.get("runs", 2), source, f"{location}.runs"),
            save_frames=_bool(
                raw.get("save_frames"),
                source,
                f"{location}.save_frames",
                True,
            ),
            save_video=_bool(
                raw.get("save_video"),
                source,
                f"{location}.save_video",
                False,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "warmups": self.warmups,
            "runs": self.runs,
            "save_frames": self.save_frames,
            "save_video": self.save_video,
        }


@dataclass(frozen=True)
class ParitySpec:
    """Full-output parity policy for native-versus-optimized comparisons."""

    policy: str = "byte_equal"
    atol: float | None = None
    rtol: float | None = None

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object, location: str
    ) -> "ParitySpec":
        if raw_value is None:
            return cls()
        raw = _mapping(raw_value, source, location)
        _unknown_fields(raw, _PARITY_FIELDS, source, location)
        policy = _text(
            raw.get("policy", "byte_equal"), source, f"{location}.policy"
        )
        if policy not in _PARITY_POLICIES:
            raise _fail(
                source,
                f"{location}.policy",
                f"must be one of {sorted(_PARITY_POLICIES)}",
            )
        atol = raw.get("atol")
        rtol = raw.get("rtol")
        return cls(
            policy=policy,
            atol=(
                None
                if atol is None
                else _finite_number(
                    atol, source, f"{location}.atol", minimum=0.0
                )
            ),
            rtol=(
                None
                if rtol is None
                else _finite_number(
                    rtol, source, f"{location}.rtol", minimum=0.0
                )
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"policy": self.policy}
        if self.atol is not None:
            payload["atol"] = self.atol
        if self.rtol is not None:
            payload["rtol"] = self.rtol
        return payload


@dataclass(frozen=True)
class PerformanceSpec:
    """Promotion thresholds for end-to-end model-level evaluation."""

    min_end_to_end_speedup: float = 1.01
    max_peak_memory_regression: float = 0.05

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object, location: str
    ) -> "PerformanceSpec":
        if raw_value is None:
            return cls()
        raw = _mapping(raw_value, source, location)
        _unknown_fields(raw, _PERFORMANCE_FIELDS, source, location)
        return cls(
            min_end_to_end_speedup=_finite_number(
                raw.get("min_end_to_end_speedup", 1.01),
                source,
                f"{location}.min_end_to_end_speedup",
                minimum=1.0,
            ),
            max_peak_memory_regression=_finite_number(
                raw.get("max_peak_memory_regression", 0.05),
                source,
                f"{location}.max_peak_memory_regression",
                minimum=0.0,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_end_to_end_speedup": self.min_end_to_end_speedup,
            "max_peak_memory_regression": self.max_peak_memory_regression,
        }


@dataclass(frozen=True)
class ModeEnvSpec:
    """Optional environment variables applied per launcher mode.

    Values are environment-variable assignments only. Do not put Python
    callables or model-specific code paths here.
    """

    native: Mapping[str, str] | None = None
    optimized: Mapping[str, str] | None = None

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object, location: str
    ) -> "ModeEnvSpec":
        if raw_value is None:
            return cls()
        raw = _mapping(raw_value, source, location)
        _unknown_fields(raw, _MODE_ENV_FIELDS, source, location)
        native = raw.get("native")
        optimized = raw.get("optimized")
        return cls(
            native=(
                None
                if native is None
                else _string_map(native, source, f"{location}.native")
            ),
            optimized=(
                None
                if optimized is None
                else _string_map(optimized, source, f"{location}.optimized")
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.native is not None:
            payload["native"] = dict(self.native)
        if self.optimized is not None:
            payload["optimized"] = dict(self.optimized)
        return payload

    def for_mode(self, mode: str) -> dict[str, str]:
        if mode == "native":
            return dict(self.native or {})
        if mode in {"optimized", "fused", "candidate"}:
            return dict(self.optimized or {})
        raise WorkloadError(f"unknown launcher mode {mode!r}")


@dataclass(frozen=True)
class WorkloadManifest:
    """Validated generation workload shared by MotionKernel and FastVideo."""

    workload_id: str
    model: ModelRef
    task: str
    sampling: SamplingSpec
    prompt: str | None = None
    prompt_file: str | None = None
    description: str | None = None
    runtime: RuntimeSpec | None = None
    measurement: MeasurementSpec | None = None
    parity: ParitySpec | None = None
    performance: PerformanceSpec | None = None
    mode_env: ModeEnvSpec | None = None
    tags: tuple[str, ...] = ()
    source: str = "<memory>"
    schema_version: int = WORKLOAD_SCHEMA_VERSION

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object = "<memory>"
    ) -> "WorkloadManifest":
        raw = _mapping(raw_value, source, "top level", non_empty=True)
        _unknown_fields(raw, _TOP_LEVEL_FIELDS, source, "top level")

        version = raw.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise _fail(source, "schema_version", "must be an integer")
        if version != WORKLOAD_SCHEMA_VERSION:
            raise _fail(
                source,
                "schema_version",
                f"unsupported version {version}; expected {WORKLOAD_SCHEMA_VERSION}",
            )

        workload_id = _text(raw.get("workload_id"), source, "workload_id")
        if not _WORKLOAD_ID_PATTERN.fullmatch(workload_id):
            raise _fail(
                source,
                "workload_id",
                "must be a short identifier of letters, digits, '.', '_', or '-'",
            )

        task = _text(raw.get("task"), source, "task").lower()
        if task not in _TASKS:
            raise _fail(source, "task", f"must be one of {sorted(_TASKS)}")

        prompt = _optional_text(raw.get("prompt"), source, "prompt")
        prompt_file = _optional_text(
            raw.get("prompt_file"), source, "prompt_file"
        )
        if prompt is None and prompt_file is None:
            raise _fail(
                source,
                "prompt",
                "exactly one of prompt or prompt_file is required",
            )
        if prompt is not None and prompt_file is not None:
            raise _fail(
                source,
                "prompt",
                "provide only one of prompt or prompt_file",
            )

        tags_raw = raw.get("tags", [])
        if not isinstance(tags_raw, Sequence) or isinstance(
            tags_raw, (str, bytes)
        ):
            raise _fail(source, "tags", "must be a list of strings")
        tags = tuple(
            _text(tag, source, f"tags[{index}]")
            for index, tag in enumerate(tags_raw)
        )
        if len(tags) != len(set(tags)):
            raise _fail(source, "tags", "contains duplicates")

        return cls(
            workload_id=workload_id,
            model=ModelRef.from_dict(
                raw.get("model"), source=source, location="model"
            ),
            task=task,
            sampling=SamplingSpec.from_dict(
                raw.get("sampling"), source=source, location="sampling"
            ),
            prompt=prompt,
            prompt_file=prompt_file,
            description=_optional_text(
                raw.get("description"), source, "description"
            ),
            runtime=RuntimeSpec.from_dict(
                raw.get("runtime"), source=source, location="runtime"
            ),
            measurement=MeasurementSpec.from_dict(
                raw.get("measurement"), source=source, location="measurement"
            ),
            parity=ParitySpec.from_dict(
                raw.get("parity"), source=source, location="parity"
            ),
            performance=PerformanceSpec.from_dict(
                raw.get("performance"),
                source=source,
                location="performance",
            ),
            mode_env=ModeEnvSpec.from_dict(
                raw.get("mode_env"), source=source, location="mode_env"
            ),
            tags=tags,
            source=str(source),
            schema_version=version,
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "workload_id": self.workload_id,
            "model": self.model.as_dict(),
            "task": self.task,
            "sampling": self.sampling.as_dict(),
            "runtime": (self.runtime or RuntimeSpec()).as_dict(),
            "measurement": (self.measurement or MeasurementSpec()).as_dict(),
            "parity": (self.parity or ParitySpec()).as_dict(),
            "performance": (self.performance or PerformanceSpec()).as_dict(),
        }
        if self.description is not None:
            payload["description"] = self.description
        if self.prompt is not None:
            payload["prompt"] = self.prompt
        if self.prompt_file is not None:
            payload["prompt_file"] = self.prompt_file
        mode_env = (self.mode_env or ModeEnvSpec()).as_dict()
        if mode_env:
            payload["mode_env"] = mode_env
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload

    def resolve_prompt(self, *, base_dir: str | Path | None = None) -> str:
        """Return the prompt text, loading ``prompt_file`` when needed."""
        if self.prompt is not None:
            return self.prompt
        assert self.prompt_file is not None
        path = Path(self.prompt_file)
        if not path.is_absolute():
            root = Path(base_dir) if base_dir is not None else Path(self.source).parent
            path = root / path
        if not path.is_file():
            raise _fail(self.source, "prompt_file", f"not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise _fail(self.source, "prompt_file", "must be non-empty")
        return text

    def generation_request(self, *, base_dir: str | Path | None = None) -> dict[str, Any]:
        """Build a FastVideo ``generate`` request dict from this workload."""
        sampling = self.sampling.as_dict()
        # FastVideo SamplingConfig does not take dtype/attention_backend.
        sampling.pop("dtype", None)
        sampling.pop("attention_backend", None)
        return {
            "prompt": self.resolve_prompt(base_dir=base_dir),
            "sampling": sampling,
            "output": {
                "save_video": (self.measurement or MeasurementSpec()).save_video,
                "return_frames": (self.measurement or MeasurementSpec()).save_frames,
            },
        }

    def generator_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``VideoGenerator.from_pretrained``."""
        runtime = self.runtime or RuntimeSpec()
        kwargs: dict[str, Any] = {
            "num_gpus": runtime.num_gpus,
            "use_fsdp_inference": runtime.use_fsdp_inference,
            "dit_cpu_offload": runtime.dit_cpu_offload,
            "vae_cpu_offload": runtime.vae_cpu_offload,
            "text_encoder_cpu_offload": runtime.text_encoder_cpu_offload,
            "image_encoder_cpu_offload": runtime.image_encoder_cpu_offload,
            "pin_cpu_memory": runtime.pin_cpu_memory,
        }
        if self.model.revision is not None:
            kwargs["revision"] = self.model.revision
        if self.model.trust_remote_code:
            kwargs["trust_remote_code"] = True
        if runtime.distributed_executor_backend is not None:
            kwargs["distributed_executor_backend"] = (
                runtime.distributed_executor_backend
            )
        if runtime.tp_size is not None:
            kwargs["tp_size"] = runtime.tp_size
        if runtime.sp_size is not None:
            kwargs["sp_size"] = runtime.sp_size
        return kwargs


def _load_raw_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised in envs without PyYAML
            raise WorkloadError(
                f"workload {path!s}: YAML support requires PyYAML; "
                "install motionkernel with PyYAML or use a .json manifest"
            ) from exc
        raw = yaml.safe_load(text)
    elif suffix == ".json":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise WorkloadError(
                f"workload {path!s}: JSON: invalid JSON: {exc}"
            ) from exc
    else:
        raise WorkloadError(
            f"workload {path!s}: file: unsupported extension {suffix!r}; "
            "use .yaml, .yml, or .json"
        )
    if not isinstance(raw, Mapping):
        raise WorkloadError(f"workload {path!s}: top level: must be an object")
    return raw


def load_workload(path: str | Path) -> WorkloadManifest:
    """Load and validate a workload manifest without importing torch."""
    file_path = Path(path)
    if not file_path.is_file():
        raise WorkloadError(f"workload {file_path!s}: file: not found")
    raw = _load_raw_mapping(file_path)
    return WorkloadManifest.from_dict(raw, source=str(file_path))


def dump_workload(
    workload: WorkloadManifest,
    path: str | Path,
    *,
    fmt: str | None = None,
) -> None:
    """Write a workload manifest as YAML or JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    format_name = (fmt or output.suffix.lstrip(".")).lower()
    payload = workload.as_dict()
    if format_name in {"yaml", "yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise WorkloadError(
                "YAML support requires PyYAML"
            ) from exc
        text = yaml.safe_dump(
            payload,
            sort_keys=False,
            default_flow_style=False,
        )
    elif format_name == "json":
        text = json.dumps(payload, indent=2) + "\n"
    else:
        raise WorkloadError(f"unsupported dump format {format_name!r}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)
