"""Versioned FastVideo generation workload manifests."""

from .result import (
    RESULT_SCHEMA_VERSION,
    GenerationRunResult,
    classify_end_to_end,
    load_generation_result,
    write_generation_result,
)
from .types import (
    WORKLOAD_SCHEMA_VERSION,
    MeasurementSpec,
    ModeEnvSpec,
    ModelRef,
    ParitySpec,
    PerformanceSpec,
    RuntimeSpec,
    SamplingSpec,
    WorkloadError,
    WorkloadManifest,
    dump_workload,
    load_workload,
)

__all__ = [
    "RESULT_SCHEMA_VERSION",
    "WORKLOAD_SCHEMA_VERSION",
    "GenerationRunResult",
    "MeasurementSpec",
    "ModeEnvSpec",
    "ModelRef",
    "ParitySpec",
    "PerformanceSpec",
    "RuntimeSpec",
    "SamplingSpec",
    "WorkloadError",
    "WorkloadManifest",
    "classify_end_to_end",
    "dump_workload",
    "load_generation_result",
    "load_workload",
    "write_generation_result",
]
