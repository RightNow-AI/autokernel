"""Universal profiling and graph-region discovery contracts."""

from .fingerprint import fingerprint_payload, graph_fingerprint
from .fx_capture import (
    CaptureResult,
    RegionCaptureSession,
    capture_callable_region,
    capture_model_regions,
    capture_module_region,
)
from .profiler_export import load_profiler_export, profiler_export_to_report
from .profiler_parse import parse_key_averages_rows
from .ranking import (
    DEFAULT_IMPACT_FLOOR,
    DEFAULT_PROMOTION_TARGET,
    RankedCandidate,
    classify_pattern_family,
    optimistic_e2e_improvement,
    rank_operators,
    rank_regions,
)
from .safety import (
    ALLOWED_ATEN_OPS,
    is_region_safe,
    normalize_op_name,
    reject_region,
)
from .types import (
    DISCOVERY_SCHEMA_VERSION,
    DiscoveryError,
    DiscoveryReport,
    GraphBreakRecord,
    GraphRegion,
    OperatorHotspot,
    TensorMeta,
    UnsupportedOpRecord,
    load_discovery_report,
    write_discovery_report,
)

__all__ = [
    "ALLOWED_ATEN_OPS",
    "DEFAULT_IMPACT_FLOOR",
    "DEFAULT_PROMOTION_TARGET",
    "DISCOVERY_SCHEMA_VERSION",
    "CaptureResult",
    "DiscoveryError",
    "DiscoveryReport",
    "GraphBreakRecord",
    "GraphRegion",
    "OperatorHotspot",
    "RankedCandidate",
    "RegionCaptureSession",
    "TensorMeta",
    "UnsupportedOpRecord",
    "capture_callable_region",
    "capture_model_regions",
    "capture_module_region",
    "classify_pattern_family",
    "fingerprint_payload",
    "graph_fingerprint",
    "is_region_safe",
    "load_discovery_report",
    "load_profiler_export",
    "normalize_op_name",
    "optimistic_e2e_improvement",
    "parse_key_averages_rows",
    "profiler_export_to_report",
    "rank_operators",
    "rank_regions",
    "reject_region",
    "write_discovery_report",
]
