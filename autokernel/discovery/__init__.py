"""Universal profiling and graph-region discovery contracts."""

from .fingerprint import fingerprint_payload, graph_fingerprint
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
    "DISCOVERY_SCHEMA_VERSION",
    "DiscoveryError",
    "DiscoveryReport",
    "GraphBreakRecord",
    "GraphRegion",
    "OperatorHotspot",
    "TensorMeta",
    "UnsupportedOpRecord",
    "fingerprint_payload",
    "graph_fingerprint",
    "is_region_safe",
    "load_discovery_report",
    "normalize_op_name",
    "reject_region",
    "write_discovery_report",
]
