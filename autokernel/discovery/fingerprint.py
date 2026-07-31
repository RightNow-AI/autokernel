"""Stable fingerprints for captured graph regions and operator sequences.

Fingerprints are derived only from operation names, tensor signatures, and
safe constants. They must be identical across repeated runs of the same
region and must never incorporate tensor values, prompts, or weights.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float):
            # Stable JSON-friendly finite floats only.
            return float(value)
        return value
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonical(item) for item in value]
    return repr(value)


def fingerprint_payload(payload: Mapping[str, Any], *, length: int = 32) -> str:
    """Hash a JSON-canonical payload into a short hex fingerprint."""
    encoded = json.dumps(
        _canonical(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return digest[:length]


def graph_fingerprint(
    *,
    operations: Sequence[str],
    input_signatures: Sequence[Mapping[str, Any]],
    output_signatures: Sequence[Mapping[str, Any]] | None = None,
    safe_constants: Mapping[str, Any] | None = None,
    parent_module: str | None = None,
) -> str:
    """Fingerprint a pure tensor region from metadata only."""
    payload = {
        "operations": list(operations),
        "inputs": list(input_signatures),
        "outputs": list(output_signatures or ()),
        "constants": dict(safe_constants or {}),
        "parent_module": parent_module or "",
    }
    return fingerprint_payload(payload)
