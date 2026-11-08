"""Local, declared benchmark identity; this module never probes a server."""

from __future__ import annotations

import json

# Dotted leaf names are also the vocabulary for --arm-delta. Additional JSON
# fields are preserved and compared, so adding evidence cannot hide a change.
STRING_FIELDS = (
    "hardware.gpu_name", "hardware.driver", "image.digest", "engine.version",
    "model.id", "model.revision", "speculation.method", "kv_dtype", "arm.label",
)
POSITIVE_INT_FIELDS = (
    "hardware.gpu_count", "hardware.sm_count", "parallelism.tp",
    "parallelism.dcp", "parallelism.ep",
)


def flatten_identity(identity: dict, prefix: str = "") -> dict:
    """Flatten object leaves, preserving empty objects and exact key boundaries."""
    flat = {}
    for key, value in identity.items():
        if not isinstance(key, str) or not key or "." in key or "," in key:
            raise ValueError("identity keys must be nonempty and contain no dots or commas")
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            flat.update(flatten_identity(value, path))
        else:
            flat[path] = value
    return flat


def validate_identity(identity: object) -> dict:
    """Require complete identity data; null is reserved for an absent identity."""
    if not isinstance(identity, dict):
        raise ValueError("identity must be a JSON object")
    try:
        json.dumps(identity, allow_nan=False)
    except (ValueError, TypeError):
        raise ValueError("identity must contain finite JSON values") from None
    flat = flatten_identity(identity)
    bad = [key for key in STRING_FIELDS
           if not isinstance(flat.get(key), str) or not flat[key].strip()]
    bad += [key for key in POSITIVE_INT_FIELDS
            if type(flat.get(key)) is not int or flat[key] < 1]
    if type(flat.get("speculation.k")) is not int or flat["speculation.k"] < 0:
        bad.append("speculation.k")
    draft = flat.get("speculation.draft_revision")
    if ("speculation.draft_revision" not in flat
            or (draft is not None and (not isinstance(draft, str) or not draft.strip()))):
        bad.append("speculation.draft_revision")
    if bad:
        raise ValueError("missing or invalid identity keys: " + ", ".join(sorted(bad)))
    return identity


def identity_differences(a: dict, b: dict) -> list[str]:
    """Name every changed leaf, including additions, removals and type changes."""
    left, right = flatten_identity(a), flatten_identity(b)
    return sorted(key for key in left.keys() | right.keys()
                  if key not in left or key not in right
                  or json.dumps(left[key], sort_keys=True) != json.dumps(right[key], sort_keys=True))
