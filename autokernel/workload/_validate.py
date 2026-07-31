"""Shared JSON-metadata validators for workload and result schemas."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class SchemaError(ValueError):
    """Malformed or unsafe schema payload."""


def fail(kind: str, source: object, location: str, message: str) -> SchemaError:
    return SchemaError(f"{kind} {source!r}: {location}: {message}")


def mapping(
    value: Any,
    source: object,
    location: str,
    *,
    kind: str,
    non_empty: bool = False,
    forbidden_keys: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or (non_empty and not value):
        qualifier = "non-empty " if non_empty else ""
        raise fail(kind, source, location, f"must be a {qualifier}object")
    forbidden = {k.lower() for k in (forbidden_keys or set())}
    for key in value:
        if not isinstance(key, str) or not key:
            raise fail(kind, source, location, "keys must be non-empty strings")
        if key.lower() in forbidden:
            raise fail(
                kind,
                source,
                f"{location}.{key}",
                "content or secret fields are forbidden",
            )
    return value


def text(value: Any, source: object, location: str, *, kind: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise fail(kind, source, location, "must be a non-empty string")
    return value.strip()


def optional_text(
    value: Any, source: object, location: str, *, kind: str
) -> str | None:
    if value is None:
        return None
    return text(value, source, location, kind=kind)


def positive_int(value: Any, source: object, location: str, *, kind: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise fail(kind, source, location, "must be a positive integer")
    return value


def non_negative_int(
    value: Any, source: object, location: str, *, kind: str
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise fail(kind, source, location, "must be a non-negative integer")
    return value


def finite_number(
    value: Any,
    source: object,
    location: str,
    *,
    kind: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise fail(kind, source, location, "must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise fail(kind, source, location, "must be a finite number")
    if minimum is not None and number < minimum:
        raise fail(kind, source, location, f"must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise fail(kind, source, location, f"must be <= {maximum}")
    return number


def finite_non_negative(
    value: Any, source: object, location: str, *, kind: str
) -> float:
    return finite_number(value, source, location, kind=kind, minimum=0.0)


def is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
