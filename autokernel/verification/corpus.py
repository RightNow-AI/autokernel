"""Production shape corpora: versioned JSON benchmark cases.

A shape corpus lets production workloads drive benchmarking without editing
Python source. The corpus contains *metadata only* -- shapes, dtypes, weights
and tags. It must never serialize model activations, weights, prompts or any
user data; the schema below has no field that could carry tensor data, and
unknown fields are rejected.

Schema version 1::

    {
      "schema_version": 1,
      "operation": "custom_affine",
      "cases": [
        {
          "name": "prod-prefill",
          "size": {"rows": 4096, "cols": 1024},
          "dtype": "float16",          // optional; default: spec primary dtype
          "weight": 37,                // optional; default: 1
          "tags": ["production"]       // optional
        }
      ]
    }

Validation happens before any GPU allocation: :func:`load_shape_corpus`
parses and structurally validates the file, and
:func:`validate_corpus_against_spec` checks it against the selected
specification (operation name, shape keys, declared dtypes, resolved
duplicates). All failures raise :class:`CorpusError` with the file path and
the offending case.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..specs.dtypes import is_canonical_dtype
from ..specs.types import KernelSpec

__all__ = [
    "CORPUS_SCHEMA_VERSION",
    "CorpusCase",
    "CorpusError",
    "ShapeCorpus",
    "load_shape_corpus",
    "validate_corpus_against_spec",
    "weighted_aggregate",
]

#: Only schema version accepted by this implementation.
CORPUS_SCHEMA_VERSION = 1

_TOP_LEVEL_KEYS = {"schema_version", "operation", "cases"}
_CASE_KEYS = {"name", "size", "dtype", "weight", "tags"}


class CorpusError(ValueError):
    """Raised when a shape corpus is malformed or incompatible with a spec."""


@dataclass(frozen=True)
class CorpusCase:
    """One production benchmark case (metadata only)."""

    name: str
    size: dict[str, int]
    dtype: str | None = None  # canonical dtype name; None -> spec primary dtype
    weight: int = 1
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShapeCorpus:
    """A validated corpus file."""

    operation: str
    cases: tuple[CorpusCase, ...]
    source: str  # path the corpus was loaded from, for diagnostics
    schema_version: int = CORPUS_SCHEMA_VERSION


def _fail(source: object, message: str) -> CorpusError:
    label = source if source else "<corpus>"
    return CorpusError(f"shape corpus {label!r}: {message}")


def _parse_case(raw: Any, index: int, source: object) -> CorpusCase:
    where = f"case #{index}"
    if not isinstance(raw, Mapping):
        raise _fail(source, f"{where} must be an object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - _CASE_KEYS)
    if unknown:
        raise _fail(source, f"{where} has unknown field(s) {unknown}; "
                            f"allowed: {sorted(_CASE_KEYS)}")

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise _fail(source, f"{where} ('name') must be a non-empty string")
    where = f"case {name!r}"

    size = raw.get("size")
    if not isinstance(size, Mapping) or not size:
        raise _fail(source, f"{where} ('size') must be a non-empty object")
    normalized_size: dict[str, int] = {}
    for key, value in size.items():
        if not isinstance(key, str) or not key:
            raise _fail(source, f"{where} size keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise _fail(
                source, f"{where} size {key!r} must be a positive integer, got {value!r}"
            )
        normalized_size[key] = value

    dtype = raw.get("dtype")
    if dtype is not None and not is_canonical_dtype(dtype):
        raise _fail(source, f"{where} has unknown dtype {dtype!r}")

    weight = raw.get("weight", 1)
    if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
        raise _fail(source, f"{where} ('weight') must be a positive integer, got {weight!r}")

    tags = raw.get("tags", ())
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
        raise _fail(source, f"{where} ('tags') must be a list of strings")
    normalized_tags: list[str] = []
    for tag in tags:
        if not isinstance(tag, str) or not tag:
            raise _fail(source, f"{where} tags must be non-empty strings, got {tag!r}")
        if tag in normalized_tags:
            raise _fail(source, f"{where} has duplicate tag {tag!r}")
        normalized_tags.append(tag)

    return CorpusCase(
        name=name,
        size=normalized_size,
        dtype=dtype,
        weight=weight,
        tags=tuple(normalized_tags),
    )


def load_shape_corpus(path: str | Path) -> ShapeCorpus:
    """Parse and structurally validate a shape corpus file.

    Raises :class:`CorpusError` for a missing file, invalid JSON, an
    unsupported schema version, or any malformed case. The check against a
    specific operation happens separately in
    :func:`validate_corpus_against_spec`.
    """
    source = str(path)
    file_path = Path(path)
    if not file_path.is_file():
        raise _fail(source, "file not found")
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _fail(source, f"invalid JSON: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise _fail(source, f"top level must be an object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        raise _fail(source, f"unknown top-level field(s) {unknown}; "
                            f"allowed: {sorted(_TOP_LEVEL_KEYS)}")

    version = raw.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise _fail(source, "('schema_version') must be an integer")
    if version != CORPUS_SCHEMA_VERSION:
        raise _fail(
            source,
            f"unsupported schema_version {version}; this harness accepts "
            f"{CORPUS_SCHEMA_VERSION}",
        )

    operation = raw.get("operation")
    if not isinstance(operation, str) or not operation:
        raise _fail(source, "('operation') must be a non-empty string")

    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise _fail(source, "('cases') must be a non-empty list")

    cases = tuple(_parse_case(item, index, source) for index, item in enumerate(raw_cases))
    seen: set[str] = set()
    for case in cases:
        if case.name in seen:
            raise _fail(source, f"duplicate case name {case.name!r}")
        seen.add(case.name)

    return ShapeCorpus(operation=operation, cases=cases, source=source)



def validate_corpus_against_spec(corpus: ShapeCorpus, spec: KernelSpec) -> None:
    """Check a loaded corpus against the selected specification.

    Raises :class:`CorpusError` when the operation does not match, a case
    uses shape keys or a dtype the spec does not declare, or two cases
    resolve to the same ``(size, dtype)`` configuration. Runs before any GPU
    allocation, so an invalid corpus never costs device memory.
    """
    if corpus.operation != spec.name:
        raise _fail(
            corpus.source,
            f"corpus operation {corpus.operation!r} does not match the selected "
            f"spec {spec.name!r}",
        )

    shape_keys = set(spec.shape_keys)
    resolved: dict[tuple[tuple[tuple[str, int], ...], str], str] = {}
    for case in corpus.cases:
        extra = sorted(set(case.size) - shape_keys)
        missing = sorted(shape_keys - set(case.size))
        if extra or missing:
            raise _fail(
                corpus.source,
                f"case {case.name!r} size keys {sorted(case.size)} do not match "
                f"shape_keys {sorted(shape_keys)} (unexpected={extra}, missing={missing})",
            )
        dtype = case.dtype if case.dtype is not None else spec.primary_dtype
        if dtype not in spec.dtypes:
            raise _fail(
                corpus.source,
                f"case {case.name!r} dtype {dtype!r} is not declared in dtypes "
                f"{list(spec.dtypes)}",
            )
        key = (tuple(sorted(case.size.items())), dtype)
        if key in resolved:
            raise _fail(
                corpus.source,
                f"cases {resolved[key]!r} and {case.name!r} resolve to the same "
                f"(size, dtype) configuration; merge their weights instead",
            )
        resolved[key] = case.name


def weighted_aggregate(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Weighted latency aggregation, grouped per dtype.

    Args:
        entries: benchmark results, each with ``dtype`` (string), ``weight``
            (positive int), ``kernel_ms`` and ``ref_ms``.

    Returns:
        ``{dtype: {"cases": n, "weight": total, "kernel_ms": weighted,
        "ref_ms": weighted, "speedup": weighted_ref / weighted_kernel}}``.
        Results from different dtypes are never mixed into one aggregate.
    """
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(str(entry["dtype"]), []).append(entry)

    out: dict[str, dict[str, float | int]] = {}
    for dtype, group in groups.items():
        total_weight = sum(int(entry["weight"]) for entry in group)
        kernel_ms = (
            sum(float(entry["kernel_ms"]) * int(entry["weight"]) for entry in group)
            / total_weight
        )
        ref_ms = (
            sum(float(entry["ref_ms"]) * int(entry["weight"]) for entry in group)
            / total_weight
        )
        out[dtype] = {
            "cases": len(group),
            "weight": total_weight,
            "kernel_ms": kernel_ms,
            "ref_ms": ref_ms,
            "speedup": (ref_ms / kernel_ms) if kernel_ms > 0 else 0.0,
        }
    return out

