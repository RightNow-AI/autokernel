"""Public specification types for kernel operations.

A :class:`KernelSpec` is the single source of truth for one benchmarkable
operation: its reference implementation, deterministic input generator, sizes,
dtypes, tolerances, edge cases, performance accounting, extraction metadata and
starter kernels.

Design constraints:

* dtypes are canonical strings (``float16``, ``bfloat16``, ``float32``) so a
  specification can be inspected on a CPU-only machine; translation to
  ``torch.dtype`` happens in runtime code only;
* importing this module must not import ``torch`` and must not touch a GPU;
* every validation error names the specification and the offending field.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .accounting import Expression
from .dtypes import CANONICAL_DTYPES, is_canonical_dtype

__all__ = [
    "STANDARD_SIZE_LABELS",
    "BytesFn",
    "EdgeCase",
    "FlopsFn",
    "InputMap",
    "KernelSpec",
    "SizeMap",
    "SpecValidationError",
    "Tolerance",
    "validate_spec",
]

#: Mapping of input name -> tensor (or other value) handed to the candidate.
InputMap = Mapping[str, Any]

#: Mapping of size key -> dimension, e.g. ``{"M": 1024, "N": 1024}``.
SizeMap = Mapping[str, int]

FlopsFn = Callable[[SizeMap], "int | float"]
BytesFn = Callable[[SizeMap, int], "int | float"]

#: Size labels every specification must define so the CLI can select a size
#: without operation-specific knowledge.
STANDARD_SIZE_LABELS: tuple[str, ...] = ("small", "medium", "large")

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SpecValidationError(ValueError):
    """Raised when a kernel specification is malformed.

    The message always identifies the specification and the invalid field.
    """


def _fail(spec_name: object, field_name: str, message: str) -> "SpecValidationError":
    label = spec_name if isinstance(spec_name, str) and spec_name else "<unnamed spec>"
    return SpecValidationError(f"kernel spec {label!r}: field {field_name!r}: {message}")


@dataclass(frozen=True)
class Tolerance:
    """Absolute and relative tolerance for one dtype.

    Both values must be finite and non-negative. A NaN or infinite tolerance
    would make every comparison pass and silently disable the correctness
    gate, so it is rejected at construction time.
    """

    atol: float
    rtol: float

    def __post_init__(self) -> None:
        for field_name in ("atol", "rtol"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SpecValidationError(
                    f"Tolerance.{field_name} must be a number, got {value!r}"
                )
            if value != value:  # NaN
                raise SpecValidationError(f"Tolerance.{field_name} must not be NaN")
            if math.isinf(value):
                raise SpecValidationError(
                    f"Tolerance.{field_name} must be finite, got {value!r}"
                )
            if value < 0:
                raise SpecValidationError(
                    f"Tolerance.{field_name} must be non-negative, got {value!r}"
                )
        object.__setattr__(self, "atol", float(self.atol))
        object.__setattr__(self, "rtol", float(self.rtol))

    def as_dict(self) -> dict[str, float]:
        """Return ``{"atol": ..., "rtol": ...}`` for legacy call sites."""
        return {"atol": self.atol, "rtol": self.rtol}


@dataclass(frozen=True)
class EdgeCase:
    """One adversarial or awkward shape that must stay covered.

    ``dtype`` of ``None`` means "the specification's first dtype".
    ``input_transform`` may post-process the generated inputs, e.g. to force a
    degenerate value distribution.
    """

    name: str
    size: SizeMap
    dtype: str | None = None
    seed: int = 42
    input_transform: Callable[[InputMap], InputMap] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", dict(self.size))


@dataclass(frozen=True, kw_only=True)
class KernelSpec:
    """Everything the harness needs to benchmark and extract one operation.

    Args:
        name: identifier-like operation name, unique within a registry.
        reference_fn: PyTorch ground truth, called as ``reference_fn(**inputs)``.
        input_generator: ``(size, dtype, device, seed) -> InputMap``, deterministic
            for a fixed seed.
        sizes: ordered label -> size mapping. Must contain ``small``, ``medium``
            and ``large``. Accepts a mapping or a sequence of ``(label, size)``
            pairs (which is also how duplicate labels are detected).
        dtypes: canonical dtype names to sweep, in benchmark order. The first
            entry is the primary dtype.
        tolerances: dtype name -> :class:`Tolerance`. Must cover every declared
            dtype; extra entries are allowed.
        edge_cases: non-power-of-two and otherwise awkward cases.
        flops_fn: ``(size) -> FLOPs``. Prefer an accounting
            :class:`~autokernel.specs.accounting.Expression` so extraction can
            serialize it.
        bytes_fn: ``(size, dt_bytes) -> bytes moved``.
        shape_keys: canonical size keys, in display order. Derived from ``sizes``
            when omitted.
        shape_aliases: external key name -> canonical key, used to parse shape
            strings coming from the profiler.
        starter_kernels: backend name -> starter kernel file.
        speedup_estimate: human-readable extraction hint, e.g. ``"2-3x"``.
        default_shape: extraction fallback when a profiled shape cannot be
            parsed. Defaults to the ``large`` size.
    """

    name: str
    reference_fn: Callable[..., Any]
    input_generator: Callable[..., InputMap]
    sizes: Mapping[str, SizeMap] | Sequence[tuple[str, SizeMap]]
    dtypes: Iterable[str]
    tolerances: Mapping[str, Any]
    flops_fn: FlopsFn
    bytes_fn: BytesFn
    edge_cases: Iterable[EdgeCase] = ()
    shape_keys: Iterable[str] = ()
    shape_aliases: Mapping[str, str] | Sequence[tuple[str, str]] = ()
    starter_kernels: Mapping[str, Any] | Sequence[tuple[str, Any]] = ()
    speedup_estimate: str | None = None
    default_shape: SizeMap | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sizes", _normalize_sizes(self.name, self.sizes))
        object.__setattr__(self, "dtypes", _normalize_dtypes(self.name, self.dtypes))
        object.__setattr__(
            self, "tolerances", _normalize_tolerances(self.name, self.tolerances)
        )
        object.__setattr__(self, "edge_cases", tuple(self.edge_cases))
        object.__setattr__(
            self, "shape_keys", _normalize_shape_keys(self.name, self.shape_keys, self.sizes)
        )
        object.__setattr__(
            self, "shape_aliases", _normalize_aliases(self.name, self.shape_aliases)
        )
        object.__setattr__(
            self, "starter_kernels", _normalize_starters(self.name, self.starter_kernels)
        )
        if self.default_shape is not None:
            object.__setattr__(self, "default_shape", dict(self.default_shape))
        # Structural validation happens eagerly. The small/medium/large
        # requirement is a *registration* rule (see KernelRegistry.register) so
        # tools can still build narrower specifications for inspection.
        validate_spec(self, require_standard_sizes=False)

    # -- convenience accessors -----------------------------------------
    @property
    def primary_dtype(self) -> str:
        """The first declared dtype, used for benchmarking and smoke tests."""
        return self.dtypes[0]

    def size_items(self) -> tuple[tuple[str, dict[str, int]], ...]:
        """Return ``((label, size), ...)`` in declaration order."""
        return tuple((label, dict(size)) for label, size in self.sizes.items())

    def tolerance_for(self, dtype: str) -> Tolerance:
        """Return the tolerance declared for a canonical dtype name."""
        from .dtypes import canonical_dtype_name

        name = canonical_dtype_name(dtype)
        try:
            return self.tolerances[name]
        except KeyError as exc:
            raise SpecValidationError(
                f"kernel spec {self.name!r}: no tolerance declared for dtype {name!r}"
            ) from exc

    def extraction_shape(self) -> dict[str, int]:
        """Shape used by extraction when a profiled shape cannot be parsed."""
        if self.default_shape is not None:
            return dict(self.default_shape)
        if "large" in self.sizes:
            return dict(self.sizes["large"])
        last_label = next(reversed(list(self.sizes)))
        return dict(self.sizes[last_label])

    def starter_kernel(self, backend: str = "triton") -> Path | None:
        """Return the starter kernel path for a backend, or None if undeclared."""
        path = self.starter_kernels.get(backend)
        return Path(path) if path is not None else None


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_sizes(
    name: object, sizes: Mapping[str, SizeMap] | Sequence[tuple[str, SizeMap]]
) -> dict[str, dict[str, int]]:
    if isinstance(sizes, Mapping):
        pairs: list[tuple[str, Any]] = list(sizes.items())
    elif isinstance(sizes, (str, bytes)) or not isinstance(sizes, Iterable):
        raise _fail(name, "sizes", f"expected a mapping or pairs, got {type(sizes).__name__}")
    else:
        pairs = []
        for item in sizes:
            if not isinstance(item, Sequence) or len(item) != 2:
                raise _fail(name, "sizes", f"expected (label, size) pairs, got {item!r}")
            pairs.append((item[0], item[1]))

    out: dict[str, dict[str, int]] = {}
    for label, size in pairs:
        if not isinstance(label, str) or not label:
            raise _fail(name, "sizes", f"size label must be a non-empty string, got {label!r}")
        if label in out:
            raise _fail(name, "sizes", f"duplicate size label {label!r}")
        if not isinstance(size, Mapping) or not size:
            raise _fail(name, "sizes", f"size {label!r} must be a non-empty mapping")
        normalized: dict[str, int] = {}
        for key, value in size.items():
            if not isinstance(key, str) or not key:
                raise _fail(name, "sizes", f"size {label!r} has a non-string key {key!r}")
            if isinstance(value, bool) or not isinstance(value, int):
                raise _fail(
                    name, "sizes", f"size {label!r} key {key!r} must be an int, got {value!r}"
                )
            if value <= 0:
                raise _fail(
                    name, "sizes", f"size {label!r} key {key!r} must be positive, got {value!r}"
                )
            normalized[key] = value
        out[label] = normalized
    return out


def _normalize_dtypes(name: object, dtypes: Iterable[str]) -> tuple[str, ...]:
    if isinstance(dtypes, (str, bytes)) or not isinstance(dtypes, Iterable):
        raise _fail(name, "dtypes", f"expected an iterable of dtype names, got {dtypes!r}")
    out: list[str] = []
    for dtype in dtypes:
        if not is_canonical_dtype(dtype):
            raise _fail(
                name,
                "dtypes",
                f"unknown dtype {dtype!r}; expected one of {', '.join(CANONICAL_DTYPES)}",
            )
        if dtype in out:
            raise _fail(name, "dtypes", f"duplicate dtype {dtype!r}")
        out.append(str(dtype))
    return tuple(out)


def _normalize_tolerances(name: object, tolerances: Mapping[str, Any]) -> dict[str, Tolerance]:
    if not isinstance(tolerances, Mapping):
        raise _fail(
            name, "tolerances", f"expected a mapping, got {type(tolerances).__name__}"
        )
    out: dict[str, Tolerance] = {}
    for dtype, tol in tolerances.items():
        if not is_canonical_dtype(dtype):
            raise _fail(
                name,
                "tolerances",
                f"unknown dtype key {dtype!r}; expected one of {', '.join(CANONICAL_DTYPES)}",
            )
        if isinstance(tol, Tolerance):
            out[str(dtype)] = tol
        elif isinstance(tol, Mapping):
            missing = {"atol", "rtol"} - set(tol)
            if missing:
                raise _fail(
                    name,
                    "tolerances",
                    f"dtype {dtype!r} is missing {sorted(missing)}",
                )
            unexpected = set(tol) - {"atol", "rtol"}
            if unexpected:
                raise _fail(
                    name,
                    "tolerances",
                    f"dtype {dtype!r} has unexpected keys {sorted(unexpected)}",
                )
            try:
                out[str(dtype)] = Tolerance(atol=tol["atol"], rtol=tol["rtol"])
            except SpecValidationError as exc:
                raise _fail(name, "tolerances", f"dtype {dtype!r}: {exc}") from exc
        else:
            raise _fail(
                name,
                "tolerances",
                f"dtype {dtype!r} must map to a Tolerance, got {type(tol).__name__}",
            )
    return out


def _normalize_shape_keys(
    name: object, shape_keys: Iterable[str], sizes: Mapping[str, SizeMap]
) -> tuple[str, ...]:
    keys = tuple(shape_keys)
    if not keys:
        # Derive from the first declared size, preserving its key order.
        for size in sizes.values():
            return tuple(size)
        return ()
    out: list[str] = []
    for key in keys:
        if not isinstance(key, str) or not key:
            raise _fail(name, "shape_keys", f"shape key must be a non-empty string, got {key!r}")
        if key in out:
            raise _fail(name, "shape_keys", f"duplicate shape key {key!r}")
        out.append(key)
    return tuple(out)


def _normalize_aliases(
    name: object, aliases: Mapping[str, str] | Sequence[tuple[str, str]]
) -> dict[str, str]:
    if isinstance(aliases, Mapping):
        pairs: list[tuple[Any, Any]] = list(aliases.items())
    elif isinstance(aliases, (str, bytes)) or not isinstance(aliases, Iterable):
        raise _fail(
            name, "shape_aliases", f"expected a mapping or pairs, got {type(aliases).__name__}"
        )
    else:
        pairs = []
        for item in aliases:
            if not isinstance(item, Sequence) or len(item) != 2:
                raise _fail(name, "shape_aliases", f"expected (alias, key) pairs, got {item!r}")
            pairs.append((item[0], item[1]))

    out: dict[str, str] = {}
    for alias, canonical in pairs:
        if not isinstance(alias, str) or not alias:
            raise _fail(name, "shape_aliases", f"alias must be a non-empty string, got {alias!r}")
        if not isinstance(canonical, str) or not canonical:
            raise _fail(
                name,
                "shape_aliases",
                f"alias {alias!r} must resolve to a non-empty string, got {canonical!r}",
            )
        if alias in out and out[alias] != canonical:
            raise _fail(
                name,
                "shape_aliases",
                f"alias {alias!r} resolves inconsistently to {out[alias]!r} and {canonical!r}",
            )
        out[alias] = canonical
    return out


def _normalize_starters(
    name: object, starters: Mapping[str, Any] | Sequence[tuple[str, Any]]
) -> dict[str, Path]:
    if isinstance(starters, Mapping):
        pairs: list[tuple[Any, Any]] = list(starters.items())
    elif isinstance(starters, (str, bytes)) or not isinstance(starters, Iterable):
        raise _fail(
            name,
            "starter_kernels",
            f"expected a mapping or pairs, got {type(starters).__name__}",
        )
    else:
        pairs = []
        for item in starters:
            if not isinstance(item, Sequence) or len(item) != 2:
                raise _fail(
                    name, "starter_kernels", f"expected (backend, path) pairs, got {item!r}"
                )
            pairs.append((item[0], item[1]))

    out: dict[str, Path] = {}
    for backend, path in pairs:
        if not isinstance(backend, str) or not backend:
            raise _fail(
                name, "starter_kernels", f"backend must be a non-empty string, got {backend!r}"
            )
        if backend in out:
            raise _fail(name, "starter_kernels", f"duplicate backend {backend!r}")
        if not isinstance(path, (str, Path)):
            raise _fail(
                name,
                "starter_kernels",
                f"backend {backend!r} must map to a path, got {type(path).__name__}",
            )
        out[backend] = Path(path)
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_spec(
    spec: KernelSpec,
    *,
    require_standard_sizes: bool = True,
    check_starter_files: bool = True,
) -> None:
    """Validate a specification, raising :class:`SpecValidationError` on failure.

    Args:
        spec: the specification to check.
        require_standard_sizes: require ``small``, ``medium`` and ``large`` size
            labels so the CLI can pick a size for any operation.
        check_starter_files: require every declared starter kernel file to exist.
    """
    if not isinstance(spec, KernelSpec):
        raise SpecValidationError(
            f"expected a KernelSpec, got {type(spec).__name__}"
        )

    name = spec.name
    if not isinstance(name, str) or not name:
        raise _fail(name, "name", "operation name must be a non-empty string")
    if not _NAME_RE.match(name):
        raise _fail(
            name,
            "name",
            "operation name must be identifier-like (letters, digits, underscore; "
            "not starting with a digit)",
        )

    if not callable(spec.reference_fn):
        raise _fail(name, "reference_fn", "reference function must be callable")
    if not callable(spec.input_generator):
        raise _fail(name, "input_generator", "input generator must be callable")
    if not callable(spec.flops_fn):
        raise _fail(name, "flops_fn", "FLOP accounting must be callable")
    if not callable(spec.bytes_fn):
        raise _fail(name, "bytes_fn", "byte accounting must be callable")

    if not spec.sizes:
        raise _fail(name, "sizes", "at least one size must be declared")
    if require_standard_sizes:
        missing = [label for label in STANDARD_SIZE_LABELS if label not in spec.sizes]
        if missing:
            raise _fail(
                name,
                "sizes",
                f"missing required size label(s) {missing}; declared {sorted(spec.sizes)}",
            )

    if not spec.dtypes:
        raise _fail(name, "dtypes", "at least one dtype must be declared")

    missing_tolerances = [d for d in spec.dtypes if d not in spec.tolerances]
    if missing_tolerances:
        raise _fail(
            name,
            "tolerances",
            f"missing tolerance for declared dtype(s) {missing_tolerances}",
        )

    shape_keys = set(spec.shape_keys)
    if not shape_keys:
        raise _fail(name, "shape_keys", "at least one shape key must be declared")
    for label, size in spec.sizes.items():
        extra = sorted(set(size) - shape_keys)
        missing = sorted(shape_keys - set(size))
        if extra or missing:
            raise _fail(
                name,
                "sizes",
                f"size {label!r} keys {sorted(size)} do not match shape_keys "
                f"{sorted(shape_keys)} (unexpected={extra}, missing={missing})",
            )

    seen_edges: set[str] = set()
    for edge in spec.edge_cases:
        if not isinstance(edge, EdgeCase):
            raise _fail(name, "edge_cases", f"expected EdgeCase, got {type(edge).__name__}")
        if not edge.name:
            raise _fail(name, "edge_cases", "edge case name must be non-empty")
        if edge.name in seen_edges:
            raise _fail(name, "edge_cases", f"duplicate edge case name {edge.name!r}")
        seen_edges.add(edge.name)
        extra = sorted(set(edge.size) - shape_keys)
        missing = sorted(shape_keys - set(edge.size))
        if extra or missing:
            raise _fail(
                name,
                "edge_cases",
                f"edge case {edge.name!r} keys {sorted(edge.size)} do not match shape_keys "
                f"{sorted(shape_keys)} (unexpected={extra}, missing={missing})",
            )
        if edge.dtype is not None:
            if not is_canonical_dtype(edge.dtype):
                raise _fail(
                    name, "edge_cases", f"edge case {edge.name!r} has unknown dtype {edge.dtype!r}"
                )
            if edge.dtype not in spec.dtypes:
                raise _fail(
                    name,
                    "edge_cases",
                    f"edge case {edge.name!r} dtype {edge.dtype!r} is not declared in dtypes "
                    f"{list(spec.dtypes)}",
                )
        if isinstance(edge.seed, bool) or not isinstance(edge.seed, int):
            raise _fail(
                name, "edge_cases", f"edge case {edge.name!r} seed must be an int, got {edge.seed!r}"
            )
        if edge.input_transform is not None and not callable(edge.input_transform):
            raise _fail(
                name, "edge_cases", f"edge case {edge.name!r} input_transform must be callable"
            )

    for alias, canonical in spec.shape_aliases.items():
        if canonical not in shape_keys:
            raise _fail(
                name,
                "shape_aliases",
                f"alias {alias!r} resolves to {canonical!r}, which is not a shape key "
                f"{sorted(shape_keys)}",
            )

    if isinstance(spec.flops_fn, Expression):
        unknown = sorted(spec.flops_fn.size_keys() - shape_keys)
        if unknown:
            raise _fail(
                name, "flops_fn", f"references unknown size key(s) {unknown}"
            )
        if spec.flops_fn.uses_dtype_bytes():
            raise _fail(name, "flops_fn", "FLOP accounting must not depend on dtype bytes")
    if isinstance(spec.bytes_fn, Expression):
        unknown = sorted(spec.bytes_fn.size_keys() - shape_keys)
        if unknown:
            raise _fail(name, "bytes_fn", f"references unknown size key(s) {unknown}")

    if spec.default_shape is not None:
        extra = sorted(set(spec.default_shape) - shape_keys)
        missing = sorted(shape_keys - set(spec.default_shape))
        if extra or missing:
            raise _fail(
                name,
                "default_shape",
                f"keys {sorted(spec.default_shape)} do not match shape_keys "
                f"{sorted(shape_keys)} (unexpected={extra}, missing={missing})",
            )

    if check_starter_files:
        for backend, path in spec.starter_kernels.items():
            if not Path(path).is_file():
                raise _fail(
                    name,
                    "starter_kernels",
                    f"backend {backend!r} starter kernel not found: {path}",
                )

    if spec.speedup_estimate is not None and not isinstance(spec.speedup_estimate, str):
        raise _fail(
            name,
            "speedup_estimate",
            f"expected a string or None, got {type(spec.speedup_estimate).__name__}",
        )
