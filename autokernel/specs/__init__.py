"""Kernel specifications: the public description of a benchmarkable operation.

Typical use::

    from autokernel.specs import create_builtin_registry, load_spec

    registry = create_builtin_registry()
    spec = registry.get("matmul")

    external = load_spec("examples/custom_ops/add.py:SPEC", registry=registry)

Importing this package never imports ``torch`` and never initializes a GPU.
"""

from __future__ import annotations

from .accounting import DT_BYTES, Expression, const, serialize_accounting, size
from .dtypes import (
    CANONICAL_DTYPES,
    DTYPE_BYTES,
    canonical_dtype_name,
    dtype_bytes,
    is_canonical_dtype,
    resolve_torch_dtype,
)
from .lazy import LazyCallable, lazy_callable
from .loader import (
    SpecCollisionError,
    SpecLoadError,
    load_spec,
    parse_locator,
    resolve_spec,
)
from .registry import (
    DuplicateSpecError,
    KernelRegistry,
    SpecNotFoundError,
    builtin_spec_names,
    create_builtin_registry,
)
from .types import (
    STANDARD_SIZE_LABELS,
    EdgeCase,
    InputMap,
    KernelSpec,
    SizeMap,
    SpecValidationError,
    Tolerance,
    validate_spec,
)

__all__ = [
    "CANONICAL_DTYPES",
    "DTYPE_BYTES",
    "DT_BYTES",
    "DuplicateSpecError",
    "EdgeCase",
    "Expression",
    "InputMap",
    "KernelRegistry",
    "KernelSpec",
    "LazyCallable",
    "STANDARD_SIZE_LABELS",
    "SizeMap",
    "SpecCollisionError",
    "SpecLoadError",
    "SpecNotFoundError",
    "SpecValidationError",
    "Tolerance",
    "builtin_spec_names",
    "canonical_dtype_name",
    "const",
    "create_builtin_registry",
    "dtype_bytes",
    "is_canonical_dtype",
    "lazy_callable",
    "load_spec",
    "parse_locator",
    "resolve_spec",
    "resolve_torch_dtype",
    "serialize_accounting",
    "size",
    "validate_spec",
]
