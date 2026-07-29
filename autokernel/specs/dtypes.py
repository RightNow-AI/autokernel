"""Canonical dtype names used at the specification boundary.

Specifications only ever name dtypes with the canonical strings below, which
keeps discovery serializable and importable without ``torch``. Runtime code
translates to ``torch.dtype`` through :func:`resolve_torch_dtype`.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "CANONICAL_DTYPES",
    "DTYPE_BYTES",
    "canonical_dtype_name",
    "dtype_bytes",
    "is_canonical_dtype",
    "resolve_torch_dtype",
]

#: Ordered tuple of dtype names a specification may declare.
CANONICAL_DTYPES: tuple[str, ...] = ("float16", "bfloat16", "float32")

#: Byte width per canonical dtype. Declared statically so byte accounting works
#: without importing torch.
DTYPE_BYTES: Mapping[str, int] = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}


def is_canonical_dtype(name: object) -> bool:
    """Return True when ``name`` is one of the canonical dtype strings."""
    return isinstance(name, str) and name in CANONICAL_DTYPES


def canonical_dtype_name(dtype: Any) -> str:
    """Normalize ``dtype`` (canonical string or ``torch.dtype``) to a canonical name.

    Raises:
        ValueError: if the dtype is not supported at the specification boundary.
    """
    if is_canonical_dtype(dtype):
        return str(dtype)

    text = str(dtype)
    if text.startswith("torch."):
        text = text[len("torch."):]
    if is_canonical_dtype(text):
        return text

    raise ValueError(
        f"unsupported dtype {dtype!r}; expected one of {', '.join(CANONICAL_DTYPES)}"
    )


def dtype_bytes(dtype: Any) -> int:
    """Return the byte width of a canonical dtype name or ``torch.dtype``."""
    return DTYPE_BYTES[canonical_dtype_name(dtype)]


def resolve_torch_dtype(dtype: Any) -> Any:
    """Translate a canonical dtype name to a ``torch.dtype``.

    ``torch`` is imported lazily so specification discovery stays torch-free.
    Passing a ``torch.dtype`` through is supported and returns it unchanged
    after validation.
    """
    import torch  # local import: keep module import torch-free

    if isinstance(dtype, torch.dtype):
        canonical_dtype_name(dtype)  # validate it is supported
        return dtype

    name = canonical_dtype_name(dtype)
    return getattr(torch, name)
