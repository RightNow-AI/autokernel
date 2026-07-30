"""Serializable arithmetic expressions for FLOP and byte accounting.

Performance accounting used to live in two places: real lambdas in ``bench.py``
and Python source strings in ``extract.py``. Both are replaced by a small
expression tree that

* evaluates like a normal callable (``expr(size)`` / ``expr(size, dt_bytes)``),
* serializes back to a safe numeric source expression for generated kernel
  files, and
* reports which size keys it references so specifications can be validated.

Nothing here evaluates untrusted text: :meth:`Expression.to_source` only emits
numbers, ``s["key"]`` lookups, ``dt_bytes`` and arithmetic operators, and the
loader never calls ``eval`` on specification content.

Example:
    >>> flops = 2 * size("M") * size("N") * size("K")
    >>> flops({"M": 2, "N": 3, "K": 4})
    48
    >>> flops.to_source()
    '2 * s[\'M\'] * s[\'N\'] * s[\'K\']'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Union

__all__ = [
    "DT_BYTES",
    "BinaryOp",
    "Constant",
    "DTypeBytes",
    "Expression",
    "SizeKey",
    "const",
    "serialize_accounting",
    "size",
]

Number = Union[int, float]

# Operator precedence, used to decide where parentheses are required.
_PRECEDENCE: Mapping[str, int] = {"+": 1, "-": 1, "*": 2, "/": 2, "**": 3}


class Expression:
    """Base class for accounting expressions.

    Instances are callable: ``expr(size)`` for FLOP accounting and
    ``expr(size, dt_bytes)`` for byte accounting.
    """

    # -- evaluation ----------------------------------------------------
    def evaluate(self, size: Mapping[str, int], dt_bytes: int | None = None) -> Number:
        raise NotImplementedError

    def __call__(self, size: Mapping[str, int], dt_bytes: int | None = None) -> Number:
        return self.evaluate(size, dt_bytes)

    # -- introspection -------------------------------------------------
    def size_keys(self) -> frozenset[str]:
        """Return every size key referenced by this expression."""
        return frozenset()

    def uses_dtype_bytes(self) -> bool:
        """Return True when evaluation requires a ``dt_bytes`` argument."""
        return False

    # -- serialization -------------------------------------------------
    def to_source(self) -> str:
        """Return an equivalent Python expression over ``s`` and ``dt_bytes``."""
        raise NotImplementedError

    @property
    def precedence(self) -> int:
        return 100

    def _wrapped_source(self, parent_precedence: int, *, is_right: bool = False) -> str:
        text = self.to_source()
        if self.precedence < parent_precedence or (
            is_right and self.precedence == parent_precedence
        ):
            return f"({text})"
        return text

    # -- operators -----------------------------------------------------
    def __add__(self, other: "Expression | Number") -> "Expression":
        return BinaryOp("+", self, _coerce(other))

    def __radd__(self, other: "Expression | Number") -> "Expression":
        return BinaryOp("+", _coerce(other), self)

    def __sub__(self, other: "Expression | Number") -> "Expression":
        return BinaryOp("-", self, _coerce(other))

    def __rsub__(self, other: "Expression | Number") -> "Expression":
        return BinaryOp("-", _coerce(other), self)

    def __mul__(self, other: "Expression | Number") -> "Expression":
        return BinaryOp("*", self, _coerce(other))

    def __rmul__(self, other: "Expression | Number") -> "Expression":
        return BinaryOp("*", _coerce(other), self)

    def __truediv__(self, other: "Expression | Number") -> "Expression":
        return BinaryOp("/", self, _coerce(other))

    def __rtruediv__(self, other: "Expression | Number") -> "Expression":
        return BinaryOp("/", _coerce(other), self)

    def __pow__(self, other: "Expression | Number") -> "Expression":
        return BinaryOp("**", self, _coerce(other))


@dataclass(frozen=True)
class Constant(Expression):
    """A literal number."""

    value: Number

    def evaluate(self, size: Mapping[str, int], dt_bytes: int | None = None) -> Number:
        return self.value

    def to_source(self) -> str:
        return repr(self.value)


@dataclass(frozen=True)
class SizeKey(Expression):
    """A lookup of one size key, serialized as ``s['key']``."""

    key: str

    def evaluate(self, size: Mapping[str, int], dt_bytes: int | None = None) -> Number:
        try:
            return size[self.key]
        except KeyError as exc:
            raise KeyError(
                f"size key {self.key!r} is missing from size mapping {sorted(size)!r}"
            ) from exc

    def size_keys(self) -> frozenset[str]:
        return frozenset({self.key})

    def to_source(self) -> str:
        return f"s[{self.key!r}]"


@dataclass(frozen=True)
class DTypeBytes(Expression):
    """The element size, in bytes, of the dtype being benchmarked."""

    def evaluate(self, size: Mapping[str, int], dt_bytes: int | None = None) -> Number:
        if dt_bytes is None:
            raise ValueError(
                "this expression needs a dtype byte width; call it as expr(size, dt_bytes)"
            )
        return dt_bytes

    def uses_dtype_bytes(self) -> bool:
        return True

    def to_source(self) -> str:
        return "dt_bytes"


@dataclass(frozen=True)
class BinaryOp(Expression):
    """An arithmetic combination of two expressions."""

    op: str
    left: Expression
    right: Expression

    def __post_init__(self) -> None:
        if self.op not in _PRECEDENCE:
            raise ValueError(f"unsupported operator {self.op!r}")

    def evaluate(self, size: Mapping[str, int], dt_bytes: int | None = None) -> Number:
        left = self.left.evaluate(size, dt_bytes)
        right = self.right.evaluate(size, dt_bytes)
        if self.op == "+":
            return left + right
        if self.op == "-":
            return left - right
        if self.op == "*":
            return left * right
        if self.op == "/":
            return left / right
        return left ** right

    def size_keys(self) -> frozenset[str]:
        return self.left.size_keys() | self.right.size_keys()

    def uses_dtype_bytes(self) -> bool:
        return self.left.uses_dtype_bytes() or self.right.uses_dtype_bytes()

    @property
    def precedence(self) -> int:
        return _PRECEDENCE[self.op]

    def to_source(self) -> str:
        precedence = self.precedence
        # ``**`` is right-associative; every other supported operator is left.
        left = self.left._wrapped_source(precedence, is_right=self.op == "**")
        right = self.right._wrapped_source(precedence, is_right=self.op != "**")
        return f"{left} {self.op} {right}"


#: Element size in bytes of the dtype currently being benchmarked.
DT_BYTES = DTypeBytes()


def size(key: str) -> SizeKey:
    """Reference one size key, e.g. ``size("M")``."""
    if not isinstance(key, str) or not key:
        raise ValueError(f"size key must be a non-empty string, got {key!r}")
    return SizeKey(key)


def const(value: Number) -> Constant:
    """Wrap a literal number as an expression."""
    return Constant(value)


def _coerce(value: "Expression | Number") -> Expression:
    if isinstance(value, Expression):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"expected an accounting Expression or a number, got {type(value).__name__}"
        )
    return Constant(value)


def serialize_accounting(fn: object) -> str | None:
    """Return the source form of an accounting callable, or None if opaque.

    Generated kernel files embed the returned text. Callables that are not
    :class:`Expression` instances (arbitrary Python functions supplied by an
    external specification) are not serializable, so ``None`` is returned and
    callers fall back to referencing the specification itself.
    """
    if isinstance(fn, Expression):
        return fn.to_source()
    return None
