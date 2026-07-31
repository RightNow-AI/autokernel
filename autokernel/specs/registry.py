"""A small, explicit registry of kernel specifications.

The registry is an ordinary object, not a module-level global: every command and
every test can build an isolated registry so registering an external
specification never leaks into another run.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from .types import KernelSpec, SpecValidationError, validate_spec

__all__ = [
    "DuplicateSpecError",
    "KernelRegistry",
    "SpecNotFoundError",
    "builtin_spec_names",
    "create_builtin_registry",
]


class DuplicateSpecError(ValueError):
    """Raised when a name is registered twice without an explicit override."""


class SpecNotFoundError(KeyError):
    """Raised when a requested specification name is not registered."""

    def __str__(self) -> str:  # KeyError repr would quote the whole message
        return self.args[0] if self.args else super().__str__()


class KernelRegistry:
    """An ordered collection of :class:`KernelSpec` objects keyed by name.

    Ordering is insertion order, which keeps ``list_names()`` deterministic and
    keeps the CLI's "available operations" listing stable.
    """

    def __init__(self, specs: Iterable[KernelSpec] = ()) -> None:
        self._specs: dict[str, KernelSpec] = {}
        for spec in specs:
            self.register(spec)

    # -- mutation ------------------------------------------------------
    def register(self, spec: KernelSpec, *, override: bool = False) -> None:
        """Register a specification.

        Args:
            spec: the specification to add.
            override: replace an existing specification with the same name.

        Raises:
            SpecValidationError: if ``spec`` is not a valid specification.
            DuplicateSpecError: if the name exists and ``override`` is False.
        """
        if not isinstance(spec, KernelSpec):
            raise SpecValidationError(
                f"can only register KernelSpec objects, got {type(spec).__name__}"
            )
        validate_spec(spec)
        if spec.name in self._specs and not override:
            raise DuplicateSpecError(
                f"kernel spec {spec.name!r} is already registered; pass override=True "
                f"to replace it"
            )
        self._specs[spec.name] = spec

    # -- lookup --------------------------------------------------------
    def get(self, name: str) -> KernelSpec:
        """Return the specification registered under ``name``."""
        try:
            return self._specs[name]
        except KeyError:
            available = ", ".join(self.list_names()) or "<none>"
            raise SpecNotFoundError(
                f"unknown kernel spec {name!r}; available: {available}"
            ) from None

    def contains(self, name: str) -> bool:
        """Return True when ``name`` is registered."""
        return name in self._specs

    def list_names(self) -> tuple[str, ...]:
        """Return registered names in registration order."""
        return tuple(self._specs)

    def specs(self) -> tuple[KernelSpec, ...]:
        """Return registered specifications in registration order."""
        return tuple(self._specs.values())

    # -- dunder --------------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._specs

    def __iter__(self) -> Iterator[KernelSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def __repr__(self) -> str:
        return f"KernelRegistry({list(self._specs)!r})"


def create_builtin_registry() -> KernelRegistry:
    """Build a fresh registry containing every built-in operation.

    Importing the built-in specifications does not import ``torch`` and does not
    initialize a GPU, so this is safe on a CPU-only machine.
    """
    from .builtins import builtin_specs  # local import avoids an import cycle

    return KernelRegistry(builtin_specs())


def builtin_spec_names() -> tuple[str, ...]:
    """Return the built-in operation names in their canonical order."""
    return create_builtin_registry().list_names()
