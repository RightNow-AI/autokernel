"""Lazily resolved callables.

Built-in specifications point at reference implementations that live in
``reference.py``, which imports ``torch``. Registry discovery must work without
importing torch (and must never touch a GPU), so specifications hold a
:class:`LazyCallable` that imports the target module on first call instead of at
module import time.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["LazyCallable", "lazy_callable"]


@dataclass(frozen=True)
class LazyCallable:
    """A callable that resolves ``module_name:attribute`` on first invocation."""

    module_name: str
    attribute: str
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def resolve(self) -> Callable[..., Any]:
        """Import and return the target callable."""
        cached = self._cache.get("fn")
        if cached is not None:
            return cached

        module = importlib.import_module(self.module_name)
        try:
            fn = getattr(module, self.attribute)
        except AttributeError as exc:
            raise AttributeError(
                f"module {self.module_name!r} has no attribute {self.attribute!r}"
            ) from exc
        if not callable(fn):
            raise TypeError(
                f"{self.module_name}:{self.attribute} is not callable "
                f"(got {type(fn).__name__})"
            )
        self._cache["fn"] = fn
        return fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.resolve()(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.module_name}:{self.attribute}"


def lazy_callable(module_name: str, attribute: str) -> LazyCallable:
    """Build a :class:`LazyCallable` for ``module_name:attribute``."""
    return LazyCallable(module_name=module_name, attribute=attribute)
