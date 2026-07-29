"""Load kernel specifications supplied from outside the repository.

Supported locators::

    package.module:SPEC
    /absolute/path/to/spec.py:SPEC
    relative/path/to/spec.py:SPEC

The selected attribute may be a :class:`~autokernel.specs.types.KernelSpec` or a
zero-argument callable returning one.

Trust boundary: loading a specification imports and executes Python supplied by
the caller, exactly like ``python -c`` would. Only pass locators you trust.
Nothing here calls ``eval``/``exec`` on specification *data*, and ``sys.path`` is
never mutated permanently -- file locators are imported through
``importlib.util.spec_from_file_location`` under a unique module name.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from .registry import KernelRegistry, create_builtin_registry
from .types import KernelSpec, SpecValidationError, validate_spec

__all__ = [
    "SpecCollisionError",
    "SpecLoadError",
    "load_spec",
    "parse_locator",
    "resolve_spec",
]


class SpecLoadError(ValueError):
    """Raised when an external specification cannot be loaded."""


class SpecCollisionError(SpecLoadError):
    """Raised when an external specification shadows a registered name."""


def parse_locator(locator: str) -> tuple[str, str]:
    """Split ``target:ATTRIBUTE`` into its two halves.

    Windows-style drive letters are handled because the split happens at the
    last colon.
    """
    if not isinstance(locator, str) or not locator.strip():
        raise SpecLoadError(
            "spec locator must be a non-empty string of the form "
            "'module:ATTRIBUTE' or 'path/to/spec.py:ATTRIBUTE'"
        )
    text = locator.strip()
    if ":" not in text:
        raise SpecLoadError(
            f"invalid spec locator {locator!r}: expected 'module:ATTRIBUTE' or "
            f"'path/to/spec.py:ATTRIBUTE' (the attribute name is required)"
        )
    target, _, attribute = text.rpartition(":")
    target = target.strip()
    attribute = attribute.strip()
    if not target or not attribute:
        raise SpecLoadError(
            f"invalid spec locator {locator!r}: both the module/path and the "
            f"attribute name are required"
        )
    return target, attribute


def _looks_like_path(target: str) -> bool:
    if target.endswith(".py"):
        return True
    if os.sep in target:
        return True
    if os.altsep and os.altsep in target:
        return True
    return Path(target).is_file()


def _import_from_path(target: str, locator: str) -> Any:
    path = Path(target).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise SpecLoadError(
            f"cannot load spec {locator!r}: file not found: {path}"
        )

    module_name = f"_autokernel_external_spec_{uuid.uuid4().hex}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise SpecLoadError(
            f"cannot load spec {locator!r}: {path} is not an importable Python file"
        )
    module = importlib.util.module_from_spec(module_spec)
    # Register before exec so dataclasses and relative lookups inside the module
    # resolve; remove it again on failure so a broken file leaves no trace.
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise SpecLoadError(
            f"cannot load spec {locator!r}: importing {path} raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return module


def _import_module(target: str, locator: str) -> Any:
    try:
        return importlib.import_module(target)
    except ModuleNotFoundError as exc:
        raise SpecLoadError(
            f"cannot load spec {locator!r}: module {target!r} not found "
            f"({exc}). Use 'path/to/spec.py:ATTRIBUTE' to load from a file."
        ) from exc
    except Exception as exc:
        raise SpecLoadError(
            f"cannot load spec {locator!r}: importing module {target!r} raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def load_spec(
    locator: str,
    *,
    registry: KernelRegistry | None = None,
    override: bool = False,
    require_standard_sizes: bool = True,
) -> KernelSpec:
    """Load, validate and return the specification named by ``locator``.

    Args:
        locator: ``module:ATTRIBUTE`` or ``path/to/spec.py:ATTRIBUTE``.
        registry: when given, the loaded name is checked against it for
            collisions.
        override: allow the loaded specification to shadow a name that already
            exists in ``registry``.
        require_standard_sizes: require ``small``/``medium``/``large`` sizes.

    Raises:
        SpecLoadError: for a missing module or file, a missing attribute, an
            attribute of the wrong type, or a factory that misbehaves.
        SpecCollisionError: when the name collides and ``override`` is False.
        SpecValidationError: when the specification itself is malformed.
    """
    target, attribute = parse_locator(locator)

    if _looks_like_path(target):
        module = _import_from_path(target, locator)
    else:
        module = _import_module(target, locator)

    if not hasattr(module, attribute):
        available = sorted(
            name
            for name, value in vars(module).items()
            if isinstance(value, KernelSpec) and not name.startswith("_")
        )
        hint = f" Available KernelSpec attributes: {', '.join(available)}." if available else ""
        raise SpecLoadError(
            f"cannot load spec {locator!r}: {getattr(module, '__name__', target)!r} has no "
            f"attribute {attribute!r}.{hint}"
        )

    obj = getattr(module, attribute)
    if isinstance(obj, KernelSpec):
        spec = obj
    elif callable(obj):
        try:
            spec = obj()
        except SpecValidationError:
            raise
        except Exception as exc:
            raise SpecLoadError(
                f"cannot load spec {locator!r}: calling {attribute!r} raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(spec, KernelSpec):
            raise SpecLoadError(
                f"cannot load spec {locator!r}: {attribute!r} returned "
                f"{type(spec).__name__}, expected a KernelSpec"
            )
    else:
        raise SpecLoadError(
            f"cannot load spec {locator!r}: {attribute!r} is a "
            f"{type(obj).__name__}, expected a KernelSpec or a zero-argument "
            f"callable returning one"
        )

    validate_spec(spec, require_standard_sizes=require_standard_sizes)

    if registry is not None and registry.contains(spec.name) and not override:
        raise SpecCollisionError(
            f"cannot load spec {locator!r}: operation name {spec.name!r} is already "
            f"registered; rename the spec or pass --spec-override to replace it"
        )

    return spec


def resolve_spec(
    *,
    spec_locator: str | None = None,
    name: str | None = None,
    registry: KernelRegistry | None = None,
    override: bool = False,
) -> tuple[KernelSpec, KernelRegistry]:
    """Select the specification for one command invocation.

    Precedence is ``spec_locator`` first, then ``name``. The returned registry is
    isolated: an externally loaded specification is registered into it and never
    into a process-wide global.

    Returns:
        ``(spec, registry)``.
    """
    registry = registry if registry is not None else create_builtin_registry()

    if spec_locator:
        spec = load_spec(spec_locator, registry=registry, override=override)
        registry.register(spec, override=True)
        return spec, registry

    if not name:
        raise SpecLoadError(
            "no operation selected: pass --spec LOCATOR or --kernel NAME"
        )

    return registry.get(name), registry
