"""AutoKernel downstream platform package.

This package holds the reusable, importable core of the project. The
command-line entry points (``bench.py``, ``extract.py``, ``profile.py``,
``verify.py``) live at the repository root and import from here.

Nothing in this package may initialize a GPU at import time: registry
discovery and specification inspection must work on a CPU-only machine.

Derived from the upstream AutoKernel project (MIT License). See LICENSE.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
