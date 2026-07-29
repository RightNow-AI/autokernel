"""Pytest bootstrap for the repository root.

The command-line entry points (``bench.py``, ``extract.py``, ``reference.py``)
live at the repository root and are imported by tests, so the root must be on
``sys.path`` regardless of where pytest is invoked from.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
