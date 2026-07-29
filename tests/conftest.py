"""Shared helpers for the CPU test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autokernel.specs import DT_BYTES, KernelSpec, Tolerance, size

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _ref(x: Any = None, y: Any = None) -> Any:
    return x


def _gen(size_map: Any, dtype: Any, device: str, seed: int = 42) -> dict:
    return {"x": None}


def spec_kwargs(**overrides: Any) -> dict:
    """Keyword arguments for a minimal valid :class:`KernelSpec`."""
    base: dict[str, Any] = {
        "name": "unit_op",
        "reference_fn": _ref,
        "input_generator": _gen,
        "sizes": {
            "small": {"rows": 4, "cols": 4},
            "medium": {"rows": 8, "cols": 8},
            "large": {"rows": 16, "cols": 16},
        },
        "dtypes": ("float16", "float32"),
        "tolerances": {
            "float16": Tolerance(atol=1e-2, rtol=1e-2),
            "float32": Tolerance(atol=1e-5, rtol=1e-5),
        },
        "flops_fn": size("rows") * size("cols"),
        "bytes_fn": 2 * size("rows") * size("cols") * DT_BYTES,
        "shape_keys": ("rows", "cols"),
    }
    base.update(overrides)
    return base


def make_spec(**overrides: Any) -> KernelSpec:
    """Build a minimal valid specification, overriding any field."""
    return KernelSpec(**spec_kwargs(**overrides))


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def in_repo_root(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run a test with the repository root as the working directory."""
    monkeypatch.chdir(REPO_ROOT)
    return REPO_ROOT


@pytest.fixture
def torch_mod():
    """The torch module, skipping the test when torch is unavailable."""
    return pytest.importorskip("torch")


def cuda_available() -> bool:
    """True when a CUDA device is usable (never raises when torch is absent)."""
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(
    not cuda_available(), reason="requires a CUDA GPU"
)

__all__ = [
    "FIXTURES_DIR",
    "REPO_ROOT",
    "cuda_available",
    "make_spec",
    "requires_gpu",
    "spec_kwargs",
]
