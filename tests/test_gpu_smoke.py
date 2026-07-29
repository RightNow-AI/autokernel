"""GPU smoke tests for the specification path.

Deselected by default with ``-m "not gpu"``. On a CUDA machine::

    uv run pytest -m gpu

These run the real starter kernels through the real harness, so a failure here
means the registry refactor changed kernel behavior.
"""

from __future__ import annotations

import pytest
from conftest import REPO_ROOT, requires_gpu

from autokernel.specs import create_builtin_registry, load_spec

pytestmark = [pytest.mark.gpu, requires_gpu]


def _load_kernel_fn(path):
    """Import ``kernel_fn`` from a starter kernel file without touching sys.path."""
    import importlib.util
    import uuid

    module_spec = importlib.util.spec_from_file_location(
        f"_starter_{uuid.uuid4().hex}", path
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.kernel_fn


@pytest.mark.parametrize("name", ["matmul", "layernorm", "rmsnorm"])
def test_builtin_starter_kernels_pass_correctness(name):
    bench = pytest.importorskip("bench")
    spec = create_builtin_registry().get(name)
    kernel_fn = _load_kernel_fn(spec.starter_kernel("triton"))

    results = bench.run_correctness(kernel_fn, spec, quick=True)
    assert results["correctness"] == "PASS", results.get("details")


def test_external_spec_runs_through_the_same_harness():
    bench = pytest.importorskip("bench")
    example = REPO_ROOT / "examples" / "custom_ops" / "add.py"
    spec = load_spec(f"{example}:SPEC", registry=create_builtin_registry())
    kernel_fn = _load_kernel_fn(spec.starter_kernel("triton"))

    results = bench.run_correctness(kernel_fn, spec, quick=False)
    assert results["correctness"] == "PASS", results.get("details")


def test_external_spec_performance_path():
    bench = pytest.importorskip("bench")
    example = REPO_ROOT / "examples" / "custom_ops" / "add.py"
    spec = load_spec(f"{example}:SPEC")
    kernel_fn = _load_kernel_fn(spec.starter_kernel("triton"))

    gpu = bench.detect_gpu()
    perf = bench.run_performance(kernel_fn, spec, gpu, sizes_filter="large")
    primary = perf["primary"]
    assert primary is not None
    assert primary["kernel_latency_us"] > 0
    assert primary["bytes"] == 3 * 4096 * 4096 * 2  # float16 primary dtype
