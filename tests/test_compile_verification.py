"""CPU tests for optional torch.compile verification."""

from __future__ import annotations

from typing import Any

import torch

from autokernel.specs import CompileSpec, Tolerance
from autokernel.verification.compile import check_compile
from conftest import make_spec


def _generator(size: dict[str, int], dtype: str, device: str, seed: int) -> dict:
    torch.manual_seed(seed)
    return {"x": torch.randn(size["rows"], size["cols"], device=device)}


def _spec(**overrides: Any):
    return make_spec(
        reference_fn=lambda x: {"output": x * 2, "metadata": 7},
        input_generator=_generator,
        dtypes=("float32",),
        tolerances={"float32": Tolerance(1e-6, 1e-6)},
        **overrides,
    )


def test_compile_uses_declared_options_and_runs_twice(monkeypatch):
    calls: list[dict[str, Any]] = []
    executions = 0

    def fake_compile(fn, **options):
        calls.append(options)

        def compiled(**inputs):
            nonlocal executions
            executions += 1
            return fn(**inputs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    report = check_compile(
        lambda x: {"output": x * 2, "metadata": 7},
        _spec(compile_spec=CompileSpec(fullgraph=True, dynamic=False)),
        device="cpu",
    )

    assert report.status == "PASS"
    assert calls == [{"fullgraph": True, "dynamic": False}]
    assert executions == 2
    assert [case.label for case in report.cases] == ["small"]


def test_dynamic_compile_reuses_callable_for_two_shapes(monkeypatch):
    compile_calls = 0
    observed_shapes: list[tuple[int, ...]] = []

    def fake_compile(fn, **options):
        nonlocal compile_calls
        compile_calls += 1

        def compiled(**inputs):
            observed_shapes.append(tuple(inputs["x"].shape))
            return fn(**inputs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    report = check_compile(
        lambda x: {"output": x * 2, "metadata": 7},
        _spec(compile_spec=CompileSpec(dynamic=True)),
        device="cpu",
    )

    assert report.status == "PASS"
    assert compile_calls == 1
    assert len(report.cases) == 2
    assert observed_shapes == [(4, 4), (4, 4), (8, 8), (8, 8)]


def test_compile_mismatch_is_not_passed(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **options: fn)
    report = check_compile(
        lambda x: {"output": x * 3, "metadata": 7},
        _spec(),
        device="cpu",
    )
    assert report.status == "FAIL"
    assert report.cases[0].status == "FAIL"
    assert "max_abs_error" in report.reason


def test_compile_unavailable_is_unsupported(monkeypatch):
    monkeypatch.delattr(torch, "compile", raising=False)
    report = check_compile(lambda x: x, _spec(), device="cpu")
    assert report.status == "UNSUPPORTED"
    assert "unavailable" in report.reason
