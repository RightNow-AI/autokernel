"""Optional backward (gradient) verification on CPU."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import autokernel.verification.backward as backward_module
from autokernel.specs import (
    DT_BYTES,
    BackwardSpec,
    KernelSpec,
    Tolerance,
    load_spec,
    resolve_torch_dtype,
    size,
)
from autokernel.verification import check_backward

torch = pytest.importorskip("torch")
bench = pytest.importorskip("bench")


def _gen(size_map: Mapping[str, int], dtype: Any, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    torch_dtype = resolve_torch_dtype(dtype)
    rows, cols = size_map["rows"], size_map["cols"]
    return {
        "x": torch.randn(rows, cols, device=device, dtype=torch_dtype),
        "scale": torch.randn(cols, device=device, dtype=torch_dtype),
        "bias": torch.randn(cols, device=device, dtype=torch_dtype),
    }


def _affine_ref(x: Any, scale: Any, bias: Any) -> dict:
    y = x * scale + bias
    return {"output": y, "aux": (y - x, 3)}


def _spec(**overrides: Any) -> KernelSpec:
    kwargs: dict[str, Any] = {
        "name": "bwd_affine",
        "reference_fn": _affine_ref,
        "input_generator": _gen,
        "sizes": {
            "small": {"rows": 4, "cols": 8},
            "medium": {"rows": 8, "cols": 8},
            "large": {"rows": 16, "cols": 16},
        },
        "dtypes": ("float32",),
        "tolerances": {"float32": Tolerance(atol=1e-5, rtol=1e-5)},
        "flops_fn": 3 * size("rows") * size("cols"),
        "bytes_fn": 5 * size("rows") * size("cols") * DT_BYTES,
        "shape_keys": ("rows", "cols"),
        "backward_spec": BackwardSpec(differentiable_inputs=("x", "scale", "bias")),
    }
    kwargs.update(overrides)
    return KernelSpec(**kwargs)


# ---------------------------------------------------------------------------
# Parity and mismatch
# ---------------------------------------------------------------------------

def test_gradient_parity_for_reference_candidate():
    report = check_backward(_affine_ref, _spec(), device="cpu")
    assert report.status == "PASS"
    assert {r.input_name for r in report.gradients} == {"x", "scale", "bias"}
    for record in report.gradients:
        assert record.status == "match"
        assert record.max_abs_error == 0.0
    # every floating tensor leaf received an upstream gradient
    assert set(report.output_paths) == {'output["aux"][0]', 'output["output"]'}


def test_empty_sizes_returns_structured_failure():
    spec = _spec()
    object.__setattr__(spec, "sizes", {})
    report = check_backward(_affine_ref, spec, device="cpu")
    assert report.status == "FAIL"
    assert "declares no sizes" in report.reason


def test_upstream_generator_fallback_generates_on_cpu_then_moves(monkeypatch):
    original_generator = torch.Generator
    moves = []
    randn_devices = []

    def generator(*args, **kwargs):
        if kwargs.get("device") == "mps":
            raise RuntimeError("device-local generators unsupported")
        return original_generator()

    class Generated:
        def to(self, device):
            moves.append(device)
            return self

    def randn(*shape, **kwargs):
        randn_devices.append(kwargs["device"])
        return Generated()

    monkeypatch.setattr(torch, "Generator", generator)
    monkeypatch.setattr(torch, "randn", randn)
    leaf = SimpleNamespace(
        shape=(2, 3),
        dtype=torch.float32,
        device=torch.device("mps"),
    )

    upstreams = backward_module._upstream_gradients(
        [("output", leaf)], "mps", seed=123
    )

    assert len(upstreams) == 1
    assert randn_devices == ["cpu"]
    assert moves == [torch.device("mps")]


def test_upstream_generation_failure_returns_structured_report(monkeypatch):
    monkeypatch.setattr(
        backward_module,
        "_upstream_gradients",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("rng failed")),
    )
    report = check_backward(_affine_ref, _spec(), device="cpu")
    assert report.status == "FAIL"
    assert "upstream gradient generation failed: RuntimeError: rng failed" in report.reason


def test_affine_example_fixture_passes_backward(repo_root):
    spec = load_spec(str(repo_root / "examples" / "custom_ops" / "affine.py") + ":SPEC")
    report = check_backward(spec.reference_fn, spec, device="cpu")
    assert report.status == "PASS"


def test_affine_example_byte_accounting(repo_root):
    spec = load_spec(str(repo_root / "examples" / "custom_ops" / "affine.py") + ":SPEC")
    size_map = {"rows": 4, "cols": 8}
    assert spec.bytes_fn(size_map, 2) == (3 * 4 + 2) * 8 * 2


def test_perturbed_candidate_reports_mismatch_with_stats():
    def perturbed(x, scale, bias):
        y = x * scale * 2 + bias  # wrong gradient wrt x and scale
        return {"output": y, "aux": (y - x, 3)}

    report = check_backward(perturbed, _spec(), device="cpu")
    assert report.status == "FAIL"
    by_name = {r.input_name: r for r in report.gradients}
    assert by_name["x"].status == "mismatch"
    assert by_name["scale"].status == "mismatch"
    assert by_name["bias"].status == "match"  # d(y)/d(bias) is unchanged
    assert by_name["x"].max_abs_error > 0
    assert by_name["x"].mean_abs_error is not None


def test_nan_gradient_is_reported():
    def nan_candidate(x, scale, bias):
        y = (x * scale + bias) * torch.where(
            torch.arange(x.shape[1], device=x.device) == 0,
            torch.tensor(float("nan"), device=x.device),
            torch.tensor(1.0, device=x.device),
        )
        return {"output": y, "aux": (y - x, 3)}

    report = check_backward(nan_candidate, _spec(), device="cpu")
    assert report.status == "FAIL"
    assert any(r.has_nan for r in report.gradients)


# ---------------------------------------------------------------------------
# Missing and unexpected gradients
# ---------------------------------------------------------------------------

def test_missing_gradient_is_reported():
    def drops_bias(x, scale, bias):
        y = x * scale  # bias unused
        return {"output": y, "aux": (y - x, 3)}

    report = check_backward(drops_bias, _spec(), device="cpu")
    assert report.status == "FAIL"
    by_name = {r.input_name: r for r in report.gradients}
    assert by_name["bias"].status == "missing"
    assert "no gradient" in by_name["bias"].reason
    assert by_name["x"].status == "match"


def test_unexpected_gradient_is_reported():
    def ref_without_bias(x, scale, bias):
        y = x * scale  # reference ignores bias...
        return {"output": y, "aux": (y - x, 3)}

    spec = _spec(reference_fn=ref_without_bias)
    # ...but the candidate uses it
    report = check_backward(_affine_ref, spec, device="cpu")
    assert report.status == "FAIL"
    by_name = {r.input_name: r for r in report.gradients}
    assert by_name["bias"].status == "unexpected"


# ---------------------------------------------------------------------------
# Unsupported and invalid declarations
# ---------------------------------------------------------------------------

def test_missing_backward_spec_fails_as_unsupported():
    spec = _spec(backward_spec=None)
    report = check_backward(_affine_ref, spec, device="cpu")
    assert report.status == "FAIL"
    assert "unsupported" in report.reason
    assert "forward-only" in report.reason
    assert "backward_spec" in report.reason


def test_declared_input_missing_from_generator_fails():
    spec = _spec(backward_spec=BackwardSpec(differentiable_inputs=("x", "nope")))
    report = check_backward(_affine_ref, spec, device="cpu")
    assert report.status == "FAIL"
    assert "'nope'" in report.reason
    assert "did not produce" in report.reason


def test_non_floating_declared_input_fails():
    def gen_with_int(size_map, dtype, device, seed=42):
        inputs = _gen(size_map, dtype, device, seed)
        inputs["bias"] = torch.ones(8, dtype=torch.int64)
        return inputs

    spec = _spec(input_generator=gen_with_int)
    report = check_backward(_affine_ref, spec, device="cpu")
    assert report.status == "FAIL"
    assert "not floating-point" in report.reason


def test_non_differentiable_candidate_fails_actionably():
    def forward_only(x, scale, bias):
        with torch.no_grad():
            y = x * scale + bias
        return {"output": y, "aux": (y - x, 3)}

    report = check_backward(forward_only, _spec(), device="cpu")
    assert report.status == "FAIL"
    assert "candidate backward failed" in report.reason


# ---------------------------------------------------------------------------
# Output path selection, determinism, accumulation
# ---------------------------------------------------------------------------

def test_output_paths_restrict_upstream_leaves():
    spec = _spec(
        backward_spec=BackwardSpec(
            differentiable_inputs=("x", "scale", "bias"),
            output_paths=('output["output"]',),
        )
    )
    report = check_backward(_affine_ref, spec, device="cpu")
    assert report.status == "PASS"
    assert report.output_paths == ('output["output"]',)


def test_output_paths_must_exist():
    spec = _spec(
        backward_spec=BackwardSpec(
            differentiable_inputs=("x",), output_paths=("output[9]",)
        )
    )
    report = check_backward(_affine_ref, spec, device="cpu")
    assert report.status == "FAIL"
    assert "does not exist" in report.reason


def test_upstream_gradients_are_deterministic():
    def perturbed(x, scale, bias):
        y = x * scale * 1.01 + bias
        return {"output": y, "aux": (y - x, 3)}

    first = check_backward(perturbed, _spec(), device="cpu")
    second = check_backward(perturbed, _spec(), device="cpu")
    assert first.status == second.status == "FAIL"
    assert [r.as_dict() for r in first.gradients] == [r.as_dict() for r in second.gradients]


def test_repeated_checks_do_not_accumulate_state():
    spec = _spec()
    for _ in range(3):
        report = check_backward(_affine_ref, spec, device="cpu")
        assert report.status == "PASS"
        assert all(r.max_abs_error == 0.0 for r in report.gradients)


def test_backward_tolerances_override_forward_tolerances():
    def slightly_off(x, scale, bias):
        y = (x * scale + bias) * 1.001  # 1e-3 relative error in grads
        return {"output": y, "aux": (y - x, 3)}

    strict = check_backward(slightly_off, _spec(), device="cpu")
    assert strict.status == "FAIL"

    loose = _spec(
        backward_spec=BackwardSpec(
            differentiable_inputs=("x", "scale", "bias"),
            tolerances={"float32": Tolerance(atol=1e-2, rtol=1e-2)},
        )
    )
    relaxed = check_backward(slightly_off, loose, device="cpu")
    assert relaxed.status == "PASS"


# ---------------------------------------------------------------------------
# bench.py wiring
# ---------------------------------------------------------------------------

def test_bench_run_backward_check_prints_verdict(monkeypatch, capsys):
    monkeypatch.setattr(bench, "BENCH_DEVICE", "cpu")
    result = bench.run_backward_check(_affine_ref, _spec())
    captured = capsys.readouterr().out
    assert result["status"] == "PASS"
    assert "BACKWARD_CORRECTNESS: PASS" in captured
    assert "grad[x]" in captured


def test_bench_run_backward_check_unsupported(monkeypatch, capsys):
    monkeypatch.setattr(bench, "BENCH_DEVICE", "cpu")
    result = bench.run_backward_check(_affine_ref, _spec(backward_spec=None))
    captured = capsys.readouterr().out
    assert result["status"] == "FAIL"
    assert "BACKWARD_CORRECTNESS: FAIL" in captured
    assert "unsupported" in captured
