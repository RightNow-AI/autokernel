"""The refactored bench harness, exercised on CPU.

No GPU is required: ``bench.BENCH_DEVICE`` is redirected to ``cpu`` so the five
correctness stages, the spec-driven size/dtype/tolerance plumbing and the
performance loop can be verified without CUDA.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from autokernel.specs import DT_BYTES, EdgeCase, KernelSpec, Tolerance, resolve_torch_dtype, size

pytest.importorskip("torch")
bench = pytest.importorskip("bench")


def _ref(x: Any, y: Any) -> Any:
    return x + y


def _gen(size_map: Mapping[str, int], dtype: Any, device: str, seed: int = 42) -> dict:
    import torch

    torch.manual_seed(seed)
    torch_dtype = resolve_torch_dtype(dtype)
    rows, cols = size_map["rows"], size_map["cols"]
    return {
        "x": torch.randn(rows, cols, device=device, dtype=torch_dtype),
        "y": torch.randn(rows, cols, device=device, dtype=torch_dtype),
    }


def _spec(**overrides: Any) -> KernelSpec:
    kwargs: dict[str, Any] = {
        "name": "cpu_add",
        "reference_fn": _ref,
        "input_generator": _gen,
        "sizes": {
            "small": {"rows": 8, "cols": 16},
            "medium": {"rows": 16, "cols": 16},
            "large": {"rows": 32, "cols": 32},
        },
        "dtypes": ("float32",),
        "tolerances": {"float32": Tolerance(atol=1e-5, rtol=1e-5)},
        "flops_fn": size("rows") * size("cols"),
        "bytes_fn": 3 * size("rows") * size("cols") * DT_BYTES,
        "edge_cases": (
            EdgeCase(name="edge_7", size={"rows": 7, "cols": 7}),
            EdgeCase(name="edge_zeros", size={"rows": 5, "cols": 5},
                     input_transform=lambda inputs: {k: v * 0 for k, v in inputs.items()}),
        ),
        "shape_keys": ("rows", "cols"),
    }
    kwargs.update(overrides)
    return KernelSpec(**kwargs)


@pytest.fixture
def cpu_device(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bench, "BENCH_DEVICE", "cpu")
    return "cpu"


def _good_kernel(x, y):
    return x + y


def _wrong_kernel(x, y):
    return x - y


def test_all_five_stages_pass_for_a_correct_candidate(cpu_device, capsys):
    results = bench.run_correctness(_good_kernel, _spec(), quick=False)
    captured = capsys.readouterr().out

    assert results["correctness"] == "PASS"
    assert results["smoke_test"] == "PASS"
    assert results["shape_sweep"].startswith("PASS")
    assert results["numerical_stability"] == "PASS"
    assert results["determinism"] == "PASS"
    assert results["edge_cases"] == "PASS"
    # The greppable stage banners the agent loop reads must stay put.
    for stage in ("Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"):
        assert stage in captured


def test_edge_cases_run_every_declared_case(cpu_device, capsys):
    bench.run_correctness(_good_kernel, _spec(), quick=False)
    captured = capsys.readouterr().out
    assert "PASS: edge_7" in captured
    assert "PASS: edge_zeros" in captured


def test_quick_mode_skips_stages_three_to_five(cpu_device):
    results = bench.run_correctness(_good_kernel, _spec(), quick=True)
    assert results["correctness"] == "PASS"
    assert results["numerical_stability"] == "SKIP (quick mode)"
    assert results["determinism"] == "SKIP (quick mode)"
    assert results["edge_cases"] == "SKIP (quick mode)"


def test_incorrect_candidate_fails_correctness(cpu_device):
    results = bench.run_correctness(_wrong_kernel, _spec(), quick=True)
    assert results["correctness"] == "FAIL"
    assert results["smoke_test"] == "FAIL"


def test_missing_edge_cases_report_skip(cpu_device):
    results = bench.run_correctness(_good_kernel, _spec(edge_cases=()), quick=False)
    assert results["edge_cases"] == "SKIP (no edge sizes defined)"
    assert results["correctness"] == "PASS"


def test_edge_case_may_pin_its_own_dtype(cpu_device, capsys):
    spec = _spec(
        dtypes=("float32", "float16"),
        tolerances={
            "float32": Tolerance(atol=1e-5, rtol=1e-5),
            "float16": Tolerance(atol=1e-2, rtol=1e-2),
        },
        edge_cases=(EdgeCase(name="edge_fp16", size={"rows": 6, "cols": 6}, dtype="float16"),),
    )
    results = bench.run_correctness(_good_kernel, spec, quick=False)
    assert results["edge_cases"] == "PASS"
    assert "PASS: edge_fp16" in capsys.readouterr().out


@pytest.fixture
def stub_timer(monkeypatch: pytest.MonkeyPatch):
    """Replace the GPU timer so the spec-driven plumbing can be checked on CPU.

    Only the timing primitive is stubbed: size selection, dtype resolution and
    FLOP/byte accounting all run for real.
    """
    calls: list[str] = []

    def fake_do_bench(fn, warmup: int = 25, rep: int = 100) -> float:
        fn()
        calls.append("bench")
        return 0.5

    monkeypatch.setattr(bench, "_do_bench", fake_do_bench)
    return calls


def test_performance_loop_uses_spec_accounting(cpu_device, stub_timer):
    spec = _spec()
    gpu = bench.GPUSpec(name="cpu-test", peak_tflops_fp16=100.0, peak_bandwidth_gb_s=1000.0)
    perf = bench.run_performance(_good_kernel, spec, gpu, sizes_filter="large")

    assert perf["primary"] is not None
    entry = perf["primary"]
    assert entry["label"] == "large"
    assert entry["flops"] == 32 * 32
    assert entry["bytes"] == 3 * 32 * 32 * 4  # float32
    assert entry["dtype"] == "torch.float32"
    assert entry["kernel_latency_us"] == pytest.approx(500.0)
    assert entry["speedup_vs_pytorch"] == pytest.approx(1.0)
    # candidate and reference are both timed
    assert len(stub_timer) == 2


def test_performance_reports_every_requested_size(cpu_device, stub_timer):
    spec = _spec()
    gpu = bench.GPUSpec(name="cpu-test", peak_tflops_fp16=100.0, peak_bandwidth_gb_s=1000.0)
    perf = bench.run_performance(_good_kernel, spec, gpu, sizes_filter="all")
    labels = [entry["label"] for entry in perf["all"]]
    assert labels == ["small", "medium", "large"]
    assert perf["primary"]["label"] == "large"


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------

def _structured_spec(**overrides: Any) -> KernelSpec:
    """Same shape contract as ``_spec`` but with a nested output tree."""
    kwargs: dict[str, Any] = {
        "name": "cpu_structured",
        "reference_fn": _structured_ref,
        "input_generator": _gen,
        "sizes": {
            "small": {"rows": 8, "cols": 16},
            "medium": {"rows": 16, "cols": 16},
            "large": {"rows": 32, "cols": 32},
        },
        "dtypes": ("float32",),
        "tolerances": {"float32": Tolerance(atol=1e-5, rtol=1e-5)},
        "flops_fn": size("rows") * size("cols"),
        "bytes_fn": 4 * size("rows") * size("cols") * DT_BYTES,
        "shape_keys": ("rows", "cols"),
    }
    kwargs.update(overrides)
    return KernelSpec(**kwargs)


def _structured_ref(x: Any, y: Any) -> Any:
    return {"output": x + y, "aux": (x - y, 2)}


def _good_structured_kernel(x: Any, y: Any) -> Any:
    return {"output": x + y, "aux": (x - y, 2)}


def _wrong_aux_kernel(x: Any, y: Any) -> Any:
    return {"output": x + y, "aux": (x - y + 1.0, 2)}


def _wrong_metadata_kernel(x: Any, y: Any) -> Any:
    return {"output": x + y, "aux": (x - y, 999)}


def _dropping_aux_kernel(x: Any, y: Any) -> Any:
    return {"output": x + y}


def test_structured_candidate_passes_all_stages(cpu_device, capsys):
    results = bench.run_correctness(_good_structured_kernel, _structured_spec(), quick=False)
    captured = capsys.readouterr().out

    assert results["correctness"] == "PASS"
    assert results["smoke_test"] == "PASS"
    assert results["determinism"] == "PASS"
    # Every tensor leaf is compared, and the paths are stable.
    paths = {record["path"] for record in results["leaf_details"]}
    assert {'output["output"]', 'output["aux"][0]', 'output["aux"][1]'} <= paths
    assert "Stage 1" in captured and "Stage 5" in captured


def test_wrong_aux_leaf_fails_with_diagnostic_path(cpu_device):
    results = bench.run_correctness(_wrong_aux_kernel, _structured_spec(), quick=True)
    assert results["correctness"] == "FAIL"
    assert any('output["aux"][0]' in record["path"] and not record["match"]
               for record in results["leaf_details"])
    assert any('output["aux"][0]' in detail for detail in results["details"])


def test_wrong_metadata_leaf_fails(cpu_device):
    results = bench.run_correctness(_wrong_metadata_kernel, _structured_spec(), quick=True)
    assert results["correctness"] == "FAIL"
    assert any("metadata mismatch" in record["reason"] for record in results["leaf_details"])


def test_dropped_output_branch_fails_structure_check(cpu_device):
    results = bench.run_correctness(_dropping_aux_kernel, _structured_spec(), quick=True)
    assert results["correctness"] == "FAIL"
    assert any("missing output path" in detail for detail in results["details"])



# ---------------------------------------------------------------------------
# Shape corpora
# ---------------------------------------------------------------------------

def _corpus_cases() -> tuple:
    from autokernel.verification import CorpusCase

    return (
        CorpusCase(name="prod-a", size={"rows": 8, "cols": 16}, weight=3),
        CorpusCase(name="prod-b", size={"rows": 7, "cols": 13}, weight=1),
    )


def test_corpus_cases_join_the_shape_sweep(cpu_device, capsys):
    results = bench.run_correctness(
        _good_kernel, _spec(), quick=True, corpus_cases=_corpus_cases()
    )
    captured = capsys.readouterr().out
    assert results["correctness"] == "PASS"
    assert "PASS: prod-a" in captured
    assert "PASS: prod-b" in captured
    assert "PASS: small" in captured  # built-in sweep still runs


def test_corpus_only_replaces_the_builtin_sweep(cpu_device, capsys):
    results = bench.run_correctness(
        _good_kernel, _spec(), quick=True,
        corpus_cases=_corpus_cases(), corpus_only=True,
    )
    captured = capsys.readouterr().out
    assert results["correctness"] == "PASS"
    assert "PASS: prod-a" in captured
    assert "PASS: small" not in captured


def test_wrong_kernel_fails_a_corpus_case(cpu_device, capsys):
    """A candidate that is correct on built-in sizes but wrong on a corpus
    shape must fail with the corpus case named."""

    def wrong_on_odd_rows(x, y):
        out = x + y
        if x.shape[0] == 7:  # prod-b is 7x13
            out = out + 1.0
        return out

    results = bench.run_correctness(
        wrong_on_odd_rows, _spec(), quick=True, corpus_cases=_corpus_cases()
    )
    captured = capsys.readouterr().out
    assert results["correctness"] == "FAIL"
    assert "FAIL: prod-b" in captured
    assert "PASS: prod-a" in captured


def test_performance_benches_corpus_cases_with_weights(cpu_device, stub_timer):
    spec = _spec()
    gpu = bench.GPUSpec(name="cpu-test", peak_tflops_fp16=100.0, peak_bandwidth_gb_s=1000.0)
    perf = bench.run_performance(
        _good_kernel, spec, gpu, sizes_filter="all", corpus_cases=_corpus_cases()
    )

    labels = [entry["label"] for entry in perf["all"]]
    assert labels == ["small", "medium", "large", "prod-a", "prod-b"]
    corpus = perf["corpus"]
    assert corpus is not None
    assert [entry["weight"] for entry in corpus["cases"]] == [3, 1]
    weighted = corpus["weighted"]["torch.float32"]
    assert weighted["cases"] == 2
    assert weighted["weight"] == 4
    # the stub timer returns 0.5 ms for every call; weight must not repeat loops
    assert weighted["kernel_ms"] == pytest.approx(0.5)
    assert len(stub_timer) == 2 * 5  # candidate + reference per configuration


def test_performance_corpus_only_skips_builtin_sizes(cpu_device, stub_timer):
    spec = _spec()
    gpu = bench.GPUSpec(name="cpu-test", peak_tflops_fp16=100.0, peak_bandwidth_gb_s=1000.0)
    perf = bench.run_performance(
        _good_kernel, spec, gpu, sizes_filter="all",
        corpus_cases=_corpus_cases(), corpus_only=True,
    )
    labels = [entry["label"] for entry in perf["all"]]
    assert labels == ["prod-a", "prod-b"]

