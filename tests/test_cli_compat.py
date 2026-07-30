"""CLI compatibility and extraction wiring.

These tests protect two things the autonomous loop depends on:

* the pre-existing command-line surface (``--kernel``, ``--sizes``, ``--quick``,
  ``--profile``, ``--report``, ``--top``, ``--kernel-type``, ``--backend``);
* extraction consuming a ``KernelSpec`` instead of operation-specific maps.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import FIXTURES_DIR, REPO_ROOT

from autokernel.specs import create_builtin_registry, load_spec

FIXTURE_LOCATOR = f"{FIXTURES_DIR / 'custom_add.py'}:SPEC"


def run_script(script: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script), *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# bench.py command line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "flag", ["--kernel", "--sizes", "--quick", "--profile", "--spec", "--spec-override"]
)
def test_bench_help_lists_expected_flags(flag):
    result = run_script("bench.py", "--help")
    assert result.returncode == 0, result.stderr
    assert flag in result.stdout


def test_bench_help_does_not_import_the_external_spec(tmp_path: Path):
    """`--help` must never execute external specification code."""
    exploding = tmp_path / "exploding_spec.py"
    exploding.write_text(
        "raise SystemExit('the external spec was imported during --help')\n"
    )
    result = run_script("bench.py", "--spec", f"{exploding}:SPEC", "--help")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage: bench.py" in result.stdout
    assert "imported during --help" not in result.stdout + result.stderr


def test_bench_reports_actionable_error_for_bad_spec():
    result = run_script("bench.py", "--spec", "/nope/missing_spec.py:SPEC")
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "/nope/missing_spec.py:SPEC" in combined
    assert "file not found" in combined
    # The greppable failure contract the agent loop parses must survive.
    assert "correctness: FAIL" in combined
    assert "throughput_tflops: 0.000" in combined


def test_bench_loads_external_spec_before_touching_the_candidate():
    """The spec is resolved (and echoed) before kernel.py is imported."""
    result = run_script("bench.py", "--spec", FIXTURE_LOCATOR, "--quick")
    combined = result.stdout + result.stderr
    assert f"kernel_spec: {FIXTURE_LOCATOR}" in combined
    assert "cannot load spec" not in combined


def test_bench_spec_collision_is_rejected_without_override():
    colliding = f"{FIXTURES_DIR / 'custom_add.py'}:COLLIDING_SPEC"
    result = run_script("bench.py", "--spec", colliding)
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "already registered" in combined
    assert "--spec-override" in combined


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------

def test_operation_precedence_spec_then_kernel_then_declared():
    bench = pytest.importorskip("bench")
    # --spec wins
    assert bench.resolve_operation_name("from_spec", "from_kernel", "declared") == "from_spec"
    # --kernel wins over kernel.py
    assert bench.resolve_operation_name(None, "from_kernel", "declared") == "from_kernel"
    # kernel.py::KERNEL_TYPE is the fallback
    assert bench.resolve_operation_name(None, None, "declared") == "declared"
    # nothing selected
    assert bench.resolve_operation_name(None, None, None) is None


def test_legacy_kernel_configs_view_is_derived_from_the_registry():
    bench = pytest.importorskip("bench")
    registry = create_builtin_registry()
    assert tuple(bench.KERNEL_CONFIGS) == registry.list_names()
    entry = bench.KERNEL_CONFIGS["matmul"]
    assert entry["spec"] is not None
    assert entry["test_sizes"][0] == ("tiny", {"M": 128, "N": 128, "K": 128})
    assert entry["edge_sizes"][0][0] == "edge_1023"


def test_bench_reads_metadata_only_from_specs():
    """No operation-specific configuration literals may return to bench.py."""
    source = (REPO_ROOT / "bench.py").read_text()
    assert "gen_matmul_inputs" not in source
    assert '"test_sizes": [' not in source
    assert "_ref_layernorm" not in source


# ---------------------------------------------------------------------------
# extract.py command line and generation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "flag", ["--report", "--top", "--kernel-type", "--backend", "--spec", "--spec-override"]
)
def test_extract_help_lists_expected_flags(flag):
    result = run_script("extract.py", "--help")
    assert result.returncode == 0, result.stderr
    assert flag in result.stdout


def test_extract_has_no_duplicated_operation_metadata():
    source = (REPO_ROOT / "extract.py").read_text()
    for removed in (
        "SHAPE_KEYS",
        "SHAPE_ALIAS_MAP",
        "TOLERANCES_MAP",
        "FLOPS_FN_SRC",
        "BYTES_FN_SRC",
        "SPEEDUP_ESTIMATES",
    ):
        assert removed not in source, f"{removed} must live in the spec registry"


def test_extract_parses_profiler_shapes_through_spec_aliases():
    extract = pytest.importorskip("extract")
    registry = create_builtin_registry()

    layernorm = registry.get("layernorm")
    assert extract.parse_shape_info("M=4096, N=2048", layernorm) == {
        "batch": 4096,
        "dim": 2048,
    }

    attention = registry.get("flash_attention")
    assert extract.parse_shape_info("B=1, H=32, N=4096, D=128", attention) == {
        "batch": 1,
        "heads": 32,
        "seq_len": 4096,
        "head_dim": 128,
    }

    matmul = registry.get("matmul")
    assert extract.parse_shape_info("M=8, N=9, K=10", matmul) == {"M": 8, "N": 9, "K": 10}
    assert extract.parse_shape_info("", matmul) is None
    assert extract.parse_shape_info("no numbers here", matmul) is None


def test_extract_default_shapes_match_pre_refactor_values():
    extract = pytest.importorskip("extract")
    registry = create_builtin_registry()
    assert extract.get_default_shape(registry.get("matmul")) == {
        "M": 2048, "N": 2048, "K": 2048
    }
    # 'reduce' deliberately keeps its pre-refactor 4096x4096 fallback even though
    # its 'large' benchmark size is 8192x8192.
    assert extract.get_default_shape(registry.get("reduce")) == {"M": 4096, "N": 4096}


def test_extract_generates_a_kernel_file_for_a_builtin():
    extract = pytest.importorskip("extract")
    spec = create_builtin_registry().get("matmul")
    starter = extract.read_starter_kernel(spec, backend="triton")
    assert starter is not None

    content = extract.generate_kernel_file(
        spec=spec,
        rank=1,
        pct_total=42.0,
        model_shape={"M": 4096, "N": 4096, "K": 4096},
        model_name="unit-test-model",
        gpu_time_ms=1.5,
        starter_code=starter,
        backend="triton",
    )

    assert 'KERNEL_TYPE = "matmul"' in content
    assert "MODEL_SHAPES = {'M': 4096, 'N': 4096, 'K': 4096}" in content
    assert "'float16': {'atol': 0.01, 'rtol': 0.01}" in content
    assert "return 2 * s['M'] * s['N'] * s['K']" in content
    assert "dt_bytes" in content
    # The generated file must be valid Python.
    compile(content, "generated_matmul.py", "exec")


def test_extract_generates_a_kernel_file_for_an_external_spec():
    extract = pytest.importorskip("extract")
    spec = load_spec(FIXTURE_LOCATOR)
    starter = extract.read_starter_kernel(spec, backend="triton")
    assert starter is not None

    content = extract.generate_kernel_file(
        spec=spec,
        rank=1,
        pct_total=0.0,
        model_shape=spec.extraction_shape(),
        model_name="external-spec",
        gpu_time_ms=0.0,
        starter_code=starter,
        backend="triton",
        spec_locator=FIXTURE_LOCATOR,
    )

    assert 'KERNEL_TYPE = "fixture_add"' in content
    assert f'KERNEL_SPEC = "{FIXTURE_LOCATOR}"' in content
    assert "return s['rows'] * s['cols']" in content
    compile(content, "generated_fixture_add.py", "exec")


def test_extract_falls_back_to_the_spec_when_accounting_is_opaque():
    extract = pytest.importorskip("extract")
    spec = load_spec(FIXTURE_LOCATOR)
    body = extract._accounting_body(lambda s: 1, spec, FIXTURE_LOCATOR)
    assert "NotImplementedError" in body
    assert FIXTURE_LOCATOR in body
    compile(f"def FLOPS_FN(s):\n    {body}\n", "opaque.py", "exec")


def test_extract_plan_uses_spec_speedup_estimates():
    extract = pytest.importorskip("extract")
    plan = extract.generate_optimization_plan(
        [
            {
                "rank": 1,
                "op_type": "matmul",
                "pct_total": 30.0,
                "gpu_time_ms": 2.0,
                "model_shape": {"M": 1, "N": 1, "K": 1},
                "output_file": "workspace/kernel_matmul_1.py",
                "estimated_speedup_potential": "2-3x",
            }
        ]
    )
    assert plan["total_optimization_targets"] == 1
    assert plan["covered_gpu_time_pct"] == 30.0
    assert plan["kernels_to_optimize"][0]["estimated_speedup_potential"] == "2-3x"


def test_extract_synthesizes_a_target_from_a_spec_alone():
    extract = pytest.importorskip("extract")
    spec = load_spec(FIXTURE_LOCATOR)
    entry = extract._synthetic_report_entry(spec)
    assert entry["op_type"] == "fixture_add"
    assert entry["autokernel_supported"] is True
    assert entry["shapes"] == spec.extraction_shape()


# ---------------------------------------------------------------------------
# bench.py shape-corpus command line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "flag", ["--shape-corpus", "--shape-corpus-only", "--check-backward"]
)
def test_bench_help_lists_verification_flags(flag):
    result = run_script("bench.py", "--help")
    assert result.returncode == 0, result.stderr
    assert flag in result.stdout


def test_bench_corpus_only_requires_corpus_path():
    result = run_script("bench.py", "--shape-corpus-only")
    assert result.returncode == 2
    assert "requires --shape-corpus" in result.stderr


def test_bench_invalid_corpus_fails_before_gpu_detection(tmp_path: Path):
    """An invalid corpus must fail before any GPU probing or allocation."""
    bad = tmp_path / "bad_corpus.json"
    bad.write_text(json.dumps({"schema_version": 1, "operation": "matmul", "cases": []}))
    result = run_script("bench.py", "--kernel", "matmul", "--shape-corpus", str(bad))
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "correctness: FAIL" in combined
    assert "throughput_tflops: 0.000" in combined
    assert str(bad) in combined
    # The GPU info block is printed only after detection; it must not appear.
    assert "gpu_name:" not in combined


def test_bench_corpus_operation_mismatch_is_actionable(tmp_path: Path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({
        "schema_version": 1,
        "operation": "some_other_op",
        "cases": [{"name": "a", "size": {"M": 4, "N": 4, "K": 4}}],
    }))
    result = run_script("bench.py", "--kernel", "matmul", "--shape-corpus", str(corpus))
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "does not match the selected spec" in combined

