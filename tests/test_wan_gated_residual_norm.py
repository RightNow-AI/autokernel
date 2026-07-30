"""CPU contract tests for the first Wan production kernel specification."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from autokernel.verification.corpus import (
    load_shape_corpus,
    validate_corpus_against_spec,
)
from models.wan_gated_residual_norm import (
    SPEC,
    gen_wan_gated_residual_norm_inputs,
    wan_gated_residual_norm_ref,
)


def test_wan_spec_covers_production_widths_and_dtypes():
    assert SPEC.sizes["small"]["hidden"] == 1536
    assert SPEC.sizes["large"]["hidden"] == 5120
    assert SPEC.dtypes == ("bfloat16", "float16")
    assert SPEC.starter_kernels["triton"].name == "wan_gated_residual_norm.py"


def test_wan_production_corpus_matches_spec(repo_root):
    corpus = load_shape_corpus(
        repo_root / "models" / "wan_gated_residual_norm_corpus.json"
    )
    validate_corpus_against_spec(corpus, SPEC)
    assert len(corpus.cases) == 5
    assert {case.size["hidden"] for case in corpus.cases} == {1536, 5120}
    assert any("sequence-parallel-4" in case.tags for case in corpus.cases)


def test_wan_input_generator_is_deterministic_on_cpu():
    size = {"batch": 2, "tokens": 3, "hidden": 5}
    first = gen_wan_gated_residual_norm_inputs(size, "bfloat16", "cpu", seed=7)
    second = gen_wan_gated_residual_norm_inputs(size, "bfloat16", "cpu", seed=7)
    assert first.keys() == second.keys()
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_wan_reference_matches_explicit_fastvideo_dtype_boundary():
    size = {"batch": 2, "tokens": 3, "hidden": 5}
    inputs = gen_wan_gated_residual_norm_inputs(
        size, "bfloat16", "cpu", seed=11
    )
    normalized, updated = wan_gated_residual_norm_ref(**inputs)

    expected_updated_fp32 = (
        inputs["residual"].float()
        + inputs["x"].float() * inputs["gate"][:, None, :]
    )
    expected_normalized = F.layer_norm(
        expected_updated_fp32,
        (size["hidden"],),
        inputs["weight"],
        inputs["bias"],
        1e-6,
    ).to(torch.bfloat16)

    assert normalized.dtype == torch.bfloat16
    assert updated.dtype == torch.bfloat16
    torch.testing.assert_close(normalized, expected_normalized, rtol=0, atol=0)
    torch.testing.assert_close(
        updated, expected_updated_fp32.to(torch.bfloat16), rtol=0, atol=0
    )
