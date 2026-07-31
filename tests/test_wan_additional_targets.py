"""CPU contracts for the remaining Wan overnight-campaign targets."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from autokernel.verification import (
    load_shape_corpus,
    validate_corpus_against_spec,
)
from models.wan_gated_residual import (
    SPEC as GATED_RESIDUAL_SPEC,
)
from models.wan_gated_residual import (
    gen_wan_gated_residual_inputs,
    wan_gated_residual_ref,
)
from models.wan_modulated_layer_norm import (
    SPEC as MODULATED_NORM_SPEC,
)
from models.wan_modulated_layer_norm import (
    gen_wan_modulated_layer_norm_inputs,
    wan_modulated_layer_norm_ref,
)


def test_additional_wan_specs_cover_production_corpora(repo_root):
    for spec, filename in (
        (GATED_RESIDUAL_SPEC, "wan_gated_residual_corpus.json"),
        (MODULATED_NORM_SPEC, "wan_modulated_layer_norm_corpus.json"),
    ):
        corpus = load_shape_corpus(repo_root / "models" / filename)
        validate_corpus_against_spec(corpus, spec)
        assert len(corpus.cases) == 5
        assert {case.size["hidden"] for case in corpus.cases} == {
            1536,
            5120,
        }


def test_wan_gated_residual_reference_matches_dtype_boundary():
    size = {"batch": 2, "tokens": 3, "hidden": 5}
    inputs = gen_wan_gated_residual_inputs(
        size, "bfloat16", "cpu", seed=13
    )
    actual = wan_gated_residual_ref(**inputs)
    expected = (
        inputs["residual"].float()
        + inputs["x"].float() * inputs["gate"][:, None, :]
    ).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_wan_modulated_norm_reference_matches_dtype_boundary():
    size = {"batch": 2, "tokens": 3, "hidden": 5}
    inputs = gen_wan_modulated_layer_norm_inputs(
        size, "bfloat16", "cpu", seed=17
    )
    actual = wan_modulated_layer_norm_ref(**inputs)
    normalized = F.layer_norm(
        inputs["x"].float(), (size["hidden"],), None, None, 1e-6
    )
    expected = (
        normalized * (1 + inputs["scale"][:, None, :])
        + inputs["shift"][:, None, :]
    ).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
