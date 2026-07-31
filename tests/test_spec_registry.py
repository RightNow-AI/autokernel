"""Registry, validation and accounting behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from autokernel.specs import (
    DT_BYTES,
    BackwardSpec,
    CompileSpec,
    DuplicateSpecError,
    EdgeCase,
    KernelRegistry,
    KernelSpec,
    OutputSpec,
    SpecNotFoundError,
    SpecValidationError,
    Tolerance,
    canonical_dtype_name,
    create_builtin_registry,
    dtype_bytes,
    serialize_accounting,
    size,
    validate_spec,
)
from conftest import make_spec, spec_kwargs

# ---------------------------------------------------------------------------
# Registry behavior
# ---------------------------------------------------------------------------

def test_register_and_get_round_trip():
    registry = KernelRegistry()
    spec = make_spec(name="alpha")
    registry.register(spec)

    assert registry.get("alpha") is spec
    assert registry.contains("alpha")
    assert "alpha" in registry
    assert len(registry) == 1
    assert list(registry) == [spec]


def test_list_names_is_registration_ordered():
    registry = KernelRegistry(
        [make_spec(name="zeta"), make_spec(name="alpha"), make_spec(name="mid")]
    )
    assert registry.list_names() == ("zeta", "alpha", "mid")
    # Repeated calls are stable.
    assert registry.list_names() == registry.list_names()


def test_duplicate_registration_is_rejected():
    registry = KernelRegistry([make_spec(name="alpha")])
    with pytest.raises(DuplicateSpecError, match="already registered"):
        registry.register(make_spec(name="alpha"))


def test_duplicate_registration_with_override_replaces():
    first = make_spec(name="alpha")
    second = make_spec(name="alpha", speedup_estimate="9x")
    registry = KernelRegistry([first])
    registry.register(second, override=True)
    assert registry.get("alpha") is second
    assert registry.list_names() == ("alpha",)


def test_unknown_name_lists_available_specs():
    registry = KernelRegistry([make_spec(name="alpha")])
    with pytest.raises(SpecNotFoundError) as exc:
        registry.get("nope")
    assert "alpha" in str(exc.value)


def test_registry_rejects_non_specs():
    registry = KernelRegistry()
    with pytest.raises(SpecValidationError):
        registry.register("not a spec")  # type: ignore[arg-type]


def test_fresh_registries_are_isolated():
    a = create_builtin_registry()
    b = create_builtin_registry()
    a.register(make_spec(name="only_in_a"))
    assert a.contains("only_in_a")
    assert not b.contains("only_in_a")
    assert b.list_names() == create_builtin_registry().list_names()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_name", ["", "  ", "2fast", "has space", "has-dash", None, 7])
def test_reject_non_identifier_names(bad_name):
    with pytest.raises(SpecValidationError, match="name"):
        make_spec(name=bad_name)


def test_reject_duplicate_size_labels():
    sizes = [
        ("small", {"rows": 1, "cols": 1}),
        ("medium", {"rows": 2, "cols": 2}),
        ("large", {"rows": 3, "cols": 3}),
        ("small", {"rows": 4, "cols": 4}),
    ]
    with pytest.raises(SpecValidationError, match="duplicate size label 'small'"):
        make_spec(sizes=sizes)


@pytest.mark.parametrize("missing", ["small", "medium", "large"])
def test_reject_missing_standard_size(missing):
    sizes = {
        label: {"rows": 4, "cols": 4}
        for label in ("small", "medium", "large")
        if label != missing
    }
    spec = make_spec(sizes=sizes)
    with pytest.raises(SpecValidationError, match="missing required size label"):
        KernelRegistry().register(spec)


def test_reject_unknown_dtype():
    with pytest.raises(SpecValidationError, match="unknown dtype"):
        make_spec(dtypes=("float8",))


def test_reject_duplicate_dtype():
    with pytest.raises(SpecValidationError, match="duplicate dtype"):
        make_spec(dtypes=("float16", "float16"))


def test_reject_missing_tolerance_for_declared_dtype():
    with pytest.raises(SpecValidationError, match="missing tolerance for declared dtype"):
        make_spec(
            dtypes=("float16", "bfloat16"),
            tolerances={"float16": Tolerance(atol=1e-2, rtol=1e-2)},
        )


@pytest.mark.parametrize("atol,rtol", [(-1e-3, 1e-3), (1e-3, -1e-3)])
def test_reject_negative_tolerances(atol, rtol):
    with pytest.raises(SpecValidationError, match="non-negative"):
        Tolerance(atol=atol, rtol=rtol)


def test_reject_negative_tolerance_through_mapping():
    with pytest.raises(SpecValidationError, match="non-negative"):
        make_spec(dtypes=("float16",), tolerances={"float16": {"atol": -1.0, "rtol": 0.0}})


@pytest.mark.parametrize("atol,rtol", [(float("nan"), 1e-3), (1e-3, float("nan"))])
def test_reject_nan_tolerances(atol, rtol):
    with pytest.raises(SpecValidationError, match="must not be NaN"):
        Tolerance(atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "atol,rtol",
    [
        (float("inf"), 1e-3),
        (1e-3, float("inf")),
        (float("-inf"), 1e-3),
        (1e-3, float("-inf")),
    ],
)
def test_reject_infinite_tolerances(atol, rtol):
    with pytest.raises(SpecValidationError, match="must be finite"):
        Tolerance(atol=atol, rtol=rtol)


def test_reject_infinite_tolerance_through_mapping():
    with pytest.raises(SpecValidationError, match="must be finite"):
        make_spec(
            dtypes=("float16",),
            tolerances={"float16": {"atol": float("inf"), "rtol": 0.0}},
        )


def test_extra_tolerances_are_allowed():
    spec = make_spec(
        dtypes=("float16",),
        tolerances={
            "float16": Tolerance(atol=1e-2, rtol=1e-2),
            "float32": Tolerance(atol=1e-5, rtol=1e-5),
        },
    )
    assert set(spec.tolerances) == {"float16", "float32"}


def test_reject_missing_starter_kernel_file(tmp_path: Path):
    missing = tmp_path / "nope.py"
    spec = make_spec(starter_kernels={"triton": missing})
    with pytest.raises(SpecValidationError, match="starter kernel not found"):
        KernelRegistry().register(spec)


def test_accept_existing_starter_kernel_file(tmp_path: Path):
    present = tmp_path / "starter.py"
    present.write_text("KERNEL_TYPE = 'unit_op'\n")
    spec = make_spec(starter_kernels={"triton": present})
    assert spec.starter_kernel("triton") == present
    assert spec.starter_kernel("cuda") is None


def test_reject_inconsistent_shape_aliases():
    aliases = [("M", "rows"), ("M", "cols")]
    with pytest.raises(SpecValidationError, match="resolves inconsistently"):
        make_spec(shape_aliases=aliases)


def test_reject_alias_to_unknown_shape_key():
    with pytest.raises(SpecValidationError, match="not a shape key"):
        make_spec(shape_aliases={"M": "not_a_key"})


@pytest.mark.parametrize("field", ["reference_fn", "input_generator"])
def test_reject_non_callable_reference_and_generator(field):
    with pytest.raises(SpecValidationError, match="must be callable"):
        make_spec(**{field: "not callable"})


def test_reject_size_keys_that_do_not_match_shape_keys():
    with pytest.raises(SpecValidationError, match="do not match shape_keys"):
        make_spec(
            sizes={
                "small": {"rows": 1, "cols": 1},
                "medium": {"rows": 2, "cols": 2},
                "large": {"rows": 3, "oops": 3},
            }
        )


def test_reject_non_positive_size():
    with pytest.raises(SpecValidationError, match="must be positive"):
        make_spec(
            sizes={
                "small": {"rows": 0, "cols": 1},
                "medium": {"rows": 2, "cols": 2},
                "large": {"rows": 3, "cols": 3},
            }
        )


@pytest.mark.parametrize("bad_dimension", [0, -1, 1.5, True])
def test_reject_invalid_edge_case_dimension(bad_dimension):
    edge = EdgeCase(
        name="bad", size={"rows": bad_dimension, "cols": 1}
    )
    with pytest.raises(SpecValidationError, match=r"edge_cases\[0\]\.size"):
        make_spec(edge_cases=(edge,))


@pytest.mark.parametrize("bad_dimension", [0, -1, 1.5, True])
def test_reject_invalid_default_shape_dimension(bad_dimension):
    with pytest.raises(SpecValidationError, match="default_shape"):
        make_spec(default_shape={"rows": bad_dimension, "cols": 1})


def test_empty_starter_kernels_are_valid_for_benchmark_only_specs():
    spec = make_spec(starter_kernels={})
    assert spec.starter_kernel("triton") is None


def test_reject_duplicate_edge_case_names():
    edges = (
        EdgeCase(name="dup", size={"rows": 1, "cols": 1}),
        EdgeCase(name="dup", size={"rows": 2, "cols": 2}),
    )
    with pytest.raises(SpecValidationError, match="duplicate edge case name"):
        make_spec(edge_cases=edges)


def test_reject_edge_case_dtype_not_declared():
    edges = (EdgeCase(name="e", size={"rows": 1, "cols": 1}, dtype="bfloat16"),)
    with pytest.raises(SpecValidationError, match="is not declared in dtypes"):
        make_spec(dtypes=("float32",), tolerances={"float32": Tolerance(1e-5, 1e-5)}, edge_cases=edges)


def test_reject_edge_case_shape_mismatch():
    edges = (EdgeCase(name="e", size={"rows": 1}),)
    with pytest.raises(SpecValidationError, match="do not match shape_keys"):
        make_spec(edge_cases=edges)


def test_reject_flops_expression_with_unknown_key():
    with pytest.raises(SpecValidationError, match="unknown size key"):
        make_spec(flops_fn=size("nope") * 2)


def test_reject_flops_expression_that_uses_dtype_bytes():
    with pytest.raises(SpecValidationError, match="must not depend on dtype bytes"):
        make_spec(flops_fn=size("rows") * DT_BYTES)


def test_validation_error_names_spec_and_field():
    with pytest.raises(SpecValidationError) as exc:
        make_spec(name="named_op", dtypes=("float8",))
    message = str(exc.value)
    assert "'named_op'" in message
    assert "'dtypes'" in message


def test_validate_spec_can_skip_standard_size_requirement():
    narrow = make_spec(sizes={"small": {"rows": 1, "cols": 1}})
    # Construction and explicit opt-out are fine...
    validate_spec(narrow, require_standard_sizes=False)
    # ...but registration enforces the standard labels.
    with pytest.raises(SpecValidationError, match="missing required size label"):
        validate_spec(narrow)


def test_spec_is_immutable():
    spec = make_spec()
    with pytest.raises(Exception):
        spec.name = "other"  # type: ignore[misc]


def test_defensive_copy_of_sizes():
    sizes = {
        "small": {"rows": 1, "cols": 1},
        "medium": {"rows": 2, "cols": 2},
        "large": {"rows": 3, "cols": 3},
    }
    spec = make_spec(sizes=sizes)
    sizes["small"]["rows"] = 999
    assert spec.sizes["small"]["rows"] == 1


# ---------------------------------------------------------------------------
# Accounting expressions
# ---------------------------------------------------------------------------

def test_accounting_expression_evaluation_and_source():
    flops = 2 * size("M") * size("N") * size("K")
    assert flops({"M": 2, "N": 3, "K": 4}) == 48
    assert flops.to_source() == "2 * s['M'] * s['N'] * s['K']"
    assert flops.size_keys() == {"M", "N", "K"}
    assert not flops.uses_dtype_bytes()


def test_accounting_expression_parenthesizes_by_precedence():
    expr = (size("a") + size("b")) * DT_BYTES
    assert expr.to_source() == "(s['a'] + s['b']) * dt_bytes"
    assert expr({"a": 1, "b": 2}, 4) == 12
    assert expr.uses_dtype_bytes()


def test_accounting_power_is_right_associative_in_source():
    expr = 4 * (size("s") ** 2)
    assert expr.to_source() == "4 * s['s'] ** 2"
    assert expr({"s": 3}) == 36


def test_accounting_supports_reflected_power():
    expr = 2 ** size("s")
    assert expr.to_source() == "2 ** s['s']"
    assert expr({"s": 4}) == 16


def test_accounting_expression_requires_dtype_bytes_when_used():
    expr = size("rows") * DT_BYTES
    with pytest.raises(ValueError, match="dtype byte width"):
        expr({"rows": 4})


def test_accounting_expression_reports_missing_size_key():
    expr = size("rows")
    with pytest.raises(KeyError, match="rows"):
        expr({"cols": 4})


def test_serialize_accounting_returns_none_for_opaque_callable():
    assert serialize_accounting(lambda s: 1) is None
    assert serialize_accounting(size("rows")) == "s['rows']"


def test_serialized_accounting_is_valid_python():
    spec = create_builtin_registry().get("matmul")
    source = serialize_accounting(spec.bytes_fn)
    compiled = compile(source, "<accounting>", "eval")
    assert eval(compiled, {"s": {"M": 2, "N": 3, "K": 4}, "dt_bytes": 2}) == (
        spec.bytes_fn({"M": 2, "N": 3, "K": 4}, 2)
    )


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

def test_dtype_helpers_are_torch_free():
    assert dtype_bytes("float16") == 2
    assert dtype_bytes("bfloat16") == 2
    assert dtype_bytes("float32") == 4
    assert canonical_dtype_name("float32") == "float32"
    with pytest.raises(ValueError, match="unsupported dtype"):
        canonical_dtype_name("float8")


def test_resolve_torch_dtype_translates_only_in_runtime(torch_mod):
    from autokernel.specs import resolve_torch_dtype

    assert resolve_torch_dtype("bfloat16") is torch_mod.bfloat16
    assert resolve_torch_dtype(torch_mod.float32) is torch_mod.float32
    assert canonical_dtype_name(torch_mod.float16) == "float16"


def test_validate_spec_rejects_non_spec_objects():
    with pytest.raises(SpecValidationError, match="expected a KernelSpec"):
        validate_spec(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Verification policy types (Week 2)
# ---------------------------------------------------------------------------

def test_verification_policies_default_to_none():
    spec = make_spec()
    assert spec.output_spec is None
    assert spec.backward_spec is None
    assert spec.compile_spec is None


def test_output_spec_accepts_mapping_coercion():
    spec = make_spec(output_spec={"included_paths": ("output[0]",), "compare_non_tensors": False})
    assert isinstance(spec.output_spec, OutputSpec)
    assert spec.output_spec.included_paths == ("output[0]",)
    assert spec.output_spec.compare_non_tensors is False


def test_output_spec_rejects_duplicate_paths():
    with pytest.raises(SpecValidationError, match="duplicate path"):
        OutputSpec(included_paths=("output", "output"))


def test_output_and_backward_specs_reject_empty_output_paths():
    with pytest.raises(SpecValidationError, match="at least one path"):
        OutputSpec(included_paths=())
    with pytest.raises(SpecValidationError, match="at least one path"):
        BackwardSpec(differentiable_inputs=("x",), output_paths=())


def test_output_spec_rejects_unknown_mapping_keys():
    with pytest.raises(SpecValidationError, match="output_spec"):
        make_spec(output_spec={"nonsense": True})


def test_backward_spec_requires_differentiable_inputs():
    with pytest.raises(SpecValidationError, match="at least one input"):
        BackwardSpec(differentiable_inputs=())


def test_backward_spec_normalizes_tolerances():
    policy = BackwardSpec(
        differentiable_inputs=("x",),
        tolerances={"float32": {"atol": 1e-4, "rtol": 1e-4}},
    )
    assert policy.tolerances["float32"] == Tolerance(atol=1e-4, rtol=1e-4)


def test_backward_spec_rejects_unknown_tolerance_dtype():
    with pytest.raises(SpecValidationError, match="unknown dtype key"):
        BackwardSpec(differentiable_inputs=("x",), tolerances={"float64": Tolerance(0.0, 0.0)})


def test_backward_spec_rejects_infinite_tolerances():
    with pytest.raises(SpecValidationError, match="must be finite"):
        BackwardSpec(
            differentiable_inputs=("x",),
            tolerances={"float32": {"atol": float("inf"), "rtol": 0.0}},
        )


def test_compile_spec_rejects_non_bool_flags():
    with pytest.raises(SpecValidationError, match="CompileSpec.fullgraph must be a bool"):
        CompileSpec(fullgraph="yes")


def test_kernel_spec_rejects_wrongly_typed_policies():
    with pytest.raises(SpecValidationError, match="backward_spec"):
        make_spec(backward_spec="not-a-spec")
