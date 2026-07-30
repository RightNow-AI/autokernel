"""Structured output-tree comparison on CPU."""

from __future__ import annotations

from collections import namedtuple

import pytest

from autokernel.specs import OutputSpec, Tolerance
from autokernel.verification import (
    OutputTreeError,
    compare_deterministic,
    compare_output_trees,
    flatten_output_tree,
    tree_has_nan_or_inf,
)

torch = pytest.importorskip("torch")

TOLS = {
    "float16": Tolerance(atol=1e-2, rtol=1e-2),
    "float32": Tolerance(atol=1e-5, rtol=1e-5),
}

Point = namedtuple("Point", ["x", "y"])


def _t(*values: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype)


# ---------------------------------------------------------------------------
# Flattening and paths
# ---------------------------------------------------------------------------

def test_flatten_single_tensor_has_root_path():
    leaves = flatten_output_tree(_t(1.0))
    assert [path for path, _ in leaves] == ["output"]


def test_flatten_tuple_and_list_paths():
    leaves = flatten_output_tree((_t(1.0), [_t(2.0), _t(3.0)]))
    assert [path for path, _ in leaves] == ["output[0]", "output[1][0]", "output[1][1]"]


def test_flatten_dict_paths_are_sorted_not_insertion_ordered():
    first = flatten_output_tree({"z": _t(1.0), "a": _t(2.0)})
    second = flatten_output_tree({"a": _t(2.0), "z": _t(1.0)})
    assert [path for path, _ in first] == ['output["a"]', 'output["z"]']
    assert [path for path, _ in first] == [path for path, _ in second]


def test_flatten_namedtuple_uses_field_names():
    leaves = flatten_output_tree(Point(x=_t(1.0), y=_t(2.0)))
    assert [path for path, _ in leaves] == ["output.x", "output.y"]


def test_flatten_nested_combination_matches_brief_example():
    tree = {"output": _t(1.0), "aux": (_t(2.0), 3)}
    leaves = flatten_output_tree(tree)
    assert [path for path, _ in leaves] == ['output["aux"][0]', 'output["aux"][1]', 'output["output"]']
    assert leaves[1][1] == 3  # metadata leaf kept as-is


def test_flatten_empty_containers_become_leaves():
    leaves = flatten_output_tree({"empty": {}, "items": []})
    assert [path for path, _ in leaves] == ['output["empty"]', 'output["items"]']


# ---------------------------------------------------------------------------
# Comparison: matching trees
# ---------------------------------------------------------------------------

def test_single_tensor_match():
    cmp = compare_output_trees(_t(1.0, 2.0), _t(1.0, 2.0), TOLS)
    assert cmp.match
    assert cmp.structure_match
    assert cmp.worst_abs_error == 0.0
    assert cmp.leaf_records()[0]["path"] == "output"


def test_nested_tree_match_with_metadata():
    ref = {"output": _t(1.0), "aux": (_t(2.0), 3)}
    cmp = compare_output_trees(ref, ref, TOLS)
    assert cmp.match
    assert len(cmp.leaves) == 3
    kinds = {leaf.path: leaf.kind for leaf in cmp.leaves}
    assert kinds['output["aux"][1]'] == "metadata"


def test_close_values_within_dtype_tolerance():
    candidate = _t(1.0, dtype=torch.float16) + 1e-3
    expected = _t(1.0, dtype=torch.float16)
    cmp = compare_output_trees(candidate, expected, TOLS)
    assert cmp.match


# ---------------------------------------------------------------------------
# Comparison: failures
# ---------------------------------------------------------------------------

def test_shape_mismatch_fails_with_path():
    cmp = compare_output_trees(
        {"output": torch.zeros(2, 3)}, {"output": torch.zeros(3, 2)}, TOLS
    )
    assert not cmp.match
    assert "shape mismatch" in cmp.reason


def test_single_tensor_reason_matches_legacy_format():
    cmp = compare_output_trees(_t(1.0), _t(2.0), TOLS)
    assert not cmp.match
    assert cmp.reason.startswith("max_abs_error=")
    assert "exceeds tol(atol=" in cmp.reason


def test_nested_failure_reason_is_prefixed_with_path():
    candidate = {"output": _t(1.0), "aux": (_t(2.0), 3)}
    expected = {"output": _t(1.0), "aux": (_t(9.0), 3)}
    cmp = compare_output_trees(candidate, expected, TOLS)
    assert not cmp.match
    assert cmp.reason.startswith('output["aux"][0]: ')


def test_missing_output_path_fails():
    cmp = compare_output_trees(_t(1.0), (_t(1.0), _t(2.0)), TOLS)
    assert not cmp.match
    assert not cmp.structure_match
    assert "missing output path(s)" in cmp.reason


def test_unexpected_output_path_fails():
    cmp = compare_output_trees((_t(1.0), _t(2.0)), _t(1.0), TOLS)
    assert not cmp.match
    assert "unexpected output path(s)" in cmp.reason


def test_leaf_kind_mismatch_fails():
    cmp = compare_output_trees({"a": _t(1.0)}, {"a": 1.0}, TOLS)
    assert not cmp.match
    assert "leaf kind mismatch" in cmp.reason


def test_dtype_mismatch_fails():
    cmp = compare_output_trees(_t(1.0, dtype=torch.float16), _t(1.0), TOLS)
    assert not cmp.match
    assert "dtype mismatch" in cmp.reason


def test_per_leaf_tolerance_selection_by_leaf_dtype():
    """A float16 leaf gets the float16 tolerance, a float32 leaf float32's."""
    candidate = (
        _t(1.0, dtype=torch.float16) + 5e-3,   # within float16 tol (1e-2)
        _t(1.0, dtype=torch.float32) + 5e-3,   # outside float32 tol (1e-5)
    )
    expected = (_t(1.0, dtype=torch.float16), _t(1.0, dtype=torch.float32))
    cmp = compare_output_trees(candidate, expected, TOLS)
    assert not cmp.match
    by_path = {leaf.path: leaf for leaf in cmp.leaves}
    assert by_path["output[0]"].match
    assert not by_path["output[1]"].match


def test_integer_tensors_compare_exactly():
    cmp = compare_output_trees(_t(1, 2, dtype=torch.int64), _t(1, 2, dtype=torch.int64), TOLS)
    assert cmp.match
    cmp = compare_output_trees(_t(1, 2, dtype=torch.int64), _t(1, 3, dtype=torch.int64), TOLS)
    assert not cmp.match
    assert "bitwise" in cmp.leaves[0].reason


# ---------------------------------------------------------------------------
# NaN / infinity
# ---------------------------------------------------------------------------

def test_nan_detected_per_path():
    candidate = {"ok": _t(1.0), "bad": _t(float("nan"))}
    expected = {"ok": _t(1.0), "bad": _t(0.0)}
    cmp = compare_output_trees(candidate, expected, TOLS)
    assert not cmp.match
    by_path = {leaf.path: leaf for leaf in cmp.leaves}
    assert by_path['output["bad"]'].has_nan
    assert not by_path['output["ok"]'].has_nan
    assert "NaN" in by_path['output["bad"]'].reason


def test_infinity_detected_per_path():
    cmp = compare_output_trees(_t(float("inf")), _t(1.0), TOLS)
    assert not cmp.match
    assert cmp.leaves[0].has_inf
    assert "infinity" in cmp.leaves[0].reason


def test_tree_has_nan_or_inf_traverses_containers():
    assert not tree_has_nan_or_inf({"a": (_t(1.0), 3)})
    assert tree_has_nan_or_inf({"a": (_t(float("nan")), 3)})
    assert tree_has_nan_or_inf((_t(1.0), [_t(float("inf"))]))


# ---------------------------------------------------------------------------
# Metadata comparison policy
# ---------------------------------------------------------------------------

def test_metadata_mismatch_fails_by_default():
    cmp = compare_output_trees((_t(1.0), 3), (_t(1.0), 4), TOLS)
    assert not cmp.match
    assert "metadata mismatch" in cmp.leaves[1].reason


def test_metadata_mismatch_ignored_when_disabled():
    policy = OutputSpec(compare_non_tensors=False)
    cmp = compare_output_trees((_t(1.0), 3), (_t(1.0), 4), TOLS, output_spec=policy)
    assert cmp.match
    assert cmp.leaves[1].reason == "not compared"


def test_included_paths_restricts_comparison():
    policy = OutputSpec(included_paths=('output["output"]',))
    candidate = {"output": _t(1.0), "aux": (_t(9.0), 4)}
    expected = {"output": _t(1.0), "aux": (_t(2.0), 3)}
    cmp = compare_output_trees(candidate, expected, TOLS, output_spec=policy)
    assert cmp.match
    assert [leaf.path for leaf in cmp.leaves] == ['output["output"]']


def test_included_paths_must_exist():
    policy = OutputSpec(included_paths=("output[9]",))
    with pytest.raises(OutputTreeError, match="not present"):
        compare_output_trees((_t(1.0),), (_t(1.0),), TOLS, output_spec=policy)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism_bitwise_identical_trees():
    tree = {"output": _t(1.5, -2.5), "aux": (_t(3.0), 3)}
    cmp = compare_deterministic(tree, tree)
    assert cmp.match


def test_determinism_detects_any_leaf_difference():
    first = {"output": _t(1.0), "aux": (_t(2.0), 3)}
    second = {"output": _t(1.0), "aux": (_t(2.0 + 1e-3), 3)}
    cmp = compare_deterministic(first, second)
    assert not cmp.match
    failure = cmp.first_failure()
    assert failure is not None
    assert failure.path == 'output["aux"][0]'
    assert failure.max_abs_error is not None


def test_determinism_reports_distinct_mean_and_max_errors():
    cmp = compare_deterministic(_t(0.0, 0.0), _t(0.0, 2.0))
    failure = cmp.first_failure()
    assert failure is not None
    assert failure.max_abs_error == pytest.approx(2.0)
    assert failure.mean_abs_error == pytest.approx(1.0)


def test_determinism_detects_metadata_change():
    cmp = compare_deterministic((_t(1.0), 3), (_t(1.0), 4))
    assert not cmp.match
    assert "metadata changed" in cmp.leaves[1].reason


def test_determinism_detects_structure_change():
    cmp = compare_deterministic(_t(1.0), (_t(1.0),))
    assert not cmp.match
    assert not cmp.structure_match


# ---------------------------------------------------------------------------
# Relax factor (numerical-stability stage)
# ---------------------------------------------------------------------------

def test_relax_multiplies_tolerances():
    candidate = _t(1.0) + 5e-5
    expected = _t(1.0)
    strict = compare_output_trees(candidate, expected, TOLS)
    assert not strict.match
    relaxed = compare_output_trees(candidate, expected, TOLS, relax=10.0)
    assert relaxed.match
