"""Shape-corpus loading, validation and weighted aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_spec

from autokernel.verification import (
    CorpusError,
    load_shape_corpus,
    validate_corpus_against_spec,
    weighted_aggregate,
)


def _write(tmp_path: Path, payload: object, name: str = "corpus.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def _valid_payload(**overrides):
    payload = {
        "schema_version": 1,
        "operation": "unit_op",
        "cases": [
            {"name": "prod-a", "size": {"rows": 100, "cols": 200}, "dtype": "float16",
             "weight": 3, "tags": ["production"]},
            {"name": "prod-b", "size": {"rows": 7, "cols": 9}},
        ],
    }
    payload.update(overrides)
    return payload


def _spec(**overrides):
    return make_spec(**overrides)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_load_valid_corpus(tmp_path):
    corpus = load_shape_corpus(_write(tmp_path, _valid_payload()))
    assert corpus.operation == "unit_op"
    assert corpus.schema_version == 1
    assert len(corpus.cases) == 2
    first, second = corpus.cases
    assert first.name == "prod-a"
    assert first.size == {"rows": 100, "cols": 200}
    assert first.dtype == "float16"
    assert first.weight == 3
    assert first.tags == ("production",)
    # defaults
    assert second.dtype is None
    assert second.weight == 1
    assert second.tags == ()


def test_valid_corpus_passes_spec_validation(tmp_path):
    corpus = load_shape_corpus(_write(tmp_path, _valid_payload()))
    validate_corpus_against_spec(corpus, _spec())  # must not raise


def test_example_corpus_matches_example_spec(repo_root):
    from autokernel.specs import load_spec

    corpus = load_shape_corpus(repo_root / "examples" / "custom_ops" / "affine_corpus.json")
    spec = load_spec(str(repo_root / "examples" / "custom_ops" / "affine.py") + ":SPEC")
    validate_corpus_against_spec(corpus, spec)


# ---------------------------------------------------------------------------
# File and schema errors
# ---------------------------------------------------------------------------

def test_missing_file_fails(tmp_path):
    with pytest.raises(CorpusError, match="file not found"):
        load_shape_corpus(tmp_path / "nope.json")


def test_invalid_json_fails(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(CorpusError, match="invalid JSON"):
        load_shape_corpus(path)


def test_top_level_must_be_object(tmp_path):
    with pytest.raises(CorpusError, match="top level must be an object"):
        load_shape_corpus(_write(tmp_path, [1, 2, 3]))


def test_unknown_top_level_field_fails(tmp_path):
    with pytest.raises(CorpusError, match="unknown top-level field"):
        load_shape_corpus(_write(tmp_path, _valid_payload(tensors={"a": [1, 2]})))


@pytest.mark.parametrize("version", [None, "1", 1.0, True])
def test_schema_version_must_be_integer(tmp_path, version):
    payload = _valid_payload()
    if version is None:
        del payload["schema_version"]
    else:
        payload["schema_version"] = version
    with pytest.raises(CorpusError, match="schema_version.*must be an integer"):
        load_shape_corpus(_write(tmp_path, payload))


def test_unsupported_schema_version_fails(tmp_path):
    with pytest.raises(CorpusError, match="unsupported schema_version 2"):
        load_shape_corpus(_write(tmp_path, _valid_payload(schema_version=2)))


def test_operation_must_be_non_empty(tmp_path):
    with pytest.raises(CorpusError, match="operation"):
        load_shape_corpus(_write(tmp_path, _valid_payload(operation="")))


def test_cases_must_be_non_empty(tmp_path):
    with pytest.raises(CorpusError, match="cases.*non-empty"):
        load_shape_corpus(_write(tmp_path, _valid_payload(cases=[])))


# ---------------------------------------------------------------------------
# Case-level errors
# ---------------------------------------------------------------------------

def _one_case(tmp_path: Path, case: dict) -> Path:
    return _write(tmp_path, _valid_payload(cases=[case]))


def test_case_must_be_object(tmp_path):
    with pytest.raises(CorpusError, match="case #0 must be an object"):
        load_shape_corpus(_write(tmp_path, _valid_payload(cases=[42])))


def test_unknown_case_field_fails(tmp_path):
    case = {"name": "a", "size": {"rows": 1, "cols": 1}, "activations": [0.1]}
    with pytest.raises(CorpusError, match="unknown field"):
        load_shape_corpus(_one_case(tmp_path, case))


@pytest.mark.parametrize("name", ["", 7, None])
def test_bad_case_name_fails(tmp_path, name):
    case = {"name": name, "size": {"rows": 1, "cols": 1}}
    with pytest.raises(CorpusError, match="name"):
        load_shape_corpus(_one_case(tmp_path, case))


def test_duplicate_case_names_fail(tmp_path):
    payload = _valid_payload()
    payload["cases"][1] = dict(payload["cases"][0])
    with pytest.raises(CorpusError, match="duplicate case name 'prod-a'"):
        load_shape_corpus(_write(tmp_path, payload))


def test_size_must_be_non_empty_mapping(tmp_path):
    with pytest.raises(CorpusError, match="size"):
        load_shape_corpus(_one_case(tmp_path, {"name": "a", "size": {}}))


@pytest.mark.parametrize("value", [0, -3, 1.5, "64", True])
def test_size_values_must_be_positive_ints(tmp_path, value):
    case = {"name": "a", "size": {"rows": value, "cols": 4}}
    with pytest.raises(CorpusError, match="positive integer"):
        load_shape_corpus(_one_case(tmp_path, case))


def test_unknown_case_dtype_fails(tmp_path):
    case = {"name": "a", "size": {"rows": 1, "cols": 1}, "dtype": "float64"}
    with pytest.raises(CorpusError, match="unknown dtype"):
        load_shape_corpus(_one_case(tmp_path, case))


@pytest.mark.parametrize("weight", [0, -2, 2.5, "3", False])
def test_weight_must_be_positive_int(tmp_path, weight):
    case = {"name": "a", "size": {"rows": 1, "cols": 1}, "weight": weight}
    with pytest.raises(CorpusError, match="weight.*positive integer"):
        load_shape_corpus(_one_case(tmp_path, case))


def test_tags_must_be_strings(tmp_path):
    case = {"name": "a", "size": {"rows": 1, "cols": 1}, "tags": ["ok", 3]}
    with pytest.raises(CorpusError, match="tags"):
        load_shape_corpus(_one_case(tmp_path, case))


def test_duplicate_tags_fail(tmp_path):
    case = {"name": "a", "size": {"rows": 1, "cols": 1}, "tags": ["x", "x"]}
    with pytest.raises(CorpusError, match="duplicate tag"):
        load_shape_corpus(_one_case(tmp_path, case))


# ---------------------------------------------------------------------------
# Spec compatibility
# ---------------------------------------------------------------------------

def test_operation_mismatch_fails(tmp_path):
    corpus = load_shape_corpus(_write(tmp_path, _valid_payload(operation="other_op")))
    with pytest.raises(CorpusError, match="does not match the selected spec"):
        validate_corpus_against_spec(corpus, _spec())


def test_size_keys_must_match_shape_keys(tmp_path):
    corpus = load_shape_corpus(
        _one_case(tmp_path, {"name": "a", "size": {"rows": 1, "wrong": 2}})
    )
    with pytest.raises(CorpusError, match="do not match shape_keys"):
        validate_corpus_against_spec(corpus, _spec())


def test_case_dtype_must_be_declared_by_spec(tmp_path):
    case = {"name": "a", "size": {"rows": 1, "cols": 1}, "dtype": "bfloat16"}
    corpus = load_shape_corpus(_one_case(tmp_path, case))
    with pytest.raises(CorpusError, match="not declared in dtypes"):
        validate_corpus_against_spec(corpus, _spec())


def test_resolved_duplicate_cases_fail(tmp_path):
    """Two cases resolving to the same (size, dtype) config are rejected."""
    payload = _valid_payload(
        cases=[
            {"name": "a", "size": {"rows": 4, "cols": 4}},  # dtype None -> primary float16
            {"name": "b", "size": {"rows": 4, "cols": 4}, "dtype": "float16"},
        ]
    )
    corpus = load_shape_corpus(_write(tmp_path, payload))
    with pytest.raises(CorpusError, match="resolve to the same"):
        validate_corpus_against_spec(corpus, _spec())


# ---------------------------------------------------------------------------
# Weighted aggregation
# ---------------------------------------------------------------------------

def test_weighted_aggregate_math():
    entries = [
        {"dtype": "torch.float16", "weight": 3, "kernel_ms": 1.0, "ref_ms": 3.0},
        {"dtype": "torch.float16", "weight": 1, "kernel_ms": 3.0, "ref_ms": 9.0},
    ]
    agg = weighted_aggregate(entries)
    assert set(agg) == {"torch.float16"}
    group = agg["torch.float16"]
    assert group["cases"] == 2
    assert group["weight"] == 4
    assert group["kernel_ms"] == pytest.approx((3 * 1.0 + 1 * 3.0) / 4)
    assert group["ref_ms"] == pytest.approx((3 * 3.0 + 1 * 9.0) / 4)
    assert group["speedup"] == pytest.approx(group["ref_ms"] / group["kernel_ms"])


def test_weighted_aggregate_never_mixes_dtypes():
    entries = [
        {"dtype": "torch.float16", "weight": 1, "kernel_ms": 1.0, "ref_ms": 2.0},
        {"dtype": "torch.float32", "weight": 1, "kernel_ms": 5.0, "ref_ms": 10.0},
    ]
    agg = weighted_aggregate(entries)
    assert set(agg) == {"torch.float16", "torch.float32"}
    assert agg["torch.float16"]["kernel_ms"] == pytest.approx(1.0)
    assert agg["torch.float32"]["kernel_ms"] == pytest.approx(5.0)


def test_weighted_aggregate_zero_kernel_latency_guards_division():
    agg = weighted_aggregate(
        [{"dtype": "torch.float16", "weight": 1, "kernel_ms": 0.0, "ref_ms": 1.0}]
    )
    assert agg["torch.float16"]["speedup"] == 0.0

