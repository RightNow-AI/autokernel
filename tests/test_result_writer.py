"""Tests for versioned atomic result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import autokernel.verification.results as results


def test_result_envelope_is_versioned():
    payload = results.result_envelope("affine", forward={"status": "PASS"})
    assert payload["schema_version"] == results.RESULT_SCHEMA_VERSION
    assert payload["operation"] == "affine"
    assert payload["created_at"].endswith("+00:00")


def test_write_result_atomic_replaces_destination(tmp_path: Path):
    destination = tmp_path / "nested" / "bench_result.json"
    destination.parent.mkdir()
    destination.write_text('{"old": true}\n')

    written = results.write_result_atomic(destination, {"new": True})

    assert written == destination
    assert json.loads(destination.read_text()) == {"new": True}
    assert list(destination.parent.glob("*.tmp")) == []
    assert list(destination.parent.glob(".*.tmp")) == []


def test_write_result_atomic_serializes_non_finite_values_safely(tmp_path: Path):
    destination = tmp_path / "bench_result.json"
    results.write_result_atomic(destination, {"bad": float("nan")})
    assert json.loads(destination.read_text()) == {"bad": "nan"}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_write_result_atomic_preserves_old_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "bench_result.json"
    destination.write_text('{"old": true}\n')

    def fail_replace(source, target):
        raise OSError("interrupted")

    monkeypatch.setattr(results.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        results.write_result_atomic(destination, {"new": True})

    assert json.loads(destination.read_text()) == {"old": True}
    assert list(tmp_path.glob(".*.tmp")) == []
