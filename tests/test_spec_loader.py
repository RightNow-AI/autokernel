"""External specification loading."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from conftest import FIXTURES_DIR, make_spec

from autokernel.specs import (
    KernelRegistry,
    KernelSpec,
    SpecCollisionError,
    SpecLoadError,
    SpecValidationError,
    create_builtin_registry,
    load_spec,
    parse_locator,
    resolve_spec,
)

FIXTURE_FILE = FIXTURES_DIR / "custom_add.py"


# ---------------------------------------------------------------------------
# Locator parsing
# ---------------------------------------------------------------------------

def test_parse_locator_splits_module_and_attribute():
    assert parse_locator("package.module:SPEC") == ("package.module", "SPEC")


def test_parse_locator_splits_at_last_colon():
    assert parse_locator(r"C:\ops\spec.py:SPEC") == (r"C:\ops\spec.py", "SPEC")


@pytest.mark.parametrize("bad", ["", "   ", "no_attribute", "module:", ":SPEC", None, 5])
def test_parse_locator_rejects_malformed_input(bad):
    with pytest.raises(SpecLoadError):
        parse_locator(bad)


# ---------------------------------------------------------------------------
# File locators
# ---------------------------------------------------------------------------

def test_load_absolute_file_locator():
    spec = load_spec(f"{FIXTURE_FILE}:SPEC")
    assert isinstance(spec, KernelSpec)
    assert spec.name == "fixture_add"


def test_load_relative_file_locator(in_repo_root):
    spec = load_spec("tests/fixtures/custom_add.py:SPEC")
    assert spec.name == "fixture_add"


def test_load_callable_factory_from_file():
    spec = load_spec(f"{FIXTURE_FILE}:SPEC_FACTORY")
    assert spec.name == "fixture_add"


def test_missing_file_reports_locator():
    locator = "/definitely/not/here/spec.py:SPEC"
    with pytest.raises(SpecLoadError) as exc:
        load_spec(locator)
    message = str(exc.value)
    assert locator in message
    assert "file not found" in message


def test_file_that_fails_to_import_reports_locator(tmp_path: Path):
    bad = tmp_path / "explodes.py"
    bad.write_text("raise RuntimeError('boom')\n")
    with pytest.raises(SpecLoadError) as exc:
        load_spec(f"{bad}:SPEC")
    assert "boom" in str(exc.value)
    assert str(bad) in str(exc.value)


def test_file_loading_does_not_mutate_sys_path(tmp_path: Path):
    before = list(sys.path)
    load_spec(f"{FIXTURE_FILE}:SPEC")
    assert sys.path == before


def test_file_loading_does_not_leak_temporary_modules():
    before = set(sys.modules)
    load_spec(f"{FIXTURE_FILE}:SPEC")
    load_spec(f"{FIXTURE_FILE}:SPEC")
    added = [name for name in set(sys.modules) - before if "external_spec" in name]
    assert added == []


# ---------------------------------------------------------------------------
# Module locators
# ---------------------------------------------------------------------------

def test_load_module_locator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module_dir = tmp_path / "pkgdir"
    module_dir.mkdir()
    (module_dir / "external_op.py").write_text(
        "from autokernel.specs import DT_BYTES, KernelSpec, Tolerance, size\n"
        "\n"
        "def ref(x):\n"
        "    return x\n"
        "\n"
        "def gen(size_map, dtype, device, seed=42):\n"
        "    return {'x': None}\n"
        "\n"
        "SPEC = KernelSpec(\n"
        "    name='module_op',\n"
        "    reference_fn=ref,\n"
        "    input_generator=gen,\n"
        "    sizes={'small': {'n': 1}, 'medium': {'n': 2}, 'large': {'n': 3}},\n"
        "    dtypes=('float32',),\n"
        "    tolerances={'float32': Tolerance(atol=1e-5, rtol=1e-5)},\n"
        "    flops_fn=size('n'),\n"
        "    bytes_fn=size('n') * DT_BYTES,\n"
        "    shape_keys=('n',),\n"
        ")\n"
    )
    monkeypatch.syspath_prepend(str(module_dir))
    spec = load_spec("external_op:SPEC")
    assert spec.name == "module_op"


def test_load_module_locator_callable_factory():
    # A built-in factory doubles as a module-locator + zero-argument factory case.
    spec = load_spec("autokernel.specs.builtins:_spec_matmul")
    assert spec.name == "matmul"


def test_missing_module_reports_locator():
    with pytest.raises(SpecLoadError) as exc:
        load_spec("definitely_not_a_module:SPEC")
    assert "definitely_not_a_module:SPEC" in str(exc.value)
    assert "not found" in str(exc.value)


# ---------------------------------------------------------------------------
# Attribute and type errors
# ---------------------------------------------------------------------------

def test_missing_attribute_lists_available_specs():
    with pytest.raises(SpecLoadError) as exc:
        load_spec(f"{FIXTURE_FILE}:NOPE")
    message = str(exc.value)
    assert "has no attribute 'NOPE'" in message
    assert "SPEC" in message


def test_attribute_of_wrong_type_is_rejected():
    with pytest.raises(SpecLoadError, match="expected a KernelSpec"):
        load_spec(f"{FIXTURE_FILE}:NOT_A_SPEC")


def test_factory_returning_wrong_type_is_rejected():
    with pytest.raises(SpecLoadError, match="returned int"):
        load_spec(f"{FIXTURE_FILE}:BAD_FACTORY")


def test_factory_that_raises_is_reported():
    with pytest.raises(SpecLoadError, match="factory exploded"):
        load_spec(f"{FIXTURE_FILE}:RAISING_FACTORY")


def test_invalid_spec_in_file_is_rejected(tmp_path: Path):
    bad = tmp_path / "bad_spec.py"
    bad.write_text(
        "from autokernel.specs import DT_BYTES, KernelSpec, Tolerance, size\n"
        "SPEC = KernelSpec(\n"
        "    name='bad op',\n"
        "    reference_fn=lambda x: x,\n"
        "    input_generator=lambda *a, **k: {},\n"
        "    sizes={'small': {'n': 1}, 'medium': {'n': 2}, 'large': {'n': 3}},\n"
        "    dtypes=('float32',),\n"
        "    tolerances={'float32': Tolerance(atol=1e-5, rtol=1e-5)},\n"
        "    flops_fn=size('n'),\n"
        "    bytes_fn=size('n') * DT_BYTES,\n"
        "    shape_keys=('n',),\n"
        ")\n"
    )
    with pytest.raises(SpecLoadError) as exc:
        load_spec(f"{bad}:SPEC")
    assert "identifier-like" in str(exc.value)


def test_starter_kernel_must_exist_for_loaded_spec(tmp_path: Path):
    bad = tmp_path / "missing_starter.py"
    bad.write_text(
        "from autokernel.specs import DT_BYTES, KernelSpec, Tolerance, size\n"
        "SPEC = KernelSpec(\n"
        "    name='no_starter',\n"
        "    reference_fn=lambda x: x,\n"
        "    input_generator=lambda *a, **k: {},\n"
        "    sizes={'small': {'n': 1}, 'medium': {'n': 2}, 'large': {'n': 3}},\n"
        "    dtypes=('float32',),\n"
        "    tolerances={'float32': Tolerance(atol=1e-5, rtol=1e-5)},\n"
        "    flops_fn=size('n'),\n"
        "    bytes_fn=size('n') * DT_BYTES,\n"
        "    shape_keys=('n',),\n"
        "    starter_kernels={'triton': '/nope/does_not_exist.py'},\n"
        ")\n"
    )
    with pytest.raises(SpecLoadError) as exc:
        load_spec(f"{bad}:SPEC")
    assert "starter kernel not found" in str(exc.value)


# ---------------------------------------------------------------------------
# Collisions and selection
# ---------------------------------------------------------------------------

def test_collision_with_builtin_is_rejected_by_default():
    registry = create_builtin_registry()
    with pytest.raises(SpecCollisionError, match="already registered"):
        load_spec(f"{FIXTURE_FILE}:COLLIDING_SPEC", registry=registry)


def test_collision_is_allowed_with_override():
    registry = create_builtin_registry()
    spec = load_spec(f"{FIXTURE_FILE}:COLLIDING_SPEC", registry=registry, override=True)
    assert spec.name == "matmul"


def test_no_collision_when_registry_not_supplied():
    spec = load_spec(f"{FIXTURE_FILE}:COLLIDING_SPEC")
    assert spec.name == "matmul"


def test_resolve_spec_prefers_locator_over_name():
    spec, registry = resolve_spec(
        spec_locator=f"{FIXTURE_FILE}:SPEC", name="matmul"
    )
    assert spec.name == "fixture_add"
    assert registry.contains("fixture_add")
    assert registry.contains("matmul")


def test_resolve_spec_falls_back_to_name():
    spec, registry = resolve_spec(name="rmsnorm")
    assert spec.name == "rmsnorm"


def test_resolve_spec_requires_a_selection():
    with pytest.raises(SpecLoadError, match="no operation selected"):
        resolve_spec()


def test_resolve_spec_registers_into_the_supplied_registry_only():
    isolated = KernelRegistry([make_spec(name="only_here")])
    spec, registry = resolve_spec(spec_locator=f"{FIXTURE_FILE}:SPEC", registry=isolated)
    assert registry is isolated
    assert isolated.list_names() == ("only_here", "fixture_add")
    assert not create_builtin_registry().contains("fixture_add")


def test_resolve_spec_collision_respects_override():
    registry = create_builtin_registry()
    with pytest.raises(SpecCollisionError):
        resolve_spec(spec_locator=f"{FIXTURE_FILE}:COLLIDING_SPEC", registry=registry)
    spec, registry2 = resolve_spec(
        spec_locator=f"{FIXTURE_FILE}:COLLIDING_SPEC",
        registry=create_builtin_registry(),
        override=True,
    )
    assert spec.name == "matmul"
    assert registry2.get("matmul") is spec


# ---------------------------------------------------------------------------
# The shipped example
# ---------------------------------------------------------------------------

def test_example_custom_op_is_discoverable(repo_root: Path):
    example = repo_root / "examples" / "custom_ops" / "add.py"
    spec = load_spec(f"{example}:SPEC", registry=create_builtin_registry())
    assert spec.name == "custom_add"
    assert set(spec.sizes) >= {"small", "medium", "large"}
    assert spec.starter_kernel("triton").is_file()
    assert spec.flops_fn(spec.sizes["large"]) == 4096 * 4096
    assert spec.bytes_fn(spec.sizes["large"], 2) == 3 * 4096 * 4096 * 2


def test_example_custom_op_inputs_are_deterministic(torch_mod, repo_root: Path):
    example = repo_root / "examples" / "custom_ops" / "add.py"
    spec = load_spec(f"{example}:SPEC")
    first = spec.input_generator(spec.sizes["small"], "float32", "cpu", 42)
    second = spec.input_generator(spec.sizes["small"], "float32", "cpu", 42)
    other = spec.input_generator(spec.sizes["small"], "float32", "cpu", 7)
    assert set(first) == {"x", "y"}
    for key in first:
        assert torch_mod.equal(first[key], second[key])
    assert not torch_mod.equal(first["x"], other["x"])
    expected = spec.reference_fn(**first)
    assert torch_mod.equal(expected, first["x"] + first["y"])
