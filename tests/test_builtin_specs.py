"""Compatibility freeze for the nine built-in operations.

Every value below was captured from ``bench.py::KERNEL_CONFIGS`` and
``extract.py``'s metadata maps *before* the registry refactor. A failure here
means benchmark coverage, tolerances or accounting changed -- which must be a
deliberate, reviewed decision, not a side effect.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from conftest import REPO_ROOT

from autokernel.specs import create_builtin_registry, dtype_bytes

BUILTIN_NAMES = (
    "matmul",
    "softmax",
    "layernorm",
    "flash_attention",
    "fused_mlp",
    "cross_entropy",
    "rotary_embedding",
    "rmsnorm",
    "reduce",
)

SIZE_LABELS = {
    "matmul": ("tiny", "small", "medium", "large", "xlarge", "tall", "wide", "deep_k",
               "llm_qkv", "llm_mlp"),
    "softmax": ("tiny", "small", "medium", "large", "xlarge", "wide", "narrow", "vocab"),
    "layernorm": ("tiny", "small", "medium", "large", "xlarge", "wide", "llm_7b", "llm_13b"),
    "flash_attention": ("tiny", "small", "medium", "large", "xlarge", "long", "gqa", "llm_7b"),
    "fused_mlp": ("tiny", "small", "medium", "large", "xlarge", "llm_7b", "llm_13b"),
    "cross_entropy": ("tiny", "small", "medium", "large", "xlarge", "llama", "gpt2"),
    "rotary_embedding": ("tiny", "small", "medium", "large", "xlarge", "llm_7b", "llm_13b"),
    "rmsnorm": ("small", "medium", "large", "llama"),
    "reduce": ("small", "medium", "large", "wide"),
}

DTYPES = {
    "matmul": ("float16", "bfloat16", "float32"),
    "softmax": ("float16", "bfloat16", "float32"),
    "layernorm": ("float16", "bfloat16", "float32"),
    "flash_attention": ("float16", "bfloat16"),
    "fused_mlp": ("float16", "bfloat16", "float32"),
    "cross_entropy": ("float16", "bfloat16", "float32"),
    "rotary_embedding": ("float16", "bfloat16", "float32"),
    "rmsnorm": ("float16", "bfloat16"),
    "reduce": ("float16", "bfloat16"),
}

TOLERANCES = {
    "matmul": {"float16": (1e-2, 1e-2), "bfloat16": (2e-2, 2e-2), "float32": (1e-4, 1e-4)},
    "softmax": {"float16": (1e-3, 1e-3), "bfloat16": (2e-3, 2e-3), "float32": (1e-5, 1e-5)},
    "layernorm": {"float16": (1e-3, 1e-3), "bfloat16": (2e-3, 2e-3), "float32": (1e-5, 1e-5)},
    "flash_attention": {"float16": (1e-2, 1e-2), "bfloat16": (2e-2, 2e-2), "float32": (1e-4, 1e-4)},
    "fused_mlp": {"float16": (1e-2, 1e-2), "bfloat16": (2e-2, 2e-2), "float32": (1e-4, 1e-4)},
    "cross_entropy": {"float16": (1e-2, 1e-2), "bfloat16": (2e-2, 2e-2), "float32": (1e-5, 1e-5)},
    "rotary_embedding": {"float16": (1e-3, 1e-3), "bfloat16": (2e-3, 2e-3), "float32": (1e-5, 1e-5)},
    "rmsnorm": {"float16": (1e-2, 1e-2), "bfloat16": (1e-1, 5e-2)},
    "reduce": {"float16": (1e-2, 1e-2), "bfloat16": (1e-1, 5e-2)},
}

LARGE_SIZES = {
    "matmul": {"M": 2048, "N": 2048, "K": 2048},
    "softmax": {"rows": 4096, "cols": 4096},
    "layernorm": {"batch": 4096, "dim": 2048},
    "flash_attention": {"batch": 2, "heads": 32, "seq_len": 1024, "head_dim": 64},
    "fused_mlp": {"batch": 2048, "dim": 2048, "hidden": 5504},
    "cross_entropy": {"batch": 4096, "vocab": 32000},
    "rotary_embedding": {"batch": 2, "heads": 32, "seq_len": 1024, "head_dim": 128},
    "rmsnorm": {"M": 4096, "N": 4096},
    "reduce": {"M": 8192, "N": 8192},
}

# FLOPs and fp16 bytes at the 'large' size, captured pre-refactor.
LARGE_FLOPS = {
    "matmul": 17179869184,
    "softmax": 83886080,
    "layernorm": 67108864,
    "flash_attention": 17179869184,
    "fused_mlp": 138512695296,
    "cross_entropy": 524288000,
    "rotary_embedding": 50331648,
    "rmsnorm": 100663296,
    "reduce": 67108864,
}
LARGE_BYTES_FP16 = {
    "matmul": 25165824,
    "softmax": 67108864,
    "layernorm": 33562624,
    "flash_attention": 33554432,
    "fused_mlp": 84410368,
    "cross_entropy": 262152192,
    "rotary_embedding": 33816576,
    "rmsnorm": 67117056,
    "reduce": 134234112,
}

EDGE_CASES = {
    "matmul": (
        ("edge_1023", {"M": 1023, "N": 1023, "K": 1023}),
        ("edge_4097", {"M": 4097, "N": 4097, "K": 512}),
        ("edge_1537", {"M": 1537, "N": 1537, "K": 1537}),
    ),
    "softmax": (
        ("edge_1023", {"rows": 1023, "cols": 1023}),
        ("edge_4097", {"rows": 4097, "cols": 4097}),
        ("edge_50257", {"rows": 1024, "cols": 50257}),
    ),
    "layernorm": (
        ("edge_1023", {"batch": 1023, "dim": 1023}),
        ("edge_4097", {"batch": 4097, "dim": 4097}),
    ),
    "flash_attention": (
        ("edge_127", {"batch": 1, "heads": 8, "seq_len": 127, "head_dim": 64}),
        ("edge_1023", {"batch": 1, "heads": 8, "seq_len": 1023, "head_dim": 64}),
    ),
    "fused_mlp": (
        ("edge_1023", {"batch": 1023, "dim": 1024, "hidden": 2048}),
        ("edge_4097", {"batch": 4097, "dim": 512, "hidden": 1024}),
    ),
    "cross_entropy": (
        ("edge_1023", {"batch": 1023, "vocab": 32000}),
        ("edge_50257", {"batch": 4096, "vocab": 50257}),
    ),
    "rotary_embedding": (
        ("edge_127", {"batch": 1, "heads": 8, "seq_len": 127, "head_dim": 64}),
        ("edge_1023", {"batch": 1, "heads": 8, "seq_len": 1023, "head_dim": 128}),
    ),
    "rmsnorm": (
        ("edge_1023", {"M": 1023, "N": 768}),
        ("edge_4097", {"M": 4097, "N": 1024}),
    ),
    "reduce": (
        ("edge_1023", {"M": 1023, "N": 1024}),
        ("edge_4097", {"M": 4096, "N": 4097}),
    ),
}

# Alias maps carried over from extract.py::SHAPE_ALIAS_MAP. The identity entries
# for matmul/rmsnorm/reduce make the mapping explicit; behavior is unchanged
# because unmapped keys always passed through.
SHAPE_ALIASES = {
    "matmul": {"M": "M", "N": "N", "K": "K"},
    "softmax": {"M": "rows", "N": "cols", "rows": "rows", "cols": "cols"},
    "layernorm": {
        "M": "batch", "N": "dim", "rows": "batch", "cols": "dim",
        "batch": "batch", "dim": "dim",
    },
    "flash_attention": {
        "B": "batch", "H": "heads", "N": "seq_len", "S": "seq_len", "D": "head_dim",
        "batch": "batch", "heads": "heads", "seq_len": "seq_len", "head_dim": "head_dim",
    },
    "fused_mlp": {
        "M": "batch", "N": "hidden", "K": "dim",
        "batch": "batch", "dim": "dim", "hidden": "hidden",
    },
    "cross_entropy": {"batch": "batch", "vocab": "vocab"},
    "rotary_embedding": {
        "B": "batch", "H": "heads", "N": "seq_len", "S": "seq_len", "D": "head_dim",
        "batch": "batch", "heads": "heads", "seq_len": "seq_len", "head_dim": "head_dim",
    },
    "rmsnorm": {"M": "M", "N": "N"},
    "reduce": {"M": "M", "N": "N"},
}

SPEEDUP_ESTIMATES = {
    "matmul": "2-3x",
    "flash_attention": "2-4x",
    "layernorm": "1.5-3x",
    "softmax": "1.5-3x",
    "cross_entropy": "1.5-2x",
    "fused_mlp": "2-3x",
    "rmsnorm": "1.5-3x",
    "reduce": "1.5-2x",
    "rotary_embedding": "1.5-2x",
}

# Extraction fallback shapes, captured from extract.py::get_default_shape.
EXTRACTION_SHAPES = {
    "matmul": {"M": 2048, "N": 2048, "K": 2048},
    "flash_attention": {"batch": 2, "heads": 32, "seq_len": 1024, "head_dim": 64},
    "layernorm": {"batch": 4096, "dim": 2048},
    "softmax": {"rows": 4096, "cols": 4096},
    "cross_entropy": {"batch": 4096, "vocab": 32000},
    "fused_mlp": {"batch": 2048, "dim": 2048, "hidden": 5504},
    "rmsnorm": {"M": 4096, "N": 4096},
    "reduce": {"M": 4096, "N": 4096},
    "rotary_embedding": {"batch": 2, "heads": 32, "seq_len": 1024, "head_dim": 128},
}

INPUT_KEYS = {
    "matmul": ("A", "B"),
    "softmax": ("x",),
    "layernorm": ("x", "weight", "bias"),
    "flash_attention": ("Q", "K", "V"),
    "fused_mlp": ("x", "w_gate", "w_up", "w_down"),
    "cross_entropy": ("logits", "targets"),
    "rotary_embedding": ("x", "cos", "sin"),
    "rmsnorm": ("x", "weight"),
    "reduce": ("x",),
}


@pytest.fixture(scope="module")
def registry():
    return create_builtin_registry()


def test_builtin_names_and_order(registry):
    assert registry.list_names() == BUILTIN_NAMES


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_size_labels(registry, name):
    assert tuple(registry.get(name).sizes) == SIZE_LABELS[name]


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_large_size_values(registry, name):
    assert registry.get(name).sizes["large"] == LARGE_SIZES[name]


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_dtypes(registry, name):
    assert registry.get(name).dtypes == DTYPES[name]
    assert registry.get(name).primary_dtype == DTYPES[name][0]


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_tolerances(registry, name):
    spec = registry.get(name)
    actual = {d: (t.atol, t.rtol) for d, t in spec.tolerances.items()}
    assert actual == TOLERANCES[name]
    for dtype in spec.dtypes:
        assert dtype in spec.tolerances


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_accounting_matches_pre_refactor_values(registry, name):
    spec = registry.get(name)
    large = spec.sizes["large"]
    assert spec.flops_fn(large) == LARGE_FLOPS[name]
    assert spec.bytes_fn(large, dtype_bytes("float16")) == LARGE_BYTES_FP16[name]


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_edge_cases(registry, name):
    spec = registry.get(name)
    actual = tuple((e.name, dict(e.size)) for e in spec.edge_cases)
    assert actual == EDGE_CASES[name]
    for edge in spec.edge_cases:
        assert edge.seed == 42
        assert edge.dtype is None
        assert edge.input_transform is None


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_shape_metadata_used_by_extraction(registry, name):
    spec = registry.get(name)
    assert spec.shape_aliases == SHAPE_ALIASES[name]
    assert spec.shape_keys == tuple(LARGE_SIZES[name])
    assert spec.speedup_estimate == SPEEDUP_ESTIMATES[name]
    assert spec.extraction_shape() == EXTRACTION_SHAPES[name]


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_starter_kernels_exist(registry, name):
    spec = registry.get(name)
    assert set(spec.starter_kernels) == {"triton", "cuda"}
    for backend in ("triton", "cuda"):
        assert spec.starter_kernel(backend).is_file()


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_input_generators_are_deterministic(registry, name, torch_mod):
    spec = registry.get(name)
    small = spec.sizes["small"]
    dtype = "float32" if "float32" in spec.dtypes else spec.primary_dtype
    first = spec.input_generator(small, dtype, "cpu", 42)
    second = spec.input_generator(small, dtype, "cpu", 42)
    third = spec.input_generator(small, dtype, "cpu", 1234)

    assert tuple(first) == INPUT_KEYS[name]
    for key in first:
        assert torch_mod.equal(first[key], second[key]), key

    changed = any(not torch_mod.equal(first[k], third[k]) for k in first)
    assert changed, "a different seed must produce different inputs"


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_reference_functions_run_on_cpu(registry, name, torch_mod):
    spec = registry.get(name)
    dtype = "float32" if "float32" in spec.dtypes else spec.primary_dtype
    inputs = spec.input_generator(spec.sizes["small"], dtype, "cpu", 42)
    output = spec.reference_fn(**inputs)
    assert isinstance(output, torch_mod.Tensor)


def test_reference_functions_come_from_reference_module(registry):
    import reference

    for name in BUILTIN_NAMES:
        lazy = registry.get(name).reference_fn
        assert lazy.module_name == "reference"
        assert getattr(reference, lazy.attribute) is lazy.resolve()


def test_discovery_does_not_import_torch():
    """Registry discovery must work on a CPU-only machine without torch."""
    code = (
        "import sys\n"
        "from autokernel.specs import create_builtin_registry\n"
        "registry = create_builtin_registry()\n"
        "assert len(registry) == 9, registry.list_names()\n"
        "for spec in registry:\n"
        "    spec.flops_fn(spec.sizes['large'])\n"
        "    spec.bytes_fn(spec.sizes['large'], 2)\n"
        "assert 'torch' not in sys.modules, 'discovery imported torch'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
