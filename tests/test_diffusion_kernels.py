"""
Unit tests for Modulated Normalization, Gated Residual Fusion, and Custom Kernel Extension.
Covers:
- Modulated LayerNorm and RMSNorm across dtypes (FP32, FP16, BF16) and broadcast modes.
- Standalone and Fused Gated Residuals.
- Custom kernel registration interface in bench.py.
- Kernel classification in profile.py.
- Module replacement in verify.py.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import bench
import reference
from kernels import modulated_norm, gated_residual
import importlib.util

profile_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profile.py")
profile_spec = importlib.util.spec_from_file_location("autokernel_profile", profile_path)
autokernel_profile = importlib.util.module_from_spec(profile_spec)
sys.modules["autokernel_profile"] = autokernel_profile
profile_spec.loader.exec_module(autokernel_profile)

from verify import (
    OptimizedModelContext,
    KernelReplacement,
    _ModulatedNormWrapper,
    _GatedResidualWrapper,
    register_replacement_strategy,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")


class MockAdaLayerNorm(nn.Module):
    """Simulates an AdaLayerNorm module from DiT / Stable Diffusion."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, scale: torch.Tensor = None, shift: torch.Tensor = None) -> torch.Tensor:
        norm = F.layer_norm(x, (self.dim,), self.weight, self.bias, self.eps)
        if scale is not None:
            norm = norm * (1.0 + scale)
        if shift is not None:
            norm = norm + shift
        return norm


class MockGatedResidual(nn.Module):
    """Simulates a GatedResidual block."""

    def __init__(self):
        super().__init__()

    def forward(self, residual: torch.Tensor, gate: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return residual + gate * update


class TestModulatedNorm:
    """Correctness and broadcast coverage for modulated_norm."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm"])
    @pytest.mark.parametrize("broadcast_scale", [True, False])
    def test_modulated_norm_modes(self, dtype, norm_type, broadcast_scale):
        batch, seq_len, dim = 2, 64, 256
        size = {
            "batch": batch,
            "seq_len": seq_len,
            "dim": dim,
            "broadcast_scale": broadcast_scale,
            "norm_type": norm_type,
        }
        inputs = bench.gen_modulated_norm_inputs(size, dtype, DEVICE, seed=123)
        expected = bench._ref_modulated_norm(inputs)
        out = modulated_norm.kernel_fn(**inputs)

        tol = bench.KERNEL_CONFIGS["modulated_norm"]["tolerances"][dtype]
        cmp = bench._compare(out, expected, **tol)
        assert cmp["match"], f"Mismatch for {dtype}, {norm_type}, broadcast={broadcast_scale}: {cmp['reason']}"

    def test_global_scale_shift(self):
        """Test with 1-D global scale and shift [dim]."""
        batch, seq_len, dim = 2, 32, 128
        x = torch.randn(batch, seq_len, dim, device=DEVICE, dtype=torch.float32)
        weight = torch.ones(dim, device=DEVICE, dtype=torch.float32)
        bias = torch.zeros(dim, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(dim, device=DEVICE, dtype=torch.float32)
        shift = torch.randn(dim, device=DEVICE, dtype=torch.float32)

        ref = reference.modulated_norm_ref(x, weight, bias, scale, shift, eps=1e-5, norm_type="layernorm")
        out = modulated_norm.kernel_fn(x, weight, bias, scale, shift, eps=1e-5, norm_type="layernorm")
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_fused_gated_residual_modulated_norm(self):
        """Test combined fused gated-residual + modulated norm kernel."""
        batch, seq_len, dim = 2, 64, 256
        residual = torch.randn(batch, seq_len, dim, device=DEVICE, dtype=torch.float32)
        update = torch.randn(batch, seq_len, dim, device=DEVICE, dtype=torch.float32)
        gate = torch.randn(batch, 1, dim, device=DEVICE, dtype=torch.float32)
        weight = torch.ones(dim, device=DEVICE, dtype=torch.float32)
        bias = torch.zeros(dim, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(batch, 1, dim, device=DEVICE, dtype=torch.float32)
        shift = torch.randn(batch, 1, dim, device=DEVICE, dtype=torch.float32)

        ref_norm, ref_updated = reference.fused_gated_residual_modulated_norm_ref(
            residual, gate, update, weight, bias, scale, shift, eps=1e-5, norm_type="layernorm"
        )
        out_norm, out_updated = modulated_norm.fused_gated_residual_modulated_norm(
            residual, gate, update, weight, bias, scale, shift, eps=1e-5, norm_type="layernorm", return_residual=True
        )

        torch.testing.assert_close(out_norm, ref_norm, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(out_updated, ref_updated, atol=1e-4, rtol=1e-4)


class TestGatedResidual:
    """Correctness and broadcast coverage for gated_residual."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("broadcast_gate", [True, False])
    def test_gated_residual_modes(self, dtype, broadcast_gate):
        batch, seq_len, dim = 2, 64, 256
        size = {
            "batch": batch,
            "seq_len": seq_len,
            "dim": dim,
            "broadcast_gate": broadcast_gate,
        }
        inputs = bench.gen_gated_residual_inputs(size, dtype, DEVICE, seed=456)
        expected = bench._ref_gated_residual(inputs)
        out = gated_residual.kernel_fn(**inputs)

        tol = bench.KERNEL_CONFIGS["gated_residual"]["tolerances"][dtype]
        cmp = bench._compare(out, expected, **tol)
        assert cmp["match"], f"Mismatch for {dtype}, broadcast={broadcast_gate}: {cmp['reason']}"


class TestCustomKernelInterface:
    """Tests for dynamic custom kernel registration and extension."""

    def test_register_and_unregister_kernel(self):
        custom_name = "custom_test_kernel_op"
        custom_config = {
            "test_sizes": [("small", {"dim": 64})],
            "test_dtypes": [torch.float32],
            "tolerances": {torch.float32: {"atol": 1e-4, "rtol": 1e-4}},
            "flops_fn": lambda s: s["dim"],
            "bytes_fn": lambda s, dt: s["dim"] * 4,
            "input_generator": lambda s, dt, dev, seed=0: {"x": torch.zeros(s["dim"], device=dev, dtype=dt)},
            "reference_fn": lambda inp: inp["x"],
        }

        bench.register_kernel(custom_name, custom_config)
        assert custom_name in bench.list_registered_kernels()
        assert bench.KERNEL_CONFIGS[custom_name] == custom_config

        bench.unregister_kernel(custom_name)
        assert custom_name not in bench.list_registered_kernels()

    def test_register_invalid_kernel_raises(self):
        with pytest.raises(ValueError, match="missing required keys"):
            bench.register_kernel("invalid_op", {"test_sizes": []})


class TestProfileClassification:
    """Tests for kernel pattern classification in profile.py."""

    def test_built_in_classifications(self):
        assert autokernel_profile.classify_kernel("ada_ln_modulation") == "modulated_norm"
        assert autokernel_profile.classify_kernel("dit_modulated_layernorm") == "modulated_norm"
        assert autokernel_profile.classify_kernel("gated_residual_fusion") == "gated_residual"

    def test_register_custom_classifier(self):
        autokernel_profile.register_kernel_classifier(["custom_diffusion_op"], "custom_op")
        assert autokernel_profile.classify_kernel("custom_diffusion_op_forward") == "custom_op"


class TestVerifyModuleReplacement:
    """Tests for module patching and wrappers in verify.py."""

    def test_modulated_norm_module_replacement(self):
        model = nn.Sequential(
            MockAdaLayerNorm(dim=128),
            MockGatedResidual(),
        ).to(DEVICE)

        repl_mod = KernelReplacement(
            kernel_type="modulated_norm",
            rank=1,
            speedup=3.5,
            optimized_path="",
            module_fn=modulated_norm.kernel_fn,
        )

        ctx = OptimizedModelContext(model, [repl_mod])
        replaced = ctx._apply_replacement(repl_mod)
        assert replaced == 1
        assert isinstance(model[0], _ModulatedNormWrapper)

    def test_gated_residual_module_replacement(self):
        model = nn.Sequential(
            MockGatedResidual(),
        ).to(DEVICE)

        repl_gate = KernelReplacement(
            kernel_type="gated_residual",
            rank=1,
            speedup=2.0,
            optimized_path="",
            module_fn=gated_residual.kernel_fn,
        )

        ctx = OptimizedModelContext(model, [repl_gate])
        replaced = ctx._apply_replacement(repl_gate)
        assert replaced == 1
        assert isinstance(model[0], _GatedResidualWrapper)
