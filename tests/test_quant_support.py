import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
import importlib.util

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Avoid colliding with Python stdlib `profile`
_profile_spec = importlib.util.spec_from_file_location("autokernel_profile", REPO_ROOT / "profile.py")
profiler = importlib.util.module_from_spec(_profile_spec)
sys.modules["autokernel_profile"] = profiler
_profile_spec.loader.exec_module(profiler)

import verify as verifier


class QuantizationSupportTests(unittest.TestCase):
    def test_build_quantization_config_none(self):
        self.assertIsNone(profiler._build_quantization_config(None))
        self.assertIsNone(profiler._build_quantization_config("none"))
        self.assertIsNone(verifier._build_quantization_config(None))
        self.assertIsNone(verifier._build_quantization_config("none"))

    def test_build_quantization_config_bnb4(self):
        cfg = profiler._build_quantization_config("bnb4", "fp16")
        self.assertIsNotNone(cfg)
        self.assertTrue(getattr(cfg, "load_in_4bit", False))
        self.assertEqual(getattr(cfg, "bnb_4bit_compute_dtype", None), torch.float16)
        self.assertEqual(getattr(cfg, "bnb_4bit_quant_type", None), "nf4")
        self.assertTrue(getattr(cfg, "bnb_4bit_use_double_quant", False))

        vcfg = verifier._build_quantization_config("bnb4", "bf16")
        self.assertIsNotNone(vcfg)
        self.assertTrue(getattr(vcfg, "load_in_4bit", False))
        self.assertEqual(getattr(vcfg, "bnb_4bit_compute_dtype", None), torch.bfloat16)

    def test_build_quantization_config_bnb8(self):
        cfg = profiler._build_quantization_config("bnb8")
        self.assertIsNotNone(cfg)
        self.assertTrue(getattr(cfg, "load_in_8bit", False))

        vcfg = verifier._build_quantization_config("bnb8")
        self.assertIsNotNone(vcfg)
        self.assertTrue(getattr(vcfg, "load_in_8bit", False))

    def test_build_quantization_config_invalid(self):
        with self.assertRaises(ValueError):
            profiler._build_quantization_config("invalid_quant")
        with self.assertRaises(ValueError):
            verifier._build_quantization_config("invalid_quant")

    def test_kernel_classification_for_quant_kernels(self):
        quant_kernel_names = [
            "ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_tn",
            "void bnb::gemv_4bit<float, 4>(...)",
            "cutlass_80_tensorop_s16816gemm_64x64_32x2_tn_align8",
            "marlin_gemm_kernel_fp16",
            "awq_gemm_kernel",
            "igemm_fwd_gtc",
            "dequantize_and_matmul_kernel",
        ]
        for name in quant_kernel_names:
            op_type = profiler.classify_kernel(name)
            self.assertEqual(
                op_type,
                "matmul",
                f"Expected '{name}' to classify as 'matmul', got '{op_type}'",
            )

    def test_prepare_model_and_input_preserves_quantized_model(self):
        class MockQuantizedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.is_quantized = True
                self.linear = nn.Linear(8, 8, dtype=torch.float16)

            def forward(self, x):
                return self.linear(x)

        model = MockQuantizedModel()
        # Ensure to() is not called on quantized model
        model.to = MagicMock(side_effect=RuntimeError(".to is not supported for quantized models"))
        prepared_model, inputs = profiler._prepare_model_and_input(
            model, [1, 8], torch.float16, "cpu"
        )
        self.assertIs(prepared_model, model)
        self.assertIn("x", inputs)

    def test_verify_replaces_linear_skips_quantized_modules(self):
        class DummyLinear4bit(nn.Linear):
            def __init__(self, in_features, out_features):
                super().__init__(in_features, out_features)
                self.is_quantized = True

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = nn.Linear(16, 16)
                self.quant = DummyLinear4bit(16, 16)

            def forward(self, x):
                return self.quant(self.dense(x))

        net = Net()
        repl = verifier.KernelReplacement(
            kernel_type="matmul",
            rank=1,
            speedup=1.5,
            optimized_path="dummy.py",
            module_fn=lambda a, b: torch.matmul(a, b),
        )
        ctx = verifier.OptimizedModelContext(net, [repl])
        count = ctx._replace_linear_modules(repl)
        # Should replace only 1 (the standard dense linear), skipping the quantized one
        self.assertEqual(count, 1)
        self.assertIn("dense", ctx._original_modules)
        self.assertNotIn("quant", ctx._original_modules)


if __name__ == "__main__":
    unittest.main()
