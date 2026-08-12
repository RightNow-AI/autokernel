#!/usr/bin/env python3
"""
AutoKernel End-to-End Verifier -- Plug optimized kernels back into the model and verify.

Usage:
    uv run verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,2048
    uv run verify.py --module transformers --class-name AutoModelForCausalLM --pretrained meta-llama/Llama-2-7b-hf
    uv run verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,2048 --diagnose

Checks:
  1. Loads the original model
  2. Runs inference with original PyTorch ops -> captures reference output
  3. Replaces bottleneck ops with optimized Triton kernels
  4. Runs inference with optimized kernels -> captures optimized output
  5. Compares outputs (tolerance check)
  6. Benchmarks both paths -> reports end-to-end speedup
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(SCRIPT_DIR, "workspace")
ORCHESTRATION_STATE = os.path.join(WORKSPACE_DIR, "orchestration_state.json")

# Benchmarking defaults
WARMUP_RUNS = 10
TIMED_RUNS = 50

# Tolerance defaults by dtype
DEFAULT_TOLERANCES: Dict[torch.dtype, Dict[str, float]] = {
    torch.float16:  {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 2e-3, "rtol": 2e-3},
    torch.float32:  {"atol": 1e-5, "rtol": 1e-5},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KernelReplacement:
    """Describes a single kernel replacement: what to replace and with what."""
    kernel_type: str          # e.g. "matmul", "layernorm", "rmsnorm"
    rank: int                 # priority rank from profiling
    speedup: float            # individual kernel speedup
    optimized_path: str       # path to optimized kernel .py file
    module_fn: Optional[Callable] = None  # loaded kernel function


@dataclass
class VerificationResult:
    """Full verification result."""
    model_name: str = ""
    input_shape: str = ""
    dtype_str: str = ""
    gpu_name: str = ""

    # Reference run
    ref_output_shape: str = ""
    ref_latency_ms: float = 0.0

    # Optimized run
    opt_output_shape: str = ""
    opt_latency_ms: float = 0.0
    kernels_replaced: List[Dict[str, Any]] = field(default_factory=list)

    # Comparison
    correctness: str = "UNKNOWN"
    max_abs_error: float = 0.0
    mean_abs_error: float = 0.0
    has_nan: bool = False
    has_inf: bool = False

    # Summary
    end_to_end_speedup: float = 0.0


# ---------------------------------------------------------------------------
# 1. Model Loading
# ---------------------------------------------------------------------------

def load_model_from_file(model_path: str, class_name: str, **kwargs) -> nn.Module:
    """Load a model from a Python file by importing it and instantiating the class."""
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    spec = importlib.util.spec_from_file_location("user_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model from: {model_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, class_name):
        available = [n for n in dir(mod) if not n.startswith("_")]
        raise AttributeError(
            f"Class '{class_name}' not found in {model_path}. "
            f"Available names: {available}"
        )

    cls = getattr(mod, class_name)
    model = cls(**kwargs)
    return model


def load_model_from_module(module_name: str, class_name: str,
                           pretrained: Optional[str] = None, **kwargs) -> nn.Module:
    """Load a model from an installed Python module (e.g. 'transformers')."""
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module '{module_name}'. Is it installed? Error: {e}"
        )

    if not hasattr(mod, class_name):
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_name}'."
        )

    cls = getattr(mod, class_name)

    if pretrained:
        # HuggingFace-style: cls.from_pretrained(...)
        if hasattr(cls, "from_pretrained"):
            model = cls.from_pretrained(pretrained, **kwargs)
        else:
            raise AttributeError(
                f"'{class_name}' has no 'from_pretrained' method. "
                f"Cannot load pretrained weights from '{pretrained}'."
            )
    else:
        model = cls(**kwargs)

    return model


def load_model(args) -> nn.Module:
    """Unified model loader from CLI args."""
    dtype = _parse_dtype(args.dtype)

    if args.model:
        print(f"Loading model from file: {args.model} (class: {args.class_name})")
        model = load_model_from_file(args.model, args.class_name)
    elif args.module:
        print(f"Loading model from module: {args.module} (class: {args.class_name})")
        extra_kwargs = {}
        if dtype == torch.float16:
            extra_kwargs["torch_dtype"] = torch.float16
        elif dtype == torch.bfloat16:
            extra_kwargs["torch_dtype"] = torch.bfloat16
        model = load_model_from_module(
            args.module, args.class_name, pretrained=args.pretrained, **extra_kwargs
        )
    else:
        raise ValueError("Must specify either --model (file path) or --module (Python module)")

    model = model.to(dtype=dtype)

    if torch.cuda.is_available():
        try:
            model = model.cuda()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"WARNING: OOM moving model to GPU. Trying with smaller footprint...")
                torch.cuda.empty_cache()
                model = model.half().cuda()
            else:
                raise

    model.eval()
    return model


# ---------------------------------------------------------------------------
# 2. Input Generation
# ---------------------------------------------------------------------------

def generate_sample_input(
    input_shape: str,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 42,
) -> torch.Tensor:
    """Generate a sample input tensor from a shape string like '1,2048'."""
    dims = [int(d.strip()) for d in input_shape.split(",")]
    torch.manual_seed(seed)

    if dtype in (torch.int32, torch.int64, torch.long):
        # For language models, generate token IDs (assume vocab size ~32000)
        return torch.randint(0, 32000, dims, device=device, dtype=dtype)
    else:
        return torch.randn(dims, device=device, dtype=dtype)


def infer_input_type(model: nn.Module) -> str:
    """Try to determine if the model expects integer token IDs or float tensors."""
    # The forward signature is the most reliable signal. HuggingFace
    # *ForCausalLM / *ForConditionalGeneration wrappers take input_ids, but
    # their children are (model, lm_head) -- so the child scan below would hit
    # lm_head and wrongly answer "float".
    try:
        sig = inspect.signature(model.forward)
        if "input_ids" in sig.parameters:
            return "token_ids"
    except (ValueError, TypeError):
        pass

    # Check if model has an embedding layer as the first module
    for name, child in model.named_children():
        if isinstance(child, nn.Embedding):
            return "token_ids"
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            return "float"
    return "float"


def infer_vocab_size(model: nn.Module, default: int = 32000) -> int:
    """Find the model's vocab size so generated token IDs are always in range."""
    config = getattr(model, "config", None)
    vocab = getattr(config, "vocab_size", None)
    if isinstance(vocab, int) and vocab > 0:
        return vocab

    for module in model.modules():
        if isinstance(module, nn.Embedding) and module.num_embeddings > 0:
            return module.num_embeddings

    return default


def make_model_input(
    model: nn.Module,
    input_shape: str,
    dtype: torch.dtype,
    device: str = "cuda",
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Create an appropriate input for the model."""
    input_type = infer_input_type(model)

    if input_type == "token_ids":
        # Language model: expects integer input_ids
        dims = [int(d.strip()) for d in input_shape.split(",")]
        torch.manual_seed(42)
        vocab_size = infer_vocab_size(model)
        input_ids = torch.randint(0, vocab_size, dims, device=device, dtype=torch.long)

        # Check if model accepts input_ids keyword
        sig = inspect.signature(model.forward)
        if "input_ids" in sig.parameters:
            return {"input_ids": input_ids}
        return input_ids
    else:
        return generate_sample_input(input_shape, dtype, device)


# ---------------------------------------------------------------------------
# 3. Benchmarking
# ---------------------------------------------------------------------------

def benchmark_model(
    model: nn.Module,
    model_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
    warmup: int = WARMUP_RUNS,
    timed: int = TIMED_RUNS,
) -> Tuple[Any, float]:
    """
    Benchmark model inference. Returns (output, median_latency_ms).
    Uses CUDA events for precise GPU timing.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking.")

    def _run():
        with torch.no_grad():
            if isinstance(model_input, dict):
                return model(**model_input)
            else:
                return model(model_input)

    # Warmup
    print(f"  Warmup: {warmup} runs...", end="", flush=True)
    for _ in range(warmup):
        output = _run()
    torch.cuda.synchronize()
    print(" done")

    # Timed runs
    print(f"  Timed: {timed} runs...", end="", flush=True)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]

    torch.cuda.synchronize()
    for i in range(timed):
        start_events[i].record()
        _run()
        end_events[i].record()
    torch.cuda.synchronize()
    print(" done")

    # Compute median
    times_ms = sorted(s.elapsed_time(e) for s, e in zip(start_events, end_events))
    median_ms = times_ms[len(times_ms) // 2]

    # Final reference output (deterministic)
    with torch.no_grad():
        output = _run()
    torch.cuda.synchronize()

    return output, median_ms


# ---------------------------------------------------------------------------
# 4. Kernel Replacement
# ---------------------------------------------------------------------------

def load_orchestration_state() -> Optional[Dict]:
    """Load workspace/orchestration_state.json if it exists."""
    if not os.path.exists(ORCHESTRATION_STATE):
        return None
    with open(ORCHESTRATION_STATE, "r") as f:
        return json.load(f)


def discover_optimized_kernels() -> List[KernelReplacement]:
    """
    Find optimized kernels from the workspace directory.
    Checks orchestration_state.json first, then scans for *_optimized.py files.
    """
    replacements: List[KernelReplacement] = []

    # Strategy 1: Read orchestration state
    state = load_orchestration_state()
    if state and "kernels" in state:
        for k in state["kernels"]:
            ktype = k.get("op_type", k.get("type", "unknown"))
            rank = k.get("rank", 0)
            speedup = k.get("speedup", k.get("best_speedup", 1.0))
            # optimized_path is not written by orchestrate.py, so derive it
            # from the kernel file path if available
            opt_path = k.get("optimized_path", "")

            if not opt_path:
                # Try to derive from the "file" key that orchestrate.py writes
                base_file = k.get("file", "")
                if base_file:
                    stem = Path(base_file).stem
                    opt_path = os.path.join(
                        WORKSPACE_DIR, f"{stem}_optimized.py"
                    )
                else:
                    # Fallback convention: workspace/kernel_{type}_{rank}_optimized.py
                    opt_path = os.path.join(
                        WORKSPACE_DIR, f"kernel_{ktype}_{rank}_optimized.py"
                    )

            if os.path.exists(opt_path) and speedup > 1.0:
                replacements.append(KernelReplacement(
                    kernel_type=ktype,
                    rank=rank,
                    speedup=speedup,
                    optimized_path=opt_path,
                ))
        return replacements

    # Strategy 2: Scan workspace directory for optimized kernel files
    if not os.path.isdir(WORKSPACE_DIR):
        return replacements

    for fname in sorted(os.listdir(WORKSPACE_DIR)):
        if fname.endswith("_optimized.py"):
            # Parse filename: kernel_{type}_{rank}_optimized.py
            # Type can be multi-word (e.g. flash_attention), so the rank
            # is always the last numeric segment before "_optimized.py".
            stem = fname.replace("_optimized.py", "")  # e.g. "kernel_flash_attention_1"
            parts = stem.split("_")
            if len(parts) >= 3 and parts[0] == "kernel":
                # Find the rank: last part that is purely numeric
                rank = 0
                rank_idx = len(parts)
                for i in range(len(parts) - 1, 0, -1):
                    if parts[i].isdigit():
                        rank = int(parts[i])
                        rank_idx = i
                        break
                # Everything between parts[1] and the rank index is the type
                ktype = "_".join(parts[1:rank_idx]) if rank_idx > 1 else parts[1]
                opt_path = os.path.join(WORKSPACE_DIR, fname)
                replacements.append(KernelReplacement(
                    kernel_type=ktype,
                    rank=rank,
                    speedup=0.0,  # Unknown without state file
                    optimized_path=opt_path,
                ))

    return replacements


def load_kernel_module(path: str) -> Any:
    """Dynamically import a kernel .py file and return the module."""
    path = os.path.abspath(path)
    module_name = f"opt_kernel_{os.path.basename(path).replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load kernel from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _LinearWrapper(nn.Module):
    """Wraps nn.Linear to use an optimized matmul kernel_fn."""

    def __init__(self, original: nn.Linear, kernel_fn: Callable):
        super().__init__()
        self.original = original
        self.kernel_fn = kernel_fn
        self.weight = original.weight
        self.bias = original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to 2D for kernel_fn, then reshape back
        orig_shape = x.shape
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x

        # kernel_fn expects (A, B) where A @ B = C
        # For nn.Linear: output = input @ weight.T + bias
        # So we call kernel_fn(input, weight.T)
        weight_t = self.weight.t().contiguous()
        out = self.kernel_fn(x_2d, weight_t)

        if self.bias is not None:
            out = out + self.bias

        if len(orig_shape) > 2:
            out = out.reshape(*orig_shape[:-1], out.shape[-1])

        return out


class _LayerNormWrapper(nn.Module):
    """Wraps nn.LayerNorm to use an optimized kernel_fn."""

    def __init__(self, original: nn.LayerNorm, kernel_fn: Callable):
        super().__init__()
        self.original = original
        self.kernel_fn = kernel_fn
        self.weight = original.weight
        self.bias = original.bias
        self.eps = original.eps
        self.normalized_shape = original.normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape if needed: kernel_fn expects (x, weight, bias[, eps])
        orig_shape = x.shape
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x

        try:
            # Try full signature: kernel_fn(x, weight, bias, eps)
            out = self.kernel_fn(x_2d, self.weight, self.bias, self.eps)
        except TypeError:
            try:
                # Try without eps: kernel_fn(x, weight, bias)
                out = self.kernel_fn(x_2d, self.weight, self.bias)
            except TypeError:
                # Fallback: just x
                out = self.kernel_fn(x_2d)

        if len(orig_shape) > 2:
            out = out.reshape(orig_shape)

        return out


class _RMSNormWrapper(nn.Module):
    """Wraps RMSNorm-like modules to use an optimized kernel_fn."""

    def __init__(self, original: nn.Module, kernel_fn: Callable):
        super().__init__()
        self.original = original
        self.kernel_fn = kernel_fn
        # RMSNorm typically has a 'weight' attribute
        self.weight = getattr(original, "weight", None)
        self.eps = getattr(original, "eps", 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x

        if self.weight is not None:
            try:
                out = self.kernel_fn(x_2d, self.weight, self.eps)
            except TypeError:
                out = self.kernel_fn(x_2d, self.weight)
        else:
            out = self.kernel_fn(x_2d)

        if len(orig_shape) > 2:
            out = out.reshape(orig_shape)

        return out


class OptimizedModelContext:
    """
    Context manager that patches a model's submodules to use optimized Triton kernels.

    Usage:
        with OptimizedModelContext(model, replacements) as patched_model:
            output = patched_model(input)
    """

    def __init__(self, model: nn.Module, replacements: List[KernelReplacement]):
        self.model = model
        self.replacements = replacements
        self._original_modules: Dict[str, nn.Module] = {}
        self._applied: List[str] = []

    def __enter__(self) -> nn.Module:
        for repl in self.replacements:
            try:
                kernel_mod = load_kernel_module(repl.optimized_path)
                if not hasattr(kernel_mod, "kernel_fn"):
                    print(f"  WARNING: {repl.optimized_path} has no kernel_fn, skipping")
                    continue
                repl.module_fn = kernel_mod.kernel_fn
            except Exception as e:
                print(f"  WARNING: Failed to load {repl.optimized_path}: {e}")
                continue

            replaced = self._apply_replacement(repl)
            if replaced > 0:
                self._applied.append(
                    f"  {repl.kernel_type} (rank {repl.rank}): "
                    f"{repl.speedup:.1f}x -> {repl.optimized_path}"
                )

        return self.model

    def __exit__(self, *exc):
        # Restore all original modules
        for name, original in self._original_modules.items():
            parts = name.split(".")
            parent = self.model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], original)
        self._original_modules.clear()
        self._applied.clear()

    def _apply_replacement(self, repl: KernelReplacement) -> int:
        """
        Replace matching modules in the model. Returns number of modules replaced.
        """
        count = 0

        if repl.kernel_type == "matmul":
            count = self._replace_linear_modules(repl)
        elif repl.kernel_type == "layernorm":
            count = self._replace_layernorm_modules(repl)
        elif repl.kernel_type == "rmsnorm":
            count = self._replace_rmsnorm_modules(repl)
        else:
            print(f"  NOTE: No replacement strategy for kernel type '{repl.kernel_type}'. "
                  f"Skipping. (Supported: matmul, layernorm, rmsnorm)")

        return count

    def _replace_linear_modules(self, repl: KernelReplacement) -> int:
        """Replace all nn.Linear modules with optimized matmul wrapper."""
        count = 0
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                # Save original
                self._original_modules[name] = module
                # Create wrapper
                wrapper = _LinearWrapper(module, repl.module_fn)
                # Install wrapper
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
                count += 1
        return count

    def _replace_layernorm_modules(self, repl: KernelReplacement) -> int:
        """Replace all nn.LayerNorm modules with optimized wrapper."""
        count = 0
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.LayerNorm):
                self._original_modules[name] = module
                wrapper = _LayerNormWrapper(module, repl.module_fn)
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
                count += 1
        return count

    def _replace_rmsnorm_modules(self, repl: KernelReplacement) -> int:
        """
        Replace RMSNorm modules. Since there is no standard nn.RMSNorm,
        we look for common class names and attributes.
        """
        count = 0
        rmsnorm_names = {"RMSNorm", "LlamaRMSNorm", "T5LayerNorm", "GemmaRMSNorm"}

        for name, module in list(self.model.named_modules()):
            cls_name = type(module).__name__
            # Match by class name or by having 'weight' but no 'bias' and a norm-like name
            is_rmsnorm = (
                cls_name in rmsnorm_names
                or (hasattr(module, "weight")
                    and hasattr(module, "eps")
                    and not hasattr(module, "bias")
                    and cls_name.lower().endswith("norm")
                    and not isinstance(module, nn.LayerNorm))
            )

            if is_rmsnorm:
                self._original_modules[name] = module
                wrapper = _RMSNormWrapper(module, repl.module_fn)
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
                count += 1
        return count

    @property
    def applied_summary(self) -> List[str]:
        return self._applied


# ---------------------------------------------------------------------------
# 5. Output Comparison
# ---------------------------------------------------------------------------

def extract_tensor(output: Any) -> torch.Tensor:
    """
    Extract a single tensor from model output, which might be a tuple, dict,
    or ModelOutput-like object.
    """
    if isinstance(output, torch.Tensor):
        return output

    # HuggingFace ModelOutput or similar dataclass-like object
    if hasattr(output, "logits"):
        return output.logits
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state

    # Tuple/list: return first tensor element
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
        # Recurse into first element
        if len(output) > 0:
            return extract_tensor(output[0])

    # Dict: try common keys
    if isinstance(output, dict):
        for key in ["logits", "last_hidden_state", "output", "hidden_states"]:
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
        # Return first tensor value
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v

    raise ValueError(
        f"Cannot extract tensor from output of type {type(output)}. "
        f"Consider adding support for this output format."
    )


def compare_outputs(
    ref_output: torch.Tensor,
    opt_output: torch.Tensor,
    dtype: torch.dtype,
    custom_atol: Optional[float] = None,
    custom_rtol: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compare reference and optimized outputs. Returns comparison metrics.
    """
    result: Dict[str, Any] = {}

    # Shape check
    result["shapes_match"] = ref_output.shape == opt_output.shape
    result["ref_shape"] = str(list(ref_output.shape))
    result["opt_shape"] = str(list(opt_output.shape))

    if not result["shapes_match"]:
        result["correctness"] = "FAIL"
        result["reason"] = f"Shape mismatch: ref={result['ref_shape']}, opt={result['opt_shape']}"
        return result

    # NaN / Inf check
    ref_float = ref_output.float()
    opt_float = opt_output.float()

    result["ref_has_nan"] = bool(torch.isnan(ref_float).any())
    result["ref_has_inf"] = bool(torch.isinf(ref_float).any())
    result["opt_has_nan"] = bool(torch.isnan(opt_float).any())
    result["opt_has_inf"] = bool(torch.isinf(opt_float).any())

    if result["opt_has_nan"] and not result["ref_has_nan"]:
        result["correctness"] = "FAIL"
        result["reason"] = "Optimized output contains NaN where reference does not"
        return result

    if result["opt_has_inf"] and not result["ref_has_inf"]:
        result["correctness"] = "FAIL"
        result["reason"] = "Optimized output contains Inf where reference does not"
        return result

    # Numerical comparison
    diff = (ref_float - opt_float).abs()

    # Mask out positions where both are NaN (those are fine)
    valid_mask = ~(torch.isnan(ref_float) & torch.isnan(opt_float))
    if valid_mask.any():
        valid_diff = diff[valid_mask]
        result["max_abs_error"] = float(valid_diff.max())
        result["mean_abs_error"] = float(valid_diff.mean())
    else:
        result["max_abs_error"] = 0.0
        result["mean_abs_error"] = 0.0

    # Tolerance check
    tols = DEFAULT_TOLERANCES.get(dtype, {"atol": 1e-4, "rtol": 1e-4})
    atol = custom_atol if custom_atol is not None else tols["atol"]
    rtol = custom_rtol if custom_rtol is not None else tols["rtol"]

    # Use allclose on the valid (non-NaN) elements
    if valid_mask.any():
        passes = torch.allclose(
            ref_float[valid_mask], opt_float[valid_mask], atol=atol, rtol=rtol
        )
    else:
        passes = True

    result["correctness"] = "PASS" if passes else "FAIL"
    result["atol"] = atol
    result["rtol"] = rtol

    if not passes:
        result["reason"] = (
            f"Values exceed tolerance (atol={atol}, rtol={rtol}). "
            f"max_abs_error={result['max_abs_error']:.6e}, "
            f"mean_abs_error={result['mean_abs_error']:.6e}"
        )

    return result


# ---------------------------------------------------------------------------
# 6. Diagnosis Mode (apply kernels one at a time)
# ---------------------------------------------------------------------------

def diagnose_kernel_failures(
    model: nn.Module,
    model_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ref_tensor: torch.Tensor,
    replacements: List[KernelReplacement],
    dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    """
    Apply each kernel replacement individually to find which one causes failure.
    """
    results = []

    for repl in replacements:
        print(f"\n  Testing kernel: {repl.kernel_type} (rank {repl.rank})...")
        ctx = OptimizedModelContext(model, [repl])

        try:
            with ctx as patched_model:
                with torch.no_grad():
                    if isinstance(model_input, dict):
                        opt_output = patched_model(**model_input)
                    else:
                        opt_output = patched_model(model_input)
                torch.cuda.synchronize()

            opt_tensor = extract_tensor(opt_output)
            comp = compare_outputs(ref_tensor, opt_tensor, dtype)

            results.append({
                "kernel_type": repl.kernel_type,
                "rank": repl.rank,
                "path": repl.optimized_path,
                "correctness": comp["correctness"],
                "max_abs_error": comp.get("max_abs_error", 0.0),
                "mean_abs_error": comp.get("mean_abs_error", 0.0),
                "reason": comp.get("reason", ""),
            })

            status = comp["correctness"]
            if status == "PASS":
                print(f"    -> PASS (max_err={comp.get('max_abs_error', 0):.6e})")
            else:
                print(f"    -> FAIL: {comp.get('reason', 'unknown')}")

        except Exception as e:
            results.append({
                "kernel_type": repl.kernel_type,
                "rank": repl.rank,
                "path": repl.optimized_path,
                "correctness": "ERROR",
                "max_abs_error": float("inf"),
                "mean_abs_error": float("inf"),
                "reason": str(e),
            })
            print(f"    -> ERROR: {e}")

    return results


# ---------------------------------------------------------------------------
# 7. Output Formatting
# ---------------------------------------------------------------------------

def format_report(result: VerificationResult, diagnose_results: Optional[List] = None) -> str:
    """Format the verification result into a human-readable report."""
    lines = []
    lines.append("")
    lines.append("=== AutoKernel End-to-End Verification ===")
    lines.append("")
    lines.append(f"Model: {result.model_name}")
    lines.append(f"Input: [{result.input_shape}], dtype={result.dtype_str}")
    lines.append(f"GPU: {result.gpu_name}")

    # Reference run
    lines.append("")
    lines.append("--- Reference Run ---")
    lines.append(f"Output shape: {result.ref_output_shape}")
    lines.append(f"Latency: {result.ref_latency_ms:.1f} ms ({TIMED_RUNS} runs, median)")

    # Optimized run
    lines.append("")
    lines.append("--- Optimized Run ---")
    if result.kernels_replaced:
        lines.append("Kernels replaced:")
        for k in result.kernels_replaced:
            lines.append(f"  {k['type']} (rank {k['rank']}): "
                         f"{k['speedup']:.1f}x -> {k['path']}")
    else:
        lines.append("Kernels replaced: none")
    lines.append(f"Output shape: {result.opt_output_shape}")
    lines.append(f"Latency: {result.opt_latency_ms:.1f} ms ({TIMED_RUNS} runs, median)")

    # Verification
    lines.append("")
    lines.append("--- Verification ---")
    lines.append(f"correctness: {result.correctness}")
    lines.append(f"max_abs_error: {result.max_abs_error:.2e}")
    lines.append(f"mean_abs_error: {result.mean_abs_error:.2e}")
    if result.has_nan:
        lines.append("WARNING: NaN detected in optimized output")
    if result.has_inf:
        lines.append("WARNING: Inf detected in optimized output")

    # Summary
    lines.append("")
    lines.append("--- Summary ---")
    lines.append(f"original_latency_ms: {result.ref_latency_ms:.1f}")
    lines.append(f"optimized_latency_ms: {result.opt_latency_ms:.1f}")
    lines.append(f"end_to_end_speedup: {result.end_to_end_speedup:.2f}x")
    lines.append(f"kernels_replaced: {len(result.kernels_replaced)}")

    # Diagnosis
    if diagnose_results:
        lines.append("")
        lines.append("--- Diagnosis (per-kernel) ---")
        for dr in diagnose_results:
            status = dr["correctness"]
            line = f"  {dr['kernel_type']} (rank {dr['rank']}): {status}"
            if status == "PASS":
                line += f" | max_err={dr['max_abs_error']:.2e}"
            if dr.get("reason"):
                line += f" | {dr['reason']}"
            lines.append(line)

    lines.append("")
    return "\n".join(lines)


def save_verification_json(result: VerificationResult, path: str) -> None:
    """Save verification results as JSON for programmatic consumption."""
    data = {
        "model": result.model_name,
        "input_shape": result.input_shape,
        "dtype": result.dtype_str,
        "gpu": result.gpu_name,
        "reference": {
            "output_shape": result.ref_output_shape,
            "latency_ms": round(result.ref_latency_ms, 2),
        },
        "optimized": {
            "output_shape": result.opt_output_shape,
            "latency_ms": round(result.opt_latency_ms, 2),
            "kernels_replaced": result.kernels_replaced,
        },
        "verification": {
            "correctness": result.correctness,
            "max_abs_error": result.max_abs_error,
            "mean_abs_error": result.mean_abs_error,
            "has_nan": result.has_nan,
            "has_inf": result.has_inf,
        },
        "summary": {
            "original_latency_ms": round(result.ref_latency_ms, 2),
            "optimized_latency_ms": round(result.opt_latency_ms, 2),
            "end_to_end_speedup": round(result.end_to_end_speedup, 3),
            "kernels_replaced": len(result.kernels_replaced),
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse a dtype string into a torch.dtype."""
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    key = dtype_str.lower().strip()
    if key not in mapping:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from: {list(mapping.keys())}")
    return mapping[key]


def _get_gpu_name() -> str:
    """Get current GPU name."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "No GPU"


def _output_shape_str(output: Any) -> str:
    """Get shape string from model output."""
    try:
        t = extract_tensor(output)
        return str(list(t.shape))
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global WORKSPACE_DIR, ORCHESTRATION_STATE, WARMUP_RUNS, TIMED_RUNS

    parser = argparse.ArgumentParser(
        description="AutoKernel End-to-End Verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model loading
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model", type=str,
        help="Path to a Python file containing the model class"
    )
    model_group.add_argument(
        "--module", type=str,
        help="Python module name (e.g. 'transformers')"
    )

    parser.add_argument(
        "--class-name", type=str, required=True,
        help="Name of the model class to instantiate"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Pretrained model name/path (for HuggingFace models)"
    )
    parser.add_argument(
        "--input-shape", type=str, default="1,2048",
        help="Comma-separated input shape, e.g. '1,2048' (default: 1,2048)"
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        help="Data type: float16, bfloat16, float32 (default: float16)"
    )

    # Benchmark tuning
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_RUNS,
        help=f"Number of warmup iterations (default: {WARMUP_RUNS})"
    )
    parser.add_argument(
        "--timed", type=int, default=TIMED_RUNS,
        help=f"Number of timed iterations (default: {TIMED_RUNS})"
    )

    # Tolerance overrides
    parser.add_argument("--atol", type=float, default=None, help="Override absolute tolerance")
    parser.add_argument("--rtol", type=float, default=None, help="Override relative tolerance")

    # Modes
    parser.add_argument(
        "--diagnose", action="store_true",
        help="On failure, test each kernel replacement individually to find the culprit"
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Save results to a JSON file at this path"
    )
    parser.add_argument(
        "--workspace", type=str, default=None,
        help="Override workspace directory (default: ./workspace)"
    )

    args = parser.parse_args()

    # Override globals if workspace specified
    if args.workspace:
        WORKSPACE_DIR = os.path.abspath(args.workspace)
        ORCHESTRATION_STATE = os.path.join(WORKSPACE_DIR, "orchestration_state.json")

    WARMUP_RUNS = args.warmup
    TIMED_RUNS = args.timed

    dtype = _parse_dtype(args.dtype)
    gpu_name = _get_gpu_name()

    print("=" * 60)
    print("  AutoKernel End-to-End Verifier")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------------
    # Step 1: Discover optimized kernels
    # -----------------------------------------------------------------------
    print("Step 1: Discovering optimized kernels...")
    replacements = discover_optimized_kernels()
    if not replacements:
        print()
        print("No optimized kernels found.")
        print(f"  Searched: {WORKSPACE_DIR}")
        print(f"  State file: {ORCHESTRATION_STATE}")
        print()
        print("Run the optimization loop first to produce optimized kernels.")
        print("Expected files: workspace/kernel_<type>_<rank>_optimized.py")
        sys.exit(1)

    print(f"  Found {len(replacements)} optimized kernel(s):")
    for r in replacements:
        print(f"    {r.kernel_type} (rank {r.rank}): speedup={r.speedup:.1f}x -> {r.optimized_path}")
    print()

    # -----------------------------------------------------------------------
    # Step 2: Load model
    # -----------------------------------------------------------------------
    print("Step 2: Loading model...")
    try:
        model = load_model(args)
        model_name = args.class_name
        if args.pretrained:
            model_name = f"{args.class_name} ({args.pretrained})"
        print(f"  Model loaded: {model_name}")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,}")
    except Exception as e:
        print(f"\nERROR: Failed to load model: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # -----------------------------------------------------------------------
    # Step 3: Create input
    # -----------------------------------------------------------------------
    print("Step 3: Creating model input...")
    try:
        model_input = make_model_input(model, args.input_shape, dtype)
        if isinstance(model_input, dict):
            for k, v in model_input.items():
                print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  Input: shape={list(model_input.shape)}, dtype={model_input.dtype}")
    except Exception as e:
        print(f"\nERROR: Failed to create input: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # -----------------------------------------------------------------------
    # Step 4: Reference run
    # -----------------------------------------------------------------------
    print("Step 4: Reference run (original PyTorch ops)...")
    try:
        ref_output, ref_latency = benchmark_model(model, model_input, WARMUP_RUNS, TIMED_RUNS)
        ref_tensor = extract_tensor(ref_output)
        ref_shape_str = str(list(ref_tensor.shape))
        print(f"  Output shape: {ref_shape_str}")
        print(f"  Median latency: {ref_latency:.1f} ms")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nERROR: GPU out of memory during reference run.")
            print("  Try a smaller --input-shape or a smaller model.")
            torch.cuda.empty_cache()
            sys.exit(1)
        else:
            raise
    except Exception as e:
        print(f"\nERROR: Reference run failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # -----------------------------------------------------------------------
    # Step 5: Optimized run
    # -----------------------------------------------------------------------
    print("Step 5: Optimized run (with Triton kernel replacements)...")
    ctx = OptimizedModelContext(model, replacements)
    try:
        with ctx as patched_model:
            if ctx.applied_summary:
                print("  Replacements applied:")
                for line in ctx.applied_summary:
                    print(f"  {line}")
            else:
                print("  WARNING: No kernel replacements could be applied to this model.")
                print("  The model may not contain modules matching the optimized kernel types.")

            opt_output, opt_latency = benchmark_model(
                patched_model, model_input, WARMUP_RUNS, TIMED_RUNS
            )
            opt_tensor = extract_tensor(opt_output)
            opt_shape_str = str(list(opt_tensor.shape))
            print(f"  Output shape: {opt_shape_str}")
            print(f"  Median latency: {opt_latency:.1f} ms")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nERROR: GPU out of memory during optimized run.")
            print("  The optimized kernels may use more memory than expected.")
            torch.cuda.empty_cache()
            sys.exit(1)
        else:
            print(f"\nERROR: Optimized run failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Optimized run failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # -----------------------------------------------------------------------
    # Step 6: Compare outputs
    # -----------------------------------------------------------------------
    print("Step 6: Comparing outputs...")
    comp = compare_outputs(ref_tensor, opt_tensor, dtype, args.atol, args.rtol)
    print(f"  correctness: {comp['correctness']}")
    print(f"  max_abs_error: {comp.get('max_abs_error', 0):.2e}")
    print(f"  mean_abs_error: {comp.get('mean_abs_error', 0):.2e}")
    if comp.get("reason"):
        print(f"  reason: {comp['reason']}")
    print()

    # -----------------------------------------------------------------------
    # Step 6b: Diagnose failures if requested
    # -----------------------------------------------------------------------
    diagnose_results = None
    if args.diagnose and comp["correctness"] == "FAIL":
        print("Step 6b: Diagnosing failure (testing each kernel individually)...")
        diagnose_results = diagnose_kernel_failures(
            model, model_input, ref_tensor, replacements, dtype
        )
        print()

    # -----------------------------------------------------------------------
    # Step 7: Build and display final report
    # -----------------------------------------------------------------------
    speedup = ref_latency / opt_latency if opt_latency > 0 else 0.0

    result = VerificationResult(
        model_name=model_name if args.pretrained else args.class_name,
        input_shape=args.input_shape,
        dtype_str=args.dtype,
        gpu_name=gpu_name,
        ref_output_shape=ref_shape_str,
        ref_latency_ms=ref_latency,
        opt_output_shape=opt_shape_str,
        opt_latency_ms=opt_latency,
        kernels_replaced=[
            {
                "type": r.kernel_type,
                "rank": r.rank,
                "speedup": r.speedup,
                "path": r.optimized_path,
            }
            for r in replacements
            if r.module_fn is not None
        ],
        correctness=comp["correctness"],
        max_abs_error=comp.get("max_abs_error", 0.0),
        mean_abs_error=comp.get("mean_abs_error", 0.0),
        has_nan=comp.get("opt_has_nan", False),
        has_inf=comp.get("opt_has_inf", False),
        end_to_end_speedup=speedup,
    )

    report = format_report(result, diagnose_results)
    print(report)

    # Save JSON if requested
    if args.json:
        json_path = os.path.abspath(args.json)
        save_verification_json(result, json_path)
        print(f"Results saved to: {json_path}")

    # Default: save to workspace
    default_json = os.path.join(WORKSPACE_DIR, "verification_result.json")
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    save_verification_json(result, default_json)
    print(f"Results saved to: {default_json}")

    # Exit code: 0 for PASS, 1 for FAIL
    if result.correctness != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
