"""
AutoKernel -- CUDA C++ kernel compilation utility.

Compiles CUDA C++ source strings into callable PyTorch extensions using
torch.utils.cpp_extension.load_inline(). Provides:
  - Hash-based caching (recompile only when source changes)
  - Architecture auto-detection (gencode for current GPU)
  - Robust error handling with source-level diagnostics
  - Thread-safe compilation via file locking
"""

import hashlib
import os
import re
import sys
import threading

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autokernel", "cuda_build")

# Default CUDA compiler flags
_DEFAULT_CUDA_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "--expt-relaxed-constexpr",
    "-std=c++17",
]

# Module-level cache: {hash -> compiled module}
_module_cache: dict = {}
_compile_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def _get_arch_flags() -> list:
    """Generate -gencode flags for the current GPU architecture."""
    if not torch.cuda.is_available():
        return []

    cap = torch.cuda.get_device_capability()
    major, minor = cap
    sm = f"{major}{minor}"

    # Hopper (sm_90a) and datacenter Blackwell (sm_100a / sm_103a) only expose
    # wgmma / tcgen05 / TMEM through the arch-specific "a" target.
    cubin_sm = f"{sm}a" if major in (9, 10) else sm

    flags = [
        # Compile for the exact GPU
        f"-gencode=arch=compute_{cubin_sm},code=sm_{cubin_sm}",
        # Also embed PTX for forward compatibility ("a" PTX is not portable)
        f"-gencode=arch=compute_{sm},code=compute_{sm}",
    ]
    return flags


# ---------------------------------------------------------------------------
# Source hashing
# ---------------------------------------------------------------------------

def _extract_forward_decl(cuda_src: str, func_name: str) -> str:
    """
    Extract a forward declaration for func_name from the CUDA source.

    load_inline auto-generates pybind11 bindings in main.cpp when
    ``functions=[func_name]`` is passed, but main.cpp has no visibility
    of functions defined in cuda.cu.  Providing a forward declaration in
    cpp_sources fixes the 'not declared in this scope' error.
    """
    # Match: return_type func_name(args) { ...
    pattern = (
        r"((?:torch::Tensor|std::vector<torch::Tensor>|at::Tensor|void)"
        r"\s+" + re.escape(func_name) + r"\s*\([^{]*?\))\s*\{"
    )
    match = re.search(pattern, cuda_src, re.DOTALL)
    if match:
        return match.group(1).strip() + ";"
    return ""


def _hash_source(cuda_src: str, cpp_src: str, extra_flags: list) -> str:
    """Create a deterministic hash of the source + flags for cache keying."""
    h = hashlib.sha256()
    h.update(cuda_src.encode("utf-8"))
    h.update(cpp_src.encode("utf-8"))
    h.update("|".join(sorted(extra_flags)).encode("utf-8"))
    # Include torch version and CUDA version in hash
    h.update(torch.__version__.encode("utf-8"))
    cuda_ver = torch.version.cuda or "unknown"
    h.update(cuda_ver.encode("utf-8"))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# C++ wrapper generation
# ---------------------------------------------------------------------------

def _generate_cpp_wrapper(func_name: str, arg_specs: list) -> str:
    """
    Generate a pybind11-compatible C++ wrapper that forwards torch::Tensor
    arguments to the CUDA kernel launcher.

    arg_specs: list of (name, type_str) pairs. Supported types:
        - "tensor" -> torch::Tensor
        - "int" -> int64_t
        - "float" -> double
        - "bool" -> bool
    """
    # Build function signature
    cpp_args = []
    for name, type_str in arg_specs:
        if type_str == "tensor":
            cpp_args.append(f"torch::Tensor {name}")
        elif type_str == "int":
            cpp_args.append(f"int64_t {name}")
        elif type_str == "float":
            cpp_args.append(f"double {name}")
        elif type_str == "bool":
            cpp_args.append(f"bool {name}")
        else:
            cpp_args.append(f"torch::Tensor {name}")

    args_str = ", ".join(cpp_args)
    forward_args = ", ".join(name for name, _ in arg_specs)

    wrapper = f"""
#include <torch/extension.h>

// Forward declaration of CUDA launcher (defined in .cu source)
torch::Tensor {func_name}_cuda({args_str});

// Python-facing wrapper
torch::Tensor {func_name}({args_str}) {{
    return {func_name}_cuda({forward_args});
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("{func_name}", &{func_name}, "{func_name}");
}}
"""
    return wrapper


# ---------------------------------------------------------------------------
# Main compilation function
# ---------------------------------------------------------------------------

def compile_cuda(
    cuda_src: str,
    func_name: str,
    cpp_src: str = "",
    extra_cuda_cflags: list = None,
    extra_cflags: list = None,
    verbose: bool = False,
):
    """
    Compile CUDA C++ source into a callable PyTorch extension module.

    Parameters
    ----------
    cuda_src : str
        The CUDA C++ source code (kernel + launcher function).
        The launcher must be named ``{func_name}_cuda`` and return a torch::Tensor.
    func_name : str
        Name of the Python-callable function. A C++ wrapper is auto-generated
        that calls ``{func_name}_cuda`` from the CUDA source.
    cpp_src : str, optional
        Custom C++ wrapper source. If empty, one is auto-generated.
    extra_cuda_cflags : list, optional
        Additional nvcc flags.
    extra_cflags : list, optional
        Additional host compiler flags.
    verbose : bool
        Print compilation output.

    Returns
    -------
    module
        Compiled extension module with ``module.{func_name}(...)`` callable.
    """
    from torch.utils.cpp_extension import load_inline

    if extra_cuda_cflags is None:
        extra_cuda_cflags = []
    if extra_cflags is None:
        extra_cflags = []

    # Build the full flag set
    cuda_flags = list(_DEFAULT_CUDA_FLAGS) + _get_arch_flags() + extra_cuda_cflags
    host_flags = ["-O3"] + extra_cflags

    # If no custom C++ source, generate a forward declaration so that the
    # auto-generated pybind11 main.cpp can see the CUDA function.
    if not cpp_src:
        fwd_decl = _extract_forward_decl(cuda_src, func_name)
        if fwd_decl:
            cpp_src_final = f"#include <torch/extension.h>\n{fwd_decl}\n"
        else:
            cpp_src_final = ""
    else:
        cpp_src_final = cpp_src

    # Hash for caching
    src_hash = _hash_source(cuda_src, cpp_src_final, cuda_flags)
    cache_key = f"{func_name}_{src_hash}"

    # Check in-memory cache first
    if cache_key in _module_cache:
        return _module_cache[cache_key]

    # Ensure build directory exists
    build_dir = os.path.join(_CACHE_DIR, cache_key)
    os.makedirs(build_dir, exist_ok=True)

    with _compile_lock:
        # Double-check after acquiring lock
        if cache_key in _module_cache:
            return _module_cache[cache_key]

        try:
            module = load_inline(
                name=cache_key,
                cpp_sources=[cpp_src_final] if cpp_src_final else [],
                cuda_sources=[cuda_src],
                functions=[func_name],
                extra_cuda_cflags=cuda_flags,
                extra_cflags=host_flags,
                build_directory=build_dir,
                verbose=verbose,
            )
        except Exception as e:
            # Print diagnostics
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"CUDA COMPILATION FAILED: {func_name}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(f"Error: {e}", file=sys.stderr)
            print(f"\nCUDA source ({len(cuda_src.splitlines())} lines):", file=sys.stderr)
            for i, line in enumerate(cuda_src.splitlines(), 1):
                print(f"  {i:4d} | {line}", file=sys.stderr)
            print(f"\nFlags: {' '.join(cuda_flags)}", file=sys.stderr)
            print(f"Build dir: {build_dir}", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
            raise

        _module_cache[cache_key] = module
        return module


# ---------------------------------------------------------------------------
# Convenience: compile with auto-generated C++ wrapper
# ---------------------------------------------------------------------------

def compile_cuda_with_wrapper(
    cuda_src: str,
    func_name: str,
    arg_specs: list,
    extra_cuda_cflags: list = None,
    verbose: bool = False,
):
    """
    Like compile_cuda, but auto-generates the C++ wrapper from arg_specs.

    Parameters
    ----------
    cuda_src : str
        CUDA source that defines ``torch::Tensor {func_name}_cuda(...)``.
    func_name : str
        Function name.
    arg_specs : list of (name, type_str) tuples
        Argument specifications. See ``_generate_cpp_wrapper`` for types.
    """
    cpp_src = _generate_cpp_wrapper(func_name, arg_specs)
    return compile_cuda(
        cuda_src=cuda_src,
        func_name=func_name,
        cpp_src=cpp_src,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )
