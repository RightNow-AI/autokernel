"""Optional full-graph ``torch.compile`` verification for kernel candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..specs.types import CompileSpec, KernelSpec
from .outputs import compare_output_trees

__all__ = ["CompileCaseRecord", "CompileReport", "check_compile"]


@dataclass(frozen=True)
class CompileCaseRecord:
    """Comparison result for one shape passed through the compiled callable."""

    label: str
    size: dict[str, int]
    status: str
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "size": dict(self.size),
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CompileReport:
    """Structured compile-verification outcome."""

    status: str  # "PASS" | "FAIL" | "UNSUPPORTED"
    reason: str
    cases: tuple[CompileCaseRecord, ...]
    environment: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "cases": [case.as_dict() for case in self.cases],
            "environment": dict(self.environment),
        }


def _environment(torch: Any, device: str, settings: CompileSpec) -> dict[str, Any]:
    try:
        import triton

        triton_version = getattr(triton, "__version__", "unknown")
    except Exception:
        triton_version = None

    gpu_name = None
    if device.startswith("cuda"):
        try:
            gpu_name = torch.cuda.get_device_name(device)
        except Exception:
            gpu_name = None
    return {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "triton_version": triton_version,
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "device": device,
        "gpu_name": gpu_name,
        "fullgraph": settings.fullgraph,
        "dynamic": settings.dynamic,
    }


def _selected_cases(spec: KernelSpec, dynamic: bool) -> list[tuple[str, dict[str, int]]]:
    items = list(spec.size_items())
    if not items:
        return []
    if not dynamic:
        label = "small" if "small" in spec.sizes else items[0][0]
        return [(label, dict(spec.sizes[label]))]

    # KernelSpec validation guarantees compatible keys. Prefer small/medium
    # because compile verification is a correctness gate, not a stress test.
    selected: list[tuple[str, dict[str, int]]] = []
    for label in ("small", "medium"):
        if label in spec.sizes:
            selected.append((label, dict(spec.sizes[label])))
    for label, size in items:
        if len(selected) >= 2:
            break
        if label not in {existing for existing, _ in selected}:
            selected.append((label, dict(size)))
    return selected[:2]


def check_compile(
    kernel_fn: Callable[..., Any],
    spec: KernelSpec,
    *,
    device: str,
) -> CompileReport:
    """Compile a candidate outside timed regions and compare it to eager.

    The same compiled callable is invoked at least twice per selected shape.
    Dynamic specifications exercise two compatible shapes through that single
    callable. All failures are returned as structured results.
    """
    import torch

    settings = spec.compile_spec or CompileSpec()
    environment = _environment(torch, device, settings)
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        return CompileReport(
            status="UNSUPPORTED",
            reason="torch.compile is unavailable in this PyTorch installation",
            cases=(),
            environment=environment,
        )

    cases = _selected_cases(spec, settings.dynamic)
    if settings.dynamic and len(cases) < 2:
        return CompileReport(
            status="FAIL",
            reason="dynamic compile verification requires at least two declared sizes",
            cases=(),
            environment=environment,
        )

    def candidate(**inputs: Any) -> Any:
        return kernel_fn(**inputs)

    try:
        compiled = compile_fn(
            candidate,
            fullgraph=settings.fullgraph,
            dynamic=settings.dynamic,
        )
    except Exception as exc:
        return CompileReport(
            status="FAIL",
            reason=f"compiler setup failed: {type(exc).__name__}: {exc}",
            cases=(),
            environment=environment,
        )

    records: list[CompileCaseRecord] = []
    for label, size in cases:
        try:
            inputs = spec.input_generator(
                size, spec.primary_dtype, device, seed=42
            )
            expected = spec.reference_fn(**inputs)
            # torch.compile is lazy. The first call may compile; the second
            # proves the compiled callable remains correct on a repeated run.
            compiled(**inputs)
            actual = compiled(**inputs)
            comparison = compare_output_trees(
                actual,
                expected,
                spec.tolerances,
                output_spec=spec.output_spec,
            )
            status = "PASS" if comparison.match else "FAIL"
            records.append(
                CompileCaseRecord(
                    label=label,
                    size=size,
                    status=status,
                    reason="" if comparison.match else comparison.reason,
                )
            )
        except Exception as exc:
            records.append(
                CompileCaseRecord(
                    label=label,
                    size=size,
                    status="FAIL",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )

    passed = bool(records) and all(record.status == "PASS" for record in records)
    reason = "" if passed else next(
        (record.reason for record in records if record.status != "PASS"),
        "no compile cases were selected",
    )
    return CompileReport(
        status="PASS" if passed else "FAIL",
        reason=reason,
        cases=tuple(records),
        environment=environment,
    )
