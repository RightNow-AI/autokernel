"""Optional gradient verification for kernel candidates.

Backward verification is opt-in: it runs only when requested with
``--check-backward`` (or when the spec's ``BackwardSpec.enabled_by_default``
is set). A specification without a :class:`~autokernel.specs.BackwardSpec`
is forward-only; requesting the check then *fails* with an actionable
unsupported message instead of silently skipping.

Protocol (per the Week 2 plan):

1. generate one canonical input mapping (``small`` size, primary dtype);
2. deep-clone tensor inputs for the reference and candidate paths so
   neither side shares autograd state;
3. set ``requires_grad=True`` only for inputs declared in
   ``BackwardSpec.differentiable_inputs``;
4. run reference and candidate independently;
5. select the tensor output leaves declared by ``output_paths``, or every
   floating tensor leaf when omitted;
6. draw deterministic upstream gradients with a fixed-seed generator --
   never only ``output.sum()``, whose symmetry can hide errors;
7. call ``torch.autograd.grad`` with matching inputs and upstreams;
8. compare every requested input gradient by name;
9. report missing gradients, unexpected gradients, shape differences, NaN,
   infinity, maximum error and mean error per input;
10. never accumulate gradients: ``autograd.grad`` returns fresh tensors and
    the generated inputs are never mutated, so repeated checks cannot
    interfere with each other.

Backward execution is correctness-only; no performance claims are made.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..specs.dtypes import canonical_dtype_name
from ..specs.types import BackwardSpec, KernelSpec, Tolerance
from .outputs import (
    DEFAULT_TOLERANCE,
    _is_tensor,
    compare_tensor_leaf,
    flatten_output_tree,
)

__all__ = [
    "BackwardReport",
    "GradientRecord",
    "check_backward",
]

#: Seed for the upstream-gradient generator. Fixed so every run of the same
#: case compares against identical upstream gradients.
UPSTREAM_SEED = 0x5EED


@dataclass(frozen=True)
class GradientRecord:
    """Gradient comparison outcome for one declared differentiable input."""

    input_name: str
    status: str  # "match" | "mismatch" | "missing" | "unexpected"
    reason: str = ""
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    has_nan: bool = False
    has_inf: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_name": self.input_name,
            "status": self.status,
            "reason": self.reason,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
        }


@dataclass(frozen=True)
class BackwardReport:
    """Aggregated backward-verification outcome."""

    status: str  # "PASS" | "FAIL"
    reason: str
    gradients: tuple[GradientRecord, ...] = ()
    output_paths: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "output_paths": list(self.output_paths),
            "gradients": [record.as_dict() for record in self.gradients],
        }


def _fail_report(reason: str) -> BackwardReport:
    return BackwardReport(status="FAIL", reason=reason)


def _validate_declared_inputs(
    spec: KernelSpec, backward: BackwardSpec, inputs: Mapping[str, Any]
) -> str | None:
    """Return an actionable failure reason, or None when inputs are usable."""
    for name in backward.differentiable_inputs:
        if name not in inputs:
            return (
                f"backward_spec declares differentiable input {name!r} but the "
                f"input generator did not produce it (got {sorted(inputs)})"
            )
        value = inputs[name]
        if not _is_tensor(value):
            return (
                f"backward_spec declares differentiable input {name!r} but the "
                f"generated value is not a tensor (got {type(value).__name__})"
            )
        if not value.is_floating_point():
            return (
                f"backward_spec declares differentiable input {name!r} but the "
                f"generated tensor is not floating-point (dtype {value.dtype})"
            )
    return None


def _clone_inputs(inputs: Mapping[str, Any], differentiable: tuple[str, ...]) -> dict:
    """Deep-clone inputs, enabling grad only on declared differentiable names."""
    cloned: dict[str, Any] = {}
    for name, value in inputs.items():
        if _is_tensor(value):
            tensor = value.detach().clone()
            if name in differentiable:
                tensor.requires_grad_(True)
            cloned[name] = tensor
        else:
            cloned[name] = copy.deepcopy(value)
    return cloned


def _select_output_leaves(
    reference_out: Any,
    candidate_out: Any,
    spec: KernelSpec,
    backward: BackwardSpec,
) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]], str | None]:
    """Pick the floating tensor leaves that receive upstream gradients.

    Returns ``(reference_leaves, candidate_leaves, failure_reason)``.
    """
    ref_leaves = flatten_output_tree(reference_out)
    cand_leaves = flatten_output_tree(candidate_out)
    cand_by_path = dict(cand_leaves)

    if backward.output_paths is not None:
        ref_by_path = dict(ref_leaves)
        selected_ref: list[tuple[str, Any]] = []
        selected_cand: list[tuple[str, Any]] = []
        for path in backward.output_paths:
            if path not in ref_by_path:
                return [], [], (
                    f"backward_spec.output_paths entry {path!r} does not exist in "
                    f"the reference output (available: {sorted(ref_by_path)})"
                )
            if path not in cand_by_path:
                return [], [], (
                    f"backward_spec.output_paths entry {path!r} does not exist in "
                    f"the candidate output"
                )
            ref_leaf = ref_by_path[path]
            if not (_is_tensor(ref_leaf) and ref_leaf.is_floating_point()):
                return [], [], (
                    f"backward_spec.output_paths entry {path!r} is not a floating "
                    f"tensor leaf and cannot receive an upstream gradient"
                )
            selected_ref.append((path, ref_leaf))
            selected_cand.append((path, cand_by_path[path]))
        return selected_ref, selected_cand, None

    included = (
        set(spec.output_spec.included_paths)
        if spec.output_spec is not None and spec.output_spec.included_paths is not None
        else None
    )
    selected_ref = [
        (path, leaf)
        for path, leaf in ref_leaves
        if (included is None or path in included)
        and _is_tensor(leaf)
        and leaf.is_floating_point()
    ]
    for path, _ in selected_ref:
        if path not in cand_by_path:
            return [], [], (
                f"candidate output is missing tensor leaf {path!r} needed "
                f"for backward comparison"
            )
    selected_cand = [(path, cand_by_path[path]) for path, _ in selected_ref]
    if not selected_ref:
        return [], [], (
            "no floating tensor output leaves available for backward comparison"
        )
    return selected_ref, selected_cand, None



def _upstream_gradients(
    leaves: list[tuple[str, Any]], device: str, seed: int
) -> list[Any]:
    """Deterministic per-leaf upstream gradients from a fixed-seed generator.

    ``randn`` breaks the symmetry that ``output.sum()``-style upstreams have,
    so sign and cancellation errors cannot hide.
    """
    import torch

    try:
        generator = torch.Generator(device=device)
        generator_is_device_local = True
    except Exception:
        generator = torch.Generator()
        generator_is_device_local = False
    generator.manual_seed(seed)
    upstreams = []
    for _, leaf in leaves:
        if generator_is_device_local:
            upstream = torch.randn(
                tuple(leaf.shape),
                dtype=leaf.dtype,
                device=leaf.device,
                generator=generator,
            )
        else:
            upstream = torch.randn(
                tuple(leaf.shape),
                dtype=leaf.dtype,
                device="cpu",
                generator=generator,
            ).to(leaf.device)
        upstreams.append(upstream)
    return upstreams


def _gradient_tolerance(
    spec: KernelSpec, backward: BackwardSpec, gradient: Any
) -> Tolerance:
    tolerances = backward.tolerances if backward.tolerances is not None else spec.tolerances
    try:
        name = canonical_dtype_name(gradient.dtype)
    except ValueError:
        return DEFAULT_TOLERANCE
    return tolerances.get(name, DEFAULT_TOLERANCE)



def check_backward(
    kernel_fn: Callable[..., Any],
    spec: KernelSpec,
    *,
    device: str,
    seed: int = UPSTREAM_SEED,
) -> BackwardReport:
    """Verify candidate gradients against the reference for one canonical case.

    Never raises for an unsupported or failing candidate: every outcome is a
    structured :class:`BackwardReport`. A spec without ``backward_spec`` is
    forward-only and yields a FAIL report with an actionable unsupported
    message.
    """
    import torch

    backward = spec.backward_spec
    if backward is None:
        return _fail_report(
            f"unsupported: kernel spec {spec.name!r} declares no backward_spec; "
            f"the operation is forward-only. Add a BackwardSpec to the spec or "
            f"drop --check-backward."
        )

    # 1. One canonical input mapping: small size, primary dtype.
    if not spec.sizes:
        return _fail_report(
            f"kernel spec {spec.name!r} declares no sizes; backward "
            f"verification needs at least one size"
        )
    size_label = "small" if "small" in spec.sizes else next(iter(spec.sizes))
    size = dict(spec.sizes[size_label])
    dtype_name = spec.primary_dtype
    base_inputs = spec.input_generator(size, dtype_name, device, seed=42)

    # Declared differentiable inputs must exist and be floating tensors.
    invalid = _validate_declared_inputs(spec, backward, base_inputs)
    if invalid is not None:
        return _fail_report(invalid)

    # 2-3. Independent deep clones; grad only on declared names.
    ref_inputs = _clone_inputs(base_inputs, backward.differentiable_inputs)
    cand_inputs = _clone_inputs(base_inputs, backward.differentiable_inputs)

    # 4. Independent forward passes.
    try:
        reference_out = spec.reference_fn(**ref_inputs)
    except Exception as exc:
        return _fail_report(
            f"reference forward failed: {type(exc).__name__}: {exc}"
        )
    try:
        candidate_out = kernel_fn(**cand_inputs)
    except Exception as exc:
        return _fail_report(
            f"candidate forward failed: {type(exc).__name__}: {exc}"
        )

    # 5. Select the output leaves that receive upstream gradients.
    ref_leaves, cand_leaves, failure = _select_output_leaves(
        reference_out, candidate_out, spec, backward
    )
    if failure is not None:
        return _fail_report(failure)

    # 6. Deterministic upstream gradients (identical for both paths).
    try:
        upstreams = _upstream_gradients(ref_leaves, device, seed)
    except Exception as exc:
        return _fail_report(
            f"upstream gradient generation failed: {type(exc).__name__}: {exc}"
        )

    # 7. Independent autograd.grad calls; allow_unused surfaces missing grads.
    grad_inputs_ref = [ref_inputs[name] for name in backward.differentiable_inputs]
    grad_inputs_cand = [cand_inputs[name] for name in backward.differentiable_inputs]
    try:
        ref_grads = torch.autograd.grad(
            [leaf for _, leaf in ref_leaves],
            grad_inputs_ref,
            upstreams,
            allow_unused=True,
        )
    except Exception as exc:
        return _fail_report(
            f"reference backward failed: {type(exc).__name__}: {exc}"
        )
    try:
        cand_grads = torch.autograd.grad(
            [leaf for _, leaf in cand_leaves],
            grad_inputs_cand,
            upstreams,
            allow_unused=True,
        )
    except Exception as exc:
        return _fail_report(
            f"candidate backward failed (is the candidate differentiable?): "
            f"{type(exc).__name__}: {exc}"
        )


    # 8-9. Compare every requested gradient by name.
    records: list[GradientRecord] = []
    for name, ref_grad, cand_grad in zip(
        backward.differentiable_inputs, ref_grads, cand_grads
    ):
        if ref_grad is None and cand_grad is None:
            records.append(
                GradientRecord(
                    input_name=name,
                    status="missing",
                    reason="no gradient in reference or candidate; the declared "
                    "differentiable input does not influence the selected outputs",
                )
            )
            continue
        if cand_grad is None:
            records.append(
                GradientRecord(
                    input_name=name,
                    status="missing",
                    reason="candidate produced no gradient but the reference did",
                )
            )
            continue
        if ref_grad is None:
            records.append(
                GradientRecord(
                    input_name=name,
                    status="unexpected",
                    reason="candidate produced a gradient where the reference "
                    "has none",
                )
            )
            continue
        leaf = compare_tensor_leaf(
            f'grad["{name}"]',
            cand_grad,
            ref_grad,
            _gradient_tolerance(spec, backward, ref_grad),
        )
        records.append(
            GradientRecord(
                input_name=name,
                status="match" if leaf.match else "mismatch",
                reason=leaf.reason,
                max_abs_error=leaf.max_abs_error,
                mean_abs_error=leaf.mean_abs_error,
                has_nan=leaf.has_nan,
                has_inf=leaf.has_inf,
            )
        )

    failures = [record for record in records if record.status != "match"]
    output_paths = tuple(path for path, _ in ref_leaves)
    if failures:
        first = failures[0]
        return BackwardReport(
            status="FAIL",
            reason=f"grad[{first.input_name!r}] {first.status}: {first.reason}",
            gradients=tuple(records),
            output_paths=output_paths,
        )
    return BackwardReport(
        status="PASS",
        reason="",
        gradients=tuple(records),
        output_paths=output_paths,
    )
