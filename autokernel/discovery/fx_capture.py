"""CPU-safe FX / symbolic-trace capture helpers for pure tensor modules.

Produces metadata-only GraphRegion objects. Never serializes tensor values,
prompts, or weights into the region. Graph breaks are recorded as strings.

Full production capture still needs a GPU profiling pass for CUDA times; this
module builds the structural graph side of the discovery report on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .ranking import classify_pattern_family
from .safety import normalize_op_name, reject_region
from .types import GraphBreakRecord, GraphRegion, TensorMeta, UnsupportedOpRecord


@dataclass(frozen=True)
class CaptureResult:
    """Result of attempting to capture one module or callable."""

    region: GraphRegion | None
    graph_breaks: tuple[GraphBreakRecord, ...]
    unsupported: tuple[UnsupportedOpRecord, ...]
    operations: tuple[str, ...]


def _tensor_meta_from_example(name: str, tensor: Any) -> TensorMeta:
    shape = tuple(int(x) for x in tensor.shape)
    # Prefer stride when available; fall back to contiguous row-major.
    if hasattr(tensor, "stride"):
        stride = tuple(int(x) for x in tensor.stride())
    else:
        stride = []
        running = 1
        for dim in reversed(shape):
            stride.append(running)
            running *= max(dim, 1)
        stride = tuple(reversed(stride))
    dtype = str(tensor.dtype).replace("torch.", "")
    device_type = str(getattr(tensor, "device", "cpu"))
    if hasattr(tensor, "device") and hasattr(tensor.device, "type"):
        device_type = tensor.device.type
    requires_grad = bool(getattr(tensor, "requires_grad", False))
    return TensorMeta(
        name=name,
        shape=shape,
        stride=stride,
        dtype=dtype,
        device_type=device_type,
        requires_grad=requires_grad,
    )


def _function_to_op_key(target: Any) -> str:
    text = str(target)
    if "aten::" in text:
        return normalize_op_name(
            "aten::" + text.split("aten::", 1)[1].split(".")[0].split("(")[0]
        )
    if "aten." in text:
        return normalize_op_name("aten::" + text.split("aten.", 1)[1].split(".")[0])
    name = getattr(target, "__name__", None) or text
    module = getattr(target, "__module__", "") or ""
    if "torch" in module or name in {
        "add",
        "mul",
        "sub",
        "div",
        "silu",
        "gelu",
        "relu",
        "sigmoid",
        "layer_norm",
        "softmax",
    }:
        return normalize_op_name(f"aten::{name}")
    return normalize_op_name(str(name))


def _ops_from_fx_graph(graph: Any) -> list[str]:
    operations: list[str] = []
    for node in graph.nodes:
        if node.op == "call_function":
            operations.append(_function_to_op_key(node.target))
        elif node.op == "call_method":
            operations.append(normalize_op_name(f"aten::{node.target}"))
        elif node.op == "call_module":
            operations.append(normalize_op_name(f"module::{node.target}"))
    return operations


def capture_module_region(
    module: Any,
    example_inputs: Sequence[Any],
    *,
    name: str,
    parent_module: str | None = None,
    tracer: str = "symbolic",
) -> CaptureResult:
    """Trace a module on CPU and build a GraphRegion when safe.

    ``example_inputs`` must be tensors (or tensor-like) used only for shapes
    and dtypes — values are never written into the region.
    """
    import torch
    import torch.fx as fx

    breaks: list[GraphBreakRecord] = []
    unsupported: list[UnsupportedOpRecord] = []
    operations: list[str] = []

    try:
        if tracer == "symbolic":
            # symbolic_trace works for many pure modules without Dynamo.
            traced = fx.symbolic_trace(module)
            operations = _ops_from_fx_graph(traced.graph)
        else:
            breaks.append(
                GraphBreakRecord(
                    scope=name,
                    reason=f"unsupported tracer {tracer!r}",
                    count=1,
                )
            )
            return CaptureResult(None, tuple(breaks), tuple(unsupported), ())
    except Exception as exc:  # noqa: BLE001 - capture failures are data
        breaks.append(
            GraphBreakRecord(
                scope=name,
                reason=f"fx_trace_failed: {type(exc).__name__}: {exc}",
                count=1,
            )
        )
        return CaptureResult(None, tuple(breaks), tuple(unsupported), ())

    if not operations:
        breaks.append(
            GraphBreakRecord(
                scope=name,
                reason="empty_graph",
                count=1,
            )
        )
        return CaptureResult(None, tuple(breaks), tuple(unsupported), ())

    rejection = reject_region(operations)
    for reason in rejection:
        if "unsupported custom" in reason or "not in pure-tensor" in reason:
            op_name = reason.rsplit(": ", 1)[0]
            unsupported.append(
                UnsupportedOpRecord(op_name=op_name, reason=reason, count=1, scope=name)
            )

    inputs = tuple(
        _tensor_meta_from_example(f"input_{i}", tensor)
        for i, tensor in enumerate(example_inputs)
    )
    # Run once to infer output meta (values discarded).
    outputs: tuple[TensorMeta, ...] = ()
    try:
        with torch.no_grad():
            out = module(*example_inputs)
        if isinstance(out, torch.Tensor):
            outputs = (_tensor_meta_from_example("output_0", out),)
        elif isinstance(out, (tuple, list)):
            outputs = tuple(
                _tensor_meta_from_example(f"output_{i}", item)
                for i, item in enumerate(out)
                if isinstance(item, torch.Tensor)
            )
    except Exception as exc:  # noqa: BLE001
        breaks.append(
            GraphBreakRecord(
                scope=name,
                reason=f"output_meta_failed: {type(exc).__name__}: {exc}",
                count=1,
            )
        )

    family = classify_pattern_family(operations)
    region = GraphRegion.build(
        name=name,
        operations=operations,
        inputs=inputs,
        outputs=outputs,
        parent_module=parent_module,
        pattern_family=family,
        rejection_reasons=tuple(rejection),
        calls=1,
        cuda_time_us=0.0,
        self_cuda_time_us=0.0,
    )
    return CaptureResult(
        region=region,
        graph_breaks=tuple(breaks),
        unsupported=tuple(unsupported),
        operations=tuple(operations),
    )


def capture_callable_region(
    fn: Callable[..., Any],
    example_inputs: Sequence[Any],
    *,
    name: str,
) -> CaptureResult:
    """Wrap a pure function as an nn.Module and capture it."""
    import torch.nn as nn

    arity = len(example_inputs)
    if arity == 1:

        class _Wrapper1(nn.Module):
            def forward(self, x):  # type: ignore[no-untyped-def]
                return fn(x)

        wrapper: nn.Module = _Wrapper1()
    elif arity == 2:

        class _Wrapper2(nn.Module):
            def forward(self, x, y):  # type: ignore[no-untyped-def]
                return fn(x, y)

        wrapper = _Wrapper2()
    elif arity == 3:

        class _Wrapper3(nn.Module):
            def forward(self, x, y, z):  # type: ignore[no-untyped-def]
                return fn(x, y, z)

        wrapper = _Wrapper3()
    else:
        return CaptureResult(
            None,
            (
                GraphBreakRecord(
                    scope=name,
                    reason=f"callable arity {arity} not supported for FX wrap",
                    count=1,
                ),
            ),
            (),
            (),
        )

    return capture_module_region(
        wrapper,
        example_inputs,
        name=name,
        parent_module=None,
    )
