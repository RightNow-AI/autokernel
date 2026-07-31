"""Model-independent Dynamo/FX region capture (metadata only).

Captures repeated module calls as executable FX tensor subgraphs for discovery
and ranking. Records only:

- ordered operations and dependencies
- input/output tensor signatures (shape, stride, dtype, device, requires_grad)
- safe scalar constants needed for semantics
- parent module scope, call counts, shape frequency
- graph breaks and unsupported operations

Never serializes tensor values, weights, prompts, model outputs, credentials,
or arbitrary Python source. Fail closed on mutation, collectives,
data-dependent control flow, unknown aliasing, and unsupported custom ops.

CUDA timings remain optional and are filled by a separate profiler path.
"""

from __future__ import annotations

import re
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

from .ranking import classify_pattern_family
from .safety import is_region_safe, normalize_op_name, reject_region
from .types import (
    DiscoveryReport,
    GraphBreakRecord,
    GraphRegion,
    TensorMeta,
    UnsupportedOpRecord,
)

_FORBIDDEN_SERIALIZED_KEYS = frozenset(
    {
        "activations",
        "credential",
        "credentials",
        "data",
        "password",
        "prompt",
        "secret",
        "secrets",
        "source",
        "source_code",
        "tensor_values",
        "token",
        "values",
        "weights",
    }
)

_SAFE_SCALAR_TYPES = (bool, int, float, str)


@dataclass(frozen=True)
class CaptureResult:
    """Result of attempting to capture one module or callable."""

    region: GraphRegion | None
    graph_breaks: tuple[GraphBreakRecord, ...]
    unsupported: tuple[UnsupportedOpRecord, ...]
    operations: tuple[str, ...]


@dataclass
class _RegionAccumulator:
    """Mutable aggregation for repeated captures of the same fingerprint."""

    region: GraphRegion
    calls: int = 0
    shape_keys: dict[str, int] = field(default_factory=dict)
    breaks: list[GraphBreakRecord] = field(default_factory=list)
    unsupported: list[UnsupportedOpRecord] = field(default_factory=list)


def _tensor_meta_from_example(name: str, tensor: Any) -> TensorMeta:
    """Layout metadata only — never reads or stores tensor values."""
    shape = tuple(int(x) for x in tensor.shape)
    if hasattr(tensor, "stride"):
        stride = tuple(int(x) for x in tensor.stride())
    else:
        stride_list: list[int] = []
        running = 1
        for dim in reversed(shape):
            stride_list.append(running)
            running *= max(dim, 1)
        stride = tuple(reversed(stride_list))
    dtype = str(tensor.dtype).replace("torch.", "")
    device_type = "cpu"
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


def _shape_frequency_key(inputs: Sequence[TensorMeta]) -> str:
    parts = []
    for item in inputs:
        shape = "x".join(str(d) for d in item.shape)
        parts.append(f"{item.name}:{shape}:{item.dtype}")
    return "|".join(parts)


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
        "tanh",
        "rsqrt",
        "sqrt",
        "pow",
        "mean",
        "var",
        "neg",
        "exp",
        "clone",
        "contiguous",
        "view",
        "reshape",
        "permute",
        "transpose",
        "unsqueeze",
        "squeeze",
        "cat",
        "stack",
        "expand",
        "type_as",
    }:
        return normalize_op_name(f"aten::{name}")
    return normalize_op_name(str(name))


def _is_safe_constant_value(value: Any) -> bool:
    if isinstance(value, _SAFE_SCALAR_TYPES):
        if isinstance(value, float):
            import math

            return math.isfinite(value)
        if isinstance(value, str):
            # Short enum-like tags only — never free-form source/prompts.
            return 0 < len(value) <= 64 and "\n" not in value
        return True
    if isinstance(value, (list, tuple)):
        if len(value) > 32:
            return False
        return all(_is_safe_constant_value(item) for item in value)
    return False


def _sanitize_safe_constants(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only finite scalars / short tags; drop anything else."""
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            continue
        if key.lower() in _FORBIDDEN_SERIALIZED_KEYS:
            continue
        if _is_safe_constant_value(value):
            result[key] = value
    return result


def _extract_graph_structure(graph: Any) -> tuple[list[str], list[str], dict[str, Any], list[str]]:
    """Return (operations, dependencies, safe_constants, structural_rejections)."""
    operations: list[str] = []
    dependencies: list[str] = []
    safe_constants: dict[str, Any] = {}
    structural: list[str] = []
    node_index: dict[str, int] = {}

    for node in graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue
        if node.op == "get_attr":
            # Parameters / buffers are weights — never export values.
            # Only allow pure Python attribute scalars if present on the module
            # via string target name recording (no value read here).
            target = str(node.target)
            if any(
                part in target.lower()
                for part in ("weight", "bias", "embed", "param", "buffer")
            ):
                structural.append(f"get_attr:{target}: weights/parameters forbidden")
            else:
                # Record attribute *name* only as a dependency tag, not its value.
                safe_constants.setdefault(f"attr:{target}", True)
            continue
        if node.op == "call_function":
            op_key = _function_to_op_key(node.target)
            idx = len(operations)
            node_index[node.name] = idx
            operations.append(op_key)
            for arg in node.args:
                arg_name = getattr(arg, "name", None)
                if arg_name in node_index:
                    dependencies.append(f"{node_index[arg_name]}->{idx}")
            for key, value in (node.kwargs or {}).items():
                if _is_safe_constant_value(value):
                    safe_constants[f"{node.name}.{key}"] = value
                elif value is not None and not hasattr(value, "op"):
                    # Non-safe non-node kwarg — fail closed (unknown constant).
                    structural.append(
                        f"{op_key}: unsafe or non-scalar constant {key!r}"
                    )
        elif node.op == "call_method":
            op_key = normalize_op_name(f"aten::{node.target}")
            idx = len(operations)
            node_index[node.name] = idx
            operations.append(op_key)
            for arg in node.args:
                arg_name = getattr(arg, "name", None)
                if arg_name in node_index:
                    dependencies.append(f"{node_index[arg_name]}->{idx}")
        elif node.op == "call_module":
            # Nested modules collapse to a single opaque node — fail closed for
            # leaf fusion search unless expanded.
            op_key = normalize_op_name(f"module::{node.target}")
            idx = len(operations)
            node_index[node.name] = idx
            operations.append(op_key)
            structural.append(f"{op_key}: nested module not expanded")
        else:
            structural.append(f"unknown_fx_op:{node.op}")

    # Unknown aliasing: multiple outputs writing overlapping views without
    # explicit clone is hard to prove; flag in-place method names already
    # covered by reject_region. Detect star-deps fan-in of getitem/scatter-like.
    for op in operations:
        if "scatter" in op or "index_put" in op or "copy_" in op:
            structural.append(f"{op}: potential aliasing mutation")

    return operations, dependencies, safe_constants, structural


def _trace_module(module: Any, *, tracer: str, example_inputs: Sequence[Any]) -> Any:
    """Return an FX GraphModule or raise."""
    import torch
    import torch.fx as fx

    if tracer == "symbolic":
        return fx.symbolic_trace(module)
    if tracer == "dynamo":
        try:
            exported = torch._dynamo.export(module)(*example_inputs)
            # dynamo.export returns (gm, guards) on some versions or an object
            if isinstance(exported, tuple):
                return exported[0]
            graph_module = getattr(exported, "graph_module", None)
            if graph_module is not None:
                return graph_module
            return exported
        except Exception:
            # Fall back to symbolic when Dynamo cannot export the module.
            return fx.symbolic_trace(module)
    raise ValueError(f"unsupported tracer {tracer!r}; use 'symbolic' or 'dynamo'")


def _assert_region_metadata_only(region: GraphRegion) -> None:
    """Hard privacy check on the produced region dict."""
    payload = region.as_dict()

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, Mapping):
            for key, value in obj.items():
                lower = str(key).lower()
                if lower in _FORBIDDEN_SERIALIZED_KEYS:
                    raise RuntimeError(
                        f"forbidden key {key!r} at {path} in captured region"
                    )
                walk(value, f"{path}.{key}")
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            for index, item in enumerate(obj):
                walk(item, f"{path}[{index}]")

    walk(payload, "region")


def capture_module_region(
    module: Any,
    example_inputs: Sequence[Any],
    *,
    name: str,
    parent_module: str | None = None,
    tracer: str = "symbolic",
    calls: int = 1,
    shape_frequency: Mapping[str, int] | None = None,
) -> CaptureResult:
    """Trace a module and build a metadata-only GraphRegion.

    ``example_inputs`` are used only for tracing shapes/dtypes and one forward
    for output signatures — values are never written into the region.
    """
    import torch

    breaks: list[GraphBreakRecord] = []
    unsupported: list[UnsupportedOpRecord] = []
    operations: list[str] = []
    dependencies: list[str] = []
    safe_constants: dict[str, Any] = {}

    try:
        traced = _trace_module(
            module, tracer=tracer, example_inputs=example_inputs
        )
        graph = getattr(traced, "graph", None)
        if graph is None:
            raise RuntimeError("trace result has no FX graph")
        operations, dependencies, safe_constants, structural = _extract_graph_structure(
            graph
        )
        for reason in structural:
            breaks.append(
                GraphBreakRecord(scope=name, reason=reason, count=1)
            )
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
            GraphBreakRecord(scope=name, reason="empty_graph", count=1)
        )
        return CaptureResult(None, tuple(breaks), tuple(unsupported), ())

    safe_constants = _sanitize_safe_constants(safe_constants)
    rejection = list(reject_region(operations))
    # Nested module / structural reasons already in breaks; fold into rejection.
    for item in breaks:
        if item.reason not in rejection and not item.reason.startswith(
            "fx_trace_failed"
        ):
            rejection.append(item.reason)

    for reason in rejection:
        if (
            "unsupported custom" in reason
            or "not in pure-tensor" in reason
            or "nested module" in reason
        ):
            op_name = reason.rsplit(": ", 1)[0]
            unsupported.append(
                UnsupportedOpRecord(
                    op_name=op_name, reason=reason, count=1, scope=name
                )
            )

    inputs = tuple(
        _tensor_meta_from_example(f"input_{i}", tensor)
        for i, tensor in enumerate(example_inputs)
    )
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

    freq = dict(shape_frequency) if shape_frequency else {}
    if not freq:
        freq[_shape_frequency_key(inputs)] = max(1, calls)

    family = classify_pattern_family(operations)
    region = GraphRegion.build(
        name=name,
        operations=operations,
        inputs=inputs,
        outputs=outputs,
        dependencies=tuple(dependencies),
        parent_module=parent_module,
        safe_constants=safe_constants or None,
        pattern_family=family,
        rejection_reasons=tuple(dict.fromkeys(rejection)),
        calls=max(1, calls),
        shape_frequency=freq,
        cuda_time_us=0.0,
        self_cuda_time_us=0.0,
    )
    _assert_region_metadata_only(region)
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
    tracer: str = "symbolic",
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
        tracer=tracer,
    )


class RegionCaptureSession:
    """Hook selected modules, capture FX regions on each forward, aggregate.

    Designed for repeated DiT block / leaf-module calls without capturing the
    entire generation pipeline as one graph.
    """

    def __init__(
        self,
        *,
        tracer: str = "symbolic",
        name_prefix: str = "region",
    ) -> None:
        self.tracer = tracer
        self.name_prefix = name_prefix
        self._accumulators: dict[str, _RegionAccumulator] = {}
        self._graph_breaks: list[GraphBreakRecord] = []
        self._unsupported: list[UnsupportedOpRecord] = []
        self._hooks: list[Any] = []
        self._call_counters: dict[str, int] = defaultdict(int)
        # Prevent re-entry when capture_module_region runs symbolic_trace /
        # a meta forward on the same hooked module.
        self._capturing: bool = False

    def register_module(
        self,
        module: Any,
        *,
        scope: str,
    ) -> None:
        """Install a forward hook that captures each invocation."""

        def _hook(_mod: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            if self._capturing:
                return
            self._call_counters[scope] += 1
            # Only tensor args are used for signatures.
            tensor_inputs = tuple(
                arg for arg in inputs if hasattr(arg, "shape") and hasattr(arg, "dtype")
            )
            if not tensor_inputs:
                self._graph_breaks.append(
                    GraphBreakRecord(
                        scope=scope,
                        reason="no_tensor_inputs",
                        count=1,
                    )
                )
                return
            capture_name = re.sub(
                r"[^A-Za-z0-9._-]", "_", f"{self.name_prefix}.{scope}"
            )
            if not re.match(r"^[A-Za-z0-9]", capture_name):
                capture_name = f"r.{capture_name}"
            # GraphRegion names must match _NAME_PATTERN (128 chars max),
            # so truncate after any prefix is applied.
            capture_name = capture_name[:128]
            self._capturing = True
            try:
                result = capture_module_region(
                    _mod,
                    tensor_inputs,
                    name=capture_name,
                    parent_module=scope,
                    tracer=self.tracer,
                )
            finally:
                self._capturing = False
            self._graph_breaks.extend(result.graph_breaks)
            self._unsupported.extend(result.unsupported)
            if result.region is None:
                return
            key = result.region.fingerprint
            shape_key = _shape_frequency_key(result.region.inputs)
            if key not in self._accumulators:
                self._accumulators[key] = _RegionAccumulator(region=result.region)
            acc = self._accumulators[key]
            acc.calls += 1
            acc.shape_keys[shape_key] = acc.shape_keys.get(shape_key, 0) + 1
            acc.breaks.extend(result.graph_breaks)
            acc.unsupported.extend(result.unsupported)

        handle = module.register_forward_hook(_hook)
        self._hooks.append(handle)

    def register_named_children(
        self,
        root: Any,
        *,
        predicate: Callable[[str, Any], bool] | None = None,
    ) -> int:
        """Register hooks on ``named_modules`` matching predicate (default: leaves)."""
        count = 0
        for name, child in root.named_modules():
            if name == "":
                continue
            if predicate is not None:
                if not predicate(name, child):
                    continue
            else:
                # Default: leaf modules with no children
                if any(True for _ in child.children()):
                    continue
            self.register_module(child, scope=name or "root")
            count += 1
        return count

    def close(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def __enter__(self) -> "RegionCaptureSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def regions(self) -> tuple[GraphRegion, ...]:
        """Materialize aggregated GraphRegion records."""
        regions: list[GraphRegion] = []
        for acc in self._accumulators.values():
            base = acc.region
            regions.append(
                GraphRegion.build(
                    name=base.name,
                    operations=base.operations,
                    inputs=base.inputs,
                    outputs=base.outputs,
                    dependencies=base.dependencies,
                    parent_module=base.parent_module,
                    safe_constants=base.safe_constants,
                    pattern_family=base.pattern_family,
                    rejection_reasons=base.rejection_reasons,
                    calls=max(1, acc.calls),
                    shape_frequency=dict(acc.shape_keys) or base.shape_frequency,
                    cuda_time_us=base.cuda_time_us,
                    self_cuda_time_us=base.self_cuda_time_us,
                    attributes=base.attributes,
                )
            )
        return tuple(regions)

    def graph_breaks(self) -> tuple[GraphBreakRecord, ...]:
        # Coalesce identical break reasons.
        counts: dict[tuple[str, str, str | None], int] = defaultdict(int)
        for item in self._graph_breaks:
            counts[(item.scope, item.reason, item.op_name)] += item.count
        return tuple(
            GraphBreakRecord(
                scope=scope, reason=reason, op_name=op_name, count=count
            )
            for (scope, reason, op_name), count in sorted(counts.items())
        )

    def unsupported(self) -> tuple[UnsupportedOpRecord, ...]:
        counts: dict[tuple[str, str, str | None], int] = defaultdict(int)
        for item in self._unsupported:
            counts[(item.op_name, item.reason, item.scope)] += item.count
        return tuple(
            UnsupportedOpRecord(
                op_name=op_name, reason=reason, count=count, scope=scope
            )
            for (op_name, reason, scope), count in sorted(counts.items())
        )

    def to_discovery_report(
        self,
        *,
        workload: Mapping[str, Any],
        environment: Mapping[str, Any] | None = None,
        producer: Mapping[str, Any] | None = None,
        total_cuda_time_us: float = 0.0,
    ) -> DiscoveryReport:
        """Build a DiscoveryReport from aggregated FX captures (CPU-safe)."""
        return DiscoveryReport.from_dict(
            {
                "schema_version": 1,
                "producer": dict(
                    producer
                    or {"name": "motionkernel.fx_capture", "version": "1"}
                ),
                "workload": dict(workload),
                "environment": dict(
                    environment
                    or {
                        "hardware_profile_id": "cpu",
                        "software_profile_id": "fx-capture",
                    }
                ),
                "total_cuda_time_us": float(total_cuda_time_us),
                "operators": [],
                "regions": [region.as_dict() for region in self.regions()],
                "graph_breaks": [item.as_dict() for item in self.graph_breaks()],
                "unsupported": [item.as_dict() for item in self.unsupported()],
            }
        )


@contextmanager
def capture_model_regions(
    root: Any,
    *,
    tracer: str = "symbolic",
    predicate: Callable[[str, Any], bool] | None = None,
    name_prefix: str = "region",
) -> Iterator[RegionCaptureSession]:
    """Context manager: hook modules, run caller forward, yield session."""
    session = RegionCaptureSession(tracer=tracer, name_prefix=name_prefix)
    try:
        session.register_named_children(root, predicate=predicate)
        yield session
    finally:
        session.close()
