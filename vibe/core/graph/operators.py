"""The ``@operator`` decorator and the in-process operator registry.

An operator is a *pure* async function whose arguments are the node's inputs and
params (bound by name at execution) and whose return type is a Pydantic model. The
decorator only records the signature and the result type — it does NOT classify
arguments as inputs vs. params. That distinction is the graph's job (see
:mod:`vibe.core.graph.model`).

Purity is a caller-enforced contract: nothing here stops an operator from reading a
clock, the network, or a RNG. If one does, the content-addressed cache will silently
serve stale results. The executor's optional ``verify_purity`` mode is the cheap guard.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
import difflib
import inspect
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel

_COLLECTION_ORIGINS = (list, tuple, set, frozenset, Sequence)


@dataclass(frozen=True)
class OperatorSpec:
    """Everything the executor needs to call and cache an operator.

    ``param_names`` is every argument (used for arity checks). ``input_names`` is the subset
    whose annotation is a Pydantic model — the arguments conventionally wired from upstream
    nodes (``inputs``); the rest are literal ``params``. This split powers the catalog shown
    to the agent so it knows how to author each node.
    """

    name: str
    func: Callable[..., Awaitable[BaseModel]]
    param_names: tuple[str, ...]
    result_type: type[BaseModel]
    input_names: tuple[str, ...] = ()
    arg_types: dict[str, str] = field(default_factory=dict)  # arg name -> readable type
    description: str = ""  # first docstring line
    library: str | None = None  # catalog-scoping tag; None = untagged (generic/kitchen-sink)
    reads_file: str | None = None  # name of a path param whose file content is fingerprinted
    defaults: dict[str, Any] = field(default_factory=dict)  # params with a signature default (optional)
    allowed_values: dict[str, tuple[str, ...]] = field(default_factory=dict)  # enum params (Literal)
    param_types: dict[str, Any] = field(default_factory=dict)  # arg name -> resolved annotation

    def literal_names(self) -> tuple[str, ...]:
        return tuple(p for p in self.param_names if p not in self.input_names)

    def required_names(self) -> frozenset[str]:
        """Params that must be supplied — everything except those with a default."""
        return frozenset(p for p in self.param_names if p not in self.defaults)


_REGISTRY: dict[str, OperatorSpec] = {}


def _readable_type(annotation: Any) -> str:
    """A short, human-readable type name (``FileContent``, ``list[Row]``, ``str``, ``list[str]|None``)."""
    if annotation is type(None):
        return "None"
    if isinstance(annotation, type):
        return annotation.__name__
    origin = get_origin(annotation)
    if origin is Literal:  # Literal["a", "b"] → "a|b" (the agent sees the allowed values)
        return "|".join(str(a) for a in get_args(annotation))
    if origin in {Union, types.UnionType}:  # X | Y → "X|Y"
        return "|".join(_readable_type(a) for a in get_args(annotation))
    if origin is not None:
        inner = ", ".join(_readable_type(a) for a in get_args(annotation))
        return f"{getattr(origin, '__name__', str(origin))}[{inner}]" if inner else str(origin)
    return str(annotation)


def _allowed_values(annotation: Any) -> tuple[str, ...] | None:
    """Allowed values for a ``Literal[...]``, ``list[Literal[...]]``, or an ``Optional`` of either,
    else ``None``. For a list-enum the values apply per element (checked element-wise by the caller).
    """
    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:  # unwrap Optional[...] / X | None
        for a in get_args(annotation):
            if a is not type(None) and (av := _allowed_values(a)) is not None:
                return av
        return None
    if origin is Literal:
        return tuple(str(a) for a in get_args(annotation))
    if origin in _COLLECTION_ORIGINS:
        args = get_args(annotation)
        if args and get_origin(args[0]) is Literal:
            return tuple(str(a) for a in get_args(args[0]))
    return None


def _is_input_annotation(annotation: Any) -> bool:
    """True if the arg is wired from a node: a BaseModel, a list/tuple/set/Sequence of one, or an
    Optional of a BaseModel (``Model | None`` — an optional wired input).
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    origin = get_origin(annotation)
    if origin in _COLLECTION_ORIGINS:
        args = get_args(annotation)
        return bool(args) and isinstance(args[0], type) and issubclass(args[0], BaseModel)
    if origin in {Union, types.UnionType}:  # Optional[Model] / Model | None → an optional input
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        return bool(non_none) and all(
            isinstance(a, type) and issubclass(a, BaseModel) for a in non_none
        )
    return False


def operator(
    func: Callable[..., Awaitable[BaseModel]] | None = None,
    *,
    name: str | None = None,
    library: str | None = None,
    reads_file: str | None = None,
) -> Any:
    """Register an async operator. Usable as ``@operator`` or ``@operator(name=...)``.

    ``library`` tags the operator for per-agent catalog scoping (``None`` = untagged, shown to
    the generic agent). ``reads_file`` names a path param whose file content the caller should
    fingerprint (see ``graph_patch``'s content-hash autofill); the op must also declare a
    ``content_fp`` param.
    """

    def wrap(fn: Callable[..., Awaitable[BaseModel]]) -> Callable[..., Awaitable[BaseModel]]:
        op_name = name or fn.__name__
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"operator {op_name!r} must be an async function")

        signature = inspect.signature(fn)
        params = tuple(signature.parameters)
        defaults = {
            name: p.default
            for name, p in signature.parameters.items()
            if p.default is not inspect.Parameter.empty
        }
        hints = get_type_hints(fn)
        result_type = hints.get("return")
        if not (isinstance(result_type, type) and issubclass(result_type, BaseModel)):
            raise TypeError(
                f"operator {op_name!r} must annotate a pydantic BaseModel return type"
            )
        if reads_file is not None and reads_file not in params:
            raise TypeError(f"operator {op_name!r}: reads_file={reads_file!r} is not a parameter")

        input_names = tuple(p for p in params if _is_input_annotation(hints.get(p)))
        arg_types = {p: _readable_type(hints[p]) for p in params if p in hints}
        param_types = {p: hints[p] for p in params if p in hints}
        allowed_values = {
            p: av for p in params if p in hints and (av := _allowed_values(hints[p])) is not None
        }
        doc = (fn.__doc__ or "").strip()
        description = doc.splitlines()[0].strip() if doc else ""
        _REGISTRY[op_name] = OperatorSpec(
            op_name,
            fn,
            params,
            result_type,
            input_names=input_names,
            arg_types=arg_types,
            description=description,
            library=library,
            reads_file=reads_file,
            defaults=defaults,
            allowed_values=allowed_values,
            param_types=param_types,
        )
        return fn

    return wrap(func) if func is not None else wrap


def get_operator(name: str) -> OperatorSpec:
    """Look up a registered operator, raising ``KeyError`` if unknown."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown operator {name!r}") from None


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def registered_operators() -> dict[str, OperatorSpec]:
    """A snapshot of all registered operators (for catalogs shown to the agent)."""
    return dict(_REGISTRY)


# --- param coercion + value checks (used by the executor's validate / the tools' coerce pass) ---


def _strip_optional(annotation: Any) -> Any:
    """Unwrap ``X | None`` to ``X`` when there's a single non-None member; else return as-is."""
    if get_origin(annotation) in {Union, types.UnionType}:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def _to_int(val: Any) -> Any:
    if isinstance(val, (bool, int)):
        return val
    if isinstance(val, float) and val.is_integer():
        return int(val)
    if isinstance(val, str):
        try:
            return int(val.strip())
        except ValueError:
            return val
    return val


def _to_float(val: Any) -> Any:
    if isinstance(val, (bool, int, float)):
        return val
    if isinstance(val, str):
        try:
            return float(val.strip())
        except ValueError:
            return val
    return val


def _to_bool(val: Any) -> Any:
    if isinstance(val, str) and val.strip().lower() in {"true", "false"}:
        return val.strip().lower() == "true"
    return val


_COERCERS: dict[type, Callable[[Any], Any]] = {int: _to_int, float: _to_float, bool: _to_bool}


def _coerce_scalar(val: Any, ann: Any) -> Any:
    """Best-effort coercion of one scalar toward ``ann``; returns ``val`` unchanged if unsafe."""
    if get_origin(ann) is Literal:  # enum domain is strings; membership checked separately
        return val if isinstance(val, str) else str(val)
    if ann in _COERCERS:
        return _COERCERS[ann](val)
    if ann is str and isinstance(val, (int, float, bool)):
        return str(val)
    return val


def coerce_value(val: Any, annotation: Any) -> Any:
    """Coerce ``val`` toward ``annotation`` for the *unambiguous* slips only (numeric string↔number,
    ``"true"/"false"``→bool, number→str, and a scalar→one-element list where a list is expected).
    Anything else is returned unchanged for :func:`param_issues` to type-check.
    """
    ann = _strip_optional(annotation)
    if get_origin(ann) in _COLLECTION_ORIGINS:
        inner = next(iter(get_args(ann)), str)
        items = val if isinstance(val, list) else [val]  # scalar → [scalar]
        return [_coerce_scalar(v, inner) for v in items]
    return _coerce_scalar(val, ann)


# int is an acceptable float; bool is neither an int nor a float here (a distinct kind).
_PRIMITIVE_CHECKS: dict[type, Callable[[Any], bool]] = {
    float: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    int: lambda v: isinstance(v, int) and not isinstance(v, bool),
    bool: lambda v: isinstance(v, bool),
    str: lambda v: isinstance(v, str),
}


def _type_ok(val: Any, annotation: Any) -> bool:
    """Whether ``val`` matches ``annotation`` (post-coercion). ``Any``/unknown annotations pass."""
    if val is None:
        return get_origin(annotation) in {Union, types.UnionType} and type(None) in get_args(annotation)
    ann = _strip_optional(annotation)
    origin = get_origin(ann)
    if origin is Literal:
        return isinstance(val, str)  # membership handled by allowed_values
    if origin in _COLLECTION_ORIGINS:
        inner = next(iter(get_args(ann)), None)
        return isinstance(val, list) and (inner is None or all(_type_ok(v, inner) for v in val))
    if ann in _PRIMITIVE_CHECKS:
        return _PRIMITIVE_CHECKS[ann](val)
    if isinstance(ann, type):
        return isinstance(val, ann)
    return True  # Any / unresolved → don't enforce


def param_issues(spec: OperatorSpec, params: dict[str, Any]) -> list[str]:
    """Human-readable problems with a node's literal params: invalid enum values (with a
    'did you mean' hint) and wrong value types. Empty list means the params are acceptable.
    """
    issues: list[str] = []
    for key, val in params.items():
        if key in spec.input_names or key not in spec.param_types:
            continue
        allowed = spec.allowed_values.get(key)
        if allowed is not None:
            for v in (val if isinstance(val, list) else [val]):
                if v not in allowed:
                    match = difflib.get_close_matches(str(v), allowed, n=1)
                    hint = f" — did you mean {match[0]!r}?" if match else ""
                    issues.append(
                        f"{spec.name}: {key} must be one of {'|'.join(allowed)}, got {v!r}{hint}"
                    )
            continue  # enum params are strings; skip the generic type check
        if not _type_ok(val, spec.param_types[key]):
            issues.append(
                f"{spec.name}: {key} expected {_readable_type(spec.param_types[key])}, "
                f"got {type(val).__name__} ({val!r})"
            )
    return issues
