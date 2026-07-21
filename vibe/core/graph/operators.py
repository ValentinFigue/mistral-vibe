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
import inspect
from typing import Any, get_args, get_origin, get_type_hints

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

    def literal_names(self) -> tuple[str, ...]:
        return tuple(p for p in self.param_names if p not in self.input_names)


_REGISTRY: dict[str, OperatorSpec] = {}


def _readable_type(annotation: Any) -> str:
    """A short, human-readable type name (``FileContent``, ``list[Row]``, ``str``)."""
    if isinstance(annotation, type):
        return annotation.__name__
    origin = get_origin(annotation)
    if origin is not None:
        inner = ", ".join(_readable_type(a) for a in get_args(annotation))
        return f"{getattr(origin, '__name__', str(origin))}[{inner}]" if inner else str(origin)
    return str(annotation)


def _is_input_annotation(annotation: Any) -> bool:
    """True if the arg is wired from a node: a BaseModel, or a list/tuple/set/Sequence of one."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    if get_origin(annotation) in _COLLECTION_ORIGINS:
        args = get_args(annotation)
        return bool(args) and isinstance(args[0], type) and issubclass(args[0], BaseModel)
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

        params = tuple(inspect.signature(fn).parameters)
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
