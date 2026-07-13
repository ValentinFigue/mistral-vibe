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

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import inspect
from typing import Any, get_type_hints

from pydantic import BaseModel


@dataclass(frozen=True)
class OperatorSpec:
    """Everything the executor needs to call and cache an operator."""

    name: str
    func: Callable[..., Awaitable[BaseModel]]
    param_names: tuple[str, ...]
    result_type: type[BaseModel]


_REGISTRY: dict[str, OperatorSpec] = {}


def operator(
    func: Callable[..., Awaitable[BaseModel]] | None = None,
    *,
    name: str | None = None,
) -> Any:
    """Register an async operator. Usable as ``@operator`` or ``@operator(name=...)``."""

    def wrap(fn: Callable[..., Awaitable[BaseModel]]) -> Callable[..., Awaitable[BaseModel]]:
        op_name = name or fn.__name__
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"operator {op_name!r} must be an async function")

        params = tuple(inspect.signature(fn).parameters)
        result_type = get_type_hints(fn).get("return")
        if not (isinstance(result_type, type) and issubclass(result_type, BaseModel)):
            raise TypeError(
                f"operator {op_name!r} must annotate a pydantic BaseModel return type"
            )

        _REGISTRY[op_name] = OperatorSpec(op_name, fn, params, result_type)
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
