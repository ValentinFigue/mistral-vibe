"""The incremental, resumable graph executor.

``execute`` validates the graph, walks it in dependency order dispatching each ready wave
concurrently, and for every node computes its fingerprint and consults the cache:

* **hit** → rehydrate the stored payload into the operator's result type and reuse it
  (this is both incremental recompute *and* crash-resume);
* **miss** → run the operator and store the result.

Concurrency here is *structural*: the topology, not hand-written orchestration, decides
what may run together. Because operators run on the single-threaded event loop, this
yields correct scheduling shape, not wall-clock parallelism — real parallelism and a
resource model are M2.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from vibe.core.llm.types import LLMCaller

from vibe.core.graph.cache import CacheStore
from vibe.core.graph.fingerprint import fingerprint_node
from vibe.core.graph.model import Graph, NodeId, NodeState, Report, Value
from vibe.core.graph.operators import (
    OperatorSpec,
    coerce_value,
    get_operator,
    is_registered,
    param_issues,
)


class GraphValidationError(Exception):
    """The graph is structurally invalid (bad reference, arity mismatch, or cycle)."""


class PurityError(Exception):
    """A cached node re-ran under ``verify_purity`` and produced a different result."""


class GraphEvent(BaseModel):
    """Per-node progress event emitted during execution."""

    node_id: NodeId
    op: str
    state: NodeState
    fingerprint: str
    duration: float


def validate(graph: Graph) -> None:
    """Check operator existence, referential integrity, arity, and acyclicity."""
    for node in graph.nodes.values():
        spec = get_operator(node.op)  # raises KeyError -> surfaced to caller
        for port, dep in node.inputs.items():
            if dep not in graph.nodes:
                raise GraphValidationError(
                    f"node {node.id!r} input {port!r} references unknown node {dep!r}"
                )
            # Type-aware wiring: catch feeding e.g. a Report into a table port here, at validate
            # time, instead of as a cryptic AttributeError mid-execution. Only enforced for a
            # plain model type (skip list[...]/unions and unknown producers — never false-reject).
            expected = spec.arg_types.get(port, "")
            if expected and "[" not in expected and "|" not in expected:
                produced = get_operator(graph.nodes[dep].op).result_type.__name__
                if produced != expected:
                    raise GraphValidationError(
                        f"node {node.id!r} input {port!r} expects {expected} but node {dep!r} "
                        f"produces {produced}"
                    )
        provided = set(node.inputs) | set(node.params)
        missing = spec.required_names() - provided
        extra = provided - set(spec.param_names)
        if missing or extra:
            raise GraphValidationError(
                f"node {node.id!r} (op {node.op!r}) argument mismatch: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        # Value-level checks: enum membership (with a "did you mean" hint) and param value types,
        # so a bad value fails here with a clear message rather than mid-execution.
        issues = param_issues(spec, node.params)
        if issues:
            raise GraphValidationError(f"node {node.id!r}: {issues[0]}")
    _topo_order(graph)  # raises GraphValidationError on a cycle


def coerce_params(graph: Graph) -> None:
    """Coerce each node's literal params toward the operator's declared types, in place — the
    unambiguous slips only (numeric string↔number, ``"true"/"false"``→bool, number→str, scalar→list;
    see :func:`vibe.core.graph.operators.coerce_value`). Run before ``validate``/fingerprinting so
    the cache key reflects the coerced values and stays stable across turns. Deterministic.
    """
    for node in graph.nodes.values():
        if not is_registered(node.op):
            continue
        spec = get_operator(node.op)
        for key in list(node.params):
            if key not in spec.input_names and key in spec.param_types:
                node.params[key] = coerce_value(node.params[key], spec.param_types[key])


def _topo_order(graph: Graph) -> list[NodeId]:
    """Kahn's algorithm; raises ``GraphValidationError`` if the graph has a cycle."""
    indegree = {nid: 0 for nid in graph.nodes}
    dependents: dict[NodeId, list[NodeId]] = {nid: [] for nid in graph.nodes}
    for node in graph.nodes.values():
        for dep in node.inputs.values():
            indegree[node.id] += 1
            dependents[dep].append(node.id)

    ready = [nid for nid, deg in indegree.items() if deg == 0]
    order: list[NodeId] = []
    while ready:
        nid = ready.pop()
        order.append(nid)
        for child in dependents[nid]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)

    if len(order) != len(graph.nodes):
        raise GraphValidationError("graph contains a cycle")
    return order


async def _run_operator(
    spec: OperatorSpec,
    graph: Graph,
    node_id: NodeId,
    results: dict[NodeId, BaseModel],
    llm: LLMCaller | None = None,
) -> BaseModel:
    """Bind inputs (upstream results) and params (literals) by name and call the operator."""
    node = graph.nodes[node_id]
    kwargs: dict[str, object] = {port: results[dep] for port, dep in node.inputs.items()}
    kwargs.update(node.params)
    # Fill any optional params the author omitted from the operator's signature defaults.
    for name, value in spec.defaults.items():
        kwargs.setdefault(name, value)
    if spec.needs_llm:  # executor-injected caller (reserved `llm` param, not a graph param)
        if llm is None:
            raise GraphValidationError(
                f"operator {node.op!r} needs an LLM — run it inside a live session "
                "(no model backend is available here)"
            )
        kwargs["llm"] = llm
    return await spec.func(**kwargs)


async def execute(
    graph: Graph,
    cache: CacheStore | None = None,
    *,
    on_event: Callable[[GraphEvent], None] | None = None,
    verify_purity: bool = False,
    expand_blocks: bool = True,
    llm: LLMCaller | None = None,
) -> tuple[dict[NodeId, Value], Report]:
    """Execute ``graph``, returning value handles per node and an execution :class:`Report`.

    With ``cache=None`` every node runs (the M0 trivial executor). With a cache, unchanged
    nodes are served from it. ``verify_purity`` re-runs each cached node and asserts the
    fresh output matches the cached one — a debug guard against secretly-impure operators.

    ``expand_blocks`` (default True) replaces any block nodes with their subgraphs first, so
    the returned handles/report are keyed by *expanded* node ids. It is identity when the
    graph has no block ops. Callers that need results keyed by authored ids should expand
    themselves (``blocks.expand``) and fold the report with ``blocks.fold_report``.
    """
    if expand_blocks:
        from vibe.core.graph.blocks import expand

        graph, _ = expand(graph)

    coerce_params(graph)  # normalize unambiguous type slips (e.g. scalar→list) before validating
    validate(graph)

    indegree = {nid: len(node.inputs) for nid, node in graph.nodes.items()}
    dependents: dict[NodeId, list[NodeId]] = {nid: [] for nid in graph.nodes}
    for node in graph.nodes.values():
        for dep in node.inputs.values():
            dependents[dep].append(node.id)

    fingerprints: dict[NodeId, str] = {}
    results: dict[NodeId, BaseModel] = {}
    report = Report()
    ready = [nid for nid, deg in indegree.items() if deg == 0]

    while ready:
        wave = ready
        ready = []
        outcomes = await asyncio.gather(
            *(
                _process_node(
                    graph, nid, cache, fingerprints, results, verify_purity, llm
                )
                for nid in wave
            )
        )
        for nid, result, state, duration in outcomes:
            results[nid] = result
            report.states[nid] = state
            report.timings[nid] = duration
            if on_event is not None:
                on_event(
                    GraphEvent(
                        node_id=nid,
                        op=graph.nodes[nid].op,
                        state=state,
                        fingerprint=fingerprints[nid],
                        duration=duration,
                    )
                )
            for child in dependents[nid]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)

    values = {
        nid: Value(type=type(results[nid]).__name__, fingerprint=fingerprints[nid])
        for nid in results
    }
    return values, report


async def _process_node(
    graph: Graph,
    node_id: NodeId,
    cache: CacheStore | None,
    fingerprints: dict[NodeId, str],
    results: dict[NodeId, BaseModel],
    verify_purity: bool,
    llm: LLMCaller | None = None,
) -> tuple[NodeId, BaseModel, NodeState, float]:
    node = graph.nodes[node_id]
    spec = get_operator(node.op)
    fp = fingerprint_node(graph, node_id, fingerprints)
    start = time.perf_counter()

    payload = cache.get(fp) if cache is not None else None
    if payload is not None:
        result: BaseModel = spec.result_type.model_validate_json(payload)
        # LLM ops are non-deterministic → exempt from the re-run purity check.
        if verify_purity and not spec.needs_llm:
            fresh = await _run_operator(spec, graph, node_id, results, llm)
            if fresh.model_dump(mode="json") != result.model_dump(mode="json"):
                raise PurityError(
                    f"operator {node.op!r} at node {node_id!r} is not pure: "
                    "cached and re-run outputs differ"
                )
        return node_id, result, "cached", time.perf_counter() - start

    result = await _run_operator(spec, graph, node_id, results, llm)
    if cache is not None:
        cache.put(fp, result.model_dump_json().encode(), op_type=node.op)
    return node_id, result, "fresh", time.perf_counter() - start
