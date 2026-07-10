from __future__ import annotations

from pydantic import BaseModel
import pytest

from vibe.core.graph.cache import CacheStore
from vibe.core.graph.executor import (
    GraphValidationError,
    PurityError,
    execute,
    validate,
)
from vibe.core.graph.model import Graph, Node
from vibe.core.graph.operators import operator


def _sum_graph(x: int, y: int) -> Graph:
    g = Graph()
    g.add(Node(id="a", op="tst_const", params={"value": x}))
    g.add(Node(id="b", op="tst_const", params={"value": y}))
    g.add(Node(id="sum", op="tst_add", inputs={"a": "a", "b": "b"}))
    return g


@pytest.mark.asyncio
async def test_basic_execution_is_correct() -> None:
    values, report = await execute(_sum_graph(2, 5))
    assert report.states == {"a": "fresh", "b": "fresh", "sum": "fresh"}
    assert values["sum"].type == "IntVal"


@pytest.mark.asyncio
async def test_no_cache_always_fresh() -> None:
    g = _sum_graph(1, 2)
    _, first = await execute(g)  # no cache
    _, second = await execute(g)  # still no cache
    assert first.fresh() == list(g.nodes)
    assert second.fresh() == list(g.nodes)


@pytest.mark.asyncio
async def test_intra_run_dedup(cache: CacheStore) -> None:
    # Two nodes with identical recipes (const(1)) share a fingerprint, so the second is a
    # cache hit *within the same run* — one compute, shared result. This pins the behavior
    # that surfaced as the earlier (1, 1) test surprise.
    g = _sum_graph(1, 1)
    _, report = await execute(g, cache)
    assert report.states["a"] == "fresh"
    assert report.states["b"] == "cached"


@pytest.mark.asyncio
async def test_cache_hit_second_run(cache: CacheStore) -> None:
    # Distinct const values so every node has a distinct fingerprint (equal values would
    # make "a" and "b" identical recipes -> "b" a legitimate intra-run dedup hit).
    g = _sum_graph(1, 2)
    _, first = await execute(g, cache)
    _, second = await execute(g, cache)
    assert set(first.fresh()) == set(g.nodes)
    assert set(second.cached()) == set(g.nodes)


def test_validate_unknown_operator() -> None:
    g = Graph()
    g.add(Node(id="x", op="does_not_exist"))
    with pytest.raises(KeyError):
        validate(g)


def test_validate_dangling_input() -> None:
    g = Graph()
    g.add(Node(id="a", op="tst_const", params={"value": 1}))
    g.add(Node(id="sum", op="tst_add", inputs={"a": "a", "b": "ghost"}))
    with pytest.raises(GraphValidationError, match="unknown node"):
        validate(g)


def test_validate_arity_mismatch() -> None:
    g = Graph()
    # tst_const needs param "value"; supply nothing.
    g.add(Node(id="a", op="tst_const"))
    with pytest.raises(GraphValidationError, match="argument mismatch"):
        validate(g)


def test_validate_cycle() -> None:
    g = Graph()
    g.add(Node(id="a", op="tst_add", inputs={"a": "b", "b": "b"}))
    g.add(Node(id="b", op="tst_add", inputs={"a": "a", "b": "a"}))
    with pytest.raises(GraphValidationError, match="cycle"):
        validate(g)


# --- purity verification -------------------------------------------------


class Counter(BaseModel):
    n: int


_impure_calls = {"n": 0}


@operator(name="tst_impure")
async def impure() -> Counter:
    _impure_calls["n"] += 1
    return Counter(n=_impure_calls["n"])


@pytest.mark.asyncio
async def test_verify_purity_flags_impure_operator(cache: CacheStore) -> None:
    g = Graph()
    g.add(Node(id="c", op="tst_impure"))
    await execute(g, cache)  # first run caches n=1
    with pytest.raises(PurityError, match="not pure"):
        await execute(g, cache, verify_purity=True)  # re-run yields n=2 != cached n=1


@pytest.mark.asyncio
async def test_verify_purity_passes_for_pure_operator(cache: CacheStore) -> None:
    g = _sum_graph(4, 4)
    await execute(g, cache)
    _, report = await execute(g, cache, verify_purity=True)
    assert set(report.cached()) == set(g.nodes)
