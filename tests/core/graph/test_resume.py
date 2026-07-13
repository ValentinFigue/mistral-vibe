from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
import pytest

from vibe.core.graph.cache import CacheStore
from vibe.core.graph.executor import execute
from vibe.core.graph.model import Graph, Node
from vibe.core.graph.operators import operator


class Num(BaseModel):
    n: int


# Toggled by the test to simulate a mid-DAG failure that is later "fixed".
_armed = {"boom": True}


@operator(name="tst_src")
async def src(value: int) -> Num:
    return Num(n=value)


@operator(name="tst_maybe_boom")
async def maybe_boom(x: Num) -> Num:
    if _armed["boom"]:
        raise RuntimeError("boom")
    return Num(n=x.n * 10)


def _graph() -> Graph:
    g = Graph()
    g.add(Node(id="a", op="tst_src", params={"value": 7}))
    g.add(Node(id="b", op="tst_maybe_boom", inputs={"x": "a"}))
    return g


@pytest.mark.asyncio
async def test_resume_from_last_good_node(tmp_path: Path) -> None:
    cache = CacheStore(tmp_path / "cache.sqlite")
    graph = _graph()

    # Run 1: upstream node "a" completes and is cached; "b" raises mid-DAG.
    _armed["boom"] = True
    with pytest.raises(RuntimeError, match="boom"):
        await execute(graph, cache)

    # Run 2: "b" is fixed. "a" resumes from cache; nothing upstream is re-paid.
    _armed["boom"] = False
    values, report = await execute(graph, cache)

    assert report.states == {"a": "cached", "b": "fresh"}
    assert values["b"].fingerprint  # produced a value
    cache.close()
