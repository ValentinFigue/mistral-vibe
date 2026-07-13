from __future__ import annotations

import pytest

from vibe.core.graph.cache import CacheStore
from vibe.core.graph.executor import execute
from vibe.core.graph.model import (
    AddNode,
    Connect,
    Disconnect,
    Graph,
    Node,
    RemoveNode,
    SetParam,
)
from vibe.core.graph.patch import PatchError, apply_patch, changed_nodes


def _sum_graph(x: int, y: int) -> Graph:
    g = Graph()
    g.add(Node(id="a", op="tst_const", params={"value": x}))
    g.add(Node(id="b", op="tst_const", params={"value": y}))
    g.add(Node(id="sum", op="tst_add", inputs={"a": "a", "b": "b"}))
    return g


def test_add_node_is_pure() -> None:
    g = Graph()
    out = apply_patch(g, [AddNode(node=Node(id="a", op="tst_const", params={"value": 1}))])
    assert "a" in out.nodes
    assert g.nodes == {}  # original untouched


def test_set_param_and_connect_and_disconnect() -> None:
    g = _sum_graph(1, 2)
    out = apply_patch(
        g,
        [
            SetParam(id="a", key="value", value=9),
            Disconnect(id="sum", port="b"),
            AddNode(node=Node(id="c", op="tst_const", params={"value": 3})),
            Connect(id="sum", port="b", source="c"),
        ],
    )
    assert out.nodes["a"].params["value"] == 9
    assert out.nodes["sum"].inputs == {"a": "a", "b": "c"}
    assert g.nodes["a"].params["value"] == 1  # pure


def test_remove_node() -> None:
    g = _sum_graph(1, 2)
    # Removing a consumed node is illegal...
    with pytest.raises(PatchError, match="still consumed"):
        apply_patch(g, [RemoveNode(id="a")])
    # ...but fine once nothing references it.
    out = apply_patch(g, [Disconnect(id="sum", port="a"), RemoveNode(id="a")])
    assert "a" not in out.nodes


def test_illegal_ops_raise() -> None:
    g = _sum_graph(1, 2)
    with pytest.raises(PatchError, match="duplicate"):
        apply_patch(g, [AddNode(node=Node(id="a", op="tst_const", params={"value": 0}))])
    with pytest.raises(PatchError, match="unknown node"):
        apply_patch(g, [SetParam(id="ghost", key="value", value=1)])
    with pytest.raises(PatchError, match="no input port"):
        apply_patch(g, [Disconnect(id="sum", port="ghost")])


def test_changed_nodes_classifies_by_fingerprint() -> None:
    g = _sum_graph(1, 2)
    out = apply_patch(
        g,
        [
            SetParam(id="a", key="value", value=9),
            AddNode(node=Node(id="c", op="tst_const", params={"value": 3})),
        ],
    )
    delta = changed_nodes(g, out)
    assert delta["added"] == ["c"]
    assert delta["removed"] == []
    assert set(delta["changed"]) == {"a", "sum"}  # "a" edited, "sum" depends on it
    assert "b" not in delta["changed"]


@pytest.mark.asyncio
async def test_incrementality_through_a_patch(cache: CacheStore) -> None:
    g = _sum_graph(1, 2)
    await execute(g, cache)  # warm cache

    patched = apply_patch(g, [SetParam(id="a", key="value", value=9)])
    _, report = await execute(patched, cache)

    assert set(report.fresh()) == {"a", "sum"}  # only the dirty subgraph
    assert report.cached() == ["b"]
