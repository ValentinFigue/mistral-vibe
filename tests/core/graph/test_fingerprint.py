from __future__ import annotations

from vibe.core.graph.fingerprint import fingerprint_node
from vibe.core.graph.model import Graph, Node


def _chain(sales_value: int, costs_value: int) -> Graph:
    """const(sales) + const(costs) -> add."""
    g = Graph()
    g.add(Node(id="a", op="tst_const", params={"value": sales_value}))
    g.add(Node(id="b", op="tst_const", params={"value": costs_value}))
    g.add(Node(id="sum", op="tst_add", inputs={"a": "a", "b": "b"}))
    return g


def test_same_recipe_same_fingerprint() -> None:
    g1, g2 = _chain(1, 2), _chain(1, 2)
    for nid in ("a", "b", "sum"):
        assert fingerprint_node(g1, nid) == fingerprint_node(g2, nid)


def test_param_change_reprints_node_and_dependents_only() -> None:
    base = _chain(1, 2)
    changed = _chain(9, 2)  # only node "a" differs

    assert fingerprint_node(changed, "b") == fingerprint_node(base, "b")
    assert fingerprint_node(changed, "a") != fingerprint_node(base, "a")
    assert fingerprint_node(changed, "sum") != fingerprint_node(base, "sum")


def test_upstream_change_propagates_downstream() -> None:
    base = _chain(1, 2)
    changed = _chain(1, 5)  # node "b" differs -> "sum" must change, "a" must not
    assert fingerprint_node(changed, "a") == fingerprint_node(base, "a")
    assert fingerprint_node(changed, "sum") != fingerprint_node(base, "sum")


def test_param_order_does_not_matter() -> None:
    g1 = Graph()
    g1.add(Node(id="n", op="tst_add", params={"a": 1, "b": 2}))
    g2 = Graph()
    g2.add(Node(id="n", op="tst_add", params={"b": 2, "a": 1}))
    assert fingerprint_node(g1, "n") == fingerprint_node(g2, "n")
