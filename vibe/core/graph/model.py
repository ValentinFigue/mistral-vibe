"""Typed data model for the Métier workflow graph.

A workflow is a DAG of :class:`Node`s. Each node names an operator and supplies
its arguments in two disjoint buckets:

* ``inputs`` — ``port -> node_id`` wiring; the value flowing along the edge is the
  cached/fresh result of the upstream node.
* ``params`` — literal scalar arguments hashed into the node's fingerprint.

Wiring is explicit here, never inferred from an operator's Python types: whether an
argument is an input or a param is decided by the ``Node``, not by its annotation.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

NodeId = str


class Node(BaseModel):
    """A single operation in the graph."""

    id: NodeId
    op: str
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, NodeId] = Field(default_factory=dict)  # port -> upstream node id


class Graph(BaseModel):
    """A DAG of nodes; edges are implied by ``Node.inputs``."""

    nodes: dict[NodeId, Node] = Field(default_factory=dict)

    def add(self, node: Node) -> NodeId:
        """Add a node and return its id (raises on duplicate id)."""
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id {node.id!r}")
        self.nodes[node.id] = node
        return node.id


class Value(BaseModel):
    """Metadata handle for a produced value. The payload lives in the cache, never here."""

    type: str
    fingerprint: str


NodeState = Literal["cached", "fresh"]


class Report(BaseModel):
    """Per-node execution outcome for one ``execute`` call."""

    states: dict[NodeId, NodeState] = Field(default_factory=dict)
    timings: dict[NodeId, float] = Field(default_factory=dict)  # seconds

    def fresh(self) -> list[NodeId]:
        return [nid for nid, s in self.states.items() if s == "fresh"]

    def cached(self) -> list[NodeId]:
        return [nid for nid, s in self.states.items() if s == "cached"]


# ---------------------------------------------------------------------------
# Patch — typed graph edit. Defined so the M2/M3 agent-authoring layer has a
# target; NOT consumed anywhere in M1 (no diff engine yet).
# ---------------------------------------------------------------------------


class AddNode(BaseModel):
    kind: Literal["add_node"] = "add_node"
    node: Node


class RemoveNode(BaseModel):
    kind: Literal["remove_node"] = "remove_node"
    id: NodeId


class SetParam(BaseModel):
    kind: Literal["set_param"] = "set_param"
    id: NodeId
    key: str
    value: Any


class Connect(BaseModel):
    kind: Literal["connect"] = "connect"
    id: NodeId
    port: str
    source: NodeId


class Disconnect(BaseModel):
    kind: Literal["disconnect"] = "disconnect"
    id: NodeId
    port: str


PatchOp = Annotated[
    AddNode | RemoveNode | SetParam | Connect | Disconnect,
    Field(discriminator="kind"),
]
Patch = list[PatchOp]
