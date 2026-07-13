"""Apply typed patches to a graph, and classify what a patch changed.

A :data:`~vibe.core.graph.model.Patch` is the agent's only action: a small, typed,
reviewable edit. ``apply_patch`` is a pure function — it returns a new graph and never
mutates the input — so the caller can validate the result before committing to it. This is
what lets the agent-authoring layer guarantee an invalid workflow is never executed.
"""

from __future__ import annotations

from vibe.core.graph.fingerprint import fingerprint_node
from vibe.core.graph.model import (
    AddNode,
    Connect,
    Disconnect,
    Graph,
    Node,
    Patch,
    RemoveNode,
    SetParam,
)


class PatchError(Exception):
    """A patch op could not be applied (illegal edit)."""


def apply_patch(graph: Graph, patch: Patch) -> Graph:
    """Return a new graph with ``patch`` applied. The input graph is left unchanged.

    Ops are applied in order. Raises :class:`PatchError` on any illegal op (duplicate id,
    unknown node, removing a node another node still consumes, disconnecting an absent port).
    """
    new = graph.model_copy(deep=True)

    for op in patch:
        if isinstance(op, AddNode):
            if op.node.id in new.nodes:
                raise PatchError(f"add_node: duplicate node id {op.node.id!r}")
            new.nodes[op.node.id] = op.node.model_copy(deep=True)

        elif isinstance(op, RemoveNode):
            _require_node(new, op.id, "remove_node")
            consumers = [
                nid
                for nid, node in new.nodes.items()
                if nid != op.id and op.id in node.inputs.values()
            ]
            if consumers:
                raise PatchError(
                    f"remove_node: {op.id!r} is still consumed by {sorted(consumers)}"
                )
            del new.nodes[op.id]

        elif isinstance(op, SetParam):
            _require_node(new, op.id, "set_param").params[op.key] = op.value

        elif isinstance(op, Connect):
            node = _require_node(new, op.id, "connect")
            _require_node(new, op.source, "connect (source)")
            node.inputs[op.port] = op.source

        elif isinstance(op, Disconnect):
            node = _require_node(new, op.id, "disconnect")
            if op.port not in node.inputs:
                raise PatchError(
                    f"disconnect: node {op.id!r} has no input port {op.port!r}"
                )
            del node.inputs[op.port]

        else:  # pragma: no cover - discriminated union is exhaustive
            raise PatchError(f"unknown patch op: {op!r}")

    return new


def changed_nodes(old: Graph, new: Graph) -> dict[str, list[str]]:
    """Classify nodes between two graphs by recipe fingerprint.

    Returns ``{"added": [...], "removed": [...], "changed": [...]}``. ``changed`` is a node
    present in both whose fingerprint differs (a param/input edit, or an upstream change).
    This is the lightweight diff that powers the approval display and the tool result.
    """
    old_fps = {nid: fingerprint_node(old, nid) for nid in old.nodes}
    new_fps = {nid: fingerprint_node(new, nid) for nid in new.nodes}

    added = [nid for nid in new_fps if nid not in old_fps]
    removed = [nid for nid in old_fps if nid not in new_fps]
    changed = [
        nid for nid in new_fps if nid in old_fps and new_fps[nid] != old_fps[nid]
    ]
    return {"added": sorted(added), "removed": sorted(removed), "changed": sorted(changed)}


def _require_node(graph: Graph, node_id: str, op_label: str) -> Node:
    try:
        return graph.nodes[node_id]
    except KeyError:
        raise PatchError(f"{op_label}: unknown node {node_id!r}") from None
