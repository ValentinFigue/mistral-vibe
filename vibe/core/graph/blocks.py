"""Reusable subgraphs — *blocks* — the mechanism behind graph reuse.

A block is a named subgraph exposed as an operator: once registered, ``Node(op="<block>")``
is a legal node whose implementation is itself a graph. Before execution the graph is
*expanded* — every block node is replaced by a namespaced copy of its subgraph — so
fingerprinting, incremental recompute, and cross-run dedup all work *through* the block,
exactly as if the nodes had been authored inline.

A block declares its interface explicitly:

* ``input_ports`` — block port name → ``(inner node id, that node's input port)``; the
  block node's ``inputs`` feed these.
* ``params`` — block param name → ``(inner node id, that node's param key)``; the block
  node's ``params`` set these.
* ``output`` — the inner node whose value is the block's result.

Blocks are single-level in this milestone: a block's subgraph may not itself contain block
ops (nested blocks are deferred).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from vibe.core.graph.model import Graph, Node, NodeId, NodeState, Report


class BlockError(Exception):
    """A block is malformed or a graph uses a block incorrectly."""


class BlockDef(BaseModel):
    """A named, reusable subgraph with a declared input/param/output interface."""

    name: str
    graph: Graph
    input_ports: dict[str, tuple[NodeId, str]] = Field(default_factory=dict)
    params: dict[str, tuple[NodeId, str]] = Field(default_factory=dict)
    output: NodeId


_BLOCKS: dict[str, BlockDef] = {}


def register_block(block: BlockDef) -> None:
    """Register a block, validating its interface references its own inner nodes."""
    inner = block.graph.nodes
    if block.output not in inner:
        raise BlockError(f"block {block.name!r}: output {block.output!r} is not an inner node")
    for port, (nid, _) in block.input_ports.items():
        if nid not in inner:
            raise BlockError(f"block {block.name!r}: port {port!r} targets unknown node {nid!r}")
    for param, (nid, _) in block.params.items():
        if nid not in inner:
            raise BlockError(f"block {block.name!r}: param {param!r} targets unknown node {nid!r}")
    for node in inner.values():
        if node.op in _BLOCKS:
            raise BlockError(
                f"block {block.name!r}: nested blocks are unsupported (inner op {node.op!r})"
            )
    _BLOCKS[block.name] = block


def get_block(name: str) -> BlockDef:
    try:
        return _BLOCKS[name]
    except KeyError:
        raise KeyError(f"unknown block {name!r}") from None


def is_block(name: str) -> bool:
    return name in _BLOCKS


def registered_blocks() -> dict[str, BlockDef]:
    """A snapshot of all registered blocks (for catalogs shown to the agent)."""
    return dict(_BLOCKS)


def expand(graph: Graph) -> tuple[Graph, dict[NodeId, NodeId]]:
    """Expand every block node into its subgraph.

    Returns the primitive-only graph and a ``expanded_id -> authored_id`` map (a block's
    inner nodes all map back to the authored block node; primitive nodes map to themselves).
    Identity when the graph contains no block ops.
    """
    # value-producing expanded id for each authored node
    produces: dict[NodeId, NodeId] = {
        nid: (f"{nid}/{get_block(node.op).output}" if is_block(node.op) else nid)
        for nid, node in graph.nodes.items()
    }

    expanded = Graph()
    fold: dict[NodeId, NodeId] = {}

    for nid, node in graph.nodes.items():
        if not is_block(node.op):
            expanded.nodes[nid] = Node(
                id=nid,
                op=node.op,
                params=dict(node.params),
                inputs={port: produces[dep] for port, dep in node.inputs.items()},
            )
            fold[nid] = nid
            continue

        block = get_block(node.op)
        if set(node.inputs) != set(block.input_ports):
            raise BlockError(
                f"block node {nid!r} ({block.name!r}) inputs {sorted(node.inputs)} "
                f"!= block ports {sorted(block.input_ports)}"
            )
        if set(node.params) != set(block.params):
            raise BlockError(
                f"block node {nid!r} ({block.name!r}) params {sorted(node.params)} "
                f"!= block params {sorted(block.params)}"
            )

        prefix = f"{nid}/"
        for inner in block.graph.nodes.values():
            eid = prefix + inner.id
            expanded.nodes[eid] = Node(
                id=eid,
                op=inner.op,
                params=dict(inner.params),
                inputs={port: prefix + dep for port, dep in inner.inputs.items()},
            )
            fold[eid] = nid

        # wire the block node's external inputs into the declared inner ports
        for block_port, external_src in node.inputs.items():
            inner_id, inner_port = block.input_ports[block_port]
            expanded.nodes[prefix + inner_id].inputs[inner_port] = produces[external_src]

        # set the block node's params on the declared inner nodes
        for block_param, value in node.params.items():
            inner_id, inner_key = block.params[block_param]
            expanded.nodes[prefix + inner_id].params[inner_key] = value

    return expanded, fold


def fold_report(report: Report, fold: dict[NodeId, NodeId]) -> Report:
    """Collapse an expanded-id report back to authored ids via a ``fold`` map.

    An authored node is ``cached`` iff every expanded child mapped to it is ``cached``
    (i.e. a block node is fresh if any of its inner nodes recomputed). Timings sum.
    """
    grouped: dict[NodeId, list[str]] = {}
    timings: dict[NodeId, float] = {}
    for eid, state in report.states.items():
        aid = fold.get(eid, eid)
        grouped.setdefault(aid, []).append(state)
        timings[aid] = timings.get(aid, 0.0) + report.timings.get(eid, 0.0)

    states: dict[NodeId, NodeState] = {
        aid: ("cached" if all(s == "cached" for s in sts) else "fresh")
        for aid, sts in grouped.items()
    }
    return Report(states=states, timings=timings)
