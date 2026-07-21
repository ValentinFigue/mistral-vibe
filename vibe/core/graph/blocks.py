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

from collections.abc import Iterable
import os
from pathlib import Path
import re
import tempfile

from pydantic import BaseModel, Field

from vibe.core.graph.model import Graph, Node, NodeId, NodeState, Report
from vibe.core.graph.operators import is_registered
from vibe.core.logger import logger
from vibe.core.paths import BLOCKS_DIR

# A block name doubles as its on-disk filename, so this regex is also the path-traversal
# guard: no slashes, dots, or separators can appear.
_BLOCK_NAME_RE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")

# A baked string param longer than this (or path-like) is flagged as a possible leak.
_MAX_SAFE_LITERAL_LEN = 200


class BlockError(Exception):
    """A block is malformed or a graph uses a block incorrectly."""


class BlockDef(BaseModel):
    """A named, reusable subgraph with a declared input/param/output interface."""

    name: str
    graph: Graph
    input_ports: dict[str, tuple[NodeId, str]] = Field(default_factory=dict)
    params: dict[str, tuple[NodeId, str]] = Field(default_factory=dict)
    output: NodeId
    library: str | None = None  # catalog-scoping tag; None = untagged (generic agent)
    description: str = ""  # one-line "when to use", shown in the catalog


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


# ---------------------------------------------------------------------------
# Agent-authored blocks: derive a block from a subgraph, persist, and reload.
# ---------------------------------------------------------------------------


def block_from_subgraph(
    name: str,
    graph: Graph,
    node_ids: Iterable[NodeId],
    *,
    expose_params: Iterable[str] = (),
    output: NodeId | None = None,
) -> BlockDef:
    """Derive a :class:`BlockDef` from a selection of nodes in ``graph``.

    The interface is auto-derived: inputs crossing the selection boundary become ports
    (``f"{node_id}_{port}"``), the single terminal node becomes the output, and every param
    stays baked unless named in ``expose_params`` (as ``"node_id.key"``). Raises
    :class:`BlockError` on any ambiguity or illegal selection.
    """
    if not _BLOCK_NAME_RE.match(name):
        raise BlockError(f"invalid block name {name!r}; use snake_case matching [a-z0-9_]")
    if is_registered(name):
        raise BlockError(f"{name!r} is already an operator; choose another block name")

    ids = list(node_ids)
    selection = set(ids)
    if not selection:
        raise BlockError("no nodes selected")
    unknown = [nid for nid in ids if nid not in graph.nodes]
    if unknown:
        raise BlockError(f"unknown nodes in selection: {sorted(unknown)}")
    for nid in ids:
        if is_block(graph.nodes[nid].op):
            raise BlockError(
                f"node {nid!r} uses block {graph.nodes[nid].op!r}; nested blocks are unsupported"
            )

    sub, input_ports = _extract_subgraph(graph, ids, selection)
    output = _resolve_output(graph, ids, selection, output)
    params = _derive_params(graph, selection, expose_params)
    return BlockDef(name=name, graph=sub, input_ports=input_ports, params=params, output=output)


def _extract_subgraph(
    graph: Graph, ids: list[NodeId], selection: set[NodeId]
) -> tuple[Graph, dict[str, tuple[NodeId, str]]]:
    """Copy selected nodes; boundary-crossing inputs become open ports."""
    sub = Graph()
    input_ports: dict[str, tuple[NodeId, str]] = {}
    for nid in ids:
        node = graph.nodes[nid]
        internal: dict[str, NodeId] = {}
        for port, dep in node.inputs.items():
            if dep in selection:
                internal[port] = dep
            else:  # crosses the boundary → block port, left open on the inner node
                input_ports[f"{nid}_{port}"] = (nid, port)
        sub.add(Node(id=nid, op=node.op, params=dict(node.params), inputs=internal))
    return sub, input_ports


def _resolve_output(
    graph: Graph, ids: list[NodeId], selection: set[NodeId], output: NodeId | None
) -> NodeId:
    if output is not None:
        if output not in selection:
            raise BlockError(f"output {output!r} is not in the selection")
        return output
    consumed = {dep for nid in ids for dep in graph.nodes[nid].inputs.values() if dep in selection}
    terminals = [nid for nid in ids if nid not in consumed]
    if len(terminals) != 1:
        raise BlockError(
            f"cannot infer a single output; terminal nodes are {sorted(terminals)} — pass output="
        )
    return terminals[0]


def _derive_params(
    graph: Graph, selection: set[NodeId], expose_params: Iterable[str]
) -> dict[str, tuple[NodeId, str]]:
    params: dict[str, tuple[NodeId, str]] = {}
    for spec in expose_params:
        if "." not in spec:
            raise BlockError(f"expose_params entry {spec!r} must be 'node_id.key'")
        nid, key = spec.split(".", 1)
        if nid not in selection:
            raise BlockError(f"expose_params: node {nid!r} is not in the selection")
        if key not in graph.nodes[nid].params:
            raise BlockError(f"expose_params: node {nid!r} has no param {key!r}")
        params[f"{nid}_{key}"] = (nid, key)
    return params


def _suspicious_baked_params(block: BlockDef) -> list[str]:
    """Baked string params that look like filesystem paths or long literals (leak risk)."""
    flagged: list[str] = []
    for nid, node in block.graph.nodes.items():
        for key, value in node.params.items():
            if isinstance(value, str) and (
                "/" in value or value.startswith("~") or len(value) > _MAX_SAFE_LITERAL_LEN
            ):
                flagged.append(f"{nid}.{key}")
    return flagged


def save_block(block: BlockDef, blocks_dir: str | Path | None = None) -> Path:
    """Write a block as ``<blocks_dir>/<name>.json`` (atomic). Returns the path.

    Warns (via the logger) when the block bakes a path-like or very long literal, since it
    persists verbatim to a file that may be shared.
    """
    if not _BLOCK_NAME_RE.match(block.name):
        raise BlockError(f"refusing to save block with unsafe name {block.name!r}")
    directory = Path(blocks_dir) if blocks_dir is not None else BLOCKS_DIR.path
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{block.name}.json"

    for leaked in _suspicious_baked_params(block):
        logger.warning("block %r bakes a path-like/long literal at %s (persists to disk)", block.name, leaked)

    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as tmp:
        tmp.write(block.model_dump_json(indent=2))
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)
    return path


def load_blocks(blocks_dir: str | Path | None = None) -> int:
    """Register every block JSON in ``blocks_dir``. Total: skips + logs bad files, never raises.

    A block is skipped (not registered) if it is malformed, structurally invalid, or
    references an operator/block not available in this environment (portability guard).
    Returns the number of blocks registered.
    """
    directory = Path(blocks_dir) if blocks_dir is not None else BLOCKS_DIR.path
    if not directory.exists():
        return 0

    count = 0
    for path in sorted(directory.glob("*.json")):
        try:
            block = BlockDef.model_validate_json(path.read_text())
        except Exception as exc:  # malformed JSON / schema mismatch
            logger.warning("skipping malformed block file %s: %s", path, exc)
            continue
        if is_block(block.name):
            # Never let a disk file override an already-registered block (a built-in, or one
            # loaded earlier). Built-ins register before load_blocks runs, so they win.
            logger.warning("skipping block %r from %s: name already registered", block.name, path)
            continue
        missing = sorted(
            {n.op for n in block.graph.nodes.values() if not is_registered(n.op) and not is_block(n.op)}
        )
        if missing:
            logger.warning("skipping block %r: unavailable operators %s", block.name, missing)
            continue
        try:
            register_block(block)
            count += 1
        except BlockError as exc:
            logger.warning("skipping invalid block %s: %s", path, exc)
    return count


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
