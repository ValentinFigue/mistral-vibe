"""UI-free renderers for inspecting a workflow graph and the block library.

Everything here returns plain strings / dicts with no terminal or Textual dependency, so the
CLI commands, a future ACP command, and tests can all share one rendering path. A Mermaid
diagram is emitted as text (the caller wraps it in a ```mermaid fence).
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Protocol

from vibe.core.graph.blocks import BlockDef, expand
from vibe.core.graph.fingerprint import fingerprint_node
from vibe.core.graph.model import Graph, NodeId, NodeState

_MAX_PREVIEW = 200


class _CacheLike(Protocol):
    def get(self, fingerprint: str) -> bytes | None: ...


def topo_order(graph: Graph) -> list[NodeId]:
    """Nodes in dependency order (Kahn); any cycle remnants appended for safety."""
    indegree = {nid: 0 for nid in graph.nodes}
    dependents: dict[NodeId, list[NodeId]] = {nid: [] for nid in graph.nodes}
    for node in graph.nodes.values():
        for dep in node.inputs.values():
            if dep in graph.nodes:
                indegree[node.id] += 1
                dependents[dep].append(node.id)
    ready = [nid for nid, d in indegree.items() if d == 0]
    order: list[NodeId] = []
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for child in dependents[nid]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    order.extend(nid for nid in graph.nodes if nid not in order)
    return order


def _safe(nid: NodeId) -> str:
    """A Mermaid-safe node id (labels keep the real id)."""
    return "n_" + re.sub(r"\W", "_", nid)


def to_mermaid(graph: Graph, states: Mapping[NodeId, NodeState] | None = None) -> str:
    """A `flowchart LR` diagram body (no ```mermaid fence). Nodes styled by last-run state."""
    lines = ["flowchart LR"]
    for nid, node in graph.nodes.items():
        lines.append(f'    {_safe(nid)}["{nid} · {node.op}"]')
    for nid, node in graph.nodes.items():
        for port, dep in node.inputs.items():
            lines.append(f"    {_safe(dep)} -->|{port}| {_safe(nid)}")
    if states:
        lines.append("    classDef fresh fill:#2d6,stroke:#1a4,color:#000")
        lines.append("    classDef cached fill:#ccd,stroke:#889,color:#000")
        for nid in graph.nodes:
            state = states.get(nid)
            if state in {"fresh", "cached"}:
                lines.append(f"    class {_safe(nid)} {state}")
    return "\n".join(lines)


def to_tree(graph: Graph, states: Mapping[NodeId, NodeState] | None = None) -> list[str]:
    """One adjacency line per node in topological order, e.g. ``report · format_report``."""
    lines: list[str] = []
    for nid in topo_order(graph):
        node = graph.nodes[nid]
        state = f"  [{states[nid]}]" if states and nid in states else ""
        wiring = ", ".join(f"{port}←{dep}" for port, dep in node.inputs.items())
        suffix = f"  ({wiring})" if wiring else ""
        lines.append(f"{nid} · {node.op}{suffix}{state}")
    return lines


def graph_status(graph: Graph, cache: _CacheLike) -> dict[NodeId, bool]:
    """Which authored nodes' results are already materialized in the cache.

    Block-aware: the cache is keyed by *expanded* recipe fingerprints, so the graph is
    expanded first and each authored node is materialized iff all of its expanded children
    are. Reuses ``expand`` (the same transform the executor runs).
    """
    expanded, fold = expand(graph)
    memo: dict[NodeId, str] = {}
    materialized: dict[NodeId, bool] = {nid: True for nid in graph.nodes}
    for eid in expanded.nodes:
        present = cache.get(fingerprint_node(expanded, eid, memo)) is not None
        authored = fold.get(eid, eid)
        materialized[authored] = materialized.get(authored, True) and present
    return materialized


def output_ids(graph: Graph) -> tuple[Graph, dict[NodeId, NodeId]]:
    """``(expanded_graph, {authored_id: expanded_id_producing_its_value})``.

    For a primitive node that's the node itself; for a block node it's the block's declared
    output child. Lets a caller fetch each authored node's output payload from the cache.
    """
    from vibe.core.graph.blocks import get_block, is_block

    expanded, _ = expand(graph)
    producing = {
        nid: (f"{nid}/{get_block(node.op).output}" if is_block(node.op) else nid)
        for nid, node in graph.nodes.items()
    }
    return expanded, producing


def _truncate(text: str) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= _MAX_PREVIEW else text[:_MAX_PREVIEW] + "…"


def nodes_table(
    graph: Graph,
    *,
    states: Mapping[NodeId, NodeState] | None = None,
    status: Mapping[NodeId, bool] | None = None,
    outputs: Mapping[NodeId, str] | None = None,
) -> str:
    """Markdown table: node · op · inputs · last run · cached · output preview."""
    rows = ["| node | op | inputs | last run | cached | output |", "|---|---|---|---|---|---|"]
    for nid in topo_order(graph):
        node = graph.nodes[nid]
        wiring = ", ".join(f"{port}←{dep}" for port, dep in node.inputs.items()) or "—"
        last = states.get(nid, "—") if states else "—"
        cached = ("✓" if status.get(nid) else "·") if status else "—"
        out = _truncate(outputs[nid]) if outputs and nid in outputs else ""
        rows.append(f"| `{nid}` | {node.op} | {wiring} | {last} | {cached} | {out} |")
    return "\n".join(rows)


def blocks_table(blocks: Mapping[str, BlockDef], sources: Mapping[str, str]) -> str:
    """Markdown table of the block library: block · source · inputs · params · output."""
    rows = ["| block | source | inputs | params | output |", "|---|---|---|---|---|"]
    for name in sorted(blocks):
        block = blocks[name]
        ports = ", ".join(block.input_ports) or "—"
        params = ", ".join(block.params) or "—"
        rows.append(f"| `{name}` | {sources.get(name, '?')} | {ports} | {params} | `{block.output}` |")
    return "\n".join(rows)
