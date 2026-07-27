"""UI-free renderers for inspecting a workflow graph and the block library.

Everything here returns plain strings / dicts with no terminal or Textual dependency, so the
CLI commands, a future ACP command, and tests can all share one rendering path. A Mermaid
diagram is emitted as text (the caller wraps it in a ```mermaid fence).
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from typing import Any, Protocol

from rich.text import Text

from vibe.core.graph.blocks import BlockDef, expand
from vibe.core.graph.fingerprint import fingerprint_node
from vibe.core.graph.model import Graph, NodeId, NodeState
from vibe.core.graph.operators import OperatorSpec

_MAX_PREVIEW = 200
_DTYPE_SAMPLE = 500  # rows scanned to infer a column's dtype
_MAX_SCHEMA_COLS = 30  # wide tables: list this many columns, then "+K more"


def infer_dtype(values: list[Any]) -> str:
    """A coarse dtype for a column: int / float / bool / str / empty (nullable noted elsewhere)."""
    seen = [v for v in values if v is not None]
    if not seen:
        return "empty"
    if all(isinstance(v, bool) for v in seen):
        return "bool"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in seen):
        return "int"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in seen):
        return "float"
    return "str"


def table_schema(payload: str) -> str | None:
    """A compact one-line schema for a Table payload — ``"col:dtype, … (N rows)"``.

    Returns ``None`` when the payload is not a ``{columns, rows}`` table (e.g. a Report or a
    chart/export handle). Wide tables are capped to the first ``_MAX_SCHEMA_COLS`` columns + a
    ``+K more`` marker; dtypes are inferred from a bounded row sample.
    """
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None
    if not (isinstance(data, dict) and "columns" in data and "rows" in data):
        return None
    columns: list[str] = list(data.get("columns") or [])
    rows: list[dict[str, Any]] = list(data.get("rows") or [])
    shown = columns[:_MAX_SCHEMA_COLS]
    parts = [f"{c}:{infer_dtype([r.get(c) for r in rows[:_DTYPE_SAMPLE]])}" for c in shown]
    if len(columns) > _MAX_SCHEMA_COLS:
        parts.append(f"+{len(columns) - _MAX_SCHEMA_COLS} more")
    return f"{', '.join(parts)} ({len(rows)} rows)"


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
    """One adjacency line per node in topological order, e.g. ``report · format_report``.

    Not used by the interactive CLI panel (which renders via :func:`node_line`); retained for
    the deferred ACP/markdown rendering surface and covered by tests.
    """
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


def materialize_node_value(graph: Graph, node_id: NodeId, cache: _CacheLike) -> str | None:
    """The decoded cached payload (canonical JSON text) for an authored node's output.

    The single fingerprint→cache-read path, shared by ``graph_inspect`` and the ``/graph``
    panel. Block-aware via :func:`output_ids` (a block node's value is produced by its declared
    output child). Returns ``None`` if the node is unknown or its value isn't cached yet.
    """
    _expanded, producing = output_ids(graph)
    eid = producing.get(node_id)
    if eid is None:
        return None
    payload = cache.get(fingerprint_node(_expanded, eid, {}))
    return payload.decode() if payload is not None else None


def _truncate(text: str) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= _MAX_PREVIEW else text[:_MAX_PREVIEW] + "…"


# --- Rich renderers for the interactive panel (shared with the widget + its tests) ---
# All build rich Text via append (never markup), so arbitrary node text can't inject markup.

_DETAIL_MAX = 400


def _state_glyph(state: str | None, cached: bool | None) -> tuple[str, str]:
    if state == "fresh":
        return "●", "green"
    if state == "cached":
        return "○", "cyan"
    if cached is True:
        return "◍", "dim"
    if cached is False:
        return "·", "dim"
    return "", ""


def node_line(
    node_id: NodeId,
    op: str,
    inputs: Mapping[str, NodeId],
    state: str | None = None,
    cached: bool | None = None,
) -> Text:
    """One-line node label for the panel's option list (glyph + id·op + upstream hint)."""
    line = Text(no_wrap=True)
    glyph, style = _state_glyph(state, cached)
    if glyph:
        line.append(f"{glyph} ", style=style)
    line.append(f"{node_id} · {op}")
    if inputs:
        line.append("  ← " + ", ".join(inputs.values()), style="dim")
    return line


def node_detail(
    node_id: NodeId,
    op: str,
    inputs: Mapping[str, NodeId],
    params: Mapping[str, object],
    state: str | None = None,
    cached: bool | None = None,
    output: str | None = None,
) -> Text:
    """Multi-line detail view for one node (right pane). Markup-safe (Text.append)."""
    detail = Text()
    detail.append(f"{node_id}\n", style="bold")
    detail.append(f"op: {op}\n")
    if state is not None:
        detail.append(f"last run: {state}\n")
    if cached is not None:
        detail.append(f"cached: {'yes' if cached else 'no'}\n")
    detail.append("\ninputs:\n", style="bold")
    detail.append("".join(f"  {port} ← {dep}\n" for port, dep in inputs.items()) or "  (none)\n")
    detail.append("\nparams:\n", style="bold")
    detail.append("".join(f"  {k} = {v}\n" for k, v in params.items()) or "  (none)\n")
    if output:
        clipped = output if len(output) <= _DETAIL_MAX else output[:_DETAIL_MAX] + "…"
        detail.append("\noutput:\n", style="bold")
        detail.append(clipped)
    return detail


def nodes_table(
    graph: Graph,
    *,
    states: Mapping[NodeId, NodeState] | None = None,
    status: Mapping[NodeId, bool] | None = None,
    outputs: Mapping[NodeId, str] | None = None,
) -> str:
    """Markdown table: node · op · inputs · last run · cached · output preview.

    Not used by the interactive CLI panel; retained for the deferred ACP/markdown surface
    and covered by tests.
    """
    rows = ["| node | op | inputs | last run | cached | output |", "|---|---|---|---|---|---|"]
    for nid in topo_order(graph):
        node = graph.nodes[nid]
        wiring = ", ".join(f"{port}←{dep}" for port, dep in node.inputs.items()) or "—"
        last = states.get(nid, "—") if states else "—"
        cached = ("✓" if status.get(nid) else "·") if status else "—"
        out = _truncate(outputs[nid]) if outputs and nid in outputs else ""
        rows.append(f"| `{nid}` | {node.op} | {wiring} | {last} | {cached} | {out} |")
    return "\n".join(rows)


# Order in which operator categories are shown; unknown categories (and untagged ops, bucketed as
# "other") are appended after, alphabetically. Grouping is opt-in: a catalog whose ops carry no
# category renders as a flat sorted list (unchanged behaviour for untagged libraries).
_CATEGORY_ORDER = (
    "load", "shape", "transform", "statistics", "inference",
    "ml", "quality", "sql", "report", "insight",
)


def _catalog_op_line(name: str, spec: OperatorSpec, verbose: bool) -> str:
    """One operator's catalog line: ``name(inputs: …; params: …) → ResultType`` (+ description)."""
    def one(a: str) -> str:  # optional params shown as name:type=default so the agent may omit them
        base = f"{a}:{spec.arg_types.get(a, '?')}"
        return f"{base}={spec.defaults[a]!r}" if a in spec.defaults else base

    def typed(argnames: tuple[str, ...]) -> str:
        return ", ".join(one(a) for a in argnames) or "—"

    line = (
        f"- {name}(inputs: {typed(spec.input_names)}; "
        f"params: {typed(spec.literal_names())}) → {spec.result_type.__name__}"
    )
    if verbose and spec.description:
        line += f"\n    {spec.description}"
    return line


def operators_catalog(
    operators: Mapping[str, OperatorSpec],
    blocks: Mapping[str, BlockDef],
    *,
    verbose: bool = False,
) -> str:
    """The operator + block catalog, as markdown.

    Each param renders as ``name:type[=default]``; for enum params the type is the allowed values
    (e.g. ``model:logreg|tree|rf``) since they're annotated ``Literal[...]``. ``verbose=True`` (used
    by the agent catalog and ``/operators``) appends each op's one-line description; ``verbose=False``
    omits it. When operators carry a ``category`` they are grouped under ``[category]`` headers (in
    ``_CATEGORY_ORDER``); an all-untagged catalog renders as one flat sorted list.
    """
    from collections import defaultdict

    lines = ["Available operators (inputs are wired from nodes; params are literals):"]
    if any(spec.category for spec in operators.values()):
        groups: dict[str, list[tuple[str, OperatorSpec]]] = defaultdict(list)
        for name, spec in operators.items():
            groups[spec.category or "other"].append((name, spec))
        ordered = [c for c in _CATEGORY_ORDER if c in groups]
        ordered += sorted(c for c in groups if c not in _CATEGORY_ORDER)  # unknowns + "other" last
        for cat in ordered:
            lines.append("")
            lines.append(f"[{cat}]")
            lines += [_catalog_op_line(n, s, verbose) for n, s in sorted(groups[cat])]
    else:
        lines += [_catalog_op_line(n, s, verbose) for n, s in sorted(operators.items())]
    lines.append("")
    lines.append("Available blocks (reusable subgraphs — inputs / params):")
    for name, block in sorted(blocks.items()):
        ports = ", ".join(block.input_ports) or "—"
        params = ", ".join(block.params) or "—"
        line = f"- {name}(inputs: {ports}; params: {params}) → {block.output}"
        if block.description:
            line += f"  — {block.description}"
        lines.append(line)
    return "\n".join(lines)


def blocks_table(blocks: Mapping[str, BlockDef], sources: Mapping[str, str]) -> str:
    """Markdown table of the block library: block · source · inputs · params · output."""
    rows = ["| block | source | inputs | params | output |", "|---|---|---|---|---|"]
    for name in sorted(blocks):
        block = blocks[name]
        ports = ", ".join(block.input_ports) or "—"
        params = ", ".join(block.params) or "—"
        rows.append(f"| `{name}` | {sources.get(name, '?')} | {ports} | {params} | `{block.output}` |")
    return "\n".join(rows)
