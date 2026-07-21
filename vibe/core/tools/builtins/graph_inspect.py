"""``graph_inspect`` — a read-only peek at a node's value in the working graph.

The analyst authors over real data whose columns it doesn't know a priori. `graph_patch`
returns only compact handles for *terminal* nodes; to author the next step the agent often
needs to *see* an intermediate — its columns, their inferred types, and a few sample rows.
`graph_inspect` reads the node's materialized value straight from the session cache (no
re-execution, no approval gate) and renders a compact schema + preview.

It never mutates anything: the value must already have been produced by a prior `graph_patch`
run. Built on the shared :func:`vibe.core.graph.render.materialize_node_value` cache-read path.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
import json
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from vibe.core.graph import render
from vibe.core.graph.cache import CacheStore
from vibe.core.graph.model import Graph
from vibe.core.graph.session_store import graph_dir as _session_graph_dir
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolResultEvent

_MAX_PREVIEW_ROWS = 50
_MAX_TEXT_CHARS = 2000
_DTYPE_SAMPLE = 500  # rows scanned to infer a column's dtype


class GraphInspectArgs(BaseModel):
    node_id: str = Field(description="The id of a node in the current graph to inspect.")
    rows: int = Field(
        default=10, ge=1, le=_MAX_PREVIEW_ROWS, description="How many sample rows to show (tables)."
    )


class GraphInspectResult(BaseModel):
    node_id: str
    kind: str  # "table" | "report" | "value" | "missing"
    columns: list[str] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)
    row_count: int | None = None
    preview: str = ""


class GraphInspectConfig(BaseToolConfig):
    # Read-only (reads the session cache, re-executes nothing) → auto-run, no approval gate.
    permission: ToolPermission = ToolPermission.ALWAYS


class GraphInspectState(BaseToolState):
    pass


def _infer_dtype(values: list[Any]) -> str:
    """A coarse dtype for a column: int / float / bool / str (nullable noted separately)."""
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


class GraphInspect(
    BaseTool[GraphInspectArgs, GraphInspectResult, GraphInspectConfig, GraphInspectState],
    ToolUIData[GraphInspectArgs, GraphInspectResult],
):
    description: ClassVar[str] = (
        "Read-only peek at a node's value in the current workflow graph: for a table, its "
        "columns with inferred dtypes, row count, and the first N rows; for a text report, its "
        "content. Use it to see what a node produced (e.g. which columns a CSV has, and which "
        "are numeric) before authoring the next graph_patch. Does not run or change anything."
    )

    @classmethod
    def get_status_text(cls) -> str:
        return "Inspecting a graph node"

    @classmethod
    def format_call_display(cls, args: GraphInspectArgs) -> ToolCallDisplay:
        return ToolCallDisplay(summary=f"Inspect node {args.node_id!r}")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if event.error:
            return ToolResultDisplay(success=False, message=event.error)
        if not isinstance(event.result, GraphInspectResult):
            return ToolResultDisplay(success=True, message="Inspected")
        r = event.result
        if r.kind == "table":
            return ToolResultDisplay(
                success=True, message=f"{r.node_id}: {r.row_count} rows × {len(r.columns)} cols"
            )
        if r.kind == "missing":
            return ToolResultDisplay(success=True, message=f"{r.node_id}: not computed yet")
        return ToolResultDisplay(success=True, message=f"{r.node_id}: {r.kind}")

    async def run(
        self, args: GraphInspectArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[GraphInspectResult, None]:
        try:
            graph_dir = _session_graph_dir(ctx)
        except ValueError as exc:
            raise ToolError(f"graph_inspect requires a session directory: {exc}") from exc

        mirror = graph_dir / "graph.json"
        if not mirror.exists():
            raise ToolError("no working graph yet — author one with graph_patch first")
        graph = Graph.model_validate_json(mirror.read_text())
        if args.node_id not in graph.nodes:
            available = ", ".join(sorted(graph.nodes)) or "(none)"
            raise ToolError(f"unknown node {args.node_id!r}; nodes are: {available}")

        cache = CacheStore(graph_dir / "cache.sqlite")
        try:
            text = render.materialize_node_value(graph, args.node_id, cache)
        finally:
            cache.close()

        if text is None:
            yield GraphInspectResult(
                node_id=args.node_id,
                kind="missing",
                preview="Not computed yet — re-run the graph with graph_patch.",
            )
            return
        yield self._render_value(args.node_id, text, args.rows)

    @staticmethod
    def _render_value(node_id: str, text: str, rows: int) -> GraphInspectResult:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return GraphInspectResult(node_id=node_id, kind="value", preview=text[:_MAX_TEXT_CHARS])

        # A tabular value: {"columns": [...], "rows": [ {..}, ... ]}.
        if isinstance(payload, dict) and "columns" in payload and "rows" in payload:
            columns: list[str] = list(payload.get("columns") or [])
            table_rows: list[dict[str, Any]] = list(payload.get("rows") or [])
            dtypes = {
                col: _infer_dtype([r.get(col) for r in table_rows[:_DTYPE_SAMPLE]])
                for col in columns
            }
            head = table_rows[:rows]
            preview = json.dumps(head, ensure_ascii=False, indent=2)[:_MAX_TEXT_CHARS]
            return GraphInspectResult(
                node_id=node_id,
                kind="table",
                columns=columns,
                dtypes=dtypes,
                row_count=len(table_rows),
                preview=preview,
            )

        # A text report: {"markdown": "..."} (or any single text field).
        if isinstance(payload, dict) and "markdown" in payload:
            return GraphInspectResult(
                node_id=node_id, kind="report", preview=str(payload["markdown"])[:_MAX_TEXT_CHARS]
            )

        return GraphInspectResult(
            node_id=node_id, kind="value", preview=text[:_MAX_TEXT_CHARS]
        )
