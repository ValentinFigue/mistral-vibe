from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.graph.model import AddNode, Node
from vibe.core.tools.base import InvokeContext, ToolError
from vibe.core.tools.builtins.graph_inspect import (
    GraphInspect,
    GraphInspectArgs,
    GraphInspectConfig,
    GraphInspectResult,
    GraphInspectState,
)
from vibe.core.tools.builtins.graph_patch import (
    GraphPatch,
    GraphPatchArgs,
    GraphPatchConfig,
    GraphPatchState,
)


def _ctx(session_dir: Path) -> InvokeContext:
    return InvokeContext(tool_call_id="t", session_dir=session_dir)


def _inspect() -> GraphInspect:
    return GraphInspect(config_getter=lambda: GraphInspectConfig(), state=GraphInspectState())


async def _authored_graph(ctx: InvokeContext) -> None:
    """Author a sample-data → group_by → report graph so its values are cached."""
    tool = GraphPatch(config_getter=lambda: GraphPatchConfig(), state=GraphPatchState())
    args = GraphPatchArgs(
        patch=[
            AddNode(node=Node(id="src", op="sample_dataset", params={"name": "sales"})),
            AddNode(node=Node(id="grp", op="group_by",
                              params={"keys": ["country"], "metric": "revenue", "aggs": ["sum"]},
                              inputs={"table": "src"})),
            AddNode(node=Node(id="rep", op="to_markdown",
                              params={"title": "T", "max_rows": 50}, inputs={"table": "grp"})),
        ]
    )
    async for _ in tool.run(args, ctx):
        pass


async def _run_inspect(node_id: str, ctx: InvokeContext) -> GraphInspectResult:
    result: GraphInspectResult | None = None
    async for item in _inspect().run(GraphInspectArgs(node_id=node_id), ctx):
        result = item
    assert result is not None
    return result


def test_permission_is_always() -> None:
    # Read-only → auto-run, no approval gate.
    from vibe.core.tools.base import ToolPermission

    assert GraphInspectConfig().permission == ToolPermission.ALWAYS


@pytest.mark.asyncio
async def test_inspect_table_shows_schema_and_dtypes(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    await _authored_graph(ctx)
    r = await _run_inspect("src", ctx)
    assert r.kind == "table"
    assert {"date", "region", "revenue"} <= set(r.columns)
    assert r.dtypes["revenue"] in ("int", "float")
    assert r.dtypes["region"] == "str"
    assert r.row_count and r.row_count > 0


@pytest.mark.asyncio
async def test_inspect_report_shows_text(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    await _authored_graph(ctx)
    r = await _run_inspect("rep", ctx)
    assert r.kind == "report"
    assert "# T" in r.preview


@pytest.mark.asyncio
async def test_inspect_unknown_node_errors(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    await _authored_graph(ctx)
    with pytest.raises(ToolError, match="unknown node"):
        await _run_inspect("nope", ctx)


@pytest.mark.asyncio
async def test_inspect_without_graph_errors(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="no working graph"):
        await _run_inspect("src", _ctx(tmp_path))
