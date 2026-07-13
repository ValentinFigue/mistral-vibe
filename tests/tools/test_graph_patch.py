from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.graph.demo.pipeline import write_fixtures
from vibe.core.graph.fingerprint import content_hash
from vibe.core.graph.model import AddNode, Node, SetParam
from vibe.core.tools.base import InvokeContext, ToolError, ToolPermission
from vibe.core.tools.builtins.graph_patch import (
    GraphPatch,
    GraphPatchArgs,
    GraphPatchConfig,
    GraphPatchResult,
    GraphPatchState,
)


def _tool() -> GraphPatch:
    config = GraphPatchConfig()
    return GraphPatch(config_getter=lambda: config, state=GraphPatchState())


def _ctx(session_dir: Path) -> InvokeContext:
    return InvokeContext(tool_call_id="t1", session_dir=session_dir)


async def _run(tool: GraphPatch, args: GraphPatchArgs, ctx: InvokeContext) -> GraphPatchResult:
    result: GraphPatchResult | None = None
    async for item in tool.run(args, ctx):
        result = item
    assert result is not None
    return result


def _build_demo_patch(sales: Path, costs: Path, title: str = "Q3") -> GraphPatchArgs:
    return GraphPatchArgs(
        patch=[
            AddNode(node=Node(id="src_sales", op="source_file",
                              params={"path": str(sales), "content_fp": content_hash(sales)})),
            AddNode(node=Node(id="src_costs", op="source_file",
                              params={"path": str(costs), "content_fp": content_hash(costs)})),
            AddNode(node=Node(id="brief", op="margin_brief",
                              inputs={"sales_file": "src_sales", "costs_file": "src_costs"},
                              params={"title": title})),
        ]
    )


def test_permission_is_ask() -> None:
    # This is what routes every call through the human approval gate in the agent loop.
    assert GraphPatchConfig().permission == ToolPermission.ASK


@pytest.mark.asyncio
async def test_apply_execute_and_reuse_block(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    tool, ctx = _tool(), _ctx(tmp_path)

    result = await _run(tool, _build_demo_patch(sales, costs, title="Q3"), ctx)

    assert result.applied
    assert set(result.fresh) == {"src_sales", "src_costs", "brief"}
    assert result.cached == []
    assert "# Q3" in result.outputs["brief"]  # terminal block output rehydrated
    assert "margin_brief" in result.catalog


@pytest.mark.asyncio
async def test_state_persists_and_reruns_incrementally(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    tool, ctx = _tool(), _ctx(tmp_path)

    await _run(tool, _build_demo_patch(sales, costs, title="Q3"), ctx)
    # Second call on the SAME tool instance: retitle via a one-op patch.
    result = await _run(tool, GraphPatchArgs(patch=[SetParam(id="brief", key="title", value="Q4")]), ctx)

    assert result.changed == ["brief"]
    assert result.fresh == ["brief"]  # only the block's report inner node recomputed
    assert set(result.cached) == {"src_sales", "src_costs"}
    assert "# Q4" in result.outputs["brief"]


@pytest.mark.asyncio
async def test_catalog_distinguishes_inputs_from_params(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    tool, ctx = _tool(), _ctx(tmp_path)
    result = await _run(tool, _build_demo_patch(sales, costs), ctx)
    # parse_table(file: FileContent, amount_col: str) -> file is an input, amount_col a param
    assert "parse_table(inputs: file; params: amount_col)" in result.catalog


@pytest.mark.asyncio
async def test_resumes_from_disk_mirror_after_restart(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    ctx = _ctx(tmp_path)

    # First tool instance authors the graph, then is discarded (simulating a restart).
    await _run(_tool(), _build_demo_patch(sales, costs, title="Q3"), ctx)

    # A fresh instance (empty state) must reload the graph from the disk mirror: a retitle
    # patch reruns only the block, proving upstream sources survived the "restart".
    fresh = _tool()
    result = await _run(fresh, GraphPatchArgs(patch=[SetParam(id="brief", key="title", value="Q4")]), ctx)
    assert result.changed == ["brief"]
    assert set(result.cached) == {"src_sales", "src_costs"}
    assert "# Q4" in result.outputs["brief"]


@pytest.mark.asyncio
async def test_empty_patch_returns_catalog(tmp_path: Path) -> None:
    # The agent's first move: an empty patch runs nothing but reveals what it can build.
    result = await _run(_tool(), GraphPatchArgs(patch=[]), _ctx(tmp_path))
    assert result.applied
    assert result.fresh == [] and result.cached == []
    assert "weekly_margin_brief" in result.catalog
    assert "sales_source" in result.catalog


@pytest.mark.asyncio
async def test_weekly_brief_block_is_one_node(tmp_path: Path) -> None:
    # The happy path: no paths, no wiring, no hashes — just a title.
    args = GraphPatchArgs(
        patch=[AddNode(node=Node(id="brief", op="weekly_margin_brief", params={"title": "Q3"}))]
    )
    result = await _run(_tool(), args, _ctx(tmp_path))
    assert result.fresh == ["brief"]
    assert "# Q3" in result.outputs["brief"]


def _raw_pipeline_patch(amount_col_sales: str = "revenue") -> GraphPatchArgs:
    """Compose the brief from raw operators (no block), like the 'compose from parts' prompt."""
    return GraphPatchArgs(
        patch=[
            AddNode(node=Node(id="s", op="sales_source")),
            AddNode(node=Node(id="c", op="costs_source")),
            AddNode(node=Node(id="ps", op="parse_table", params={"amount_col": amount_col_sales}, inputs={"file": "s"})),
            AddNode(node=Node(id="pc", op="parse_table", params={"amount_col": "cost"}, inputs={"file": "c"})),
            AddNode(node=Node(id="m", op="join_margin", inputs={"sales": "ps", "costs": "pc"})),
            AddNode(node=Node(id="r", op="format_report", params={"title": "V1"}, inputs={"margins": "m"})),
        ]
    )


@pytest.mark.asyncio
async def test_compose_from_raw_operators(tmp_path: Path) -> None:
    result = await _run(_tool(), _raw_pipeline_patch(), _ctx(tmp_path))
    assert "r" in result.fresh and "# V1" in result.outputs["r"]


@pytest.mark.asyncio
async def test_operator_runtime_error_is_recoverable(tmp_path: Path) -> None:
    # Wrong column name (the failure the user hit) → a clear, recoverable ToolError.
    with pytest.raises(ToolError, match=r"execution failed.*not found.*available columns"):
        await _run(_tool(), _raw_pipeline_patch(amount_col_sales="amount"), _ctx(tmp_path))


@pytest.mark.asyncio
async def test_invalid_patch_returns_tool_error(tmp_path: Path) -> None:
    tool, ctx = _tool(), _ctx(tmp_path)
    args = GraphPatchArgs(patch=[AddNode(node=Node(id="x", op="does_not_exist"))])
    with pytest.raises(ToolError, match="invalid graph"):
        await _run(tool, args, ctx)
