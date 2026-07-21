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


def test_format_call_display_shows_per_op_diff() -> None:
    from vibe.core.graph.model import Connect, Disconnect, RemoveNode, SetParam
    from vibe.core.tools.builtins.graph_patch import GraphPatch

    args = GraphPatchArgs(
        patch=[
            AddNode(node=Node(id="r", op="format_report", params={"title": "Q3"})),
            SetParam(id="r", key="title", value="Q4"),
            Connect(id="r", port="margins", source="m"),
            Disconnect(id="r", port="margins"),
            RemoveNode(id="old"),
        ]
    )
    display = GraphPatch.format_call_display(args)
    assert "1×add_node" in display.summary
    assert display.content is not None
    assert "+ add r (format_report)" in display.content
    assert "~ set r.title = 'Q4'" in display.content
    assert "→ connect r.margins ← m" in display.content
    assert "⊘ disconnect r.margins" in display.content
    assert "− remove old" in display.content


def test_graph_patch_approval_widget_registered() -> None:
    from vibe.cli.textual_ui.widgets.tool_widgets import (
        APPROVAL_WIDGETS,
        GraphPatchApprovalWidget,
    )

    assert APPROVAL_WIDGETS["graph_patch"] is GraphPatchApprovalWidget


@pytest.mark.asyncio
async def test_result_glimpse_includes_output(tmp_path: Path) -> None:
    from vibe.core.tools.builtins.graph_patch import GraphPatch
    from vibe.core.types import ToolResultEvent

    sales, costs = write_fixtures(tmp_path)
    result = await _run(_tool(), _build_demo_patch(sales, costs, title="Q3"), _ctx(tmp_path))
    display = GraphPatch.get_result_display(
        ToolResultEvent(tool_name="graph_patch", tool_call_id="t", tool_class=GraphPatch, result=result)
    )
    assert "3 ran" in display.message
    assert "# Q3" in display.message  # terminal output glimpse


@pytest.mark.asyncio
async def test_persists_graph_and_report(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    await _run(_tool(), _build_demo_patch(sales, costs), _ctx(tmp_path))
    graph_dir = tmp_path / "graph"
    assert (graph_dir / "graph.json").exists()
    report = (graph_dir / "report.json").read_text()
    assert "states" in report and "brief" in report  # folded report keyed by authored ids


@pytest.mark.asyncio
async def test_catalog_distinguishes_inputs_from_params(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    tool, ctx = _tool(), _ctx(tmp_path)
    result = await _run(tool, _build_demo_patch(sales, costs), ctx)
    # parse_table(file: FileContent, amount_col: str) -> file is an input, amount_col a param
    # Catalog now carries arg types.
    assert "parse_table(inputs: file:FileContent; params: amount_col:str)" in result.catalog


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
async def test_reset_discards_prior_graph(tmp_path: Path) -> None:
    tool, ctx = _tool(), _ctx(tmp_path)
    await _run(tool, _raw_pipeline_patch(), ctx)  # graph now has s, c, ps, pc, m, r

    # reset + a single block node: prior nodes are gone, no teardown ordering needed.
    result = await _run(
        tool,
        GraphPatchArgs(
            reset=True,
            patch=[AddNode(node=Node(id="brief", op="weekly_margin_brief", params={"title": "Q1"}))],
        ),
        ctx,
    )
    assert set(result.fresh) == {"brief"}
    assert "s" not in result.fresh and "s" not in result.cached  # old graph discarded


@pytest.mark.asyncio
async def test_invalid_patch_returns_tool_error(tmp_path: Path) -> None:
    tool, ctx = _tool(), _ctx(tmp_path)
    args = GraphPatchArgs(patch=[AddNode(node=Node(id="x", op="does_not_exist"))])
    with pytest.raises(ToolError, match="invalid graph"):
        await _run(tool, args, ctx)


@pytest.mark.asyncio
async def test_read_csv_content_fp_autofilled_and_incremental(tmp_path: Path) -> None:
    # The agent supplies only a path; graph_patch injects content_fp = hash(file) each turn.
    csv = tmp_path / "d.csv"
    csv.write_text("a,b\n1,x\n")
    tool, ctx = _tool(), _ctx(tmp_path)
    build = GraphPatchArgs(
        patch=[
            AddNode(node=Node(id="src", op="read_csv", params={"path": str(csv)})),
            AddNode(node=Node(id="rep", op="to_markdown",
                              params={"title": "T", "max_rows": 10}, inputs={"table": "src"})),
        ]
    )
    r1 = await _run(tool, build, ctx)
    assert set(r1.fresh) == {"src", "rep"}  # ran despite the agent omitting content_fp

    r2 = await _run(tool, GraphPatchArgs(patch=[]), ctx)  # unchanged file → all cached
    assert set(r2.cached) == {"src", "rep"} and r2.fresh == []

    csv.write_text("a,b\n1,x\n2,y\n")  # edit the file
    r3 = await _run(tool, GraphPatchArgs(patch=[]), ctx)  # re-hash → dirty rerun
    assert set(r3.fresh) == {"src", "rep"}


@pytest.mark.asyncio
async def test_library_scoping_catalog_and_focus_guard(tmp_path: Path) -> None:
    config = GraphPatchConfig(library="analysis")
    tool = GraphPatch(config_getter=lambda: config, state=GraphPatchState())
    ctx = _ctx(tmp_path)

    # The scoped catalog shows analysis ops/blocks only — not the margin demo.
    catalog = (await _run(tool, GraphPatchArgs(patch=[]), ctx)).catalog
    assert "sample_dataset" in catalog and "sales_source" not in catalog

    # Referencing an out-of-library (untagged margin) op is rejected.
    with pytest.raises(ToolError, match="not in the 'analysis' library"):
        await _run(tool, GraphPatchArgs(patch=[AddNode(node=Node(id="s", op="sales_source"))]), ctx)

    # An in-library op is accepted.
    r = await _run(
        tool,
        GraphPatchArgs(patch=[AddNode(node=Node(id="src", op="sample_dataset", params={"name": "sales"}))]),
        ctx,
    )
    assert "src" in r.fresh
