from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.graph import blocks as blocks_mod
from vibe.core.graph.blocks import is_block, load_blocks
from vibe.core.graph.model import AddNode, Node
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.graph_patch import (
    GraphPatch,
    GraphPatchArgs,
    GraphPatchConfig,
    GraphPatchState,
)
from vibe.core.tools.builtins.graph_save_block import (
    GraphSaveBlock,
    GraphSaveBlockArgs,
    GraphSaveBlockConfig,
    GraphSaveBlockResult,
)


@pytest.fixture(autouse=True)
def _isolate_vibe_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Redirect ~/.vibe (hence BLOCKS_DIR) to a tmp dir so tests never touch the real home.
    monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibehome"))


def _ctx(session_dir: Path) -> InvokeContext:
    return InvokeContext(tool_call_id="t", session_dir=session_dir)


async def _author_pipeline(ctx: InvokeContext, title: str = "V1") -> None:
    """Author the raw margin pipeline via graph_patch (writes the session graph mirror)."""
    tool = GraphPatch(config_getter=lambda: GraphPatchConfig(), state=GraphPatchState())
    patch = [
        AddNode(node=Node(id="s", op="sales_source")),
        AddNode(node=Node(id="c", op="costs_source")),
        AddNode(node=Node(id="ps", op="parse_table", params={"amount_col": "revenue"}, inputs={"file": "s"})),
        AddNode(node=Node(id="pc", op="parse_table", params={"amount_col": "cost"}, inputs={"file": "c"})),
        AddNode(node=Node(id="m", op="join_margin", inputs={"sales": "ps", "costs": "pc"})),
        AddNode(node=Node(id="r", op="format_report", params={"title": title}, inputs={"margins": "m"})),
    ]
    async for _ in tool.run(GraphPatchArgs(patch=patch), ctx):
        pass


async def _save_block(ctx: InvokeContext, args: GraphSaveBlockArgs) -> GraphSaveBlockResult:
    tool = GraphSaveBlock(config_getter=lambda: GraphSaveBlockConfig(), state=BaseToolState())
    result: GraphSaveBlockResult | None = None
    async for item in tool.run(args, ctx):
        result = item
    assert result is not None
    return result


@pytest.mark.asyncio
async def test_save_derives_interface_and_persists(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    await _author_pipeline(ctx)

    result = await _save_block(
        ctx,
        GraphSaveBlockArgs(name="my_brief", nodes=["ps", "pc", "m", "r"], expose_params=["r.title"]),
    )

    assert result.input_ports == ["pc_file", "ps_file"]
    assert result.params == ["r_title"]
    assert result.output == "r"
    assert result.overwritten is False
    assert is_block("my_brief")
    assert Path(result.path).exists()


@pytest.mark.asyncio
async def test_saved_block_is_reusable(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    await _author_pipeline(ctx)
    await _save_block(ctx, GraphSaveBlockArgs(name="reuse_brief", nodes=["ps", "pc", "m", "r"], expose_params=["r.title"]))

    # reset the graph, then wire the saved block to fresh sources.
    patch = GraphPatch(config_getter=lambda: GraphPatchConfig(), state=GraphPatchState())
    use = GraphPatchArgs(
        reset=True,
        patch=[
            AddNode(node=Node(id="s2", op="sales_source")),
            AddNode(node=Node(id="c2", op="costs_source")),
            AddNode(node=Node(id="b", op="reuse_brief",
                              inputs={"ps_file": "s2", "pc_file": "c2"}, params={"r_title": "Reused"})),
        ],
    )
    result = None
    async for item in patch.run(use, ctx):
        result = item
    assert result is not None
    assert result.fresh == ["b"]
    assert "# Reused" in result.outputs["b"]
    assert "reuse_brief" in result.catalog


@pytest.mark.asyncio
async def test_rejects_no_graph_reserved_and_operator_names(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    # No graph authored yet.
    with pytest.raises(ToolError, match="no graph to save"):
        await _save_block(ctx, GraphSaveBlockArgs(name="whatever"))

    await _author_pipeline(ctx)
    with pytest.raises(ToolError, match="already an operator"):
        await _save_block(ctx, GraphSaveBlockArgs(name="parse_table", nodes=["r"]))
    with pytest.raises(ToolError, match="built-in block"):
        await _save_block(ctx, GraphSaveBlockArgs(name="margin_brief", nodes=["r"]))


@pytest.mark.asyncio
async def test_persists_across_sessions(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    await _author_pipeline(ctx)
    result = await _save_block(ctx, GraphSaveBlockArgs(name="cross_session", nodes=["ps", "pc", "m", "r"]))

    # Simulate a fresh process: drop the in-memory registration, then reload from disk.
    blocks_mod._BLOCKS.pop("cross_session", None)
    assert not is_block("cross_session")
    load_blocks(Path(result.path).parent)
    assert is_block("cross_session")
