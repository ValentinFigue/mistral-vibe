from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import OptionList

from vibe.cli.textual_ui.widgets.graph_app import GraphApp, NodeView


class _Host(App):
    def __init__(self, panel: GraphApp) -> None:
        super().__init__()
        self._panel = panel
        self.closed = False

    def compose(self) -> ComposeResult:
        yield self._panel

    async def on_graph_app_closed(self, _message: GraphApp.Closed) -> None:
        self.closed = True


def _nodes() -> list[NodeView]:
    return [
        NodeView(id="a", op="sales_source", state="cached", cached=True),
        NodeView(
            id="b",
            op="format_report",
            inputs={"margins": "a"},
            params={"title": "[bold]x[/]"},  # markup-looking → must render literally
            state="fresh",
            cached=True,
            output='{"md":"[x] Q3"}',
        ),
    ]


@pytest.mark.asyncio
async def test_panel_lists_nodes_and_updates_detail_on_highlight() -> None:
    panel = GraphApp("Graph — 2 nodes", _nodes(), "flowchart LR")
    async with _Host(panel).run_test() as pilot:
        await pilot.pause()
        options = panel.query_one(OptionList)
        assert options.option_count == 2

        options.highlighted = 1
        await pilot.pause()
        detail = panel.detail_text
        assert "format_report" in detail
        assert "[x] Q3" in detail and "[bold]x[/]" in detail  # markup-safe


@pytest.mark.asyncio
async def test_escape_closes_the_panel() -> None:
    panel = GraphApp("Graph", _nodes(), "flowchart LR")
    host = _Host(panel)
    async with host.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert host.closed is True


@pytest.mark.asyncio
async def test_empty_and_block_views_render() -> None:
    # Empty graph → no option list, a placeholder note instead.
    empty = GraphApp("Graph", [], "")
    async with _Host(empty).run_test() as pilot:
        await pilot.pause()
        assert not empty.query(OptionList)

    # Block view: no run data (state/cached/output all None) still renders every node.
    block_nodes = [
        NodeView(id="parse", op="parse_table", params={"amount_col": "revenue"}),
        NodeView(id="report", op="format_report", inputs={"margins": "parse"}),
    ]
    panel = GraphApp("Block `x`", block_nodes, "flowchart LR")
    async with _Host(panel).run_test() as pilot:
        await pilot.pause()
        assert panel.query_one(OptionList).option_count == 2
        assert "parse_table" in panel.detail_text and "last run" not in panel.detail_text
