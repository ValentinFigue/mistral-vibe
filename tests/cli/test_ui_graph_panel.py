from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_app, build_test_vibe_config
from vibe.cli.textual_ui.app import BottomApp
from vibe.cli.textual_ui.widgets.graph_app import GraphApp
from vibe.core.graph.model import Graph, Node


@pytest.mark.asyncio
async def test_graph_command_opens_panel_and_escape_returns_to_input() -> None:
    app = build_test_vibe_app(config=build_test_vibe_config())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        # Stub the disk/cache reads so the launch path (which is what we're testing) runs
        # regardless of session state.
        graph = Graph()
        graph.add(Node(id="a", op="sales_source"))
        app._load_graph_and_states = lambda _sd: (graph, None)  # type: ignore[method-assign]
        app._graph_cache_view = lambda _g, _sd: (None, None)  # type: ignore[method-assign]

        await app._show_graph()
        await pilot.pause(0.1)
        assert app._current_bottom_app == BottomApp.Graph
        assert len(app.query(GraphApp)) == 1

        # Escape must dismiss the panel and return to the input app (the trap-the-user guard).
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert app._current_bottom_app == BottomApp.Input
        assert len(app.query(GraphApp)) == 0


@pytest.mark.asyncio
async def test_graph_command_no_graph_shows_message() -> None:
    app = build_test_vibe_app(config=build_test_vibe_config())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        app._load_graph_and_states = lambda _sd: (None, None)  # type: ignore[method-assign]

        await app._show_graph()
        await pilot.pause(0.1)
        # Stays in the input app (no panel) when there's no graph.
        assert app._current_bottom_app == BottomApp.Input
        assert len(app.query(GraphApp)) == 0
