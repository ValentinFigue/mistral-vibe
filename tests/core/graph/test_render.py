from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.graph.cache import CacheStore
from vibe.core.graph.demo.blocks import build_graph_with_block
from vibe.core.graph.demo.pipeline import build_graph, write_fixtures
from vibe.core.graph.executor import execute
from vibe.core.graph.render import (
    blocks_table,
    graph_status,
    nodes_table,
    to_mermaid,
    to_tree,
)


def test_to_mermaid_has_a_line_per_edge(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    g = build_graph(sales, costs)
    diagram = to_mermaid(g)
    assert diagram.startswith("flowchart LR")
    # every input edge appears (6-node demo has 5 edges)
    edge_count = sum(len(n.inputs) for n in g.nodes.values())
    assert diagram.count("-->") == edge_count
    assert "n_report" in diagram  # sanitized id present


def test_to_tree_is_topologically_ordered(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    lines = to_tree(build_graph(sales, costs))
    joined = "\n".join(lines)
    # sources before their consumers
    assert joined.index("src_sales") < joined.index("parse_sales")
    assert joined.index("margin") < joined.index("report")


def test_mermaid_states_styling(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    g = build_graph(sales, costs)
    diagram = to_mermaid(g, states={"report": "fresh", "margin": "cached"})
    assert "classDef fresh" in diagram
    assert "class n_report fresh" in diagram
    assert "class n_margin cached" in diagram


@pytest.mark.asyncio
async def test_graph_status_marks_materialized_vs_absent(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    cache = CacheStore(tmp_path / "cache.sqlite")
    g = build_graph(sales, costs)

    # Nothing run yet → every node absent.
    assert graph_status(g, cache) == {nid: False for nid in g.nodes}

    await execute(g, cache)
    # After a full run → every node materialized.
    assert all(graph_status(g, cache).values())
    cache.close()


@pytest.mark.asyncio
async def test_graph_status_is_block_aware(tmp_path: Path) -> None:
    # The cache is keyed by expanded fingerprints; a block node must still report correctly.
    sales, costs = write_fixtures(tmp_path)
    cache = CacheStore(tmp_path / "cache.sqlite")
    gb = build_graph_with_block(sales, costs)  # has a 'brief' block node

    assert graph_status(gb, cache)["brief"] is False
    await execute(gb, cache)
    assert graph_status(gb, cache)["brief"] is True
    cache.close()


def test_nodes_table_covers_every_node(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    g = build_graph(sales, costs)
    table = nodes_table(
        g,
        states={"report": "fresh"},
        status={"report": True},
        outputs={"report": "# Q1\nline"},
    )
    for nid in g.nodes:
        assert f"`{nid}`" in table
    assert "# Q1 line" in table  # newline flattened in preview


def test_blocks_table_lists_blocks() -> None:
    from vibe.core.graph.blocks import registered_blocks

    blocks = registered_blocks()
    table = blocks_table(blocks, {name: "built-in" for name in blocks})
    assert "margin_brief" in table
    assert "weekly_margin_brief" in table


def test_node_line_and_detail_are_markup_safe() -> None:
    from vibe.core.graph.render import node_detail, node_line

    line = node_line("report", "format_report", {"margins": "m"}, state="fresh")
    assert "report · format_report" in str(line)
    assert "← m" in str(line)

    # Arbitrary bracketed text (looks like Rich markup) must render as literal, not crash.
    detail = node_detail(
        "report", "format_report", {"margins": "m"}, {"title": "[bold]x[/]"},
        state="fresh", cached=True, output='{"md":"[x] Q3"}',
    )
    text = str(detail)
    assert "[bold]x[/]" in text and "[x] Q3" in text and "last run: fresh" in text


def test_node_detail_omits_run_data_when_none() -> None:
    from vibe.core.graph.render import node_detail

    text = str(node_detail("a", "src", {}, {}, state=None, cached=None, output=None))
    assert "last run" not in text and "cached" not in text
