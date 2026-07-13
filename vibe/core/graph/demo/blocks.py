"""A demo reusable block: ``margin_brief``.

Packages the `{parse_sales, parse_costs, margin, report}` subgraph of the weekly-margin
pipeline as a single named operator. A graph of `source_file → margin_brief` expands to —
and executes identically to — the hand-wired six-node ``build_graph``, proving that reuse
preserves fingerprints and incrementality.
"""

from __future__ import annotations

from pathlib import Path

from vibe.core.graph.blocks import BlockDef, is_block, register_block
from vibe.core.graph.demo import (
    pipeline as _pipeline,  # noqa: F401  (registers operators)
)
from vibe.core.graph.fingerprint import content_hash
from vibe.core.graph.model import Graph, Node


def _margin_brief_subgraph() -> Graph:
    g = Graph()
    # `file` inputs are open (exposed as block ports); `title` is set via a block param.
    g.add(Node(id="parse_sales", op="parse_table", params={"amount_col": "revenue"}))
    g.add(Node(id="parse_costs", op="parse_table", params={"amount_col": "cost"}))
    g.add(Node(id="margin", op="join_margin", inputs={"sales": "parse_sales", "costs": "parse_costs"}))
    g.add(Node(id="report", op="format_report", inputs={"margins": "margin"}))
    return g


MARGIN_BRIEF = BlockDef(
    name="margin_brief",
    graph=_margin_brief_subgraph(),
    input_ports={"sales_file": ("parse_sales", "file"), "costs_file": ("parse_costs", "file")},
    params={"title": ("report", "title")},
    output="report",
)


def register() -> None:
    """Register the demo block (idempotent)."""
    if not is_block(MARGIN_BRIEF.name):
        register_block(MARGIN_BRIEF)


register()


def build_graph_with_block(
    sales_path: Path, costs_path: Path, *, title: str = "Weekly Margin Brief"
) -> Graph:
    """The demo pipeline built from two sources plus the ``margin_brief`` block."""
    register()
    g = Graph()
    g.add(
        Node(
            id="src_sales",
            op="source_file",
            params={"path": str(sales_path), "content_fp": content_hash(sales_path)},
        )
    )
    g.add(
        Node(
            id="src_costs",
            op="source_file",
            params={"path": str(costs_path), "content_fp": content_hash(costs_path)},
        )
    )
    g.add(
        Node(
            id="brief",
            op="margin_brief",
            inputs={"sales_file": "src_sales", "costs_file": "src_costs"},
            params={"title": title},
        )
    )
    return g
