"""Pure operators and graph builder for the weekly-margin demo pipeline.

The DAG::

    source_file(sales.csv) -> parse_table(revenue) ┐
                                                    ├-> join_margin -> format_report
    source_file(costs.csv) -> parse_table(cost)     ┘

Every operator is a pure function of its declared inputs and params. Editing ``sales.csv``
re-fingerprints only ``source_file(sales)``, ``parse_table(sales)``, ``join_margin`` and
``format_report``; the costs branch stays a cache hit.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from vibe.core.graph.fingerprint import content_hash
from vibe.core.graph.model import Graph, Node
from vibe.core.graph.operators import operator


class FileContent(BaseModel):
    text: str


class Table(BaseModel):
    by_region: dict[str, float]


class Margins(BaseModel):
    by_region: dict[str, float]


class ReportDoc(BaseModel):
    markdown: str


@operator
async def source_file(path: str, content_fp: str) -> FileContent:
    """Read a file. ``content_fp`` (the file's content hash) participates only in the
    fingerprint (purity discipline: external content is lifted into an explicit param); it
    is not used at runtime, so a changed file invalidates this node via a rebuilt graph.
    """
    return FileContent(text=Path(path).read_text())


@operator
async def parse_table(file: FileContent, amount_col: str) -> Table:
    """Parse a two-column ``region,<amount_col>`` CSV into ``{region: amount}``."""
    lines = [line for line in file.text.strip().splitlines() if line]
    header = lines[0].split(",")
    region_idx = header.index("region")
    amount_idx = header.index(amount_col)
    by_region: dict[str, float] = {}
    for line in lines[1:]:
        cells = line.split(",")
        by_region[cells[region_idx]] = float(cells[amount_idx])
    return Table(by_region=by_region)


@operator
async def join_margin(sales: Table, costs: Table) -> Margins:
    """Margin per region = revenue - cost."""
    regions = set(sales.by_region) | set(costs.by_region)
    return Margins(
        by_region={
            region: sales.by_region.get(region, 0.0) - costs.by_region.get(region, 0.0)
            for region in sorted(regions)
        }
    )


@operator
async def format_report(margins: Margins, title: str) -> ReportDoc:
    """Render the margins as a small markdown brief."""
    lines = [f"# {title}", ""]
    lines.extend(f"- {region}: {margin:,.2f}" for region, margin in sorted(margins.by_region.items()))
    return ReportDoc(markdown="\n".join(lines))


_SALES_CSV = "region,revenue\nemea,1200\namer,2100\napac,900\n"
_COSTS_CSV = "region,cost\nemea,500\namer,1300\napac,400\n"


@operator
async def sales_source() -> FileContent:
    """The demo sales data, bundled — an agent-authorable source needing no path or hash.

    Content is baked into the operator, so its fingerprint is stable across a session (the
    demo cache is per-session). Unlike ``source_file`` this takes no filesystem path, so it
    is safe to expose to an agent.
    """
    return FileContent(text=_SALES_CSV)


@operator
async def costs_source() -> FileContent:
    """The demo costs data, bundled (see :func:`sales_source`)."""
    return FileContent(text=_COSTS_CSV)


def write_fixtures(work_dir: Path) -> tuple[Path, Path]:
    """Write the demo's ``sales.csv`` / ``costs.csv`` fixtures and return their paths."""
    work_dir.mkdir(parents=True, exist_ok=True)
    sales = work_dir / "sales.csv"
    costs = work_dir / "costs.csv"
    sales.write_text(_SALES_CSV)
    costs.write_text(_COSTS_CSV)
    return sales, costs


def build_graph(sales_path: Path, costs_path: Path, *, title: str = "Weekly Margin Brief") -> Graph:
    """Wire the demo DAG over two source CSVs, folding file content into source fingerprints."""
    graph = Graph()
    graph.add(
        Node(
            id="src_sales",
            op="source_file",
            params={"path": str(sales_path), "content_fp": content_hash(sales_path)},
        )
    )
    graph.add(
        Node(
            id="src_costs",
            op="source_file",
            params={"path": str(costs_path), "content_fp": content_hash(costs_path)},
        )
    )
    graph.add(
        Node(
            id="parse_sales",
            op="parse_table",
            params={"amount_col": "revenue"},
            inputs={"file": "src_sales"},
        )
    )
    graph.add(
        Node(
            id="parse_costs",
            op="parse_table",
            params={"amount_col": "cost"},
            inputs={"file": "src_costs"},
        )
    )
    graph.add(
        Node(
            id="margin",
            op="join_margin",
            inputs={"sales": "parse_sales", "costs": "parse_costs"},
        )
    )
    graph.add(
        Node(
            id="report",
            op="format_report",
            params={"title": title},
            inputs={"margins": "margin"},
        )
    )
    return graph
