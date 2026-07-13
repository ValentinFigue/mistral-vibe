from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.graph.blocks import (
    BlockDef,
    BlockError,
    block_from_subgraph,
    expand,
    get_block,
    is_block,
    load_blocks,
    save_block,
)
from vibe.core.graph.cache import CacheStore
from vibe.core.graph.demo.blocks import build_graph_with_block
from vibe.core.graph.demo.pipeline import build_graph, write_fixtures
from vibe.core.graph.executor import execute
from vibe.core.graph.model import Graph, Node

_INNER = ["parse_sales", "parse_costs", "margin", "report"]


def test_auto_derives_ports_and_output(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    block = block_from_subgraph("brief_a", build_graph(sales, costs), _INNER)
    assert set(block.input_ports) == {"parse_sales_file", "parse_costs_file"}
    assert block.input_ports["parse_sales_file"] == ("parse_sales", "file")
    assert block.output == "report"
    assert block.params == {}  # title stays baked by default


def test_expose_params(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    block = block_from_subgraph("brief_b", build_graph(sales, costs), _INNER, expose_params=["report.title"])
    assert block.params == {"report_title": ("report", "title")}


def test_whole_graph_has_no_ports(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    g = build_graph(sales, costs)
    block = block_from_subgraph("brief_c", g, list(g.nodes))
    assert block.input_ports == {}
    assert block.output == "report"


def test_multi_terminal_requires_output() -> None:
    g = Graph()
    g.add(Node(id="a", op="tst_const", params={"value": 1}))
    g.add(Node(id="b", op="tst_const", params={"value": 2}))
    with pytest.raises(BlockError, match="cannot infer a single output"):
        block_from_subgraph("two_terminals", g, ["a", "b"])
    # ...unless output is given explicitly
    block = block_from_subgraph("two_terminals", g, ["a", "b"], output="a")
    assert block.output == "a"


def test_rejects_block_ops_and_bad_names(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    blocked = build_graph_with_block(sales, costs)  # 'brief' is a block node
    with pytest.raises(BlockError, match="nested blocks are unsupported"):
        block_from_subgraph("nested", blocked, ["brief"])

    g = build_graph(sales, costs)
    with pytest.raises(BlockError, match="invalid block name"):
        block_from_subgraph("Bad Name!", g, _INNER)
    with pytest.raises(BlockError, match="already an operator"):
        block_from_subgraph("parse_table", g, _INNER)


def test_expose_params_validation(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    g = build_graph(sales, costs)
    with pytest.raises(BlockError, match="no param 'nope'"):
        block_from_subgraph("brief_d", g, _INNER, expose_params=["report.nope"])
    with pytest.raises(BlockError, match="not in the selection"):
        block_from_subgraph("brief_e", g, _INNER, expose_params=["src_sales.path"])


def test_save_load_roundtrip_and_use(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    blocks_dir = tmp_path / "blocks"
    block = block_from_subgraph(
        "roundtrip_brief", build_graph(sales, costs), _INNER, expose_params=["report.title"]
    )
    assert not is_block("roundtrip_brief")  # derivation does not register

    path = save_block(block, blocks_dir)
    assert path == blocks_dir / "roundtrip_brief.json"
    assert load_blocks(blocks_dir) >= 1
    assert is_block("roundtrip_brief")


def test_load_skips_block_with_unavailable_operator(tmp_path: Path) -> None:
    blocks_dir = tmp_path / "blocks"
    # A structurally valid block whose inner op does not exist in this environment.
    ghost = Graph()
    ghost.add(Node(id="x", op="no_such_op"))
    save_block(BlockDef(name="ghostly", graph=ghost, output="x"), blocks_dir)
    assert load_blocks(blocks_dir) == 0
    assert not is_block("ghostly")


def test_load_does_not_override_registered_block(tmp_path: Path) -> None:
    # margin_brief is a registered demo block; a rogue file with that name must be ignored.
    assert is_block("margin_brief")
    blocks_dir = tmp_path / "blocks"
    ghost = Graph()
    ghost.add(Node(id="x", op="no_such_op"))
    save_block(BlockDef(name="margin_brief", graph=ghost, output="x"), blocks_dir)

    load_blocks(blocks_dir)

    inner_ops = {n.op for n in get_block("margin_brief").graph.nodes.values()}
    assert "no_such_op" not in inner_ops  # still the real demo block, not the ghost


@pytest.mark.asyncio
async def test_loaded_block_executes(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    blocks_dir = tmp_path / "blocks"
    block = block_from_subgraph(
        "usable_brief", build_graph(sales, costs), _INNER, expose_params=["report.title"]
    )
    save_block(block, blocks_dir)
    load_blocks(blocks_dir)

    # Wire the loaded block to bundled sources and run it.
    g = Graph()
    g.add(Node(id="s", op="sales_source"))
    g.add(Node(id="c", op="costs_source"))
    g.add(
        Node(
            id="b",
            op="usable_brief",
            inputs={"parse_sales_file": "s", "parse_costs_file": "c"},
            params={"report_title": "Loaded"},
        )
    )
    cache = CacheStore(tmp_path / "cache.sqlite")
    expanded, fold = expand(g)
    values, report = await execute(expanded, cache, expand_blocks=False)
    payload = cache.get(values["b/report"].fingerprint)
    assert payload is not None and "# Loaded" in payload.decode()
    cache.close()
