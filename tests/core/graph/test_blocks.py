from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.graph.blocks import expand, fold_report
from vibe.core.graph.cache import CacheStore
from vibe.core.graph.demo.blocks import build_graph_with_block
from vibe.core.graph.demo.pipeline import build_graph, write_fixtures
from vibe.core.graph.executor import execute
from vibe.core.graph.fingerprint import fingerprint_node


def test_expand_is_identity_without_blocks(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    g = build_graph(sales, costs)
    expanded, fold = expand(g)
    assert set(expanded.nodes) == set(g.nodes)
    assert fold == {nid: nid for nid in g.nodes}


def test_block_expands_to_primitive_nodes(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    gbk = build_graph_with_block(sales, costs)
    expanded, fold = expand(gbk)

    assert set(expanded.nodes) == {
        "src_sales",
        "src_costs",
        "brief/parse_sales",
        "brief/parse_costs",
        "brief/margin",
        "brief/report",
    }
    # inner nodes fold back to the authored block node
    assert fold["brief/report"] == "brief"
    assert fold["src_sales"] == "src_sales"


def test_block_graph_matches_hand_wired_fingerprints(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    title = "Weekly Margin Brief"
    hand = build_graph(sales, costs, title=title)
    blocked, _ = expand(build_graph_with_block(sales, costs, title=title))

    # The terminal recipe is identical regardless of node ids.
    assert fingerprint_node(blocked, "brief/report") == fingerprint_node(hand, "report")


@pytest.mark.asyncio
async def test_block_graph_dedups_against_hand_wired(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    cache = CacheStore(tmp_path / "cache.sqlite")

    await execute(build_graph(sales, costs), cache)  # warm cache via hand-wired graph
    _, report = await execute(build_graph_with_block(sales, costs), cache)

    # Every expanded node is a cache hit — identical fingerprints, shared results.
    assert set(report.cached()) == set(report.states)
    cache.close()


@pytest.mark.asyncio
async def test_fold_report_collapses_block_states(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    cache = CacheStore(tmp_path / "cache.sqlite")

    gbk = build_graph_with_block(sales, costs)
    expanded, fold = expand(gbk)
    _, cold = await execute(expanded, cache, expand_blocks=False)
    assert fold_report(cold, fold).states == {"src_sales": "fresh", "src_costs": "fresh", "brief": "fresh"}

    # Edit only costs; the sales source stays cached, the block recomputes.
    costs.write_text("region,cost\nemea,999\namer,1300\napac,400\n")
    expanded2, fold2 = expand(build_graph_with_block(sales, costs))
    _, warm = await execute(expanded2, cache, expand_blocks=False)
    folded = fold_report(warm, fold2)
    assert folded.states["src_sales"] == "cached"
    assert folded.states["src_costs"] == "fresh"
    assert folded.states["brief"] == "fresh"  # inner costs branch recomputed
    cache.close()



# --- computed block params (templates) ---

from pydantic import BaseModel

from vibe.core.graph.blocks import BlockDef, BlockError, is_block, register_block
from vibe.core.graph.model import Graph, Node
from vibe.core.graph.operators import operator


class _Echo(BaseModel):
    label: str


@operator(name="tst_echo")
async def _echo(label: str) -> _Echo:
    return _Echo(label=label)


def _register_templated_block() -> None:
    if is_block("tst_tmpl"):
        return
    sub = Graph()
    sub.add(Node(id="a", op="tst_echo", params={"label": ""}))          # exposed target for `tag`
    sub.add(Node(id="b", op="tst_echo", params={"label": "{tag}_x"}))   # template referencing `tag`
    register_block(
        BlockDef(name="tst_tmpl", graph=sub, params={"tag": ("a", "label")}, output="b")
    )


def test_template_resolves_from_block_param() -> None:
    _register_templated_block()
    g = Graph()
    g.add(Node(id="blk", op="tst_tmpl", params={"tag": "rev"}))
    expanded, _ = expand(g)
    assert expanded.nodes["blk/b"].params["label"] == "rev_x"   # template resolved
    assert expanded.nodes["blk/a"].params["label"] == "rev"     # exposed value verbatim


def test_template_unknown_ref_rejected_at_registration() -> None:
    sub = Graph()
    sub.add(Node(id="k", op="tst_echo", params={"label": "{nope}_x"}))
    with pytest.raises(BlockError, match="unknown param 'nope'"):
        register_block(BlockDef(name="tst_bad_tmpl", graph=sub, params={}, output="k"))


def test_agent_value_is_not_templated() -> None:
    if not is_block("tst_passthru"):
        sub = Graph()
        sub.add(Node(id="k", op="tst_echo", params={"label": "x"}))
        register_block(BlockDef(name="tst_passthru", graph=sub, params={"label": ("k", "label")}, output="k"))
    g = Graph()
    g.add(Node(id="blk", op="tst_passthru", params={"label": "{literal}"}))
    expanded, _ = expand(g)
    assert expanded.nodes["blk/k"].params["label"] == "{literal}"  # exposed value, not templated
