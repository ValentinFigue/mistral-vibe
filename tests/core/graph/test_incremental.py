from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.graph.cache import CacheStore
from vibe.core.graph.demo.pipeline import build_graph, write_fixtures
from vibe.core.graph.executor import execute


@pytest.mark.asyncio
async def test_editing_one_input_reruns_only_dirty_subgraph(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    cache = CacheStore(tmp_path / "cache.sqlite")

    graph = build_graph(sales, costs)
    _, first = await execute(graph, cache)
    assert set(first.fresh()) == set(graph.nodes)  # cold cache: all fresh

    # Edit only sales.csv and rebuild (source fingerprints fold in file content).
    sales.write_text("region,revenue\nemea,1500\namer,2100\napac,900\n")
    graph = build_graph(sales, costs)
    _, second = await execute(graph, cache)

    assert set(second.fresh()) == {"src_sales", "parse_sales", "margin", "report"}
    assert set(second.cached()) == {"src_costs", "parse_costs"}

    cache.close()


@pytest.mark.asyncio
async def test_untouched_rerun_is_all_cached(tmp_path: Path) -> None:
    sales, costs = write_fixtures(tmp_path)
    cache = CacheStore(tmp_path / "cache.sqlite")
    graph = build_graph(sales, costs)

    await execute(graph, cache)
    _, second = await execute(graph, cache)
    assert set(second.cached()) == set(graph.nodes)

    cache.close()
