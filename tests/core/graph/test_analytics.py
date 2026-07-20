from __future__ import annotations

import pytest

from vibe.core.graph.cache import CacheStore
from vibe.core.graph.demo.analytics import build_analytics_graph
from vibe.core.graph.demo.research import build_research_graph
from vibe.core.graph.executor import execute


@pytest.mark.asyncio
async def test_analytics_deterministic_and_incremental(cache: CacheStore) -> None:
    graph = build_analytics_graph()

    # Cold: every node fresh.
    _, cold = await execute(graph, cache)
    assert set(cold.fresh()) == set(graph.nodes)
    assert cold.cached() == []

    # Warm: identical graph → identical fingerprints (deterministic ops) → all cached.
    _, warm = await execute(graph, cache)
    assert set(warm.cached()) == set(graph.nodes)
    assert warm.fresh() == []

    # Edit the group_by key: only the group→rank→report tail recomputes; load/filter/join hit.
    edited = build_analytics_graph(key="region")
    _, rerun = await execute(edited, cache)
    assert set(rerun.fresh()) == {"grouped", "ranked", "report"}
    assert set(rerun.cached()) == {"events", "users", "filtered", "enriched"}


@pytest.mark.asyncio
async def test_analytics_report_ranks_groups(cache: CacheStore) -> None:
    graph = build_analytics_graph(n=3, key="country")
    values, _ = await execute(graph, cache)
    report = cache.get(values["report"].fingerprint)
    assert report is not None
    text = report.decode()
    assert "# Top markets by purchase revenue" in text
    # top_n(3) keeps three groups.
    assert text.count("- ") == 3


@pytest.mark.asyncio
async def test_research_incremental(cache: CacheStore) -> None:
    graph = build_research_graph()
    _, cold = await execute(graph, cache)
    assert set(cold.fresh()) == set(graph.nodes)

    # Change the extract keyword: fetch stays cached, the extract→summarize→synthesize tail reruns.
    edited = build_research_graph(keyword="pricing")
    _, rerun = await execute(edited, cache)
    assert set(rerun.cached()) == {"docs"}
    assert set(rerun.fresh()) == {"passages", "summary", "brief"}
