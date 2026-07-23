from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.graph.demo.compare import (
    build_analytics_graph,
    build_docreview_graph,
    build_research_graph,
    measure_workflow,
)


@pytest.mark.asyncio
async def test_analytics_comparison_wins_on_context_and_recompute(tmp_path: Path) -> None:
    m = await measure_workflow(
        "analytics", build_analytics_graph, {"key": "region"}, "group_by key", work_dir=tmp_path
    )
    # Context: the graph keeps payloads out of context, so it is far smaller both turns.
    assert m.cold.graph_tokens < m.cold.transcript_tokens
    assert m.rerun.graph_tokens < m.rerun.transcript_tokens
    assert m.cold.reduction_pct > 80.0
    # Recompute: the transcript recomputes everything; the graph only the dirty subgraph.
    assert m.rerun.transcript_nodes == m.node_count
    assert m.rerun.graph_nodes < m.node_count


@pytest.mark.asyncio
async def test_research_comparison_reduces_context(tmp_path: Path) -> None:
    m = await measure_workflow(
        "research", build_research_graph, {"keyword": "pricing"}, "extract keyword", work_dir=tmp_path
    )
    assert m.cold.graph_tokens < m.cold.transcript_tokens
    assert m.rerun.graph_nodes < m.rerun.transcript_nodes


@pytest.mark.asyncio
async def test_docreview_comparison_reduces_context(tmp_path: Path) -> None:
    # A different artifact (Document, not Table) flows through the same engine: the large parsed
    # document stays in the cache, so the graph context is far smaller and only the dirty subgraph
    # recomputes on the filter edit.
    m = await measure_workflow(
        "doc-review", build_docreview_graph, {"contains": "termination"}, "filter sections",
        work_dir=tmp_path,
    )
    assert m.cold.graph_tokens < m.cold.transcript_tokens
    assert m.rerun.graph_nodes < m.node_count


@pytest.mark.asyncio
async def test_rerun_context_smaller_than_cold_for_graph(tmp_path: Path) -> None:
    # The re-run drops the catalog (shown once), so the graph's per-turn context shrinks further.
    m = await measure_workflow(
        "analytics", build_analytics_graph, {"key": "region"}, "group_by key", work_dir=tmp_path
    )
    assert m.rerun.graph_tokens < m.cold.graph_tokens
