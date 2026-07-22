from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.tools.base import InvokeContext, ToolError
from vibe.core.tools.builtins.graph_patch import GraphPatchResult
from vibe.core.tools.builtins.run_pipeline import (
    RunPipeline,
    RunPipelineArgs,
    RunPipelineConfig,
    RunPipelineState,
)


def _tool(library: str | None = "analysis") -> RunPipeline:
    return RunPipeline(config_getter=lambda: RunPipelineConfig(library=library), state=RunPipelineState())


def _ctx(session_dir: Path) -> InvokeContext:
    return InvokeContext(tool_call_id="t", session_dir=session_dir)


async def _run(tool: RunPipeline, program: str, ctx: InvokeContext) -> GraphPatchResult:
    result: GraphPatchResult | None = None
    async for item in tool.run(RunPipelineArgs(pipeline=program), ctx):
        result = item
    assert result is not None
    return result


_PIPELINE = (
    'sample_dataset(name="sales")'
    ' | sql(query="SELECT country, sum(revenue) AS rev FROM t1 GROUP BY country ORDER BY rev DESC")'
    ' | to_markdown(title="Top")'
)


@pytest.mark.asyncio
async def test_run_pipeline_executes_and_returns_handles(tmp_path: Path) -> None:
    result = await _run(_tool(), _PIPELINE, _ctx(tmp_path))
    assert result.applied
    assert len(result.fresh) == 3 and result.cached == []
    # terminal output is the report; its handle preview shows the rendered markdown
    (term, handle), = result.handles.items()
    assert "# Top" in handle.preview and "revenue".split()  # report rendered
    assert (tmp_path / "graph" / "graph.json").exists()


@pytest.mark.asyncio
async def test_run_pipeline_incremental_rerun(tmp_path: Path) -> None:
    tool, ctx = _tool(), _ctx(tmp_path)
    await _run(tool, _PIPELINE, ctx)
    # Resubmit identical program → all cached (declarative diff, same fingerprints).
    again = await _run(tool, _PIPELINE, ctx)
    assert again.fresh == [] and len(again.cached) == 3
    # Change only the report title → only that terminal step reruns.
    edited = _PIPELINE.replace('title="Top"', 'title="Best"')
    r = await _run(tool, edited, ctx)
    assert len(r.fresh) == 1 and len(r.cached) == 2


@pytest.mark.asyncio
async def test_run_pipeline_reuses_cache_across_sessions(tmp_path: Path) -> None:
    # The result cache is shared under VIBE_HOME, not per session-dir. A brand-new session
    # (fresh tool state + a different session dir) submitting the same recipe recomputes nothing.
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    first = await _run(_tool(), _PIPELINE, _ctx(a))
    assert len(first.fresh) == 3 and first.cached == []  # session A: cold cache
    second = await _run(_tool(), _PIPELINE, _ctx(b))
    assert second.fresh == [] and len(second.cached) == 3  # session B: all cross-session hits


@pytest.mark.asyncio
async def test_run_pipeline_library_scoping(tmp_path: Path) -> None:
    # An out-of-library op (the margin demo's sales_source) is rejected for the analyst.
    with pytest.raises(ToolError, match="not in the 'analysis' library"):
        await _run(_tool(), "sales_source()", _ctx(tmp_path))


@pytest.mark.asyncio
async def test_run_pipeline_reports_parse_errors(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="pipeline error"):
        await _run(_tool(), 'nope(x=1)', _ctx(tmp_path))


@pytest.mark.asyncio
async def test_run_pipeline_catalog_shown_once(tmp_path: Path) -> None:
    tool, ctx = _tool(), _ctx(tmp_path)
    r1 = await _run(tool, _PIPELINE, ctx)
    first = tool.get_llm_content(r1)
    assert first is not None and "read_csv" in first  # full catalog on the first turn
    second = tool.get_llm_content(r1)
    assert second is not None and "read_csv" not in second and "catalog unchanged" in second
