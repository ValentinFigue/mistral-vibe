"""``run_pipeline`` — the analyst's authoring action: a compact pipeline DSL.

The agent submits a short program (`read_csv(...) | sql(...) | to_markdown()`) instead of typed
patch nodes. Submission is **declarative**: the whole program is parsed to a graph, diffed
against the stored graph, and only the dirty subgraph reruns — so editing is "resubmit with the
tweak." The heavy execution/handle/catalog machinery is shared with ``graph_patch`` via
``build_result``/``format_llm_content``; this tool only adds the DSL front-end.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import ClassVar

from pydantic import BaseModel, Field

from vibe.core.graph.dsl import DSLError, parse_pipeline
from vibe.core.graph.model import Graph
from vibe.core.graph.session_store import graph_dir as _session_graph_dir
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)

# Reuse the shared graph execution + model-facing rendering (and the analysis library registers
# via this import, as with graph_patch).
from vibe.core.tools.builtins.graph_patch import (
    GraphPatch,
    GraphPatchResult,
    build_result,
    format_llm_content,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolResultEvent


class RunPipelineArgs(BaseModel):
    pipeline: str = Field(
        description="A pipeline program. Each line is `op(key=value, …)` steps joined by `|` "
        "(the previous step's table feeds the next); use `name = …` to name a step and refer to "
        "it later (e.g. as an extra table for `sql`/`join`). Reference only operators/blocks from "
        "the catalog. Resubmit the whole program to edit — only changed steps recompute."
    )


class RunPipelineConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    library: str | None = None  # scope operators/blocks (the analyst sets "analysis")


class RunPipelineState(BaseToolState):
    graph_json: str = ""
    catalog_shown: bool = False


class RunPipeline(
    BaseTool[RunPipelineArgs, GraphPatchResult, RunPipelineConfig, RunPipelineState],
    ToolUIData[RunPipelineArgs, GraphPatchResult],
):
    description: ClassVar[str] = (
        "Author or refine a data-analysis workflow by submitting a compact pipeline program "
        "(load → transform → sql → report/export). It is parsed into a cached graph and executed "
        "incrementally — resubmit the whole program to change a step; only the affected steps "
        "recompute. Returns which steps ran vs. were cached and handles to the terminal outputs."
    )

    @classmethod
    def get_status_text(cls) -> str:
        return "Running analysis pipeline"

    @classmethod
    def format_call_display(cls, args: RunPipelineArgs) -> ToolCallDisplay:
        return ToolCallDisplay(summary="Run analysis pipeline", content=args.pipeline.strip())

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        # The result shape is shared with graph_patch, so its "N ran, M cached · glimpse" display
        # applies verbatim.
        return GraphPatch.get_result_display(event)

    async def run(
        self, args: RunPipelineArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[GraphPatchResult, None]:
        try:
            graph_dir = _session_graph_dir(ctx)
        except ValueError as exc:
            raise ToolError(f"run_pipeline requires a session directory: {exc}") from exc

        # Declarative: the submitted program IS the whole graph. Diff it against the prior graph
        # (in-memory state, falling back to the disk mirror after a restart) for incremental rerun.
        graph_json = self.state.graph_json
        if not graph_json:
            mirror = graph_dir / "graph.json"
            if mirror.exists():
                graph_json = mirror.read_text()
                self.state.graph_json = graph_json
        current = Graph.model_validate_json(graph_json) if graph_json else Graph()

        try:
            new = parse_pipeline(args.pipeline)
        except DSLError as exc:
            raise ToolError(f"pipeline error: {exc}") from exc

        # A live session's ctx carries a sampling handler → an LLM caller for narrate/classify;
        # None when headless (those ops then raise a clear "needs a live session" error).
        llm = ctx.sampling_callback.complete_text if (ctx and ctx.sampling_callback) else None
        result = await build_result(new, current, graph_dir, self.config.library, llm=llm)
        self.state.graph_json = new.model_dump_json()
        yield result

    def get_llm_content(self, result: GraphPatchResult) -> str | None:
        """Compact handle-oriented model text; the catalog is sent once per session."""
        show_catalog = not self.state.catalog_shown
        text = format_llm_content(result, show_catalog=show_catalog)
        if text is not None and show_catalog:
            self.state.catalog_shown = True
        return text
