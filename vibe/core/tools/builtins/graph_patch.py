"""``graph_patch`` — the agent's single action for authoring a workflow graph.

Instead of an opaque stream of tool calls, the agent emits a typed :data:`Patch`: a small,
reviewable edit to a persistent graph. Each call applies the patch to the graph carried in
tool state, validates the result (an invalid workflow is rejected before anything runs),
executes it incrementally (only the dirty subgraph recomputes), and returns the run report
plus a catalog of operators and reusable blocks the agent can wire next.

The graph persists across turns in :class:`GraphPatchState` and is mirrored to
``<session_dir>/graph/`` for audit and resume. Because the tool's permission is ``ASK``,
the human reviews the typed diff at the approval gate before it is applied.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from vibe.core.graph.blocks import (
    BlockError,
    expand,
    fold_report,
    get_block,
    is_block,
    registered_blocks,
)
from vibe.core.graph.cache import CacheStore

# Importing the demo module registers a usable operator library + the `margin_brief` block.
# A real deployment would register its own operator library at startup instead.
import vibe.core.graph.demo.blocks  # noqa: F401
from vibe.core.graph.executor import GraphValidationError, execute, validate
from vibe.core.graph.model import Graph, NodeId, Patch, Value
from vibe.core.graph.operators import registered_operators
from vibe.core.graph.patch import PatchError, apply_patch, changed_nodes
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolResultEvent

_MAX_OUTPUT_CHARS = 800


class GraphPatchArgs(BaseModel):
    patch: Patch = Field(
        description="Ordered list of typed edits to apply to the current workflow graph.",
    )
    reset: bool = Field(
        default=False,
        description="Discard the current graph and apply this patch to an empty one. Use "
        "to start a fresh, unrelated workflow instead of tearing down existing nodes.",
    )


class GraphPatchResult(BaseModel):
    applied: bool
    added: list[str] = Field(default_factory=list)
    changed: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    fresh: list[str] = Field(default_factory=list)
    cached: list[str] = Field(default_factory=list)
    outputs: dict[str, str] = Field(default_factory=dict)
    catalog: str = ""


class GraphPatchConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class GraphPatchState(BaseToolState):
    graph_json: str = ""


class GraphPatch(
    BaseTool[GraphPatchArgs, GraphPatchResult, GraphPatchConfig, GraphPatchState],
    ToolUIData[GraphPatchArgs, GraphPatchResult],
):
    description: ClassVar[str] = (
        "Edit the persistent workflow graph by applying a typed patch (add/remove nodes, "
        "set params, connect/disconnect ports). The patch is validated, then the graph is "
        "executed incrementally — only nodes whose inputs changed recompute. Returns which "
        "nodes ran vs. were cached, terminal outputs, and the catalog of available "
        "operators and reusable blocks. This is your only action for building a workflow."
    )

    @classmethod
    def get_status_text(cls) -> str:
        return "Updating workflow graph"

    @classmethod
    def format_call_display(cls, args: GraphPatchArgs) -> ToolCallDisplay:
        kinds = ", ".join(op.kind for op in args.patch) or "no-op"
        return ToolCallDisplay(summary=f"Patch graph: {kinds}")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if event.error:
            return ToolResultDisplay(success=False, message=event.error)
        if not isinstance(event.result, GraphPatchResult):
            return ToolResultDisplay(success=True, message="Graph updated")
        r = event.result
        return ToolResultDisplay(
            success=True,
            message=f"{len(r.fresh)} ran, {len(r.cached)} cached",
        )

    async def run(
        self, args: GraphPatchArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[GraphPatchResult, None]:
        graph_dir = self._graph_dir(ctx)

        if args.reset:
            # Start a fresh workflow — ignore any prior graph.
            current = Graph()
        else:
            # Prefer in-memory state; fall back to the disk mirror so an agent-authored
            # graph resumes after a restart (fresh tool instance with empty state).
            graph_json = self.state.graph_json
            if not graph_json:
                mirror = graph_dir / "graph.json"
                if mirror.exists():
                    graph_json = mirror.read_text()
                    self.state.graph_json = graph_json
            current = Graph.model_validate_json(graph_json) if graph_json else Graph()

        # Apply the patch (pure) and reject an invalid result before executing anything.
        try:
            new = apply_patch(current, args.patch)
        except PatchError as exc:
            raise ToolError(f"invalid patch: {exc}") from exc
        try:
            expanded, fold = expand(new)
            validate(expanded)
        except (GraphValidationError, BlockError, KeyError) as exc:
            raise ToolError(f"patch produces an invalid graph: {exc}") from exc

        delta = changed_nodes(current, new)

        cache = CacheStore(graph_dir / "cache.sqlite")
        try:
            try:
                values, report = await execute(expanded, cache, expand_blocks=False)
            except Exception as exc:  # operator raised at runtime — surface as recoverable
                raise ToolError(f"graph execution failed: {exc}") from exc
            folded = fold_report(report, fold)
            outputs = self._collect_outputs(new, values, cache)
        finally:
            cache.close()

        # Persist the new graph to state (across turns) and disk (audit / resume).
        self.state.graph_json = new.model_dump_json()
        (graph_dir / "graph.json").write_text(self.state.graph_json)

        yield GraphPatchResult(
            applied=True,
            added=delta["added"],
            changed=delta["changed"],
            removed=delta["removed"],
            fresh=folded.fresh(),
            cached=folded.cached(),
            outputs=outputs,
            catalog=_render_catalog(),
        )

    @staticmethod
    def _graph_dir(ctx: InvokeContext | None) -> Path:
        base = (ctx.session_dir or ctx.scratchpad_dir) if ctx else None
        if base is None:
            raise ToolError("graph_patch requires a session or scratchpad directory")
        graph_dir = base / "graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        return graph_dir

    @staticmethod
    def _collect_outputs(
        graph: Graph, values: dict[NodeId, Value], cache: CacheStore
    ) -> dict[str, str]:
        """Rehydrate the terminal nodes' payloads (truncated) for the agent to inspect."""
        consumed = {dep for node in graph.nodes.values() for dep in node.inputs.values()}
        outputs: dict[str, str] = {}
        for nid, node in graph.nodes.items():
            if nid in consumed:
                continue
            producing = f"{nid}/{get_block(node.op).output}" if is_block(node.op) else nid
            value = values.get(producing)
            if value is None:
                continue
            payload = cache.get(value.fingerprint)
            if payload is not None:
                outputs[nid] = payload.decode()[:_MAX_OUTPUT_CHARS]
        return outputs


def _render_catalog() -> str:
    """Markdown list of available operators and blocks, injected into each result."""
    lines = ["Available operators (inputs are wired from nodes; params are literals):"]
    for name, spec in sorted(registered_operators().items()):
        inputs = ", ".join(spec.input_names) or "—"
        params = ", ".join(spec.literal_names()) or "—"
        lines.append(f"- {name}(inputs: {inputs}; params: {params}) → {spec.result_type.__name__}")
    lines.append("")
    lines.append("Available blocks (reusable subgraphs — inputs / params):")
    for name, block in sorted(registered_blocks().items()):
        ports = ", ".join(block.input_ports)
        params = ", ".join(block.params)
        lines.append(f"- {name}(inputs: {ports}; params: {params})")
    return "\n".join(lines)
