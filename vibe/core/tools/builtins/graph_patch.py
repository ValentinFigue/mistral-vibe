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

from collections import Counter
from collections.abc import AsyncGenerator
from textwrap import indent
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

# Importing the demo modules registers a usable operator library for the agent: the
# weekly-margin pipeline + `margin_brief` block, and the analytics / research operator kits
# (larger, realistic workflows). A real deployment would register its own library at startup.
import vibe.core.graph.demo.analytics  # noqa: F401
import vibe.core.graph.demo.blocks  # noqa: F401
import vibe.core.graph.demo.research  # noqa: F401
from vibe.core.graph.executor import GraphValidationError, execute, validate
from vibe.core.graph.model import Graph, NodeId, Patch, Value
from vibe.core.graph.operators import registered_operators
from vibe.core.graph.patch import PatchError, apply_patch, changed_nodes
from vibe.core.graph.session_store import graph_dir as _session_graph_dir
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
_MAX_GLIMPSE_CHARS = 120


def _op_line(op: object) -> str:
    """A readable one-line description of a single patch op (for the approval preview)."""
    kind = getattr(op, "kind", "?")
    if kind == "add_node":
        return f"+ add {op.node.id} ({op.node.op})"  # type: ignore[attr-defined]
    if kind == "remove_node":
        return f"− remove {op.id}"  # type: ignore[attr-defined]
    if kind == "set_param":
        return f"~ set {op.id}.{op.key} = {op.value!r}"  # type: ignore[attr-defined]
    if kind == "connect":
        return f"→ connect {op.id}.{op.port} ← {op.source}"  # type: ignore[attr-defined]
    if kind == "disconnect":
        return f"⊘ disconnect {op.id}.{op.port}"  # type: ignore[attr-defined]
    return f"? {kind}"


class GraphPatchArgs(BaseModel):
    patch: Patch = Field(
        description="Ordered list of typed edits to apply to the current workflow graph.",
    )
    reset: bool = Field(
        default=False,
        description="Discard the current graph and apply this patch to an empty one. Use "
        "to start a fresh, unrelated workflow instead of tearing down existing nodes.",
    )


class OutputHandle(BaseModel):
    """A content-addressed reference to a terminal node's value.

    The payload lives in the cache, keyed by ``fingerprint``. ``preview`` is an excerpt of the
    terminal value (up to ``_MAX_OUTPUT_CHARS``) so the agent can actually read its final
    result; the context win comes from *intermediate* payloads never entering context, not from
    starving terminals — only terminal nodes get a handle.
    """

    type: str
    fingerprint: str
    preview: str  # excerpt of the terminal payload (up to _MAX_OUTPUT_CHARS)


class GraphPatchResult(BaseModel):
    applied: bool
    added: list[str] = Field(default_factory=list)
    changed: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    fresh: list[str] = Field(default_factory=list)
    cached: list[str] = Field(default_factory=list)
    outputs: dict[str, str] = Field(default_factory=dict)  # terminal node -> full-ish payload (UI)
    handles: dict[str, OutputHandle] = Field(default_factory=dict)  # terminal node -> handle (model)
    catalog: str = ""


class GraphPatchConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class GraphPatchState(BaseToolState):
    graph_json: str = ""
    catalog_shown: bool = False  # the full catalog is sent to the model once per session


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
        if not args.patch:
            return ToolCallDisplay(summary="Graph: read catalog (empty patch)")
        counts = Counter(op.kind for op in args.patch)
        summary = "Patch graph: " + ", ".join(f"{n}×{kind}" for kind, n in counts.items())
        lines = [_op_line(op) for op in args.patch]
        if args.reset:
            lines.insert(0, "reset (discard current graph)")
        return ToolCallDisplay(summary=summary, content="\n".join(lines))

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if event.error:
            return ToolResultDisplay(success=False, message=event.error)
        if not isinstance(event.result, GraphPatchResult):
            return ToolResultDisplay(success=True, message="Graph updated")
        r = event.result
        message = f"{len(r.fresh)} ran, {len(r.cached)} cached"
        # Glimpse of the terminal output(s), truncated (crit #6).
        preview = next(iter(r.outputs.values()), "")
        if preview:
            first_line = preview.replace("\n", " ").strip()[:_MAX_GLIMPSE_CHARS]
            message = f"{message} · {first_line}"
        return ToolResultDisplay(success=True, message=message)

    async def run(
        self, args: GraphPatchArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[GraphPatchResult, None]:
        try:
            graph_dir = _session_graph_dir(ctx)
        except ValueError as exc:
            raise ToolError(f"graph_patch requires a session directory: {exc}") from exc

        # An empty patch is the agent's explicit "re-list the catalog" request: force the
        # full catalog back into the model-facing content on this turn (see get_llm_content).
        if not args.patch:
            self.state.catalog_shown = False

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
            collected = self._collect_outputs(new, values, cache)
        finally:
            cache.close()

        outputs = {nid: text[:_MAX_OUTPUT_CHARS] for nid, (_, text) in collected.items()}
        handles = {
            nid: OutputHandle(
                type=value.type,
                fingerprint=value.fingerprint,
                preview=text.strip()[:_MAX_OUTPUT_CHARS],
            )
            for nid, (value, text) in collected.items()
        }

        # Persist the new graph to state (across turns) and disk (audit / resume), plus the
        # folded run report (authored ids) so `/graph` can show the actual last-run states.
        self.state.graph_json = new.model_dump_json()
        (graph_dir / "graph.json").write_text(self.state.graph_json)
        (graph_dir / "report.json").write_text(folded.model_dump_json())

        yield GraphPatchResult(
            applied=True,
            added=delta["added"],
            changed=delta["changed"],
            removed=delta["removed"],
            fresh=folded.fresh(),
            cached=folded.cached(),
            outputs=outputs,
            handles=handles,
            catalog=_render_catalog(),
        )

    def get_llm_content(self, result: GraphPatchResult) -> str | None:
        """Compact, handle-oriented text for the model — not the full field dump.

        The graph's payoff is that intermediate payloads live in the cache, not in context:
        the model reasons over content-addressed handles (``ref:<fp> <Type> — <preview>``)
        while the full tables stay out of the transcript. The heavy catalog is sent only on
        the first turn of a session (or when the agent asks to re-list via an empty patch);
        afterwards a one-line pointer stands in. The full result (payloads, catalog) still
        reaches the UI and the cache untouched — this only shapes the model-facing string.
        """
        if not result.applied:
            return None
        lines: list[str] = []
        delta = [
            f"{label} {ids}"
            for label, ids in (("added", result.added), ("changed", result.changed),
                               ("removed", result.removed))
            if ids
        ]
        lines.append(
            f"applied · {len(result.fresh)} ran, {len(result.cached)} cached"
            + ("; " + "; ".join(delta) if delta else "")
        )
        if result.handles:
            lines.append(
                "terminal outputs (ref = content-addressed cache handle; "
                "intermediate payloads stay in the cache, out of context):"
            )
            for nid, h in result.handles.items():
                lines.append(f"  {nid} → ref:{h.fingerprint[:12]} {h.type}")
                lines.append(indent(h.preview, "    "))
        if self.state.catalog_shown:
            lines.append("(catalog unchanged — send an empty patch to re-list operators/blocks)")
        else:
            lines.append("")
            lines.append(result.catalog)
            self.state.catalog_shown = True
        return "\n".join(lines)

    @staticmethod
    def _collect_outputs(
        graph: Graph, values: dict[NodeId, Value], cache: CacheStore
    ) -> dict[str, tuple[Value, str]]:
        """Rehydrate each terminal node's ``(handle, decoded payload)`` for the agent.

        Returns the :class:`Value` handle (type + fingerprint) alongside the full decoded
        payload so the caller can build both the UI preview and the model-facing handle
        without reading the cache twice.
        """
        consumed = {dep for node in graph.nodes.values() for dep in node.inputs.values()}
        outputs: dict[str, tuple[Value, str]] = {}
        for nid, node in graph.nodes.items():
            if nid in consumed:
                continue
            producing = f"{nid}/{get_block(node.op).output}" if is_block(node.op) else nid
            value = values.get(producing)
            if value is None:
                continue
            payload = cache.get(value.fingerprint)
            if payload is not None:
                outputs[nid] = (value, payload.decode())
        return outputs


def _render_catalog() -> str:
    """Compact operator + block catalog injected into each result (no descriptions)."""
    from vibe.core.graph import render

    return render.operators_catalog(registered_operators(), registered_blocks(), verbose=False)
