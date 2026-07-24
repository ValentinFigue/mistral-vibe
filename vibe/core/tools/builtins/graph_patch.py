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
from pathlib import Path
from textwrap import indent
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

from vibe.core.graph.blocks import (
    BlockError,
    expand,
    fold_report,
    get_block,
    is_block,
    registered_blocks,
)
from vibe.core.graph.cache import SHARED_CACHE_MAX_BYTES, CacheStore, open_shared_cache

# Registering the operator libraries the agents may use. The `analysis` library (data-analysis
# kit) is the `analyst` profile's scoped catalog; the demo margin/research kits round out the
# generic `graph` agent. A real deployment would register its own libraries at startup.
import vibe.core.graph.demo.blocks  # noqa: F401  (weekly-margin pipeline + blocks)
import vibe.core.graph.demo.research  # noqa: F401
from vibe.core.graph.executor import (
    GraphValidationError,
    coerce_params,
    execute,
    validate,
)
from vibe.core.graph.fingerprint import content_hash
import vibe.core.graph.library.analysis  # noqa: F401  (data-analysis operator kit + blocks)
import vibe.core.graph.library.doc_reviewer  # noqa: F401  (document-review operator kit + blocks)
import vibe.core.graph.library.teaching  # noqa: F401  (lesson-authoring operator kit + blocks)
from vibe.core.graph.model import Graph, NodeId, Patch, Value
from vibe.core.graph.operators import get_operator, is_registered, registered_operators
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

if TYPE_CHECKING:
    from vibe.core.llm.types import LLMCaller

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
    schemas: dict[str, str] = Field(default_factory=dict)  # added/changed table node -> "col:dtype … (N rows)"
    catalog: str = ""


class GraphPatchConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    # Catalog-scoping: when set (e.g. the `analyst` profile sets "analysis"), the agent only
    # sees — and may only reference — operators/blocks tagged with this library. None = generic
    # agent, full catalog (unchanged behavior).
    library: str | None = None


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

        llm = ctx.sampling_callback.complete_text if (ctx and ctx.sampling_callback) else None
        result = await build_result(new, current, graph_dir, self.config.library, llm=llm)
        # Persist the (autofilled) graph to in-memory state so it resumes across turns.
        self.state.graph_json = new.model_dump_json()
        yield result

    @staticmethod
    def _autofill_content_fp(graph: Graph) -> None:
        """For every file-source node, stamp ``content_fp = content_hash(path)``.

        Lets the agent supply only a ``path``; the tool fingerprints the file's bytes so an
        edited file re-fingerprints (and reruns) while an unchanged file stays a cache hit.
        """
        for nid, node in graph.nodes.items():
            if is_block(node.op) or not is_registered(node.op):
                continue
            spec = get_operator(node.op)
            if spec.reads_file is None:
                continue
            path = node.params.get(spec.reads_file)
            if not isinstance(path, str) or not path:
                raise ToolError(
                    f"node {nid!r} ({node.op}) needs a '{spec.reads_file}' file path param"
                )
            try:
                node.params["content_fp"] = content_hash(path)
            except OSError as exc:
                raise ToolError(f"node {nid!r}: cannot read file {path!r}: {exc}") from exc

    def get_llm_content(self, result: GraphPatchResult) -> str | None:
        """Compact, handle-oriented model text (handles + previews, catalog once per session)."""
        show_catalog = not self.state.catalog_shown
        text = format_llm_content(result, show_catalog=show_catalog)
        if text is not None and show_catalog:
            self.state.catalog_shown = True
        return text

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


def _visible(item_library: str | None, agent_library: str | None) -> bool:
    """Catalog visibility: the generic agent (``None``) sees all; a scoped agent sees only
    operators/blocks tagged with its own library (untagged items are generic-only).
    """
    return agent_library is None or item_library == agent_library


def _render_catalog(library: str | None = None) -> str:
    """Operator + block catalog injected into each result, scoped to ``library``.

    ``verbose=True`` so each op carries its one-line description — combined with the enum values now
    in ``arg_types`` (via ``Literal``), the agent sees each param's type, default, allowed values,
    and purpose, which is what it needs to author valid nodes.
    """
    from vibe.core.graph import render

    ops = {n: s for n, s in registered_operators().items() if _visible(s.library, library)}
    blocks = {n: b for n, b in registered_blocks().items() if _visible(b.library, library)}
    return render.operators_catalog(ops, blocks, verbose=True)


def _enforce_library(graph: Graph, library: str | None) -> None:
    """A scoped agent may only reference operators/blocks tagged with ``library``."""
    if library is None:
        return
    for nid, node in graph.nodes.items():
        if is_block(node.op):
            item_lib: str | None = get_block(node.op).library
        elif is_registered(node.op):
            item_lib = get_operator(node.op).library
        else:
            continue  # unknown op — let validate raise the clearer "unknown operator" error
        if item_lib != library:
            raise ToolError(
                f"operator {node.op!r} (node {nid!r}) is not in the {library!r} library; "
                "use only the operators and blocks shown in the catalog"
            )


def _collect_schemas(
    new: Graph, values: dict[NodeId, Value], cache: CacheStore, node_ids: set[str]
) -> dict[str, str]:
    """A compact ``col:dtype … (N rows)`` schema per Table node in ``node_ids`` (block-aware).

    Only the added/changed nodes are passed, so the agent sees the shape of what it just built —
    which is what stops it guessing column names / operator output shapes on the next turn. Cheap:
    column names + a bounded dtype sample (no rows retained); non-Table nodes are skipped.
    """
    from vibe.core.graph import render

    schemas: dict[str, str] = {}
    for nid in node_ids:
        node = new.nodes.get(nid)
        if node is None:
            continue
        eid = f"{nid}/{get_block(node.op).output}" if is_block(node.op) else nid
        value = values.get(eid)
        if value is None or value.type != "Table":
            continue
        payload = cache.get(value.fingerprint)
        if payload is not None and (sch := render.table_schema(payload.decode())):
            schemas[nid] = sch
    return schemas


async def build_result(
    new: Graph,
    current: Graph,
    graph_dir: Path,
    library: str | None,
    llm: LLMCaller | None = None,
) -> GraphPatchResult:
    """Validate + execute the authored graph incrementally and build the shared result.

    Shared by ``graph_patch`` (typed patches) and ``run_pipeline`` (the DSL): enforces the
    library scope, fingerprints file sources, diffs vs the prior graph, runs only the dirty
    subgraph, and returns handles + fresh/cached + the scoped catalog. Persists the graph +
    folded report to ``graph_dir`` (in-memory tool state is set by the caller).
    """
    _enforce_library(new, library)
    GraphPatch._autofill_content_fp(new)
    try:
        expanded, fold = expand(new)
        coerce_params(expanded)  # fix unambiguous type slips before fingerprint/validate
        validate(expanded)
    except (GraphValidationError, BlockError, KeyError) as exc:
        raise ToolError(f"invalid graph: {exc}") from exc

    delta = changed_nodes(current, new)
    cache = open_shared_cache()  # shared cross-session store under VIBE_HOME
    # Trim the shared store to its soft budget once per run (a cheap SUM when under budget), so
    # cross-session accumulation stays bounded. This run's fresh results are newest → not evicted.
    cache.evict_to(SHARED_CACHE_MAX_BYTES)
    try:
        try:
            values, report = await execute(expanded, cache, expand_blocks=False, llm=llm)
        except Exception as exc:  # operator raised at runtime — surface as recoverable
            raise ToolError(f"graph execution failed: {exc}") from exc
        folded = fold_report(report, fold)
        collected = GraphPatch._collect_outputs(new, values, cache)
        schemas = _collect_schemas(new, values, cache, set(delta["added"]) | set(delta["changed"]))
    finally:
        cache.close()

    outputs = {nid: text[:_MAX_OUTPUT_CHARS] for nid, (_, text) in collected.items()}
    handles = {
        nid: OutputHandle(
            type=value.type, fingerprint=value.fingerprint, preview=text.strip()[:_MAX_OUTPUT_CHARS]
        )
        for nid, (value, text) in collected.items()
    }
    (graph_dir / "graph.json").write_text(new.model_dump_json())
    (graph_dir / "report.json").write_text(folded.model_dump_json())
    return GraphPatchResult(
        applied=True,
        added=delta["added"], changed=delta["changed"], removed=delta["removed"],
        fresh=folded.fresh(), cached=folded.cached(),
        outputs=outputs, handles=handles, schemas=schemas, catalog=_render_catalog(library),
    )


def format_llm_content(result: GraphPatchResult, *, show_catalog: bool) -> str | None:
    """Compact, handle-oriented model text: a summary + terminal handles (ref + preview), with
    the catalog included only when ``show_catalog`` (once per session). Intermediate payloads
    stay in the cache, out of context. Shared by graph_patch and run_pipeline.
    """
    if not result.applied:
        return None
    delta = [
        f"{label} {ids}"
        for label, ids in (("added", result.added), ("changed", result.changed),
                           ("removed", result.removed))
        if ids
    ]
    lines = [
        f"applied · {len(result.fresh)} ran, {len(result.cached)} cached"
        + ("; " + "; ".join(delta) if delta else "")
    ]
    if result.schemas:
        # The shape of what was just built/changed, so the next step wires real columns (and the
        # agent rarely needs a separate graph_inspect just to learn columns/dtypes).
        lines.append("schema of new/changed steps (columns:dtype):")
        for nid, sch in result.schemas.items():
            lines.append(f"  {nid}: {sch}")
    if result.handles:
        lines.append(
            "terminal outputs (ref = content-addressed cache handle; "
            "intermediate payloads stay in the cache, out of context):"
        )
        for nid, h in result.handles.items():
            lines.append(f"  {nid} → ref:{h.fingerprint[:12]} {h.type}")
            lines.append(indent(h.preview, "    "))
    if show_catalog:
        lines.append("")
        lines.append(result.catalog)
    else:
        lines.append("(catalog unchanged — omitted to save context)")
    return "\n".join(lines)
