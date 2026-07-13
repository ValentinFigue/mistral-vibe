"""``graph_save_block`` — promote a subgraph the agent authored into a reusable block.

Reads the working graph mirrored by `graph_patch` at ``<session_dir>/graph/graph.json``,
derives a block interface from the selected nodes (ports from boundary inputs, output from
the terminal, params baked unless exposed), registers it for the rest of this session, and
writes it to ``~/.vibe/blocks/<name>.json`` so later sessions can wire it in by name.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import ClassVar

from pydantic import BaseModel, Field

from vibe.core.graph.blocks import (
    BlockError,
    block_from_subgraph,
    is_block,
    register_block,
    save_block,
)
from vibe.core.graph.model import Graph
from vibe.core.graph.session_store import graph_json_path
from vibe.core.paths import BLOCKS_DIR
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


class GraphSaveBlockArgs(BaseModel):
    name: str = Field(description="Block name (snake_case). Becomes its catalog + file name.")
    nodes: list[str] = Field(
        default_factory=list,
        description="Node ids to include. Empty = the whole current graph.",
    )
    expose_params: list[str] = Field(
        default_factory=list,
        description="Params to keep configurable, as 'node_id.key'. Others are baked in.",
    )
    output: str | None = Field(
        default=None,
        description="Output node id. Omit to infer the single terminal node of the selection.",
    )


class GraphSaveBlockResult(BaseModel):
    name: str
    input_ports: list[str] = Field(default_factory=list)
    params: list[str] = Field(default_factory=list)
    output: str = ""
    path: str = ""
    overwritten: bool = False


class GraphSaveBlockConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class GraphSaveBlock(
    BaseTool[GraphSaveBlockArgs, GraphSaveBlockResult, GraphSaveBlockConfig, BaseToolState],
    ToolUIData[GraphSaveBlockArgs, GraphSaveBlockResult],
):
    description: ClassVar[str] = (
        "Save part of the current workflow graph as a reusable, named block. Pick the nodes "
        "to include (or all of them); the block's inputs are the connections crossing the "
        "selection boundary and its output is the terminal node. Saved blocks persist and "
        "appear in the catalog so they can be wired into future graphs by name. Prefer "
        "excluding source nodes and exposing the params you want configurable."
    )

    @classmethod
    def get_status_text(cls) -> str:
        return "Saving reusable block"

    @classmethod
    def format_call_display(cls, args: GraphSaveBlockArgs) -> ToolCallDisplay:
        return ToolCallDisplay(summary=f"Save block: {args.name}")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if event.error:
            return ToolResultDisplay(success=False, message=event.error)
        if not isinstance(event.result, GraphSaveBlockResult):
            return ToolResultDisplay(success=True, message="Block saved")
        r = event.result
        verb = "Replaced" if r.overwritten else "Saved"
        return ToolResultDisplay(success=True, message=f"{verb} block {r.name}")

    async def run(
        self, args: GraphSaveBlockArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[GraphSaveBlockResult, None]:
        try:
            mirror = graph_json_path(ctx)
        except ValueError as exc:
            raise ToolError(f"graph_save_block requires a session directory: {exc}") from exc
        if not mirror.exists():
            raise ToolError("no graph to save yet — build one with graph_patch first")

        graph = Graph.model_validate_json(mirror.read_text())
        node_ids = args.nodes or list(graph.nodes)

        blocks_dir = BLOCKS_DIR.path
        file_path = blocks_dir / f"{args.name}.json"
        overwritten = file_path.exists()
        # A registered block with no backing file is a built-in/demo block — protect it.
        if is_block(args.name) and not overwritten:
            raise ToolError(f"{args.name!r} is a built-in block; choose another name")

        try:
            block = block_from_subgraph(
                args.name,
                graph,
                node_ids,
                expose_params=args.expose_params,
                output=args.output,
            )
        except BlockError as exc:
            raise ToolError(str(exc)) from exc

        register_block(block)
        path = save_block(block, blocks_dir)

        yield GraphSaveBlockResult(
            name=block.name,
            input_ports=sorted(block.input_ports),
            params=sorted(block.params),
            output=block.output,
            path=str(path),
            overwritten=overwritten,
        )
