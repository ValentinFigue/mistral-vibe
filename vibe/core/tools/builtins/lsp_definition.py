from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, final
from pydantic import BaseModel, Field

from vibe.core.lsp import LSPMode, get_lsp_manager, uri_to_path
from vibe.core.lsp._client import LSPError
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolStreamEvent

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig
    from vibe.core.types import ToolResultEvent


def _location_to_dict(loc: dict[str, Any]) -> dict[str, Any]:
    start = loc.get("range", {}).get("start", {})
    return {
        "file": uri_to_path(loc.get("uri", "")),
        "line": start.get("line", 0) + 1,
        "character": start.get("character", 0),
    }


class LspDefinitionArgs(BaseModel):
    file_path: str = Field(description="Absolute path to the file.")
    line: int = Field(description="1-indexed line number.", ge=1)
    character: int = Field(description="0-indexed character offset.", ge=0)


class LspDefinitionResult(BaseModel):
    file_path: str
    line: int
    character: int
    locations: list[dict[str, Any]]


class LspDefinitionConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


@final
class LspDefinition(
    BaseTool[LspDefinitionArgs, LspDefinitionResult, LspDefinitionConfig, BaseToolState],
    ToolUIData[LspDefinitionArgs, LspDefinitionResult],
):
    description: ClassVar[str] = (
        "Go to the definition of the symbol at a given position."
    )

    def get_tool_call_display(self, args: LspDefinitionArgs) -> ToolCallDisplay:
        return ToolCallDisplay(
            summary=f"Definition: {Path(args.file_path).name}:{args.line}:{args.character}",
            content=args.file_path,
        )

    def get_tool_result_display(self, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, LspDefinitionResult):
            locs = event.result.locations
            if locs:
                first = locs[0]
                return ToolResultDisplay(message=f"{first['file']}:{first['line']}")
            return ToolResultDisplay(message="No definition found")
        return ToolResultDisplay(message="")

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        if config is None:
            return False
        return config.lsp.mode != LSPMode.OFF and bool(config.lsp.servers)

    async def run(
        self, args: LspDefinitionArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | LspDefinitionResult, None]:
        lsp = get_lsp_manager()
        if lsp is None:
            raise ToolError("LSP manager not started.")

        client = lsp.get_client_for_file(args.file_path)
        if client is None:
            raise ToolError(f"No LSP server for '{Path(args.file_path).suffix}' files.")

        await lsp.open_or_sync_file(args.file_path)
        uri = lsp.path_to_uri(args.file_path)

        try:
            result = await client.request("textDocument/definition", {
                "textDocument": {"uri": uri},
                "position": {"line": args.line - 1, "character": args.character},
            })
        except (LSPError, TimeoutError) as exc:
            raise ToolError(f"LSP definition failed: {exc}") from exc

        locations: list[dict[str, Any]] = []
        if isinstance(result, list):
            locations = [_location_to_dict(loc) for loc in result]
        elif isinstance(result, dict):
            locations = [_location_to_dict(result)]

        yield LspDefinitionResult(
            file_path=args.file_path,
            line=args.line,
            character=args.character,
            locations=locations,
        )
