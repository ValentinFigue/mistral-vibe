from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, final

from pydantic import BaseModel, Field

from vibe.core.lsp import LSPMode, get_lsp_manager
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


class LspHoverArgs(BaseModel):
    file_path: str = Field(description="Absolute path to the file.")
    line: int = Field(description="1-indexed line number.", ge=1)
    character: int = Field(description="0-indexed character offset.", ge=0)


class LspHoverResult(BaseModel):
    file_path: str
    line: int
    character: int
    contents: str | None


class LspHoverConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


@final
class LspHover(
    BaseTool[LspHoverArgs, LspHoverResult, LspHoverConfig, BaseToolState],
    ToolUIData[LspHoverArgs, LspHoverResult],
):
    description: ClassVar[str] = (
        "Get type information or documentation for the symbol at a given position."
    )

    def get_tool_call_display(self, args: LspHoverArgs) -> ToolCallDisplay:
        return ToolCallDisplay(
            summary=f"Hover: {Path(args.file_path).name}:{args.line}:{args.character}",
            content=args.file_path,
        )

    def get_tool_result_display(self, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, LspHoverResult):
            contents = event.result.contents or "(no info)"
            return ToolResultDisplay(message=contents[:120])
        return ToolResultDisplay(message="")

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        if config is None:
            return False
        return config.lsp.mode != LSPMode.OFF and bool(config.lsp.servers)

    async def run(
        self, args: LspHoverArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | LspHoverResult, None]:
        lsp = get_lsp_manager()
        if lsp is None:
            raise ToolError("LSP manager not started.")

        client = lsp.get_client_for_file(args.file_path)
        if client is None:
            raise ToolError(f"No LSP server for '{Path(args.file_path).suffix}' files.")

        await lsp.open_or_sync_file(args.file_path)
        uri = lsp.path_to_uri(args.file_path)

        try:
            result = await client.request("textDocument/hover", {
                "textDocument": {"uri": uri},
                "position": {"line": args.line - 1, "character": args.character},
            })
        except (LSPError, TimeoutError) as exc:
            raise ToolError(f"LSP hover failed: {exc}") from exc

        contents: str | None = None
        if result:
            raw = result.get("contents")
            if isinstance(raw, str):
                contents = raw
            elif isinstance(raw, dict):
                contents = raw.get("value")
            elif isinstance(raw, list):
                parts = []
                for item in raw:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        parts.append(item.get("value", ""))
                contents = "\n".join(p for p in parts if p)

        yield LspHoverResult(
            file_path=args.file_path,
            line=args.line,
            character=args.character,
            contents=contents,
        )
