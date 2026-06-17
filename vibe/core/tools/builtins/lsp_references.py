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


class LspReferencesArgs(BaseModel):
    file_path: str = Field(description="Absolute path to the file.")
    line: int = Field(description="1-indexed line number.", ge=1)
    character: int = Field(description="0-indexed character offset.", ge=0)
    include_declaration: bool = Field(
        default=False, description="Include the symbol declaration itself in results."
    )


class LspReferencesResult(BaseModel):
    file_path: str
    line: int
    character: int
    reference_count: int
    references: list[dict[str, Any]]


class LspReferencesConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


@final
class LspReferences(
    BaseTool[LspReferencesArgs, LspReferencesResult, LspReferencesConfig, BaseToolState],
    ToolUIData[LspReferencesArgs, LspReferencesResult],
):
    description: ClassVar[str] = (
        "Find all references to the symbol at a given position across the workspace."
    )

    def get_tool_call_display(self, args: LspReferencesArgs) -> ToolCallDisplay:
        return ToolCallDisplay(
            summary=f"References: {Path(args.file_path).name}:{args.line}:{args.character}",
            content=args.file_path,
        )

    def get_tool_result_display(self, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, LspReferencesResult):
            n = event.result.reference_count
            return ToolResultDisplay(message=f"{n} reference{'s' if n != 1 else ''}")
        return ToolResultDisplay(message="")

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        if config is None:
            return False
        return config.lsp.mode != LSPMode.OFF and bool(config.lsp.servers)

    async def run(
        self, args: LspReferencesArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | LspReferencesResult, None]:
        lsp = get_lsp_manager()
        if lsp is None:
            raise ToolError("LSP manager not started.")

        client = lsp.get_client_for_file(args.file_path)
        if client is None:
            raise ToolError(f"No LSP server for '{Path(args.file_path).suffix}' files.")

        await lsp.open_or_sync_file(args.file_path)
        uri = lsp.path_to_uri(args.file_path)

        try:
            result = await client.request("textDocument/references", {
                "textDocument": {"uri": uri},
                "position": {"line": args.line - 1, "character": args.character},
                "context": {"includeDeclaration": args.include_declaration},
            })
        except (LSPError, TimeoutError) as exc:
            raise ToolError(f"LSP references failed: {exc}") from exc

        references: list[dict[str, Any]] = []
        if isinstance(result, list):
            for loc in result:
                start = loc.get("range", {}).get("start", {})
                references.append({
                    "file": uri_to_path(loc.get("uri", "")),
                    "line": start.get("line", 0) + 1,
                    "character": start.get("character", 0),
                })

        yield LspReferencesResult(
            file_path=args.file_path,
            line=args.line,
            character=args.character,
            reference_count=len(references),
            references=references,
        )
