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


def _count_edits(workspace_edit: dict[str, Any]) -> tuple[list[str], int]:
    files_changed: list[str] = []
    edit_count = 0
    changes = workspace_edit.get("changes", {})
    for uri, edits in changes.items():
        files_changed.append(uri_to_path(uri))
        edit_count += len(edits)
    doc_changes = workspace_edit.get("documentChanges", [])
    for change in doc_changes:
        if isinstance(change, dict):
            uri = change.get("textDocument", {}).get("uri", "")
            path = uri_to_path(uri)
            if path not in files_changed:
                files_changed.append(path)
            edits = change.get("edits", [])
            edit_count += len(edits)
    return files_changed, edit_count


class LspRenameArgs(BaseModel):
    file_path: str = Field(description="Absolute path to the file containing the symbol.")
    line: int = Field(description="1-indexed line number.", ge=1)
    character: int = Field(description="0-indexed character offset.", ge=0)
    new_name: str = Field(description="The new name for the symbol.")


class LspRenameResult(BaseModel):
    file_path: str
    old_position: str
    new_name: str
    files_changed: list[str]
    edit_count: int
    workspace_edit: dict[str, Any]


class LspRenameConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


@final
class LspRename(
    BaseTool[LspRenameArgs, LspRenameResult, LspRenameConfig, BaseToolState],
    ToolUIData[LspRenameArgs, LspRenameResult],
):
    description: ClassVar[str] = (
        "Compute the rename workspace-edit for a symbol. "
        "Returns a WorkspaceEdit that you must apply using the edit tool. "
        "Use lsp_references first to understand the blast radius."
    )

    def get_tool_call_display(self, args: LspRenameArgs) -> ToolCallDisplay:
        return ToolCallDisplay(
            summary=f"Rename at {Path(args.file_path).name}:{args.line}:{args.character} → {args.new_name}",
            content=args.file_path,
        )

    def get_tool_result_display(self, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, LspRenameResult):
            r = event.result
            n_files = len(r.files_changed)
            return ToolResultDisplay(
                message=f"{r.edit_count} edit{'s' if r.edit_count != 1 else ''} across {n_files} file{'s' if n_files != 1 else ''}"
            )
        return ToolResultDisplay(message="")

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        if config is None:
            return False
        return config.lsp.mode != LSPMode.OFF and bool(config.lsp.servers)

    async def run(
        self, args: LspRenameArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | LspRenameResult, None]:
        lsp = get_lsp_manager()
        if lsp is None:
            raise ToolError("LSP manager not started.")

        client = lsp.get_client_for_file(args.file_path)
        if client is None:
            raise ToolError(f"No LSP server for '{Path(args.file_path).suffix}' files.")

        await lsp.open_or_sync_file(args.file_path)
        uri = lsp.path_to_uri(args.file_path)

        try:
            result = await client.request("textDocument/rename", {
                "textDocument": {"uri": uri},
                "position": {"line": args.line - 1, "character": args.character},
                "newName": args.new_name,
            })
        except (LSPError, TimeoutError) as exc:
            raise ToolError(f"LSP rename failed: {exc}") from exc

        if result is None:
            raise ToolError("The language server returned no rename edits. The symbol may not be renameable at this position.")

        files_changed, edit_count = _count_edits(result)

        yield LspRenameResult(
            file_path=args.file_path,
            old_position=f"{args.file_path}:{args.line}:{args.character}",
            new_name=args.new_name,
            files_changed=files_changed,
            edit_count=edit_count,
            workspace_edit=result,
        )
