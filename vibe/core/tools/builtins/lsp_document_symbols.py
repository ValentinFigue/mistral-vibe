from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, final

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


_SYMBOL_KIND_NAMES = {
    1: "File", 2: "Module", 3: "Namespace", 4: "Package", 5: "Class",
    6: "Method", 7: "Property", 8: "Field", 9: "Constructor", 10: "Enum",
    11: "Interface", 12: "Function", 13: "Variable", 14: "Constant",
    15: "String", 16: "Number", 17: "Boolean", 18: "Array", 19: "Object",
    20: "Key", 21: "Null", 22: "EnumMember", 23: "Struct", 24: "Event",
    25: "Operator", 26: "TypeParameter",
}


def _normalize_symbols(
    symbols: list[dict[str, Any]], depth: int = 0
) -> list[dict[str, Any]]:
    """Flatten DocumentSymbol[] or SymbolInformation[] to a uniform list."""
    result = []
    for sym in symbols:
        if "location" in sym:
            loc = sym["location"]
            start = loc.get("range", {}).get("start", {})
            entry = {
                "name": sym.get("name", ""),
                "kind": _SYMBOL_KIND_NAMES.get(sym.get("kind", 0), "Unknown"),
                "line": start.get("line", 0) + 1,
                "character": start.get("character", 0),
                "depth": depth,
            }
        else:
            start = sym.get("range", {}).get("start", {})
            entry = {
                "name": sym.get("name", ""),
                "kind": _SYMBOL_KIND_NAMES.get(sym.get("kind", 0), "Unknown"),
                "line": start.get("line", 0) + 1,
                "character": start.get("character", 0),
                "detail": sym.get("detail", ""),
                "depth": depth,
            }
        result.append(entry)
        children = sym.get("children", [])
        if children:
            result.extend(_normalize_symbols(children, depth + 1))
    return result


class LspDocumentSymbolsArgs(BaseModel):
    file_path: str = Field(description="Absolute path to the file.")


class LspDocumentSymbolsResult(BaseModel):
    file_path: str
    symbols: list[dict[str, Any]]
    symbol_count: int


class LspDocumentSymbolsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


@final
class LspDocumentSymbols(
    BaseTool[LspDocumentSymbolsArgs, LspDocumentSymbolsResult, LspDocumentSymbolsConfig, BaseToolState],
    ToolUIData[LspDocumentSymbolsArgs, LspDocumentSymbolsResult],
):
    description: ClassVar[str] = (
        "Get the outline of a file: all classes, functions, variables, and other symbols."
    )

    def get_tool_call_display(self, args: LspDocumentSymbolsArgs) -> ToolCallDisplay:
        return ToolCallDisplay(
            summary=f"Symbols: {Path(args.file_path).name}",
            content=args.file_path,
        )

    def get_tool_result_display(self, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, LspDocumentSymbolsResult):
            n = event.result.symbol_count
            return ToolResultDisplay(message=f"{n} symbol{'s' if n != 1 else ''}")
        return ToolResultDisplay(message="")

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        if config is None:
            return False
        return config.lsp.mode != LSPMode.OFF and bool(config.lsp.servers)

    async def run(
        self, args: LspDocumentSymbolsArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | LspDocumentSymbolsResult, None]:
        lsp = get_lsp_manager()
        if lsp is None:
            raise ToolError("LSP manager not started.")

        client = lsp.get_client_for_file(args.file_path)
        if client is None:
            raise ToolError(f"No LSP server for '{Path(args.file_path).suffix}' files.")

        await lsp.open_or_sync_file(args.file_path)
        uri = lsp.path_to_uri(args.file_path)

        try:
            result = await client.request("textDocument/documentSymbol", {
                "textDocument": {"uri": uri},
            })
        except (LSPError, TimeoutError) as exc:
            raise ToolError(f"LSP document symbols failed: {exc}") from exc

        symbols = _normalize_symbols(result or [])

        yield LspDocumentSymbolsResult(
            file_path=args.file_path,
            symbols=symbols,
            symbol_count=len(symbols),
        )
