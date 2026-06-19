from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, final

from pydantic import BaseModel, Field

from vibe.core.lsp import LSPMode, get_lsp_manager
from vibe.core.lsp._diagnostics import format_diagnostic_line, severity_name
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


class LspDiagnosticsArgs(BaseModel):
    file_path: str = Field(description="Absolute path to the file to check.")


class LspDiagnosticsResult(BaseModel):
    file_path: str
    diagnostics: list[dict[str, Any]]
    error_count: int
    warning_count: int
    formatted: str


class LspDiagnosticsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


@final
class LspDiagnostics(
    BaseTool[LspDiagnosticsArgs, LspDiagnosticsResult, LspDiagnosticsConfig, BaseToolState],
    ToolUIData[LspDiagnosticsArgs, LspDiagnosticsResult],
):
    description: ClassVar[str] = (
        "Get type errors, syntax errors, and warnings from the language server for a file."
    )

    def get_tool_call_display(self, args: LspDiagnosticsArgs) -> ToolCallDisplay:
        return ToolCallDisplay(
            summary=f"Checking diagnostics: {Path(args.file_path).name}",
            content=args.file_path,
        )

    def get_tool_result_display(self, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, LspDiagnosticsResult):
            r = event.result
            suffix = f"{r.error_count} errors, {r.warning_count} warnings"
            return ToolResultDisplay(message=suffix, content=r.formatted)
        return ToolResultDisplay(message="")

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        if config is None:
            return False
        return config.lsp.mode != LSPMode.OFF and bool(config.lsp.servers)

    async def run(
        self, args: LspDiagnosticsArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | LspDiagnosticsResult, None]:
        lsp = get_lsp_manager()
        if lsp is None:
            raise ToolError("LSP manager not started. Configure lsp.servers in config.toml.")

        client = lsp.get_client_for_file(args.file_path)
        if client is None:
            ext = Path(args.file_path).suffix
            raise ToolError(
                f"No LSP server configured for '{ext}' files. "
                f"Add a server entry to [[lsp.servers]] in your config."
            )

        await lsp.open_or_sync_file(args.file_path)
        diagnostics = await lsp.fetch_diagnostics(args.file_path)

        lsp_cfg = None
        if ctx and hasattr(ctx, "config"):
            lsp_cfg = getattr(ctx.config, "lsp", None)
        min_sev = 2
        if lsp_cfg:
            from vibe.core.lsp._diagnostics import _MIN_SEVERITY_THRESHOLDS
            min_sev = _MIN_SEVERITY_THRESHOLDS.get(lsp_cfg.min_severity, 2)

        filtered = [d for d in diagnostics if d.get("severity", 2) <= min_sev]

        errors = [d for d in filtered if d.get("severity", 2) == 1]
        warnings = [d for d in filtered if d.get("severity", 2) == 2]

        limit = lsp_cfg.max_diagnostics_shown if lsp_cfg else 20

        shown = filtered[:limit]
        lines = [format_diagnostic_line(d) for d in shown]
        if len(filtered) > limit:
            lines.append(f"  +{len(filtered) - limit} more (use /diag to see all)")

        formatted = "\n".join(lines) if lines else "No issues found."

        lsp.update_last_diagnostics(args.file_path, diagnostics)

        yield LspDiagnosticsResult(
            file_path=args.file_path,
            diagnostics=filtered,
            error_count=len(errors),
            warning_count=len(warnings),
            formatted=formatted,
        )
