from __future__ import annotations

from vibe.core.lsp._client import AsyncLSPClient, LSPError, ServerStatus
from vibe.core.lsp._config import LSPConfig, LSPMode, LSPSeverity, LSPServerConfig, LSPTrigger
from vibe.core.lsp._diagnostics import DiagnosticDelta, compute_delta, format_code_frame, format_diagnostic_line
from vibe.core.lsp._manager import LSPManager

__all__ = [
    "AsyncLSPClient",
    "DiagnosticDelta",
    "LSPConfig",
    "LSPError",
    "LSPManager",
    "LSPMode",
    "LSPServerConfig",
    "LSPSeverity",
    "LSPTrigger",
    "ServerStatus",
    "compute_delta",
    "format_code_frame",
    "format_diagnostic_line",
    "get_lsp_manager",
    "set_lsp_manager",
    "uri_to_path",
]


def uri_to_path(uri: str) -> str:
    """Convert a file:// URI to an absolute filesystem path."""
    from urllib.parse import unquote

    if uri.startswith("file://"):
        path = uri[7:]
        if path.startswith("//"):
            path = path[2:]
        return unquote(path)
    return uri

_current_manager: LSPManager | None = None


def get_lsp_manager() -> LSPManager | None:
    """Return the session-scoped LSPManager, or None if not started."""
    return _current_manager


def set_lsp_manager(manager: LSPManager | None) -> None:
    """Set (or clear) the session-scoped LSPManager."""
    global _current_manager
    _current_manager = manager
