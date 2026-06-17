from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import Static

if TYPE_CHECKING:
    from vibe.core.lsp._client import ServerStatus

_ICON = {
    "STARTING": "⟳",
    "INDEXING": "⟳",
    "READY": "✓",
    "FAILED": "✗",
    "STOPPED": "·",
}

_STYLE = {
    "STARTING": "dim",
    "INDEXING": "dim",
    "READY": "green",
    "FAILED": "red",
    "STOPPED": "dim",
}


class LSPStatusChip(Static):
    """Compact [py✓ ts⟳] status indicator for active LSP servers."""

    DEFAULT_CSS = """
    LSPStatusChip {
        margin: 0 1;
        padding: 0;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__("", **kwargs)
        self._statuses: dict[str, str] = {}
        self._listener = self._on_status_change

    def on_mount(self) -> None:
        from vibe.core.lsp import get_lsp_manager

        lsp = get_lsp_manager()
        if lsp is not None:
            for entry in lsp.status_summary():
                self._statuses[entry["name"]] = entry["status"].name
            lsp.add_status_listener(self._listener)
        self._refresh()

    def on_unmount(self) -> None:
        from vibe.core.lsp import get_lsp_manager

        lsp = get_lsp_manager()
        if lsp is not None:
            lsp.remove_status_listener(self._listener)

    def _on_status_change(self, name: str, status: ServerStatus) -> None:
        self._statuses[name] = status.name
        self.app.call_from_thread(self._refresh)

    def _refresh(self) -> None:
        if not self._statuses:
            self.update("")
            self.display = False
            return

        parts = []
        for name, status_name in sorted(self._statuses.items()):
            icon = _ICON.get(status_name, "?")
            style = _STYLE.get(status_name, "")
            short = name[:2]
            parts.append(f"[{style}]{short}{icon}[/{style}]")

        self.update(f"[dim][[/dim]{' '.join(parts)}[dim]][/dim]")
        self.display = True
