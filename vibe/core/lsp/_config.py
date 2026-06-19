from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class LSPMode(StrEnum):
    OFF = "off"
    MANUAL = "manual"
    AUTO = "auto"
    STRICT = "strict"


class LSPTrigger(StrEnum):
    ON_EDIT = "on-edit"
    ON_FINISH = "on-finish"


class LSPSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFORMATION = "information"
    HINT = "hint"


class LSPServerConfig(BaseModel):
    name: str = Field(description="Server identifier, e.g. 'pyright' or 'ts'.")
    command: list[str] = Field(
        description="Command and arguments to launch the server, e.g. ['pyright', '--stdio']."
    )
    language_ids: list[str] = Field(
        description="LSP language identifiers, e.g. ['python'] or ['typescript']."
    )
    extensions: list[str] = Field(
        description="File extensions handled by this server, e.g. ['.py'] or ['.ts', '.tsx']."
    )
    startup_timeout_sec: float = Field(
        default=10.0, description="Seconds to wait for the server to send its initialize response."
    )
    request_timeout_sec: float = Field(
        default=15.0, description="Seconds to wait for individual LSP requests."
    )
    enabled: bool = Field(default=True, description="Set false to disable without removing config.")
    workspace_root: str | None = Field(
        default=None,
        description="Override workspace root URI; defaults to cwd() at session start.",
    )
    initialization_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Passed verbatim as initializationOptions in the LSP initialize request.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables injected into the server process.",
    )
    mode_override: LSPMode | None = Field(
        default=None,
        description="Per-server mode; overrides the global lsp.mode for files this server handles.",
    )


class LSPConfig(BaseModel):
    mode: LSPMode = Field(
        default=LSPMode.MANUAL,
        description=(
            "LSP integration mode: "
            "'off' (disabled), "
            "'manual' (agent calls tools on demand), "
            "'auto' (diagnostics run after edits, results shown + fed to agent), "
            "'strict' (same as auto, new errors block the agent until fixed)."
        ),
    )
    trigger: LSPTrigger = Field(
        default=LSPTrigger.ON_EDIT,
        description=(
            "'on-edit': run diagnostics after each Edit/WriteFile tool call; "
            "'on-finish': run once before the agent declares the task done."
        ),
    )
    min_severity: LSPSeverity = Field(
        default=LSPSeverity.WARNING,
        description="Minimum severity level to show; 'error' mutes warnings/info/hints.",
    )
    max_diagnostics_shown: int = Field(
        default=20,
        description="Cap on diagnostics shown per file; excess shown as '+N more'.",
    )
    verbose: bool = Field(
        default=False,
        description="When true, inject brief LSP context into agent turns in auto/strict modes.",
    )
    servers: list[LSPServerConfig] = Field(
        default_factory=list,
        description="Language servers to start. Each maps file extensions to an LSP server command.",
    )

    def is_active(self) -> bool:
        return self.mode != LSPMode.OFF and bool(self.servers)

    def active_servers(self) -> list[LSPServerConfig]:
        return [s for s in self.servers if s.enabled]
