from __future__ import annotations

import asyncio
from collections.abc import Callable
from enum import StrEnum
import json
import os
from pathlib import Path
import re
from typing import Any

from vibe.core.logger import logger


class ServerStatus(StrEnum):
    STARTING = "starting"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"
    STOPPED = "stopped"


_CONTENT_LENGTH_RE = re.compile(rb"Content-Length:\s*(\d+)", re.IGNORECASE)


class LSPError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class AsyncLSPClient:
    """Async LSP client using stdio transport and JSON-RPC 2.0."""

    def __init__(
        self,
        name: str,
        command: list[str],
        env_overrides: dict[str, str] | None = None,
        startup_timeout_sec: float = 10.0,
        request_timeout_sec: float = 15.0,
        on_status_change: Callable[[ServerStatus], None] | None = None,
    ) -> None:
        self._name = name
        self._command = command
        self._env_overrides = env_overrides or {}
        self._startup_timeout_sec = startup_timeout_sec
        self._request_timeout_sec = request_timeout_sec
        self._on_status_change = on_status_change

        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None

        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._request_id = 0

        self._status: ServerStatus = ServerStatus.STARTING
        self._server_capabilities: dict[str, Any] = {}

        self._diagnostics_cache: dict[str, list[dict[str, Any]]] = {}
        self._stop_requested = False

    @property
    def status(self) -> ServerStatus:
        return self._status

    @property
    def name(self) -> str:
        return self._name

    @property
    def server_capabilities(self) -> dict[str, Any]:
        return self._server_capabilities

    def _set_status(self, status: ServerStatus) -> None:
        if self._status != status:
            self._status = status
            if self._on_status_change:
                try:
                    self._on_status_change(status)
                except Exception:
                    pass

    async def start(self, workspace_root: Path) -> None:
        env = {**os.environ, **self._env_overrides}
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as exc:
            self._set_status(ServerStatus.FAILED)
            cmd = self._command[0]
            raise FileNotFoundError(
                f"{self._name}: '{cmd}' not found. "
                f"Install it first (e.g. 'npm i -g pyright' or 'npm i -g typescript-language-server typescript')."
            ) from exc

        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"lsp-reader-{self._name}"
        )
        self._stderr_task = asyncio.create_task(
            self._stderr_drain(), name=f"lsp-stderr-{self._name}"
        )

        self._set_status(ServerStatus.INDEXING)

        try:
            await asyncio.wait_for(
                self._initialize(workspace_root),
                timeout=self._startup_timeout_sec,
            )
        except asyncio.TimeoutError:
            self._set_status(ServerStatus.FAILED)
            raise TimeoutError(
                f"{self._name}: server did not respond within {self._startup_timeout_sec}s. "
                "Check that the server binary is correct and can reach the workspace."
            )

        self._set_status(ServerStatus.READY)

    async def stop(self) -> None:
        self._stop_requested = True
        self._set_status(ServerStatus.STOPPED)

        if self._process is None:
            return

        try:
            await asyncio.wait_for(
                self._graceful_shutdown(), timeout=3.0
            )
        except (asyncio.TimeoutError, Exception):
            if self._process.returncode is None:
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass

        for task in (self._reader_task, self._stderr_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    async def _graceful_shutdown(self) -> None:
        if self._process and self._process.returncode is None:
            try:
                await self._send({"jsonrpc": "2.0", "id": self._next_id(), "method": "shutdown", "params": None})
                await self._send({"jsonrpc": "2.0", "method": "exit", "params": None})
            except Exception:
                pass

    async def request(
        self,
        method: str,
        params: Any,
        *,
        timeout: float | None = None,
    ) -> Any:
        if (
            self._status in (ServerStatus.STOPPED, ServerStatus.FAILED)
            or self._process is None
            or self._process.returncode is not None
        ):
            raise LSPError(-32003, f"{self._name}: server is not running")

        req_id = self._next_id()
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[req_id] = future

        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params

        await self._send(msg)
        try:
            return await asyncio.wait_for(
                future, timeout=timeout or self._request_timeout_sec
            )
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise

    async def notify(self, method: str, params: Any) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._send(msg)

    async def did_open(self, uri: str, language_id: str, text: str, version: int = 1) -> None:
        await self.notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": language_id,
                "version": version,
                "text": text,
            }
        })

    async def did_change(self, uri: str, text: str, version: int) -> None:
        await self.notify("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": version},
            "contentChanges": [{"text": text}],
        })

    async def did_close(self, uri: str) -> None:
        await self.notify("textDocument/didClose", {
            "textDocument": {"uri": uri}
        })
        self._diagnostics_cache.pop(uri, None)

    def get_diagnostics_cache(self, uri: str) -> list[dict[str, Any]]:
        return self._diagnostics_cache.get(uri, [])

    def supports_pull_diagnostics(self) -> bool:
        return bool(self._server_capabilities.get("diagnosticProvider"))

    # ── internals ────────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send(self, message: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise LSPError(-32003, f"{self._name}: no stdin")
        body = json.dumps(message).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._process.stdin.write(header + body)
        await self._process.stdin.drain()

    async def _read_loop(self) -> None:
        assert self._process and self._process.stdout
        stdout = self._process.stdout
        try:
            while True:
                # Read headers until blank line
                header_lines: list[bytes] = []
                while True:
                    line = await stdout.readline()
                    if not line:
                        return
                    stripped = line.strip()
                    if not stripped:
                        break
                    header_lines.append(stripped)

                header_block = b"\r\n".join(header_lines)
                m = _CONTENT_LENGTH_RE.search(header_block)
                if not m:
                    logger.debug(f"lsp/{self._name}: malformed header, skipping")
                    continue

                length = int(m.group(1))
                body = await stdout.readexactly(length)
                try:
                    msg = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError:
                    logger.debug(f"lsp/{self._name}: JSON decode error")
                    continue

                self._dispatch(msg)
        except asyncio.IncompleteReadError:
            pass
        except Exception as exc:
            logger.debug(f"lsp/{self._name}: read loop error: {exc}")
        finally:
            if not self._stop_requested and self._status not in (
                ServerStatus.STOPPED, ServerStatus.FAILED
            ):
                self._set_status(ServerStatus.FAILED)
                logger.warning(f"lsp/{self._name}: server exited unexpectedly")

    def _dispatch(self, msg: dict[str, Any]) -> None:
        msg_id = msg.get("id")
        method = msg.get("method")

        if method == "textDocument/publishDiagnostics":
            params = msg.get("params", {})
            uri = params.get("uri", "")
            self._diagnostics_cache[uri] = params.get("diagnostics", [])
            return

        if method in ("$/progress", "window/logMessage", "window/showMessage"):
            if method == "$/progress":
                value = msg.get("params", {}).get("value", {})
                kind = value.get("kind")
                if kind == "end" and self._status == ServerStatus.INDEXING:
                    self._set_status(ServerStatus.READY)
            return

        if msg_id is not None and msg_id in self._pending:
            future = self._pending.pop(msg_id)
            if future.done():
                return
            if "error" in msg:
                err = msg["error"]
                future.set_exception(
                    LSPError(err.get("code", -1), err.get("message", "LSP error"), err.get("data"))
                )
            else:
                future.set_result(msg.get("result"))

    async def _stderr_drain(self) -> None:
        assert self._process and self._process.stderr
        try:
            async for line in self._process.stderr:
                if line:
                    logger.debug(f"lsp/{self._name}/stderr: {line.decode('utf-8', errors='replace').rstrip()}")
        except Exception:
            pass

    async def _initialize(self, workspace_root: Path) -> None:
        root_uri = workspace_root.resolve().as_uri()
        result = await self.request("initialize", {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "workspaceFolders": [{"uri": root_uri, "name": workspace_root.name}],
            "capabilities": {
                "textDocument": {
                    "synchronization": {
                        "dynamicRegistration": False,
                        "didSave": False,
                        "willSave": False,
                    },
                    "hover": {
                        "dynamicRegistration": False,
                        "contentFormat": ["markdown", "plaintext"],
                    },
                    "definition": {"dynamicRegistration": False},
                    "references": {"dynamicRegistration": False},
                    "documentSymbol": {
                        "dynamicRegistration": False,
                        "hierarchicalDocumentSymbolSupport": True,
                    },
                    "rename": {
                        "dynamicRegistration": False,
                        "prepareSupport": False,
                    },
                    "diagnostic": {
                        "dynamicRegistration": False,
                        "relatedDocumentSupport": False,
                    },
                    "publishDiagnostics": {
                        "relatedInformation": False,
                    },
                },
                "workspace": {
                    "applyEdit": False,
                    "workspaceFolders": True,
                },
            },
        }, timeout=self._startup_timeout_sec)

        if result:
            self._server_capabilities = result.get("capabilities", {})

        await self.notify("initialized", {})
