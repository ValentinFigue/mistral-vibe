from __future__ import annotations

import asyncio
from collections.abc import Callable
import hashlib
from pathlib import Path
import shutil
from typing import Any

from vibe.core.logger import logger
from vibe.core.lsp._client import AsyncLSPClient, LSPError, ServerStatus
from vibe.core.lsp._config import LSPConfig, LSPMode, LSPServerConfig
from vibe.core.lsp._diagnostics import DiagnosticDelta, compute_delta


class LSPManager:
    """Manages the lifecycle of LSP server processes for a session."""

    def __init__(self) -> None:
        self._clients: dict[str, AsyncLSPClient] = {}
        self._config: LSPConfig | None = None
        self._workspace_root: Path = Path.cwd()
        self._open_files: dict[str, tuple[int, bytes]] = {}
        self._last_diagnostics: dict[str, list[dict[str, Any]]] = {}
        self._status_listeners: list[Callable[[str, ServerStatus], None]] = []

    def add_status_listener(self, fn: Callable[[str, ServerStatus], None]) -> None:
        self._status_listeners.append(fn)

    def remove_status_listener(self, fn: Callable[[str, ServerStatus], None]) -> None:
        self._status_listeners.discard(fn) if hasattr(self._status_listeners, "discard") else None
        try:
            self._status_listeners.remove(fn)
        except ValueError:
            pass

    def _on_server_status_change(self, name: str) -> Callable[[ServerStatus], None]:
        def handler(status: ServerStatus) -> None:
            for listener in self._status_listeners:
                try:
                    listener(name, status)
                except Exception:
                    pass
        return handler

    async def start_all(self, config: LSPConfig, workspace_root: Path) -> None:
        self._config = config
        self._workspace_root = workspace_root

        servers = config.active_servers()
        if not servers:
            return

        tasks = [
            asyncio.create_task(
                self._start_one(server, workspace_root),
                name=f"lsp-start-{server.name}",
            )
            for server in servers
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        ready = [name for name, c in self._clients.items() if c.status == ServerStatus.READY]
        failed = [name for name, c in self._clients.items() if c.status == ServerStatus.FAILED]
        if ready:
            logger.info(f"LSP: started {', '.join(ready)}")
        if failed:
            logger.warning(f"LSP: failed to start {', '.join(failed)}")

    async def _start_one(self, server: LSPServerConfig, workspace_root: Path) -> None:
        if not _check_binary(server.command[0]):
            cmd = server.command[0]
            logger.warning(
                f"LSP: '{cmd}' not found on PATH for server '{server.name}'. "
                f"Install it to enable LSP support."
            )
            client = AsyncLSPClient(
                name=server.name,
                command=server.command,
                on_status_change=self._on_server_status_change(server.name),
            )
            client._set_status(ServerStatus.FAILED)
            self._clients[server.name] = client
            return

        workspace = Path(server.workspace_root) if server.workspace_root else workspace_root
        client = AsyncLSPClient(
            name=server.name,
            command=server.command,
            env_overrides=server.env,
            startup_timeout_sec=server.startup_timeout_sec,
            request_timeout_sec=server.request_timeout_sec,
            on_status_change=self._on_server_status_change(server.name),
        )
        self._clients[server.name] = client
        try:
            await client.start(workspace)
        except (FileNotFoundError, TimeoutError) as exc:
            logger.warning(f"LSP: {exc}")
        except Exception as exc:
            logger.warning(f"LSP: server '{server.name}' failed to start: {exc}")

    async def stop_all(self) -> None:
        tasks = [
            asyncio.create_task(client.stop(), name=f"lsp-stop-{name}")
            for name, client in self._clients.items()
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._clients.clear()
        self._open_files.clear()

    async def restart_server(self, name: str) -> None:
        client = self._clients.get(name)
        if client is None:
            logger.warning(f"LSP: unknown server '{name}'")
            return

        logger.info(f"LSP: restarting {name}…")
        await client.stop()

        server_cfg = self._find_server_config(name)
        if server_cfg is None or self._config is None:
            return

        previously_open = {
            uri: data
            for uri, data in self._open_files.items()
            if self._get_client_for_uri(uri) is None or self._get_client_for_uri(uri).name == name
        }
        for uri in list(previously_open.keys()):
            self._open_files.pop(uri, None)

        await self._start_one(server_cfg, self._workspace_root)

        for uri in previously_open:
            path = _uri_to_path(uri)
            if path:
                try:
                    await self.open_or_sync_file(path)
                except Exception:
                    pass

    async def auto_restart_if_crashed(self, name: str) -> None:
        logger.warning(f"LSP: {name} crashed, restarting…")
        await self.restart_server(name)

    def get_client_for_file(self, path: str) -> AsyncLSPClient | None:
        ext = Path(path).suffix.lower()
        if not self._config:
            return None
        for server in self._config.active_servers():
            if ext in [e.lower() for e in server.extensions]:
                client = self._clients.get(server.name)
                if client and client.status in (ServerStatus.READY, ServerStatus.INDEXING):
                    return client
        return None

    def effective_mode_for_file(self, path: str) -> LSPMode:
        if not self._config:
            return LSPMode.OFF
        ext = Path(path).suffix.lower()
        for server in self._config.active_servers():
            if ext in [e.lower() for e in server.extensions]:
                if server.mode_override is not None:
                    return server.mode_override
        return self._config.mode

    async def open_or_sync_file(self, path: str) -> None:
        client = self.get_client_for_file(path)
        if client is None:
            return

        file_path = Path(path).resolve()
        if not file_path.exists():
            return

        content_bytes = file_path.read_bytes()
        content_hash = hashlib.md5(content_bytes).digest()
        uri = file_path.as_uri()

        if uri in self._open_files:
            version, last_hash = self._open_files[uri]
            if last_hash == content_hash:
                return
            new_version = version + 1
            self._open_files[uri] = (new_version, content_hash)
            await client.did_change(uri, content_bytes.decode("utf-8", errors="replace"), new_version)
        else:
            lang_id = self._detect_language_id(path)
            self._open_files[uri] = (1, content_hash)
            await client.did_open(uri, lang_id, content_bytes.decode("utf-8", errors="replace"), 1)

    async def close_file(self, path: str) -> None:
        file_path = Path(path).resolve()
        uri = file_path.as_uri()
        if uri not in self._open_files:
            return
        self._open_files.pop(uri)
        client = self.get_client_for_file(path)
        if client:
            await client.did_close(uri)

    def get_last_diagnostics(self, path: str) -> list[dict[str, Any]]:
        uri = Path(path).resolve().as_uri()
        return self._last_diagnostics.get(uri, [])

    def update_last_diagnostics(self, path: str, diagnostics: list[dict[str, Any]]) -> None:
        uri = Path(path).resolve().as_uri()
        self._last_diagnostics[uri] = diagnostics

    async def fetch_diagnostics(self, path: str) -> list[dict[str, Any]]:
        """Fetch diagnostics for a file, using pull or push model as appropriate."""
        client = self.get_client_for_file(path)
        if client is None:
            return []

        uri = Path(path).resolve().as_uri()

        if client.supports_pull_diagnostics():
            try:
                result = await client.request(
                    "textDocument/diagnostic",
                    {
                        "textDocument": {"uri": uri},
                    },
                )
                if result and "items" in result:
                    return result["items"]
                elif result and isinstance(result, list):
                    return result
            except (LSPError, asyncio.TimeoutError):
                pass

        # Fallback: pushed diagnostics from cache
        # Give the server a moment to push after did_open/did_change
        await asyncio.sleep(0.3)
        return client.get_diagnostics_cache(uri)

    def get_client_status(self, name: str) -> ServerStatus | None:
        client = self._clients.get(name)
        return client.status if client else None

    def status_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "status": client.status,
                "extensions": self._get_server_extensions(name),
            }
            for name, client in self._clients.items()
        ]

    def path_to_uri(self, path: str) -> str:
        return Path(path).resolve().as_uri()

    def _detect_language_id(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        if not self._config:
            return "plaintext"
        for server in self._config.active_servers():
            if ext in [e.lower() for e in server.extensions]:
                return server.language_ids[0] if server.language_ids else "plaintext"
        return "plaintext"

    def _find_server_config(self, name: str) -> LSPServerConfig | None:
        if not self._config:
            return None
        for s in self._config.servers:
            if s.name == name:
                return s
        return None

    def _get_server_extensions(self, name: str) -> list[str]:
        cfg = self._find_server_config(name)
        return cfg.extensions if cfg else []

    def _get_client_for_uri(self, uri: str) -> AsyncLSPClient | None:
        path = _uri_to_path(uri)
        if path:
            return self.get_client_for_file(path)
        return None


def _check_binary(name: str) -> bool:
    return shutil.which(name) is not None


def _uri_to_path(uri: str) -> str | None:
    if uri.startswith("file://"):
        from urllib.parse import unquote
        path = uri[7:]
        if path.startswith("//"):
            path = path[2:]
        return unquote(path)
    return None
