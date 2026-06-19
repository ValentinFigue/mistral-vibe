from __future__ import annotations

"""Tests for AsyncLSPClient using a minimal in-process fake server."""

import asyncio
import json
import sys
import textwrap
from pathlib import Path

import pytest

from vibe.core.lsp._client import AsyncLSPClient, LSPError, ServerStatus


# ── minimal fake LSP server (runs as a subprocess) ───────────────────────────

_FAKE_SERVER_SRC = textwrap.dedent("""\
    import json, sys

    def _frame(body: bytes) -> bytes:
        return b"Content-Length: " + str(len(body)).encode() + b"\\r\\n\\r\\n" + body

    def _send(msg: dict) -> None:
        raw = json.dumps(msg).encode()
        sys.stdout.buffer.write(_frame(raw))
        sys.stdout.buffer.flush()

    def _read_message() -> dict | None:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        length = int(line.split(b":")[1].strip())
        sys.stdin.buffer.readline()          # blank line
        body = sys.stdin.buffer.read(length)
        return json.loads(body)

    while True:
        msg = _read_message()
        if msg is None:
            break
        method = msg.get("method", "")
        msg_id  = msg.get("id")

        if method == "initialize":
            _send({"jsonrpc": "2.0", "id": msg_id, "result": {
                "capabilities": {},
                "serverInfo": {"name": "fake", "version": "0.0.1"},
            }})
        elif method == "initialized":
            pass  # notification, no reply
        elif method == "shutdown":
            _send({"jsonrpc": "2.0", "id": msg_id, "result": None})
        elif method == "exit":
            sys.exit(0)
        elif msg_id is not None:
            # Echo any other request back as a result containing the method name.
            _send({"jsonrpc": "2.0", "id": msg_id, "result": {"echo": method}})
""")


def _fake_server_command() -> list[str]:
    return [sys.executable, "-c", _FAKE_SERVER_SRC]


# ── helpers ───────────────────────────────────────────────────────────────────

async def _make_ready_client(tmp_path: Path) -> AsyncLSPClient:
    client = AsyncLSPClient("fake", _fake_server_command())
    await client.start(tmp_path)
    return client


# ── lifecycle tests ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_client_reaches_ready_after_start(tmp_path: Path) -> None:
    client = await _make_ready_client(tmp_path)
    assert client.status == ServerStatus.READY
    await client.stop()


@pytest.mark.asyncio
async def test_client_reaches_stopped_after_stop(tmp_path: Path) -> None:
    client = await _make_ready_client(tmp_path)
    await client.stop()
    assert client.status == ServerStatus.STOPPED


@pytest.mark.asyncio
async def test_client_status_change_callback_fires(tmp_path: Path) -> None:
    transitions: list[ServerStatus] = []
    client = AsyncLSPClient(
        "fake",
        _fake_server_command(),
        on_status_change=transitions.append,
    )
    await client.start(tmp_path)
    await client.stop()
    assert ServerStatus.INDEXING in transitions
    assert ServerStatus.READY in transitions
    assert ServerStatus.STOPPED in transitions


@pytest.mark.asyncio
async def test_client_start_raises_on_missing_binary(tmp_path: Path) -> None:
    client = AsyncLSPClient("bad", ["__no_such_binary__"])
    with pytest.raises(FileNotFoundError, match="not found"):
        await client.start(tmp_path)
    assert client.status == ServerStatus.FAILED


# ── request / notify ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_client_request_returns_echo_result(tmp_path: Path) -> None:
    client = await _make_ready_client(tmp_path)
    result = await client.request("textDocument/hover", {"textDocument": {"uri": "file:///x.py"}})
    assert result == {"echo": "textDocument/hover"}
    await client.stop()


@pytest.mark.asyncio
async def test_client_notify_does_not_raise(tmp_path: Path) -> None:
    client = await _make_ready_client(tmp_path)
    # Notifications are fire-and-forget; the fake server ignores them silently.
    await client.notify("textDocument/didOpen", {
        "textDocument": {"uri": "file:///x.py", "languageId": "python", "version": 1, "text": "x = 1\n"}
    })
    await client.stop()


@pytest.mark.asyncio
async def test_client_request_raises_when_not_running(tmp_path: Path) -> None:
    client = await _make_ready_client(tmp_path)
    await client.stop()
    with pytest.raises(LSPError):
        await client.request("textDocument/hover", {})


# ── diagnostics cache ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_diagnostics_cache_empty_before_any_push(tmp_path: Path) -> None:
    client = await _make_ready_client(tmp_path)
    assert client.get_diagnostics_cache("file:///x.py") == []
    await client.stop()


@pytest.mark.asyncio
async def test_did_close_evicts_diagnostics_cache(tmp_path: Path) -> None:
    client = await _make_ready_client(tmp_path)
    uri = "file:///x.py"
    # Inject a fake entry directly (simulates a push notification landing).
    client._diagnostics_cache[uri] = [{"severity": 1, "message": "err"}]
    await client.did_close(uri)
    assert client.get_diagnostics_cache(uri) == []
    await client.stop()


# ── capabilities ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_supports_pull_diagnostics_false_when_not_advertised(tmp_path: Path) -> None:
    client = await _make_ready_client(tmp_path)
    # Fake server returns empty capabilities, so pull diagnostics are not advertised.
    assert client.supports_pull_diagnostics() is False
    await client.stop()
