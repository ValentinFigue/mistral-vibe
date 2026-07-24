from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from vibe.core.tools.mcp_sampling import MCPSamplingHandler
from vibe.core.types import LLMMessage, Role


class _FakeBackend:
    """A backend whose ``complete`` returns a fixed assistant message."""

    def __init__(self, message: LLMMessage) -> None:
        self._message = message

    async def complete(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(message=self._message)


def _handler(message: LLMMessage) -> MCPSamplingHandler:
    model = SimpleNamespace(name="test-model", temperature=0.0)
    return MCPSamplingHandler(
        backend_getter=lambda: _FakeBackend(message),
        config_getter=lambda: SimpleNamespace(get_active_model=lambda: model),
    )


@pytest.mark.asyncio
async def test_complete_text_returns_content() -> None:
    handler = _handler(LLMMessage(role=Role.assistant, content="hello"))
    assert await handler.complete_text("hi") == "hello"


@pytest.mark.asyncio
async def test_complete_text_warns_on_empty_content_with_reasoning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A reasoning model can spend the whole budget reasoning and emit empty content — the caller
    # should see a warning, not a silent "".
    handler = _handler(
        LLMMessage(role=Role.assistant, content="", reasoning_content="thinking hard...")
    )
    with caplog.at_level(logging.WARNING, logger="vibe"):
        out = await handler.complete_text("hi", max_tokens=512)
    assert out == ""
    assert "empty content but non-empty reasoning" in caplog.text


@pytest.mark.asyncio
async def test_complete_text_no_warning_when_both_empty(caplog: pytest.LogCaptureFixture) -> None:
    handler = _handler(LLMMessage(role=Role.assistant, content=""))
    with caplog.at_level(logging.WARNING, logger="vibe"):
        assert await handler.complete_text("hi") == ""
    assert "empty content but non-empty reasoning" not in caplog.text
