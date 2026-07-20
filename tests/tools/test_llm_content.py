from __future__ import annotations

from pydantic import BaseModel

from vibe.core.agent_loop import _tool_result_text
from vibe.core.tools.base import BaseTool
from vibe.core.tools.builtins.graph_patch import (
    GraphPatch,
    GraphPatchConfig,
    GraphPatchResult,
    GraphPatchState,
    OutputHandle,
)


def _tool() -> GraphPatch:
    return GraphPatch(config_getter=lambda: GraphPatchConfig(), state=GraphPatchState())


def _result() -> GraphPatchResult:
    return GraphPatchResult(
        applied=True,
        fresh=["report"],
        cached=["events", "users"],
        outputs={"report": "X" * 3000},  # the full payload the UI keeps
        handles={
            "report": OutputHandle(
                type="AnalyticsReport",
                fingerprint="abcd1234ef567890" * 4,
                preview="# Top markets by purchase revenue",
            )
        },
        catalog="Available operators:\n" + "- load_events(...)\n" * 20,
    )


def test_base_get_llm_content_defaults_to_none() -> None:
    # The base implementation returns None, so non-overriding tools keep the default field dump.
    assert BaseTool.get_llm_content(_tool(), _result()) is None


def test_graph_patch_content_is_handles_not_payloads() -> None:
    result = _result()
    content = _tool().get_llm_content(result)
    assert content is not None
    # The model gets a content-addressed handle + preview, never the full payload.
    assert "ref:abcd1234ef56" in content
    assert "AnalyticsReport" in content
    assert "# Top markets by purchase revenue" in content
    assert "X" * 3000 not in content
    # And it is far smaller than the default full-field dump would be.
    dump = "\n".join(f"{k}: {v}" for k, v in result.model_dump().items())
    assert len(content) < len(dump)


def test_graph_patch_catalog_shown_once() -> None:
    tool = _tool()
    result = _result()

    first = tool.get_llm_content(result)
    assert first is not None and "load_events" in first  # full catalog on the first turn

    second = tool.get_llm_content(result)
    assert second is not None
    assert "load_events" not in second  # catalog dropped afterwards
    assert "catalog unchanged" in second


def test_terminal_preview_is_readable_multiline() -> None:
    # Regression (temper #1): terminal values must be shown in full enough for the agent to
    # read its own result — not clipped to a one-line glimpse.
    report = "# Top markets\n- fr: 1,234.00 across 42 events\n- us: 900.00 across 30 events"
    result = _result()
    result.handles["report"] = OutputHandle(
        type="AnalyticsReport", fingerprint="deadbeef" * 8, preview=report
    )
    content = _tool().get_llm_content(result)
    assert content is not None
    for line in report.splitlines():
        assert line in content  # every line of the terminal report survives


# --- the agent-loop wiring (temper #4): _tool_result_text ---


class _Res(BaseModel):
    a: int = 1
    b: str = "x"


class _PlainTool:
    """A tool that overrides neither hook — must get the default full-field dump."""

    def get_llm_content(self, result: _Res) -> str | None:
        return None

    def get_result_extra(self, result: _Res) -> str | None:
        return None


class _CompactTool:
    """A tool that supplies compact model-facing text and an appended extra."""

    def get_llm_content(self, result: _Res) -> str | None:
        return "COMPACT"

    def get_result_extra(self, result: _Res) -> str | None:
        return "EXTRA"


def test_loop_falls_back_to_full_dump_when_no_llm_content() -> None:
    assert _tool_result_text(_PlainTool(), _Res()) == "a: 1\nb: x"  # type: ignore[arg-type]


def test_loop_uses_llm_content_and_appends_extra() -> None:
    # get_llm_content replaces the dump; get_result_extra is still appended after it.
    assert _tool_result_text(_CompactTool(), _Res()) == "COMPACT\n\nEXTRA"  # type: ignore[arg-type]
