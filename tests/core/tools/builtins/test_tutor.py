from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibe.core.graph.library.teaching import HintRung, TutorContract
from vibe.core.tools.base import InvokeContext, ToolError
from vibe.core.tools.builtins._tutor import (
    Reply,
    TutorState,
    attempt_limit,
    check_hint_ladder,
    check_move,
    enforce_contract,
    load_contract,
    load_contract_for_prompt,
    reveal_allowed,
)
from vibe.core.tools.builtins.escalate_to_teacher import (
    EscalateToTeacher,
    EscalateToTeacherArgs,
    EscalateToTeacherConfig,
    EscalateToTeacherState,
)
from vibe.core.tools.builtins.tutor_reply import (
    _HELD_BACK,
    TutorReply,
    TutorReplyArgs,
    TutorReplyConfig,
    TutorReplyResult,
    TutorReplyState,
)


def _contract(
    policy: str = "after_attempts", after: int = 3, triggers: list[str] | None = None
) -> TutorContract:
    return TutorContract(
        objectives=["Identify the main idea"],
        item_scope=["Q1"],
        reveal_policy=policy,  # type: ignore[arg-type]
        reveal_after_attempts=after,
        hint_ladder=[HintRung(order=0, text="restate it"), HintRung(order=1, text="point to the line"),
                     HintRung(order=2, text="give the first step")],
        escalation_triggers=["3 failed attempts", "signs of distress"] if triggers is None else triggers,
        done_criteria="explains it in their own words",
    )


class _Judge:
    """A stub reveal-judge matching complete_text's shape; returns a fixed verdict."""

    def __init__(self, verdict: str) -> None:
        self.verdict, self.calls = verdict, 0

    async def __call__(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024) -> str:
        self.calls += 1
        return self.verdict


class _StubSampling:
    def __init__(self, verdict: str) -> None:
        self._judge = _Judge(verdict)

    async def complete_text(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024) -> str:
        return await self._judge(prompt, system=system, max_tokens=max_tokens)


# --- structural checks (pure) --------------------------------------------------------------


def test_check_move_rejects_forbidden_and_off_menu() -> None:
    c = _contract()
    assert check_move("answer", c) is not None  # forbidden by default
    assert check_move("hint", c) is None
    c2 = TutorContract(allowed_moves=["hint"], forbidden_moves=["answer"])
    assert check_move("worked_example", c2) is not None  # not in allowed_moves


def test_check_hint_ladder_blocks_when_exhausted() -> None:
    c = _contract()
    assert check_hint_ladder("hint", TutorState(hint_rung=0), c) is None
    assert check_hint_ladder("hint", TutorState(hint_rung=3), c) is not None  # past the last rung
    assert check_hint_ladder("encourage", TutorState(hint_rung=99), c) is None  # non-hint unaffected


def test_attempt_limit_parsed_from_triggers() -> None:
    assert attempt_limit(_contract()) == 3
    assert attempt_limit(TutorContract(escalation_triggers=["off-topic"])) is None


def test_reveal_allowed_matches_policy() -> None:
    assert reveal_allowed("Q1", TutorState(), _contract("never"), requested=True) is False
    assert reveal_allowed("Q1", TutorState(), _contract("on_request"), requested=True) is True
    assert reveal_allowed("Q1", TutorState(), _contract("on_request"), requested=False) is False
    below = TutorState(attempts={"Q1": 2})
    at = TutorState(attempts={"Q1": 3})
    assert reveal_allowed("Q1", below, _contract("after_attempts", 3), requested=False) is False
    assert reveal_allowed("Q1", at, _contract("after_attempts", 3), requested=False) is True


# --- enforce_contract (async orchestrator) -------------------------------------------------


@pytest.mark.asyncio
async def test_forbidden_move_blocked() -> None:
    v = await enforce_contract(Reply(move="answer", draft="It's 42."), _contract(), TutorState(), _Judge("GUIDE"))
    assert v.action == "block" and "allowed" in v.correction.lower()


@pytest.mark.asyncio
async def test_reveal_blocked_when_policy_never() -> None:
    judge = _Judge("REVEAL")
    v = await enforce_contract(
        Reply(move="worked_example", draft="The full solution is …", item="Q1"),
        _contract("never"), TutorState(), judge,
    )
    assert v.action == "block" and judge.calls == 1
    assert "reveal" in v.reason.lower() and v.student_text == ""  # draft withheld


@pytest.mark.asyncio
async def test_reveal_allowed_after_threshold_skips_judge() -> None:
    judge = _Judge("REVEAL")  # would block if consulted
    state = TutorState(attempts={"Q1": 3})
    # escalation trigger has no numeric limit, so only the reveal threshold applies here
    v = await enforce_contract(
        Reply(move="worked_example", draft="Here's the worked solution.", item="Q1"),
        _contract("after_attempts", 3, triggers=["signs of distress"]), state, judge,
    )
    assert v.action == "allow" and judge.calls == 0  # reveal permitted → judge not consulted
    assert v.student_text == "Here's the worked solution."


@pytest.mark.asyncio
async def test_no_judge_fails_closed() -> None:
    v = await enforce_contract(
        Reply(move="hint", draft="maybe the answer is X", item="Q1"),
        _contract("never"), TutorState(), judge=None,
    )
    assert v.action == "block"  # cannot verify without a live session → withhold


@pytest.mark.asyncio
async def test_answer_leaked_via_low_risk_move_is_blocked() -> None:
    # Regression for the reveal-judge bypass: an answer smuggled into an "encourage" turn must still
    # be judged and blocked — every move is checked when a reveal is disallowed.
    judge = _Judge("REVEAL")
    v = await enforce_contract(
        Reply(move="encourage", draft="Great effort — and the answer is 42!", item="Q1"),
        _contract("never"), TutorState(), judge,
    )
    assert v.action == "block" and judge.calls == 1


@pytest.mark.asyncio
async def test_guiding_low_risk_move_allowed() -> None:
    judge = _Judge("GUIDE")
    v = await enforce_contract(
        Reply(move="encourage", draft="Great effort — keep going!", item="Q1"),
        _contract("never"), TutorState(), judge,
    )
    assert v.action == "allow" and judge.calls == 1  # judged, but guiding → allowed


@pytest.mark.asyncio
async def test_escalates_at_attempt_limit() -> None:
    state = TutorState(attempts={"Q1": 3})
    v = await enforce_contract(
        Reply(move="hint", draft="try restating it", item="Q1"),
        _contract("after_attempts", 5), state, _Judge("GUIDE"),
    )
    assert v.action == "escalate" and "escalate_to_teacher" in v.correction


@pytest.mark.asyncio
async def test_guide_verdict_is_allowed() -> None:
    v = await enforce_contract(
        Reply(move="hint", draft="What is the paragraph mostly about?", item="Q1"),
        _contract("never"), TutorState(), _Judge("GUIDE"),
    )
    assert v.action == "allow" and v.student_text.startswith("What is")


# --- the TutorReply tool end-to-end (contract pre-loaded into state) -----------------------


def _tool(contract: TutorContract) -> TutorReply:
    return TutorReply(
        config_getter=lambda: TutorReplyConfig(),
        state=TutorReplyState(contract=contract),
    )


async def _run(tool: TutorReply, args: TutorReplyArgs, ctx: InvokeContext | None) -> TutorReplyResult:
    results = [r async for r in tool.run(args, ctx)]
    assert len(results) == 1
    return results[0]


@pytest.mark.asyncio
async def test_tool_allows_and_shows_draft_to_student() -> None:
    tool = _tool(_contract())
    ctx = InvokeContext(tool_call_id="t", sampling_callback=_StubSampling("GUIDE"))  # type: ignore[arg-type]
    res = await _run(tool, TutorReplyArgs(move="encourage", draft="Nice work so far!"), ctx)
    assert res.delivered and res.student_message == "Nice work so far!"
    # llm-facing content is terse, not the whole draft
    assert "Delivered" in (tool.get_llm_content(res) or "")


@pytest.mark.asyncio
async def test_tool_blocks_reveal_and_hides_draft() -> None:
    tool = _tool(_contract("never"))
    ctx = InvokeContext(tool_call_id="t", sampling_callback=_StubSampling("REVEAL"))  # type: ignore[arg-type]
    res = await _run(tool, TutorReplyArgs(move="worked_example", draft="The answer is 42."), ctx)
    assert res.blocked and not res.delivered and res.student_message == ""
    # the student never sees the withheld draft; the model gets a correction
    from vibe.core.types import ToolResultEvent

    event = ToolResultEvent(tool_name="tutor_reply", tool_class=TutorReply, tool_call_id="t", result=res)
    display = TutorReply.get_result_display(event)
    assert display.message == _HELD_BACK and "42" not in display.message
    assert "BLOCKED" in (tool.get_llm_content(res) or "")


@pytest.mark.asyncio
async def test_tool_advances_hint_rung_on_allow() -> None:
    tool = _tool(_contract("never"))
    ctx = InvokeContext(tool_call_id="t", sampling_callback=_StubSampling("GUIDE"))  # type: ignore[arg-type]
    assert tool.state.hint_rung == 0
    await _run(tool, TutorReplyArgs(move="hint", draft="What's the paragraph about?", item="Q1"), ctx)
    assert tool.state.hint_rung == 1


@pytest.mark.asyncio
async def test_tool_counts_attempts_and_reveals_after_threshold() -> None:
    tool = _tool(_contract("after_attempts", 2, triggers=["signs of distress"]))
    # a guiding judge so the pre-threshold turns are delivered (they are still judged now)
    ctx_guide = InvokeContext(tool_call_id="t", sampling_callback=_StubSampling("GUIDE"))  # type: ignore[arg-type]
    await _run(tool, TutorReplyArgs(move="check_understanding", draft="Show me your thinking?", item="Q1", student_attempted=True), ctx_guide)
    await _run(tool, TutorReplyArgs(move="check_understanding", draft="And now?", item="Q1", student_attempted=True), ctx_guide)
    assert tool.state.attempts["Q1"] == 2
    # now a reveal is permitted (attempts >= 2), so the judge is skipped even under a REVEAL stub
    ctx_reveal = InvokeContext(tool_call_id="t", sampling_callback=_StubSampling("REVEAL"))  # type: ignore[arg-type]
    res = await _run(tool, TutorReplyArgs(move="worked_example", draft="Here's how it works …", item="Q1"), ctx_reveal)
    assert res.delivered


def test_tutor_agent_scoping() -> None:
    from vibe.core.agents.models import BUILTIN_AGENTS, BuiltinAgentName

    profile = BUILTIN_AGENTS[BuiltinAgentName.TUTOR]
    enabled = profile.overrides["enabled_tools"]
    assert "tutor_reply" in enabled and "escalate_to_teacher" in enabled
    assert "run_pipeline" not in enabled and "graph_patch" not in enabled
    assert profile.overrides["system_prompt_id"] == "tutor"


# --- contract loading ----------------------------------------------------------------------


def test_load_contract_from_json_path(tmp_path: Path) -> None:
    p = tmp_path / "contract.json"
    p.write_text(_contract("never").model_dump_json())
    loaded = load_contract(None, str(p))
    assert loaded.reveal_policy == "never" and loaded.objectives == ["Identify the main idea"]


def test_load_contract_missing_json_errors(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="not found"):
        load_contract(None, str(tmp_path / "nope.json"))


def test_load_contract_without_session_errors() -> None:
    # no *.json, no fingerprint, no session dir → an actionable error, not a crash
    with pytest.raises(ToolError, match="no session directory"):
        load_contract(None, None)


def test_load_contract_for_prompt_never_raises(tmp_path: Path) -> None:
    # a good *.json loads; anything unresolvable returns None (the prompt degrades, the tool still gates)
    p = tmp_path / "c.json"
    p.write_text(_contract().model_dump_json())
    assert load_contract_for_prompt(str(p), None) is not None
    assert load_contract_for_prompt(str(tmp_path / "missing.json"), None) is None
    assert load_contract_for_prompt(None, None) is None


# --- escalate_to_teacher -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalate_writes_record(tmp_path: Path) -> None:
    tool = EscalateToTeacher(config_getter=lambda: EscalateToTeacherConfig(), state=EscalateToTeacherState())
    ctx = InvokeContext(tool_call_id="t", session_dir=tmp_path)
    results = [r async for r in tool.run(EscalateToTeacherArgs(reason="signs of distress"), ctx)]
    res = results[0]
    assert res.recorded and res.student_message
    log = tmp_path / "tutor" / "escalations.jsonl"
    assert log.exists()
    record = json.loads(log.read_text().splitlines()[0])
    assert record["reason"] == "signs of distress"


@pytest.mark.asyncio
async def test_escalate_requires_a_directory() -> None:
    tool = EscalateToTeacher(config_getter=lambda: EscalateToTeacherConfig(), state=EscalateToTeacherState())
    with pytest.raises(ToolError, match="requires a session"):
        _ = [r async for r in tool.run(EscalateToTeacherArgs(reason="x"), None)]
