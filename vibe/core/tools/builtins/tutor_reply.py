"""``tutor_reply`` — the bounded tutor's **only** way to address the student.

Every student-facing message goes through this tool, and every call is validated in code against the
teacher-authored :class:`~vibe.core.graph.library.teaching.TutorContract` (see
:mod:`vibe.core.tools.builtins._tutor`) before the draft can reach the student. A draft that would
reveal the solution outside the contract's policy is **withheld** — the student never sees it — and the
model is told to give a smaller hint or escalate. This is the hard gate: the model cannot hand over an
answer by calling this tool, because the tool refuses to pass it on.

Attempt counts and the current hint rung live in this tool's **session-scoped state** (never the shared
cache), so no student data is persisted cross-session.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.graph.library.teaching import TutorContract
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolPermission,
)
from vibe.core.tools.builtins._tutor import (
    Reply,
    TutorState,
    enforce_contract,
    load_contract,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolResultEvent

_HELD_BACK = "(held back — reformulating)"


class TutorReplyArgs(BaseModel):
    move: Literal[
        "hint", "scaffold", "worked_example", "check_understanding", "socratic_prompt", "encourage"
    ] = Field(description="The pedagogical move this message makes.")
    draft: str = Field(description="The message for the student.")
    item: str = Field(default="", description="Which item/problem this addresses (its stem), if any.")
    student_attempted: bool = Field(
        default=False, description="True if the student just made a scored attempt on `item` (counts toward the reveal/escalation thresholds)."
    )
    reveal_requested: bool = Field(
        default=False, description="True if the student explicitly asked to be shown the answer."
    )


class TutorReplyResult(BaseModel):
    delivered: bool
    move: str = ""
    student_message: str = ""
    blocked: bool = False
    escalate: bool = False
    correction: str = ""
    reason: str = ""


class TutorReplyConfig(BaseToolConfig):
    # The gate is enforced in code (not a human approval prompt), so the tool auto-runs.
    permission: ToolPermission = ToolPermission.ALWAYS
    # Locator for the TutorContract: a node id in the session graph, a *.json path, or None (auto).
    contract: str | None = None


class TutorReplyState(BaseToolState):
    model_config = ConfigDict(extra="forbid", validate_default=True, arbitrary_types_allowed=True)

    contract: TutorContract | None = None
    attempts: dict[str, int] = Field(default_factory=dict)
    hint_rung: int = 0


class TutorReply(
    BaseTool[TutorReplyArgs, TutorReplyResult, TutorReplyConfig, TutorReplyState],
    ToolUIData[TutorReplyArgs, TutorReplyResult],
):
    description: ClassVar[str] = (
        "The ONLY way to speak to the student. Every message is checked against the teacher's tutor "
        "contract before the student sees it: an off-contract move, or a draft that gives away the "
        "answer against the reveal policy, is withheld and you must revise. Choose the pedagogical "
        "`move`, put the student-facing text in `draft`, and set `item`/`student_attempted` so the "
        "reveal and escalation thresholds track correctly. There is deliberately no way to hand over "
        "the answer."
    )

    @classmethod
    def get_status_text(cls) -> str:
        return "Tutoring"

    @classmethod
    def format_call_display(cls, args: TutorReplyArgs) -> ToolCallDisplay:
        return ToolCallDisplay(summary=f"Tutor · {args.move}")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if event.error:
            return ToolResultDisplay(success=False, message=event.error)
        if not isinstance(event.result, TutorReplyResult):
            return ToolResultDisplay(success=True, message="")
        r = event.result
        if r.delivered or r.escalate:
            return ToolResultDisplay(success=True, message=r.student_message)
        # Blocked: the student must NOT see the withheld draft or the model-facing correction.
        return ToolResultDisplay(success=False, message=_HELD_BACK)

    def get_llm_content(self, result: TutorReplyResult) -> str | None:
        if result.delivered:
            return f"Delivered a {result.move}. Wait for the student's next message."
        if result.escalate:
            return f"Escalation raised to the teacher. {result.correction}"
        return f"BLOCKED by the tutor contract — the student did NOT see your draft. {result.correction}"

    async def run(
        self, args: TutorReplyArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[TutorReplyResult, None]:
        if self.state.contract is None:
            self.state.contract = load_contract(ctx, self.config.contract)
        contract = self.state.contract

        if args.student_attempted and args.item:
            self.state.attempts[args.item] = self.state.attempts.get(args.item, 0) + 1

        judge = ctx.sampling_callback.complete_text if (ctx and ctx.sampling_callback) else None
        verdict = await enforce_contract(
            Reply(
                move=args.move, draft=args.draft, item=args.item,
                student_attempted=args.student_attempted, reveal_requested=args.reveal_requested,
            ),
            contract,
            TutorState(attempts=self.state.attempts, hint_rung=self.state.hint_rung),
            judge,
        )

        if verdict.action == "allow":
            if args.move == "hint":
                self.state.hint_rung = min(self.state.hint_rung + 1, len(contract.hint_ladder))
            yield TutorReplyResult(delivered=True, move=args.move, student_message=verdict.student_text)
            return
        if verdict.action == "escalate":
            yield TutorReplyResult(
                delivered=False, escalate=True, student_message=verdict.student_text,
                correction=verdict.correction, reason=verdict.reason,
            )
            return
        yield TutorReplyResult(
            delivered=False, blocked=True, correction=verdict.correction, reason=verdict.reason
        )
