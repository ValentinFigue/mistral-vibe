"""``escalate_to_teacher`` — the tutor's exit to a human when it hits the contract's boundary.

The tutor calls this when the contract's escalation conditions are met (repeated failure, signs of
distress, off-topic, or anything it can't handle within its allowed moves). It records the escalation
to a per-session teacher inbox (``<session_dir>/tutor/escalations.jsonl``) — never the shared
cross-session cache — and tells the student a teacher is being brought in. This is the escalation-loop
v1: a durable session artifact a teacher-facing surface can read.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
import json
from typing import ClassVar

from pydantic import BaseModel, Field

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolResultEvent

_STUDENT_MESSAGE = "I've let your teacher know so they can help with this — let's pause here for now."


class EscalateToTeacherArgs(BaseModel):
    reason: str = Field(description="Why this needs the teacher (e.g. 'repeated failure', 'distress', 'off-topic').")
    student_context: str = Field(
        default="", description="Short, factual context for the teacher — no unnecessary personal detail."
    )


class EscalateToTeacherResult(BaseModel):
    recorded: bool
    student_message: str
    path: str = ""


class EscalateToTeacherConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class EscalateToTeacherState(BaseToolState):
    pass


class EscalateToTeacher(
    BaseTool[EscalateToTeacherArgs, EscalateToTeacherResult, EscalateToTeacherConfig, EscalateToTeacherState],
    ToolUIData[EscalateToTeacherArgs, EscalateToTeacherResult],
):
    description: ClassVar[str] = (
        "Hand off to the student's teacher when you hit the contract's boundary — repeated failed "
        "attempts, signs of distress, an off-topic or unsafe request, or anything your allowed moves "
        "can't handle. Records the escalation for the teacher and pauses the session."
    )

    @classmethod
    def get_status_text(cls) -> str:
        return "Escalating to the teacher"

    @classmethod
    def format_call_display(cls, args: EscalateToTeacherArgs) -> ToolCallDisplay:
        return ToolCallDisplay(summary=f"Escalate to teacher · {args.reason}")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if event.error:
            return ToolResultDisplay(success=False, message=event.error)
        if not isinstance(event.result, EscalateToTeacherResult):
            return ToolResultDisplay(success=True, message=_STUDENT_MESSAGE)
        return ToolResultDisplay(success=True, message=event.result.student_message)

    def get_llm_content(self, result: EscalateToTeacherResult) -> str | None:
        return "Escalation recorded for the teacher; the session is paused. Do not continue tutoring this item."

    async def run(
        self, args: EscalateToTeacherArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[EscalateToTeacherResult, None]:
        base = (ctx.session_dir or ctx.scratchpad_dir) if ctx else None
        if base is None:
            raise ToolError("escalate_to_teacher requires a session or scratchpad directory")
        inbox = base / "tutor"
        inbox.mkdir(parents=True, exist_ok=True)
        record = {"reason": args.reason, "student_context": args.student_context}
        path = inbox / "escalations.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        yield EscalateToTeacherResult(recorded=True, student_message=_STUDENT_MESSAGE, path=str(path))
