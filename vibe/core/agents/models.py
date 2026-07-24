from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
import tomllib
from typing import TYPE_CHECKING, Any

from vibe.core.paths import PLANS_DIR
from vibe.core.utils import name_matches

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class AgentSafety(StrEnum):
    SAFE = auto()
    NEUTRAL = auto()
    DESTRUCTIVE = auto()
    YOLO = auto()


class AgentType(StrEnum):
    AGENT = auto()
    SUBAGENT = auto()


class BuiltinAgentName(StrEnum):
    DEFAULT = "default"
    CHAT = "chat"
    PLAN = "plan"
    ACCEPT_EDITS = "accept-edits"
    AUTO_APPROVE = "auto-approve"
    EXPLORE = "explore"
    LEAN = "lean"
    GRAPH = "graph"
    ANALYST = "analyst"
    DOC_REVIEWER = "doc-reviewer"
    TEACHER = "teacher"
    TUTOR = "tutor"


@dataclass(frozen=True)
class AgentProfile:
    name: str
    display_name: str
    description: str
    safety: AgentSafety
    agent_type: AgentType = AgentType.AGENT
    overrides: dict[str, Any] = field(default_factory=dict)
    install_required: bool = False

    def apply_to_config(self, base: VibeConfig) -> VibeConfig:
        from vibe.core.config import VibeConfig as VC

        merged = _deep_merge(
            base.model_dump(),
            {k: v for k, v in self.overrides.items() if k != "base_disabled"},
        )
        base_disabled = self.overrides.get("base_disabled")
        if isinstance(base_disabled, list):
            merged["disabled_tools"] = list({
                *base_disabled,
                *merged.get("disabled_tools", []),
            })

        # Environment-level disables (set by ACP/programmatic mode) must take
        # precedence over an agent's enabled_tools allowlist
        if base.disabled_tools and merged.get("enabled_tools"):
            merged["enabled_tools"] = [
                t
                for t in merged["enabled_tools"]
                if not name_matches(t, base.disabled_tools)
            ]

        return VC.model_validate(merged)

    @classmethod
    def from_toml(cls, path: Path) -> AgentProfile:
        with path.open("rb") as f:
            data = tomllib.load(f)
        return cls(
            name=path.stem,
            display_name=data.pop("display_name", path.stem.replace("-", " ").title()),
            description=data.pop("description", ""),
            safety=AgentSafety(data.pop("safety", AgentSafety.NEUTRAL)),
            agent_type=AgentType(data.pop("agent_type", AgentType.AGENT)),
            overrides=data,
        )


CHAT_AGENT_TOOLS = ["grep", "read", "ask_user_question", "task"]


def _plan_overrides() -> dict[str, Any]:
    plans_pattern = str(PLANS_DIR.path / "*")
    return {
        "tools": {
            "write_file": {"permission": "never", "allowlist": [plans_pattern]},
            "edit": {"permission": "never", "allowlist": [plans_pattern]},
        }
    }


DEFAULT = AgentProfile(
    BuiltinAgentName.DEFAULT,
    "Default",
    "Requires approval for tool executions",
    AgentSafety.NEUTRAL,
    overrides={"base_disabled": ["exit_plan_mode"]},
)
PLAN = AgentProfile(
    BuiltinAgentName.PLAN,
    "Plan",
    "Read-only agent for exploration and planning",
    AgentSafety.SAFE,
    overrides=_plan_overrides(),
)
CHAT = AgentProfile(
    BuiltinAgentName.CHAT,
    "Chat",
    "Read-only conversational mode for questions and discussions",
    AgentSafety.SAFE,
    overrides={"bypass_tool_permissions": True, "enabled_tools": CHAT_AGENT_TOOLS},
)
ACCEPT_EDITS = AgentProfile(
    BuiltinAgentName.ACCEPT_EDITS,
    "Accept Edits",
    "Auto-approves file edits only",
    AgentSafety.DESTRUCTIVE,
    overrides={
        "base_disabled": ["exit_plan_mode"],
        "tools": {
            "write_file": {"permission": "always"},
            "edit": {"permission": "always"},
        },
    },
)
AUTO_APPROVE = AgentProfile(
    BuiltinAgentName.AUTO_APPROVE,
    "Auto Approve",
    "Auto-approves all tool executions",
    AgentSafety.YOLO,
    overrides={"bypass_tool_permissions": True, "base_disabled": ["exit_plan_mode"]},
)

EXPLORE = AgentProfile(
    name=BuiltinAgentName.EXPLORE,
    display_name="Explore",
    description="Read-only subagent for codebase exploration",
    safety=AgentSafety.SAFE,
    agent_type=AgentType.SUBAGENT,
    overrides={"enabled_tools": ["grep", "read"], "system_prompt_id": "explore"},
)

LEAN = AgentProfile(
    name=BuiltinAgentName.LEAN,
    display_name="Lean",
    description="Specialized mode for Lean 4 code analysis, proof assistance, and theorem proving",
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.AGENT,
    install_required=True,
    overrides={
        "system_prompt_id": "lean",
        "active_model": "leanstral",
        "providers": [
            {
                "name": "mistral-testing",
                "api_base": "https://api.mistral.ai/v1",
                "api_key_env_var": "MISTRAL_API_KEY",
                "backend": "mistral",
            }
        ],
        "models": [
            {
                "name": "labs-leanstral-2603",
                "provider": "mistral-testing",
                "alias": "leanstral",
                "thinking": "high",
                "temperature": 1.0,
                "auto_compact_threshold": 168_000,
            }
        ],
        "compaction_model": {
            "name": "mistral-small-latest",
            "provider": "mistral-testing",
            "alias": "devstral-compact",
            "temperature": 0.2,
            "thinking": "off",
        },
        "tools": {"bash": {"default_timeout": 1200}},
        "base_disabled": ["exit_plan_mode"],
    },
)

GRAPH = AgentProfile(
    name=BuiltinAgentName.GRAPH,
    display_name="Graph",
    description="Author and iterate a workflow graph via typed patches",
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.AGENT,
    overrides={
        "enabled_tools": ["graph_patch", "graph_save_block", "ask_user_question"],
        "system_prompt_id": "graph",
    },
)

ANALYST = AgentProfile(
    name=BuiltinAgentName.ANALYST,
    display_name="Data Analyst",
    description="Analyze data by authoring a workflow graph (load → clean → aggregate → report)",
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.AGENT,
    overrides={
        "enabled_tools": [
            "run_pipeline",
            "graph_inspect",
            "graph_save_block",
            "ask_user_question",
        ],
        "system_prompt_id": "analyst",
        # Scope the pipeline tool to the data-analysis operator library — the analyst sees and
        # may reference only that library's operators and blocks.
        "tools": {
            "run_pipeline": {"library": "analysis"},
            "graph_save_block": {"library": "analysis"},
        },
    },
)

DOC_REVIEWER = AgentProfile(
    name=BuiltinAgentName.DOC_REVIEWER,
    display_name="Document Reviewer",
    description="Review documents (contracts, papers, policies, …) by authoring a workflow graph (load → extract → classify → report)",
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.AGENT,
    overrides={
        "enabled_tools": [
            "run_pipeline",
            "graph_inspect",
            "graph_save_block",
            "ask_user_question",
        ],
        "system_prompt_id": "doc_reviewer",
        # Scope the pipeline tool to the document-review operator library — the reviewer sees and
        # may reference only that library's operators and blocks (a different artifact than the
        # analyst's tables, proving the engine is library-agnostic).
        "tools": {
            "run_pipeline": {"library": "documents"},
            "graph_save_block": {"library": "documents"},
        },
    },
)

TEACHER = AgentProfile(
    name=BuiltinAgentName.TEACHER,
    display_name="Teacher",
    description="Author lessons by building a workflow graph (frame → generate → adapt → assess → bound)",
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.AGENT,
    overrides={
        "enabled_tools": [
            "run_pipeline",
            "graph_inspect",
            "graph_save_block",
            "ask_user_question",
        ],
        "system_prompt_id": "teacher",
        # Scope the pipeline tool to the lesson-authoring operator library — the teacher sees and
        # may reference only that library's operators and blocks (a third artifact family on the
        # same engine, and the source of the TutorContract that would bound a student tutor).
        "tools": {
            "run_pipeline": {"library": "teaching"},
            "graph_save_block": {"library": "teaching"},
        },
    },
)

TUTOR = AgentProfile(
    name=BuiltinAgentName.TUTOR,
    display_name="Tutor",
    description="Tutor a student inside a teacher-authored contract (bounded moves, no answers)",
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.AGENT,
    overrides={
        # Deliberately NOT a graph author: no run_pipeline, no graph_patch. The tutor addresses the
        # student only through `tutor_reply` (its hard gate) and may escalate to a human.
        "enabled_tools": [
            "tutor_reply",
            "escalate_to_teacher",
            "graph_inspect",
            "ask_user_question",
        ],
        "system_prompt_id": "tutor",
        # Where to load the TutorContract that bounds this session: a node id in the session's
        # authored graph, a *.json path, or omitted (auto-detect the contract node). Same per-tool
        # config channel the analyst/teacher use to scope run_pipeline.
        "tools": {
            "tutor_reply": {"contract": None},
        },
    },
)

BUILTIN_AGENTS: dict[str, AgentProfile] = {
    BuiltinAgentName.DEFAULT: DEFAULT,
    BuiltinAgentName.PLAN: PLAN,
    BuiltinAgentName.ACCEPT_EDITS: ACCEPT_EDITS,
    BuiltinAgentName.AUTO_APPROVE: AUTO_APPROVE,
    BuiltinAgentName.EXPLORE: EXPLORE,
    BuiltinAgentName.LEAN: LEAN,
    BuiltinAgentName.GRAPH: GRAPH,
    BuiltinAgentName.ANALYST: ANALYST,
    BuiltinAgentName.DOC_REVIEWER: DOC_REVIEWER,
    BuiltinAgentName.TEACHER: TEACHER,
    BuiltinAgentName.TUTOR: TUTOR,
}
