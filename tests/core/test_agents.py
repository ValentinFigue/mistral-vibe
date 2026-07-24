from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.agents.manager import AgentManager
from vibe.core.agents.models import (
    ANALYST,
    BUILTIN_AGENTS,
    DOC_REVIEWER,
    EXPLORE,
    AgentSafety,
    AgentType,
)


class TestAnalystAgent:
    def test_analyst_registered(self) -> None:
        assert BUILTIN_AGENTS["analyst"] is ANALYST
        assert ANALYST.agent_type == AgentType.AGENT

    def test_analyst_scopes_tools_and_catalog(self) -> None:
        config = ANALYST.apply_to_config(build_test_vibe_config())
        # The analyst authors via the pipeline DSL (run_pipeline), not raw graph_patch.
        assert set(config.enabled_tools) == {
            "run_pipeline",
            "graph_save_block",
            "graph_inspect",
            "ask_user_question",
        }
        assert config.system_prompt_id == "analyst"
        # run_pipeline is scoped to the analysis library (the catalog + focus guard).
        assert config.tools["run_pipeline"]["library"] == "analysis"
        assert config.tools["graph_save_block"]["library"] == "analysis"

    def test_analyst_prompt_is_data_analysis(self) -> None:
        config = ANALYST.apply_to_config(build_test_vibe_config())
        prompt = config.system_prompt.lower()
        assert "data analyst" in prompt
        assert "graph_inspect" in prompt

    def test_analyst_tools_resolve_through_manager(self) -> None:
        # run_pipeline + graph_inspect are auto-discovered; the allow-list scopes to four tools.
        from vibe.core.tools.manager import ToolManager

        config = ANALYST.apply_to_config(build_test_vibe_config())
        available = set(ToolManager(lambda: config, defer_mcp=True).available_tools)
        assert available == {
            "run_pipeline",
            "graph_save_block",
            "graph_inspect",
            "ask_user_question",
        }

    def test_analyst_prompt_embeds_scoped_catalog(self) -> None:
        # The catalog is in the system prompt (no empty-patch round-trip), scoped to analysis.
        from vibe.core.system_prompt import _get_graph_catalog_section
        from vibe.core.tools.manager import ToolManager

        base = build_test_vibe_config()
        applied = ANALYST.apply_to_config(base)
        tm = ToolManager(lambda: applied, defer_mcp=True)
        am = AgentManager(lambda: base, initial_agent="analyst")
        section = _get_graph_catalog_section(tm, applied, am)
        assert "Workflow catalog" in section
        assert "read_csv" in section and "quick_profile" in section
        assert "sales_source" not in section and "fetch_docs" not in section  # out of library

        # The default agent has graph_patch enabled too, but must NOT get the catalog section.
        dm = AgentManager(lambda: base, initial_agent="default")
        assert _get_graph_catalog_section(ToolManager(lambda: base, defer_mcp=True), base, dm) == ""


class TestAgentProfile:
    def test_explore_agent_is_subagent(self) -> None:
        """Test that EXPLORE agent has SUBAGENT type."""
        assert EXPLORE.agent_type == AgentType.SUBAGENT

    def test_explore_agent_has_safe_safety(self) -> None:
        """Test that EXPLORE agent has SAFE safety level."""
        assert EXPLORE.safety == AgentSafety.SAFE

    def test_explore_agent_has_enabled_tools(self) -> None:
        """Test that EXPLORE agent has expected enabled tools."""
        enabled_tools = EXPLORE.overrides.get("enabled_tools", [])
        assert "grep" in enabled_tools
        assert "read" in enabled_tools

    def test_builtin_agents_contains_explore(self) -> None:
        """Test that BUILTIN_AGENTS includes explore."""
        assert "explore" in BUILTIN_AGENTS
        assert BUILTIN_AGENTS["explore"] is EXPLORE


class TestAgentManager:
    @pytest.fixture
    def manager(self) -> AgentManager:
        config = build_test_vibe_config(
            include_project_context=False, include_prompt_detail=False
        )
        return AgentManager(lambda: config)

    def test_get_subagents_returns_only_subagents(self, manager: AgentManager) -> None:
        """Test that only SUBAGENT type agents are returned."""
        subagents = manager.get_subagents()

        for agent in subagents:
            assert agent.agent_type == AgentType.SUBAGENT

    def test_get_subagents_includes_explore(self, manager: AgentManager) -> None:
        """Test that EXPLORE is included in subagents."""
        subagents = manager.get_subagents()
        names = [a.name for a in subagents]

        assert "explore" in names

    def test_get_subagents_excludes_agents(self, manager: AgentManager) -> None:
        """Test that AGENT type agents are not returned."""
        subagents = manager.get_subagents()
        names = [a.name for a in subagents]

        # These are AGENT type
        assert "default" not in names
        assert "plan" not in names
        assert "auto-approve" not in names

    def test_get_builtin_agent(self, manager: AgentManager) -> None:
        """Test getting a builtin agent by name."""
        agent = manager.get_agent("explore")

        assert agent is EXPLORE
        assert agent.agent_type == AgentType.SUBAGENT

    def test_get_nonexistent_agent_raises(self, manager: AgentManager) -> None:
        """Test that getting a nonexistent agent raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            manager.get_agent("nonexistent-agent")

    def test_get_default_agent(self, manager: AgentManager) -> None:
        """Test getting the default agent."""
        agent = manager.get_agent("default")

        assert agent.name == "default"
        assert agent.agent_type == AgentType.AGENT

    def test_initial_agent_rejects_subagent(self) -> None:
        """Test that creating AgentManager with a subagent as initial_agent raises."""
        config = build_test_vibe_config(
            include_project_context=False, include_prompt_detail=False
        )
        with pytest.raises(ValueError, match="cannot be used as the primary agent"):
            AgentManager(lambda: config, initial_agent="explore")

    def test_initial_agent_accepts_subagent_when_allowed(self) -> None:
        """Test that allow_subagent=True permits subagent as initial_agent."""
        config = build_test_vibe_config(
            include_project_context=False, include_prompt_detail=False
        )
        manager = AgentManager(
            lambda: config, initial_agent="explore", allow_subagent=True
        )
        assert manager.active_profile.name == "explore"

    def test_initial_agent_accepts_agent_type(self) -> None:
        """Test that creating AgentManager with an agent-type agent works."""
        config = build_test_vibe_config(
            include_project_context=False, include_prompt_detail=False
        )
        manager = AgentManager(lambda: config, initial_agent="plan")
        assert manager.active_profile.name == "plan"

    def test_initial_agent_raises_when_agent_is_disabled(self) -> None:
        config = build_test_vibe_config(
            include_project_context=False,
            include_prompt_detail=False,
            disabled_agents=["plan"],
        )
        with pytest.raises(ValueError, match="disabled_agents") as exc_info:
            AgentManager(lambda: config, initial_agent="plan")
        message = str(exc_info.value)
        assert "default_agent" not in message
        assert message.startswith("Agent 'plan'")

    def test_explicit_agent_excluded_by_enabled_agents_does_not_blame_default(
        self,
    ) -> None:
        config = build_test_vibe_config(
            include_project_context=False,
            include_prompt_detail=False,
            enabled_agents=["default"],
        )
        with pytest.raises(ValueError, match="enabled_agents") as exc_info:
            AgentManager(lambda: config, initial_agent="plan")
        message = str(exc_info.value)
        assert "default_agent" not in message
        assert message.startswith("Agent 'plan'")

    def test_initial_agent_raises_when_agent_does_not_exist(self) -> None:
        config = build_test_vibe_config(
            include_project_context=False, include_prompt_detail=False
        )
        with pytest.raises(ValueError, match="not found"):
            AgentManager(lambda: config, initial_agent="nonexistent-agent")

    def test_default_agent_excluded_by_enabled_agents_raises_config_contradiction(
        self,
    ) -> None:
        config = build_test_vibe_config(
            include_project_context=False,
            include_prompt_detail=False,
            enabled_agents=["plan"],
        )
        with pytest.raises(ValueError, match="enabled_agents") as exc_info:
            AgentManager(lambda: config)
        message = str(exc_info.value)
        assert "default" in message
        assert "default_agent" in message

    def test_default_agent_excluded_by_disabled_agents_raises_config_contradiction(
        self,
    ) -> None:
        config = build_test_vibe_config(
            include_project_context=False,
            include_prompt_detail=False,
            disabled_agents=["default"],
        )
        with pytest.raises(ValueError, match="disabled_agents") as exc_info:
            AgentManager(lambda: config)
        assert "default_agent" in str(exc_info.value)

    def test_disabled_agents_ignored_entirely_when_enabled_agents_set(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = build_test_vibe_config(
            include_project_context=False,
            include_prompt_detail=False,
            enabled_agents=["plan"],
            disabled_agents=["plan"],
        )
        with caplog.at_level("WARNING"):
            manager = AgentManager(lambda: config, initial_agent="plan")
        assert manager.active_profile.name == "plan"
        assert caplog.text == ""

    def test_docreviewer_registered_and_scoped(self) -> None:
        assert BUILTIN_AGENTS["doc-reviewer"] is DOC_REVIEWER
        config = DOC_REVIEWER.apply_to_config(build_test_vibe_config())
        assert set(config.enabled_tools) == {
            "run_pipeline",
            "graph_save_block",
            "graph_inspect",
            "ask_user_question",
        }
        assert config.system_prompt_id == "doc_reviewer"
        assert config.tools["run_pipeline"]["library"] == "documents"
        assert config.tools["graph_save_block"]["library"] == "documents"

    def test_docreviewer_catalog_is_documents_only(self) -> None:
        # The generality guarantee: the reviewer sees its own library and NOT the analyst's ops.
        from vibe.core.system_prompt import _get_graph_catalog_section
        from vibe.core.tools.manager import ToolManager

        base = build_test_vibe_config()
        applied = DOC_REVIEWER.apply_to_config(base)
        tm = ToolManager(lambda: applied, defer_mcp=True)
        am = AgentManager(lambda: base, initial_agent="doc-reviewer")
        section = _get_graph_catalog_section(tm, applied, am)
        assert "extract_segments" in section and "contract_review" in section  # documents library
        assert "read_csv" not in section and "quick_profile" not in section  # analysis library excluded

    def test_install_required_agent_reports_install_not_disabled_agents(self) -> None:
        # 'lean' is install_required and enabled but not installed: the message
        # must point to installation, not blame disabled_agents.
        config = build_test_vibe_config(
            include_project_context=False,
            include_prompt_detail=False,
            enabled_agents=["lean"],
        )
        with pytest.raises(ValueError, match="requires installation") as exc_info:
            AgentManager(lambda: config, initial_agent="lean")
        assert "disabled_agents" not in str(exc_info.value)
