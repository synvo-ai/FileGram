"""Agent type definitions and registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..permission import PermissionAction, PermissionRule
else:
    # Avoid circular import at runtime
    from ..permission.permission import PermissionAction, PermissionRule

from ..prompts import load_agent_prompt

# Load agent prompts from txt files (like OpenCode's static import)
EXPLORE_AGENT_PROMPT = load_agent_prompt("explore")
COMPACTION_AGENT_PROMPT = load_agent_prompt("compaction")


class AgentMode(Enum):
    """Agent mode types."""

    PRIMARY = "primary"  # Main conversation agent
    SUBAGENT = "subagent"  # Can be spawned as sub-task
    ALL = "all"  # Both primary and subagent


@dataclass
class AgentInfo:
    """Information about an agent type."""

    name: str
    description: str
    mode: AgentMode = AgentMode.SUBAGENT
    permission: list[PermissionRule] = field(default_factory=list)
    prompt: str | None = None  # Custom system prompt addition
    temperature: float = 0.7
    top_p: float = 1.0
    model: str | None = None  # Specific model to use
    max_steps: int | None = None  # Maximum agentic steps
    hidden: bool = False  # Hidden from user selection
    tools_allowed: list[str] | None = None  # Specific tools allowed (None = all)
    tools_denied: list[str] | None = None  # Specific tools denied

    def can_use_tool(self, tool_name: str) -> bool:
        """Check if this agent can use a specific tool."""
        if self.tools_denied and tool_name in self.tools_denied:
            return False
        if self.tools_allowed is not None and tool_name not in self.tools_allowed:
            return False
        return True


# Built-in agent types (prompts loaded from txt files above)


class AgentRegistry:
    """Registry for managing agent types."""

    def __init__(self):
        self._agents: dict[str, AgentInfo] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default agent types."""
        # Primary build agent
        self.register(
            AgentInfo(
                name="build",
                description="The default agent. Executes tools based on configured permissions.",
                mode=AgentMode.PRIMARY,
                permission=[
                    PermissionRule("*", "*", PermissionAction.ALLOW),
                ],
            )
        )

        # Explore agent (read-only)
        self.register(
            AgentInfo(
                name="explore",
                description=(
                    "Fast agent specialized for exploring codebases. Use for finding files, "
                    "searching code, or answering questions about the codebase."
                ),
                mode=AgentMode.SUBAGENT,
                prompt=EXPLORE_AGENT_PROMPT,
                permission=[
                    PermissionRule("*", "*", PermissionAction.DENY),
                    PermissionRule("grep", "*", PermissionAction.ALLOW),
                    PermissionRule("glob", "*", PermissionAction.ALLOW),
                    PermissionRule("read", "*", PermissionAction.ALLOW),
                    PermissionRule("bash", "ls *", PermissionAction.ALLOW),
                    PermissionRule("bash", "find *", PermissionAction.ALLOW),
                    PermissionRule("bash", "cat *", PermissionAction.ALLOW),
                ],
                tools_denied=["write", "edit", "task"],
            )
        )

        # General purpose subagent
        self.register(
            AgentInfo(
                name="general",
                description="General-purpose agent for researching complex questions and executing multi-step tasks.",
                mode=AgentMode.SUBAGENT,
                permission=[
                    PermissionRule("*", "*", PermissionAction.ALLOW),
                ],
            )
        )

        # Compaction agent (internal)
        self.register(
            AgentInfo(
                name="compaction",
                description="Internal agent for summarizing conversations.",
                mode=AgentMode.PRIMARY,
                prompt=COMPACTION_AGENT_PROMPT,
                hidden=True,
                permission=[
                    PermissionRule("*", "*", PermissionAction.DENY),
                ],
                tools_allowed=[],  # No tools
            )
        )

    def register(self, agent: AgentInfo) -> None:
        """Register an agent type."""
        self._agents[agent.name] = agent

    def get(self, name: str) -> AgentInfo | None:
        """Get an agent by name."""
        return self._agents.get(name)

    def list(self, include_hidden: bool = False) -> list[AgentInfo]:
        """List all registered agents."""
        agents = list(self._agents.values())
        if not include_hidden:
            agents = [a for a in agents if not a.hidden]
        return agents

    def list_subagents(self) -> list[AgentInfo]:
        """List agents that can be used as subagents."""
        return [a for a in self._agents.values() if a.mode in (AgentMode.SUBAGENT, AgentMode.ALL) and not a.hidden]

    def get_primary(self) -> AgentInfo | None:
        """Get the primary agent."""
        for agent in self._agents.values():
            if agent.mode == AgentMode.PRIMARY and not agent.hidden:
                return agent
        return None


# Global registry instance
_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry


def register_agent(agent: AgentInfo) -> None:
    """Register an agent in the global registry."""
    get_agent_registry().register(agent)


def get_agent(name: str) -> AgentInfo | None:
    """Get an agent from the global registry."""
    return get_agent_registry().get(name)
