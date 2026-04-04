"""Task tool for spawning sub-agents."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from .base import BaseTool, ToolContext

if TYPE_CHECKING:
    from ..agent.types import AgentInfo

# Load base description from txt file (like OpenCode's static import)
TASK_TOOL_DESCRIPTION = load_tool_prompt("task")


class TaskTool(BaseTool):
    """Tool for spawning sub-agents to handle complex tasks."""

    def __init__(
        self,
        agent_executor: Callable[[str, str, AgentInfo, ToolContext], Awaitable[str]] | None = None,
    ):
        """
        Initialize TaskTool.

        Args:
            agent_executor: Async function to execute sub-agent tasks.
                            Signature: (prompt, description, agent_info, context) -> result_text
        """
        self._agent_executor = agent_executor

    @property
    def name(self) -> str:
        return "task"

    def _get_registry(self):
        """Get agent registry (lazy import to avoid circular dependency)."""
        from ..agent.types import get_agent_registry

        return get_agent_registry()

    @property
    def description(self) -> str:
        # Return description loaded from txt file
        return TASK_TOOL_DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        registry = self._get_registry()
        agent_names = [a.name for a in registry.list_subagents()]

        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A short (3-5 words) description of the task",
                },
                "prompt": {
                    "type": "string",
                    "description": "The detailed task for the agent to perform",
                },
                "subagent_type": {
                    "type": "string",
                    "description": "The type of agent to use",
                    "enum": agent_names,
                },
            },
            "required": ["description", "prompt", "subagent_type"],
        }

    def set_executor(
        self,
        executor: Callable[[str, str, AgentInfo, ToolContext], Awaitable[str]],
    ) -> None:
        """Set the agent executor function."""
        self._agent_executor = executor

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute task by spawning a sub-agent."""
        description = arguments.get("description", "")
        prompt = arguments.get("prompt", "")
        subagent_type = arguments.get("subagent_type", "general")

        if not prompt:
            return self._make_result(
                tool_use_id,
                "Error: prompt is required",
                is_error=True,
            )

        # Get agent info
        registry = self._get_registry()
        agent_info = registry.get(subagent_type)

        if not agent_info:
            available = [a.name for a in registry.list_subagents()]
            return self._make_result(
                tool_use_id,
                f"Error: Unknown agent type '{subagent_type}'. Available: {available}",
                is_error=True,
            )

        if not self._agent_executor:
            return self._make_result(
                tool_use_id,
                "Error: Task execution not configured. Agent executor not set.",
                is_error=True,
            )

        try:
            # Execute sub-agent
            result = await self._agent_executor(prompt, description, agent_info, context)

            # Format output
            output = f"""Task completed: {description}

Agent: {subagent_type}
Result:
{result}"""

            return self._make_result(
                tool_use_id,
                output,
                metadata={
                    "description": description,
                    "subagent_type": subagent_type,
                    "task_id": str(uuid.uuid4()),
                },
            )

        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Task execution failed: {str(e)}",
                is_error=True,
            )
