"""Plan mode tools for structured planning workflows.

Plan mode provides a read-only environment where the agent can:
- Analyze the codebase
- Design implementation strategies
- Write plans to a plan file
- Get user approval before implementation

This prevents accidental code modifications during the planning phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from .base import BaseTool, ToolContext

# Load descriptions from txt files (like OpenCode's static import)
PLAN_ENTER_DESCRIPTION = load_tool_prompt("plan_enter")
PLAN_EXIT_DESCRIPTION = load_tool_prompt("plan_exit")


class AgentMode(str, Enum):
    """Current agent mode."""

    BUILD = "build"
    PLAN = "plan"


@dataclass
class PlanState:
    """Tracks plan mode state for a session."""

    mode: AgentMode = AgentMode.BUILD
    plan_file: str | None = None
    entered_at: float | None = None

    def enter_plan(self, plan_file: str) -> None:
        """Enter plan mode."""
        self.mode = AgentMode.PLAN
        self.plan_file = plan_file
        self.entered_at = datetime.now().timestamp()

    def exit_plan(self) -> None:
        """Exit plan mode."""
        self.mode = AgentMode.BUILD
        # Keep plan_file for reference

    @property
    def is_plan_mode(self) -> bool:
        return self.mode == AgentMode.PLAN


# Session state storage
_plan_states: dict[str, PlanState] = {}


def get_plan_state(session_id: str) -> PlanState:
    """Get or create plan state for a session."""
    if session_id not in _plan_states:
        _plan_states[session_id] = PlanState()
    return _plan_states[session_id]


def get_plan_file_path(target_directory: Path, session_id: str) -> Path:
    """Generate the path for the plan file."""
    plans_dir = target_directory / ".synvocode" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return plans_dir / f"plan_{timestamp}_{session_id[:8]}.md"


# ============== Plan Enter Tool ==============


class PlanEnterTool(BaseTool):
    """Tool for entering plan mode."""

    @property
    def name(self) -> str:
        return "plan_enter"

    @property
    def description(self) -> str:
        return PLAN_ENTER_DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        state = get_plan_state(context.session_id)

        # Check if already in plan mode
        if state.is_plan_mode:
            return self._make_result(
                tool_use_id,
                f"Already in plan mode. Plan file: {state.plan_file}\nUse plan_exit to exit plan mode.",
                metadata={"mode": "plan", "plan_file": state.plan_file},
            )

        # Create plan file
        plan_file = get_plan_file_path(context.target_directory, context.session_id)

        # Initialize plan file with template
        template = f"""# Implementation Plan

Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Session: {context.session_id}

## Task Summary

[Describe the task here]

## Files to Modify

- [ ] file1.py - description of changes
- [ ] file2.py - description of changes

## Implementation Steps

1. Step one
2. Step two
3. Step three

## Considerations

- Risk 1
- Risk 2

## Questions for User

- Question 1?

---
*This plan was created in plan mode. Use plan_exit when ready for approval.*
"""

        try:
            with open(plan_file, "w", encoding="utf-8") as f:
                f.write(template)
        except OSError as e:
            return self._make_result(
                tool_use_id,
                f"Failed to create plan file: {e}",
                is_error=True,
            )

        # Enter plan mode
        state.enter_plan(str(plan_file))

        output = f"""Entered PLAN MODE.

📝 Plan file created: {plan_file}

IMPORTANT - While in plan mode:
- You can ONLY read files (read, grep, glob, bash for ls/cat/find)
- You can ONLY write to the plan file: {plan_file}
- All other edit/write operations are BLOCKED
- Use this time to explore and design your implementation

When your plan is complete, call plan_exit to request user approval."""

        return self._make_result(
            tool_use_id,
            output,
            metadata={
                "mode": "plan",
                "plan_file": str(plan_file),
                "entered_at": state.entered_at,
            },
        )


# ============== Plan Exit Tool ==============


class PlanExitTool(BaseTool):
    """Tool for exiting plan mode."""

    @property
    def name(self) -> str:
        return "plan_exit"

    @property
    def description(self) -> str:
        return PLAN_EXIT_DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        state = get_plan_state(context.session_id)

        # Check if in plan mode
        if not state.is_plan_mode:
            return self._make_result(
                tool_use_id,
                "Not currently in plan mode. Use plan_enter to enter plan mode.",
                is_error=True,
            )

        plan_file = state.plan_file

        # Read plan content for display
        plan_content = ""
        if plan_file and Path(plan_file).exists():
            try:
                with open(plan_file, encoding="utf-8") as f:
                    plan_content = f.read()
            except OSError:
                pass

        # Exit plan mode
        state.exit_plan()

        output = f"""Exited PLAN MODE - returned to BUILD mode.

📋 Plan file: {plan_file}

The plan is ready for user review. You can now:
- Wait for user approval
- If approved, proceed with implementation
- If rejected, discuss modifications

---
Plan Summary:
{plan_content[:1000]}{"..." if len(plan_content) > 1000 else ""}"""

        return self._make_result(
            tool_use_id,
            output,
            metadata={
                "mode": "build",
                "plan_file": plan_file,
                "plan_content": plan_content,
            },
        )


# ============== Plan Mode Permission Check ==============


def is_operation_allowed_in_plan_mode(
    session_id: str,
    tool_name: str,
    target: str,
) -> tuple[bool, str]:
    """
    Check if an operation is allowed in plan mode.

    Returns:
        Tuple of (allowed, reason)
    """
    state = get_plan_state(session_id)

    # If not in plan mode, everything is allowed
    if not state.is_plan_mode:
        return True, ""

    # Read-only tools are always allowed
    read_only_tools = {"read", "grep", "glob", "todoread"}
    if tool_name in read_only_tools:
        return True, ""

    # Bash with read-only commands is allowed
    if tool_name == "bash":
        read_only_commands = [
            "ls",
            "cat",
            "head",
            "tail",
            "find",
            "tree",
            "pwd",
            "echo",
            "wc",
        ]
        # Simple check - could be more sophisticated
        for cmd in read_only_commands:
            if target.strip().startswith(cmd):
                return True, ""
        return (
            False,
            "In plan mode, only read-only bash commands (ls, cat, find, etc.) are allowed.",
        )

    # Edit is only allowed for the plan file
    if tool_name in {"edit", "write"}:
        if state.plan_file and target and Path(target).resolve() == Path(state.plan_file).resolve():
            return True, ""
        return (
            False,
            f"In plan mode, you can only edit the plan file: {state.plan_file}",
        )

    # plan_exit is allowed
    if tool_name == "plan_exit":
        return True, ""

    # plan_enter should not be called in plan mode
    if tool_name == "plan_enter":
        return False, "Already in plan mode. Use plan_exit to exit."

    # Task/subagent tools are blocked
    if tool_name == "task":
        return False, "Sub-agent spawning is not allowed in plan mode."

    # Default: block unknown tools
    return False, f"Tool '{tool_name}' is not allowed in plan mode."


__all__ = [
    "AgentMode",
    "PlanState",
    "PlanEnterTool",
    "PlanExitTool",
    "get_plan_state",
    "get_plan_file_path",
    "is_operation_allowed_in_plan_mode",
]
