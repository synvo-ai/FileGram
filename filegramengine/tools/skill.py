"""Skill tool for loading and executing skills."""

from pathlib import Path
from typing import Any

from ..models.tool import ToolResult
from ..skill import SkillLoader, SkillNotFoundError, get_default_loader
from .base import BaseTool, ToolContext


class SkillTool(BaseTool):
    """Tool for loading skills.

    Skills provide specialized instructions for specific tasks.
    The description is dynamically generated to list available skills.
    """

    def __init__(
        self,
        project_dir: str | Path | None = None,
        custom_paths: list[str | Path] | None = None,
        disable_claude_skills: bool = False,
    ):
        """
        Initialize the skill tool.

        Args:
            project_dir: Project directory to scan for skills.
            custom_paths: Additional paths to scan for skills.
            disable_claude_skills: Whether to disable Claude Code skills.
        """
        self._loader = SkillLoader(
            project_dir=project_dir,
            custom_paths=custom_paths,
            disable_claude_skills=disable_claude_skills,
        )
        self._cached_description: str | None = None

    @property
    def name(self) -> str:
        return "skill"

    @property
    def description(self) -> str:
        """Generate description listing available skills."""
        if self._cached_description is not None:
            return self._cached_description

        skills = self._loader.all()

        if not skills:
            return "Load a skill to get detailed instructions for a specific task. No skills are currently available."

        # Build description with available skills
        lines = [
            "Load a skill to get detailed instructions for a specific task.",
            "Skills provide specialized knowledge and step-by-step guidance.",
            "Use this when a task matches an available skill's description.",
            "Only the skills listed here are available:",
            "<available_skills>",
        ]

        for skill in skills:
            lines.extend(
                [
                    "  <skill>",
                    f"    <name>{skill.name}</name>",
                    f"    <description>{skill.description}</description>",
                    "  </skill>",
                ]
            )

        lines.append("</available_skills>")

        self._cached_description = " ".join(lines)
        return self._cached_description

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for skill tool parameters."""
        skills = self._loader.all()
        examples = [f"'{s.name}'" for s in skills[:3]]
        hint = f" (e.g., {', '.join(examples)}, ...)" if examples else ""

        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": f"The skill identifier from available_skills{hint}",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        }

    def reload_skills(self) -> None:
        """Reload skills and clear cached description."""
        self._loader.reload()
        self._cached_description = None

    def get_available_skills(self) -> list[str]:
        """Get list of available skill names."""
        return [s.name for s in self._loader.all()]

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """
        Execute the skill tool.

        Args:
            tool_use_id: Unique ID for this tool use.
            arguments: Should contain 'name' key with skill name.
            context: Execution context.

        Returns:
            ToolResult with skill content or error.
        """
        skill_name = arguments.get("name", "")

        if not skill_name:
            return self._make_result(
                tool_use_id=tool_use_id,
                output="Error: skill name is required",
                is_error=True,
            )

        try:
            # Get skill content
            content = self._loader.get_content(skill_name)
            skill_info = self._loader.get(skill_name)

            # Get the skill directory for metadata
            skill_dir = str(Path(skill_info.location).parent) if skill_info else ""

            return self._make_result(
                tool_use_id=tool_use_id,
                output=content,
                is_error=False,
                metadata={
                    "name": skill_name,
                    "dir": skill_dir,
                },
            )

        except SkillNotFoundError as e:
            return self._make_result(
                tool_use_id=tool_use_id,
                output=str(e),
                is_error=True,
            )
        except Exception as e:
            return self._make_result(
                tool_use_id=tool_use_id,
                output=f"Failed to load skill '{skill_name}': {e}",
                is_error=True,
            )


class DynamicSkillTool(BaseTool):
    """A skill tool that uses the default global loader.

    This version recalculates available skills on each call,
    which is useful when skills may be added/removed during runtime.
    """

    @property
    def name(self) -> str:
        return "skill"

    @property
    def description(self) -> str:
        """Generate description listing available skills."""
        loader = get_default_loader()
        skills = loader.all()

        if not skills:
            return "Load a skill to get detailed instructions for a specific task. No skills are currently available."

        lines = [
            "Load a skill to get detailed instructions for a specific task.",
            "Skills provide specialized knowledge and step-by-step guidance.",
            "Use this when a task matches an available skill's description.",
            "Only the skills listed here are available:",
            "<available_skills>",
        ]

        for skill in skills:
            lines.extend(
                [
                    "  <skill>",
                    f"    <name>{skill.name}</name>",
                    f"    <description>{skill.description}</description>",
                    "  </skill>",
                ]
            )

        lines.append("</available_skills>")

        return " ".join(lines)

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for skill tool parameters."""
        loader = get_default_loader()
        skills = loader.all()
        examples = [f"'{s.name}'" for s in skills[:3]]
        hint = f" (e.g., {', '.join(examples)}, ...)" if examples else ""

        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": f"The skill identifier from available_skills{hint}",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the skill tool."""
        skill_name = arguments.get("name", "")

        if not skill_name:
            return self._make_result(
                tool_use_id=tool_use_id,
                output="Error: skill name is required",
                is_error=True,
            )

        try:
            loader = get_default_loader()
            content = loader.get_content(skill_name)
            skill_info = loader.get(skill_name)
            skill_dir = str(Path(skill_info.location).parent) if skill_info else ""

            return self._make_result(
                tool_use_id=tool_use_id,
                output=content,
                is_error=False,
                metadata={
                    "name": skill_name,
                    "dir": skill_dir,
                },
            )

        except SkillNotFoundError as e:
            return self._make_result(
                tool_use_id=tool_use_id,
                output=str(e),
                is_error=True,
            )
        except Exception as e:
            return self._make_result(
                tool_use_id=tool_use_id,
                output=f"Failed to load skill '{skill_name}': {e}",
                is_error=True,
            )
