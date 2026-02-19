"""Base tool class and tool context."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ..models.tool import ToolDefinition, ToolResult

if TYPE_CHECKING:
    from ..behavior import BehaviorCollector


@dataclass
class ToolContext:
    """Context provided to tools during execution."""

    session_id: str
    target_directory: Path
    max_output_chars: int = 30000
    default_timeout: int = 120000  # 2 minutes in ms
    sandbox_enabled: bool = True  # If True, restrict file access to target_directory
    behavior_collector: Optional["BehaviorCollector"] = None  # Behavioral signal collector

    def resolve_path(self, path_str: str) -> Path:
        """Resolve a path string to an absolute path within target_directory.

        Args:
            path_str: Path string (absolute or relative)

        Returns:
            Resolved absolute path

        Raises:
            ValueError: If sandbox is enabled and path escapes target_directory
        """
        path = Path(path_str)

        # Convert relative path to absolute
        if not path.is_absolute():
            path = self.target_directory / path

        # Resolve to get canonical path (handles .. and symlinks)
        resolved = path.resolve()

        # Check if path is within target_directory when sandbox is enabled
        if self.sandbox_enabled:
            target_resolved = self.target_directory.resolve()
            try:
                resolved.relative_to(target_resolved)
            except ValueError:
                # Auto-correct: LLM sometimes generates absolute paths with a
                # wrong root prefix (e.g. "Claude Code" instead of "FileGram")
                # but the relative portion within the sandbox dir is correct.
                # Try to extract the relative part and remap into sandbox.
                corrected = self._auto_correct_path(path, target_resolved)
                if corrected is not None:
                    return corrected
                raise ValueError(
                    f"Access denied: '{path_str}' is outside the allowed directory. Allowed: {target_resolved}"
                )

        return resolved

    def _auto_correct_path(self, bad_path: Path, target_dir: Path) -> Path | None:
        """Try to remap an out-of-sandbox path back into the sandbox.

        Looks for the sandbox directory name in the bad path's components.
        If found, extracts everything after it as a relative path and
        reconstructs using the real target_dir.
        """
        target_name = target_dir.name
        parts = bad_path.parts
        for i, part in enumerate(parts):
            if part == target_name:
                # Build relative path from everything after the match
                if i + 1 < len(parts):
                    relative = Path(*parts[i + 1 :])
                    corrected = (target_dir / relative).resolve()
                else:
                    corrected = target_dir.resolve()
                # Verify corrected path is actually within sandbox
                try:
                    corrected.relative_to(target_dir)
                    return corrected
                except ValueError:
                    continue
        return None

    def is_path_allowed(self, path_str: str) -> bool:
        """Check if a path is within the allowed directory.

        Args:
            path_str: Path string to check

        Returns:
            True if path is allowed, False otherwise
        """
        try:
            self.resolve_path(path_str)
            return True
        except ValueError:
            return False


class BaseTool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in API calls."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the LLM."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        pass

    def get_definition(self) -> ToolDefinition:
        """Get the tool definition for OpenAI API."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    @abstractmethod
    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """
        Execute the tool with given arguments.

        Args:
            tool_use_id: Unique ID for this tool use
            arguments: Arguments passed to the tool
            context: Execution context

        Returns:
            ToolResult with output or error
        """
        pass

    def _make_result(
        self,
        tool_use_id: str,
        output: str,
        is_error: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Helper to create a ToolResult."""
        return ToolResult(
            tool_use_id=tool_use_id,
            name=self.name,
            output=output,
            is_error=is_error,
            metadata=metadata,
        )
