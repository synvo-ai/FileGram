"""Edit tool for modifying file contents with advanced replacement strategies."""

from typing import Any

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from .base import BaseTool, ToolContext
from .replacer import replace

# Load description from txt file (like OpenCode's static import)
DESCRIPTION = load_tool_prompt("edit")


class EditTool(BaseTool):
    """Tool for editing file contents via advanced string replacement.

    Uses 9 progressive replacement strategies:
    1. SimpleReplacer - exact match
    2. LineTrimmedReplacer - handles indentation changes
    3. BlockAnchorReplacer - uses first/last lines as anchors
    4. WhitespaceNormalizedReplacer - normalizes whitespace
    5. IndentationFlexibleReplacer - ignores overall indentation
    6. EscapeNormalizedReplacer - handles escape sequences
    7. TrimmedBoundaryReplacer - handles boundary whitespace
    8. ContextAwareReplacer - uses context for matching
    9. MultiOccurrenceReplacer - handles multiple matches
    """

    @property
    def name(self) -> str:
        return "edit"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to edit",
                },
                "old_string": {
                    "type": "string",
                    "description": "The text to replace (can be fuzzy matched)",
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace it with (must be different from old_string)",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false)",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        file_path_str = arguments.get("file_path", "")
        old_string = arguments.get("old_string", "")
        new_string = arguments.get("new_string", "")
        replace_all = arguments.get("replace_all", False)

        if not file_path_str:
            return self._make_result(
                tool_use_id,
                "No file path provided",
                is_error=True,
            )

        if not old_string:
            # Empty old_string means create file with new content
            return await self._create_file(tool_use_id, file_path_str, new_string, context)

        if old_string == new_string:
            return self._make_result(
                tool_use_id,
                "old_string and new_string are identical",
                is_error=True,
            )

        try:
            file_path = context.resolve_path(file_path_str)
        except ValueError as e:
            return self._make_result(
                tool_use_id,
                str(e),
                is_error=True,
            )

        if not file_path.exists():
            return self._make_result(
                tool_use_id,
                f"File not found: {file_path}",
                is_error=True,
            )

        try:
            # Read file content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Store original for behavioral tracking
            original_content = content

            # Normalize line endings
            content = content.replace("\r\n", "\n")
            old_string = old_string.replace("\r\n", "\n")
            new_string = new_string.replace("\r\n", "\n")

            # Use advanced replacement
            try:
                new_content = replace(
                    content,
                    old_string,
                    new_string,
                    replace_all=replace_all,
                )
            except ValueError as e:
                return self._make_result(
                    tool_use_id,
                    str(e),
                    is_error=True,
                )

            # Calculate diff for metadata
            diff_info = self._calculate_diff(content, new_content)

            # Write back
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            # Record behavioral signal
            if context.behavior_collector:
                context.behavior_collector.record_file_edit(
                    file_path=str(file_path),
                    edit_tool="edit",
                    lines_added=diff_info["additions"],
                    lines_deleted=diff_info["deletions"],
                    diff_summary=f"+{diff_info['additions']} -{diff_info['deletions']}",
                    before_content=original_content,
                    after_content=new_content,
                )

            output = (
                f"Successfully edited {file_path}\nChanges: +{diff_info['additions']} -{diff_info['deletions']} lines"
            )

            return self._make_result(
                tool_use_id,
                output,
                metadata={
                    "file_path": str(file_path),
                    "additions": diff_info["additions"],
                    "deletions": diff_info["deletions"],
                },
            )

        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Failed to edit file: {str(e)}",
                is_error=True,
            )

    async def _create_file(
        self,
        tool_use_id: str,
        file_path_str: str,
        content: str,
        context: ToolContext,
    ) -> ToolResult:
        """Create a new file with the given content."""
        try:
            file_path = context.resolve_path(file_path_str)
        except ValueError as e:
            return self._make_result(
                tool_use_id,
                str(e),
                is_error=True,
            )

        try:
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            existed = file_path.exists()

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            action = "Updated" if existed else "Created"
            line_count = len(content.split("\n"))

            return self._make_result(
                tool_use_id,
                f"{action} file: {file_path} ({line_count} lines)",
                metadata={
                    "file_path": str(file_path),
                    "created": not existed,
                    "lines": line_count,
                },
            )

        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Failed to create file: {str(e)}",
                is_error=True,
            )

    def _calculate_diff(self, old_content: str, new_content: str) -> dict[str, int]:
        """Calculate line additions and deletions."""
        old_lines = set(old_content.split("\n"))
        new_lines = set(new_content.split("\n"))

        additions = len(new_lines - old_lines)
        deletions = len(old_lines - new_lines)

        return {"additions": additions, "deletions": deletions}
