"""MultiEdit tool for applying multiple edits to a single file."""

from typing import Any

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from .base import BaseTool, ToolContext
from .replacer import replace

# Load description from txt file
DESCRIPTION = load_tool_prompt("multiedit")


class MultiEditTool(BaseTool):
    """Tool for applying multiple edits to a single file atomically."""

    @property
    def name(self) -> str:
        return "multiedit"

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
                "edits": {
                    "type": "array",
                    "description": "List of edits to apply",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": {
                                "type": "string",
                                "description": "The text to replace",
                            },
                            "new_string": {
                                "type": "string",
                                "description": "The text to replace it with",
                            },
                        },
                        "required": ["old_string", "new_string"],
                    },
                },
            },
            "required": ["file_path", "edits"],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        file_path_str = arguments.get("file_path", "")
        edits = arguments.get("edits", [])

        if not file_path_str:
            return self._make_result(
                tool_use_id,
                "No file path provided",
                is_error=True,
            )

        if not edits:
            return self._make_result(
                tool_use_id,
                "No edits provided",
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
                original_content = f.read()

            # Normalize line endings
            content = original_content.replace("\r\n", "\n")

            # Apply edits sequentially
            applied_edits = []
            for i, edit in enumerate(edits):
                old_string = edit.get("old_string", "")
                new_string = edit.get("new_string", "")

                if not old_string:
                    return self._make_result(
                        tool_use_id,
                        f"Edit {i + 1}: old_string is empty",
                        is_error=True,
                    )

                if old_string == new_string:
                    return self._make_result(
                        tool_use_id,
                        f"Edit {i + 1}: old_string and new_string are identical",
                        is_error=True,
                    )

                # Normalize edit strings
                old_string = old_string.replace("\r\n", "\n")
                new_string = new_string.replace("\r\n", "\n")

                # Apply the edit using the advanced replacer
                try:
                    content = replace(content, old_string, new_string, replace_all=False)
                    applied_edits.append(
                        {
                            "index": i + 1,
                            "old_preview": old_string[:50] + "..." if len(old_string) > 50 else old_string,
                            "new_preview": new_string[:50] + "..." if len(new_string) > 50 else new_string,
                        }
                    )
                except ValueError as e:
                    return self._make_result(
                        tool_use_id,
                        f"Edit {i + 1} failed: {str(e)}",
                        is_error=True,
                    )

            # Calculate diff
            diff_info = self._calculate_diff(original_content, content)

            # Write back
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Record behavioral signal
            if context.behavior_collector:
                context.behavior_collector.record_file_edit(
                    file_path=str(file_path),
                    edit_tool="multiedit",
                    lines_added=diff_info["additions"],
                    lines_deleted=diff_info["deletions"],
                    diff_summary=f"{len(edits)} edits, +{diff_info['additions']} -{diff_info['deletions']}",
                    before_content=original_content,
                    after_content=content,
                )

            output = (
                f"Successfully applied {len(edits)} edits to {file_path}\n"
                f"Changes: +{diff_info['additions']} -{diff_info['deletions']} lines"
            )

            return self._make_result(
                tool_use_id,
                output,
                metadata={
                    "file_path": str(file_path),
                    "edits_applied": len(edits),
                    "additions": diff_info["additions"],
                    "deletions": diff_info["deletions"],
                },
            )

        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Failed to apply edits: {str(e)}",
                is_error=True,
            )

    def _calculate_diff(self, old_content: str, new_content: str) -> dict[str, int]:
        """Calculate line additions and deletions."""
        old_lines = old_content.split("\n")
        new_lines = new_content.split("\n")

        # Count actual line changes
        additions = len(new_lines) - len(old_lines) if len(new_lines) > len(old_lines) else 0
        deletions = len(old_lines) - len(new_lines) if len(old_lines) > len(new_lines) else 0

        # Count modified lines
        old_line_set = set(old_lines)
        new_line_set = set(new_lines)

        additions += len(new_line_set - old_line_set)
        deletions += len(old_line_set - new_line_set)

        return {"additions": additions, "deletions": deletions}
