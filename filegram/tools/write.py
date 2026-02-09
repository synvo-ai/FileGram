"""Write tool for writing file contents."""

from typing import Any

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from .base import BaseTool, ToolContext

# Load description from txt file (like OpenCode's static import)
DESCRIPTION = load_tool_prompt("write")


class WriteTool(BaseTool):
    """Tool for writing file contents."""

    @property
    def name(self) -> str:
        return "write"

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
                    "description": "The absolute path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        file_path_str = arguments.get("file_path", "")
        content = arguments.get("content", "")

        if not file_path_str:
            return self._make_result(
                tool_use_id,
                "No file path provided",
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

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file exists for behavioral tracking
            before_content = None
            file_existed = file_path.exists()
            if file_existed:
                try:
                    with open(file_path, encoding="utf-8", errors="replace") as f:
                        before_content = f.read()
                except Exception:
                    pass

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

            # Record behavioral signal
            if context.behavior_collector:
                context.behavior_collector.record_file_write(
                    file_path=str(file_path),
                    operation="overwrite" if file_existed else "create",
                    content_length=len(content),
                    before_content=before_content,
                    after_content=content,
                )

            return self._make_result(
                tool_use_id,
                f"Successfully wrote {len(content)} bytes ({line_count} lines) to {file_path}",
                metadata={
                    "file_path": str(file_path),
                    "bytes_written": len(content),
                    "lines": line_count,
                },
            )

        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Failed to write file: {str(e)}",
                is_error=True,
            )
