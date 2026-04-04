"""Glob tool for matching file patterns."""

from typing import Any

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from ..utils.truncate import truncate_output
from .base import BaseTool, ToolContext

# Load description from txt file (like OpenCode's static import)
DESCRIPTION = load_tool_prompt("glob")


class GlobTool(BaseTool):
    """Tool for finding files by pattern."""

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern to match files against (e.g., '**/*.py')",
                },
                "path": {
                    "type": "string",
                    "description": "Base directory for the search (default: target directory)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 100)",
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        pattern = arguments.get("pattern", "")
        path = arguments.get("path", str(context.target_directory))
        max_results = arguments.get("max_results", 100)

        if not pattern:
            return self._make_result(
                tool_use_id,
                "No pattern provided",
                is_error=True,
            )

        try:
            search_path = context.resolve_path(path)
        except ValueError as e:
            return self._make_result(
                tool_use_id,
                str(e),
                is_error=True,
            )

        if not search_path.exists():
            return self._make_result(
                tool_use_id,
                f"Path not found: {search_path}",
                is_error=True,
            )

        try:
            matches = list(search_path.glob(pattern))

            matches = [m for m in matches if m.is_file()]

            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            total_matches = len(matches)
            if total_matches > max_results:
                matches = matches[:max_results]

            relative_paths = []
            for match in matches:
                try:
                    rel_path = match.relative_to(context.target_directory)
                    relative_paths.append(str(rel_path))
                except ValueError:
                    relative_paths.append(str(match))

            if not relative_paths:
                # Record behavioral signal for empty search
                if context.behavior_collector:
                    context.behavior_collector.record_file_search(
                        search_type="glob",
                        query=pattern,
                        files_matched=0,
                    )
                return self._make_result(
                    tool_use_id,
                    f"No files found matching pattern: {pattern}",
                    metadata={"matches": 0},
                )

            output = "\n".join(relative_paths)

            if total_matches > max_results:
                output += f"\n\n[Showing first {max_results} of {total_matches} matches]"

            truncated_output = truncate_output(output, context.max_output_chars)

            # Record behavioral signal
            if context.behavior_collector:
                context.behavior_collector.record_file_search(
                    search_type="glob",
                    query=pattern,
                    files_matched=min(total_matches, max_results),
                )

            return self._make_result(
                tool_use_id,
                truncated_output,
                metadata={"matches": min(total_matches, max_results)},
            )

        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Glob search failed: {str(e)}",
                is_error=True,
            )
