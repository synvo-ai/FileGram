"""Grep tool for searching file contents."""

import asyncio
import subprocess
from typing import Any

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from ..utils.truncate import truncate_output
from .base import BaseTool, ToolContext

# Load description from txt file (like OpenCode's static import)
DESCRIPTION = load_tool_prompt("grep")


class GrepTool(BaseTool):
    """Tool for searching file contents using grep/ripgrep."""

    @property
    def name(self) -> str:
        return "grep"

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
                    "description": "The regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (default: target directory)",
                },
                "include": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.py')",
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Case insensitive search (default: false)",
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
        include = arguments.get("include")
        ignore_case = arguments.get("ignore_case", False)
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

        rg_available = await self._check_command_available("rg")

        if rg_available:
            cmd = ["rg", "--line-number", "--no-heading", f"--max-count={max_results}"]
            if ignore_case:
                cmd.append("-i")
            if include:
                cmd.extend(["--glob", include])
            cmd.append(pattern)
            cmd.append(str(search_path))
        else:
            cmd = ["grep", "-rn"]
            if ignore_case:
                cmd.append("-i")
            if include:
                cmd.extend(["--include", include])
            cmd.append(pattern)
            cmd.append(str(search_path))

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(context.target_directory),
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30,
            )

            output = stdout.decode("utf-8", errors="replace")

            if not output.strip():
                # Record behavioral signal for empty search
                if context.behavior_collector:
                    context.behavior_collector.record_file_search(
                        search_type="grep",
                        query=pattern,
                        files_matched=0,
                    )
                return self._make_result(
                    tool_use_id,
                    f"No matches found for pattern: {pattern}",
                    metadata={"matches": 0},
                )

            lines = output.strip().split("\n")
            match_count = len(lines)

            if match_count > max_results:
                lines = lines[:max_results]
                output = "\n".join(lines) + f"\n\n[Showing first {max_results} of {match_count} matches]"
            else:
                output = "\n".join(lines)

            truncated_output = truncate_output(output, context.max_output_chars)

            # Record behavioral signal
            if context.behavior_collector:
                context.behavior_collector.record_file_search(
                    search_type="grep",
                    query=pattern,
                    files_matched=min(match_count, max_results),
                )

            return self._make_result(
                tool_use_id,
                truncated_output,
                metadata={"matches": min(match_count, max_results)},
            )

        except asyncio.TimeoutError:
            return self._make_result(
                tool_use_id,
                "Search timed out after 30 seconds",
                is_error=True,
            )
        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Search failed: {str(e)}",
                is_error=True,
            )

    async def _check_command_available(self, command: str) -> bool:
        """Check if a command is available in PATH."""
        try:
            process = await asyncio.create_subprocess_exec(
                "which",
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            await process.communicate()
            return process.returncode == 0
        except Exception:
            return False
