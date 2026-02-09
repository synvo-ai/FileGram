"""Bash tool for executing shell commands."""

import asyncio
import subprocess
from typing import Any

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from ..utils.truncate import truncate_output
from .base import BaseTool, ToolContext

# Load description from txt file (like OpenCode's static import)
DESCRIPTION = load_tool_prompt("bash")


class BashTool(BaseTool):
    """Tool for executing shell commands."""

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default: 120000)",
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory for the command (optional)",
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        command = arguments.get("command", "")
        timeout_ms = arguments.get("timeout", context.default_timeout)
        workdir = arguments.get("workdir", str(context.target_directory))

        if not command:
            return self._make_result(
                tool_use_id,
                "No command provided",
                is_error=True,
            )

        timeout_seconds = timeout_ms / 1000

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=workdir,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                return self._make_result(
                    tool_use_id,
                    f"Command timed out after {timeout_seconds}s",
                    is_error=True,
                )

            output_parts = []
            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                output_parts.append(f"stderr:\n{stderr.decode('utf-8', errors='replace')}")

            output = "\n".join(output_parts) if output_parts else "(no output)"

            if process.returncode != 0:
                output = f"Exit code: {process.returncode}\n{output}"

            truncated_output = truncate_output(output, context.max_output_chars)

            return self._make_result(
                tool_use_id,
                truncated_output,
                is_error=process.returncode != 0,
                metadata={"exit_code": process.returncode},
            )

        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Failed to execute command: {str(e)}",
                is_error=True,
            )
