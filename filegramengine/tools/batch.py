"""Batch tool for executing multiple tool calls in parallel."""

import asyncio
import json
from typing import Any

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from .base import BaseTool, ToolContext

# Load description from txt file
DESCRIPTION = load_tool_prompt("batch")

# Tools that cannot be batched
DISALLOWED = {"batch"}
MAX_BATCH_SIZE = 25


class BatchTool(BaseTool):
    """Tool for executing multiple tool calls in parallel.

    This improves efficiency by running independent operations concurrently.
    """

    def __init__(self, registry=None):
        """Initialize with a reference to the tool registry.

        Args:
            registry: The ToolRegistry instance to use for executing tools
        """
        self._registry = registry

    def set_registry(self, registry):
        """Set the tool registry (called after registration)."""
        self._registry = registry

    @property
    def name(self) -> str:
        return "batch"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tool_calls": {
                    "type": "array",
                    "description": "Array of tool calls to execute in parallel",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {
                                "type": "string",
                                "description": "The name of the tool to execute",
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Parameters for the tool",
                            },
                        },
                        "required": ["tool", "parameters"],
                    },
                    "minItems": 1,
                },
            },
            "required": ["tool_calls"],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        if self._registry is None:
            return self._make_result(
                tool_use_id,
                "Batch tool not properly initialized: registry not set",
                is_error=True,
            )

        tool_calls = arguments.get("tool_calls", [])

        # LLM sometimes sends tool_calls as a JSON string instead of an array
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except (json.JSONDecodeError, TypeError):
                return self._make_result(
                    tool_use_id,
                    f"Invalid tool_calls format: expected array, got string",
                    is_error=True,
                )

        if not tool_calls:
            return self._make_result(
                tool_use_id,
                "No tool calls provided",
                is_error=True,
            )

        # Limit batch size
        if len(tool_calls) > MAX_BATCH_SIZE:
            tool_calls = tool_calls[:MAX_BATCH_SIZE]
            discarded = len(arguments.get("tool_calls", [])) - MAX_BATCH_SIZE
        else:
            discarded = 0

        async def execute_call(call: dict[str, Any], index: int) -> dict[str, Any]:
            """Execute a single tool call."""
            tool_name = call.get("tool", "")
            params = call.get("parameters", {})

            # Check if tool is disallowed
            if tool_name in DISALLOWED:
                return {
                    "index": index,
                    "tool": tool_name,
                    "success": False,
                    "error": f"Tool '{tool_name}' cannot be used in batch",
                }

            # Get the tool from registry
            tool = self._registry.get(tool_name)
            if tool is None:
                available = self._registry.list_tools()
                available = [t for t in available if t not in DISALLOWED]
                return {
                    "index": index,
                    "tool": tool_name,
                    "success": False,
                    "error": f"Tool '{tool_name}' not found. Available: {', '.join(available)}",
                }

            try:
                # Execute the tool
                result = await tool.execute(
                    f"{tool_use_id}_batch_{index}",
                    params,
                    context,
                )

                return {
                    "index": index,
                    "tool": tool_name,
                    "success": not result.is_error,
                    "output": result.output,
                    "error": result.output if result.is_error else None,
                }
            except Exception as e:
                return {
                    "index": index,
                    "tool": tool_name,
                    "success": False,
                    "error": str(e),
                }

        # Execute all calls in parallel
        tasks = [execute_call(call, i) for i, call in enumerate(tool_calls)]
        results = await asyncio.gather(*tasks)

        # Count successes and failures
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        # Build output message
        if failed > 0:
            output = f"Executed {successful}/{len(results)} tools successfully. {failed} failed."
            # Add error details
            for r in results:
                if not r["success"]:
                    output += f"\n- {r['tool']}: {r['error']}"
        else:
            output = (
                f"All {successful} tools executed successfully.\n\nKeep using the batch tool for optimal performance!"
            )

        if discarded > 0:
            output += f"\n\nNote: {discarded} tool calls were discarded (max {MAX_BATCH_SIZE} per batch)"

        return self._make_result(
            tool_use_id,
            output,
            metadata={
                "total_calls": len(results),
                "successful": successful,
                "failed": failed,
                "tools": [call.get("tool") for call in tool_calls],
                "results": results,
            },
        )
