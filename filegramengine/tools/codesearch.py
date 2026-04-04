"""CodeSearch tool for searching programming documentation and examples."""

from typing import Any

import httpx

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from .base import BaseTool, ToolContext

# Load description from txt file
DESCRIPTION = load_tool_prompt("codesearch")

# Exa Code API configuration
API_CONFIG = {
    "BASE_URL": "https://mcp.exa.ai",
    "ENDPOINT": "/mcp",
}


class CodeSearchTool(BaseTool):
    """Tool for searching code documentation and examples using Exa Code API.

    Provides high-quality context for libraries, SDKs, and APIs.
    """

    def __init__(self, timeout: int = 30):
        """Initialize the CodeSearch tool.

        Args:
            timeout: Request timeout in seconds
        """
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "codesearch"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query to find relevant context for APIs, Libraries, and SDKs. "
                        "For example, 'React useState hook examples', 'Python pandas dataframe filtering', "
                        "'Express.js middleware', 'Next js partial prerendering configuration'"
                    ),
                },
                "tokens_num": {
                    "type": "integer",
                    "minimum": 1000,
                    "maximum": 50000,
                    "default": 5000,
                    "description": (
                        "Number of tokens to return (1000-50000). Default is 5000 tokens. "
                        "Adjust this value based on how much context you need - use lower values "
                        "for focused queries and higher values for comprehensive documentation."
                    ),
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        query = arguments.get("query", "")
        tokens_num = arguments.get("tokens_num", 5000)

        if not query:
            return self._make_result(
                tool_use_id,
                "No search query provided",
                is_error=True,
            )

        # Validate tokens_num
        tokens_num = max(1000, min(50000, tokens_num))

        try:
            # Build MCP request for Exa Code API
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "get_code_context_exa",
                    "arguments": {
                        "query": query,
                        "tokensNum": tokens_num,
                    },
                },
            }

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{API_CONFIG['BASE_URL']}{API_CONFIG['ENDPOINT']}",
                    json=mcp_request,
                    headers={
                        "Accept": "application/json, text/event-stream",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()

                response_text = response.text

                # Parse SSE response
                for line in response_text.split("\n"):
                    if line.startswith("data: "):
                        import json

                        data = json.loads(line[6:])
                        if "result" in data and "content" in data["result"] and len(data["result"]["content"]) > 0:
                            content = data["result"]["content"][0].get("text", "")
                            return self._make_result(
                                tool_use_id,
                                content,
                                metadata={
                                    "query": query,
                                    "tokens_num": tokens_num,
                                },
                            )

                # No results found
                return self._make_result(
                    tool_use_id,
                    (
                        "No code snippets or documentation found. Please try a different query, "
                        "be more specific about the library or programming concept, or check the spelling "
                        "of framework names."
                    ),
                    metadata={
                        "query": query,
                        "tokens_num": tokens_num,
                    },
                )

        except httpx.TimeoutException:
            return self._make_result(
                tool_use_id,
                f"Code search request timed out after {self._timeout} seconds",
                is_error=True,
            )
        except httpx.HTTPStatusError as e:
            return self._make_result(
                tool_use_id,
                f"Code search error ({e.response.status_code}): {e.response.text}",
                is_error=True,
            )
        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Code search failed: {str(e)}",
                is_error=True,
            )
