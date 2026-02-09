"""WebSearch tool for searching the web."""

import os
from typing import Any

import httpx

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from .base import BaseTool, ToolContext

# Load description from txt file
DESCRIPTION = load_tool_prompt("websearch")


class WebSearchTool(BaseTool):
    """Tool for searching the web using a search API.

    Supports multiple search providers:
    - Exa (default, requires EXA_API_KEY)
    - SerpAPI (requires SERPAPI_API_KEY)
    - DuckDuckGo (no API key required, basic functionality)
    """

    def __init__(
        self,
        max_results: int = 10,
        api_key: str | None = None,
        provider: str = "duckduckgo",
    ):
        """Initialize the WebSearch tool.

        Args:
            max_results: Maximum number of results to return
            api_key: API key for the search provider
            provider: Search provider to use (exa, serpapi, duckduckgo)
        """
        self._max_results = max_results
        self._api_key = api_key or os.environ.get("EXA_API_KEY") or os.environ.get("SERPAPI_API_KEY")
        self._provider = provider

    @property
    def name(self) -> str:
        return "websearch"

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
                    "description": "The search query",
                },
                "allowed_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only include results from these domains",
                },
                "blocked_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exclude results from these domains",
                },
            },
            "required": ["query"],
        }

    async def _search_duckduckgo(
        self,
        query: str,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Search using DuckDuckGo HTML interface (no API key required)."""
        # Modify query for domain filtering
        if allowed_domains:
            site_filter = " OR ".join([f"site:{d}" for d in allowed_domains])
            query = f"({query}) ({site_filter})"

        if blocked_domains:
            for domain in blocked_domains:
                query = f"{query} -site:{domain}"

        async with httpx.AsyncClient(timeout=30) as client:
            # Use DuckDuckGo HTML version for basic search
            response = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={
                    "User-Agent": "FileGram/1.0 (Web Search Tool)",
                },
            )
            response.raise_for_status()

            # Parse results from HTML (basic extraction)
            import re

            results = []

            # Find result links
            pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
            matches = re.findall(pattern, response.text, re.DOTALL | re.IGNORECASE)

            for url, title in matches[: self._max_results]:
                # Clean up title
                title = re.sub(r"<[^>]+>", "", title).strip()
                if url and title:
                    results.append(
                        {
                            "title": title,
                            "url": url,
                            "snippet": "",  # DuckDuckGo HTML doesn't always have snippets easily
                        }
                    )

            # Also try to extract snippets
            snippet_pattern = r'<a class="result__snippet"[^>]*>(.*?)</a>'
            snippets = re.findall(snippet_pattern, response.text, re.DOTALL | re.IGNORECASE)

            for i, snippet in enumerate(snippets):
                if i < len(results):
                    results[i]["snippet"] = re.sub(r"<[^>]+>", "", snippet).strip()

            return results

    async def _search_exa(
        self,
        query: str,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Search using Exa API."""
        if not self._api_key:
            raise ValueError("EXA_API_KEY environment variable is required for Exa search")

        async with httpx.AsyncClient(timeout=30) as client:
            payload = {
                "query": query,
                "numResults": self._max_results,
                "contents": {"text": {"maxCharacters": 500}},
            }

            if allowed_domains:
                payload["includeDomains"] = allowed_domains
            if blocked_domains:
                payload["excludeDomains"] = blocked_domains

            response = await client.post(
                "https://api.exa.ai/search",
                json=payload,
                headers={
                    "x-api-key": self._api_key,
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()

            data = response.json()
            results = []

            for result in data.get("results", []):
                results.append(
                    {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("text", "")[:300] if result.get("text") else "",
                    }
                )

            return results

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        query = arguments.get("query", "")
        allowed_domains = arguments.get("allowed_domains")
        blocked_domains = arguments.get("blocked_domains")

        if not query:
            return self._make_result(
                tool_use_id,
                "No search query provided",
                is_error=True,
            )

        try:
            # Try Exa if API key is available
            if self._api_key and os.environ.get("EXA_API_KEY"):
                results = await self._search_exa(query, allowed_domains, blocked_domains)
            else:
                # Fall back to DuckDuckGo
                results = await self._search_duckduckgo(query, allowed_domains, blocked_domains)

            if not results:
                return self._make_result(
                    tool_use_id,
                    f"No results found for: {query}",
                    metadata={"query": query, "result_count": 0},
                )

            # Format results
            output_parts = [f"# Search Results for: {query}\n"]

            for i, result in enumerate(results, 1):
                output_parts.append(f"## {i}. {result['title']}")
                output_parts.append(f"URL: {result['url']}")
                if result.get("snippet"):
                    output_parts.append(f"Snippet: {result['snippet']}")
                output_parts.append("")  # Empty line between results

            output = "\n".join(output_parts)

            return self._make_result(
                tool_use_id,
                output,
                metadata={
                    "query": query,
                    "result_count": len(results),
                    "provider": "exa" if self._api_key else "duckduckgo",
                },
            )

        except httpx.HTTPStatusError as e:
            return self._make_result(
                tool_use_id,
                f"Search API error: {e.response.status_code}",
                is_error=True,
            )
        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Search failed: {str(e)}",
                is_error=True,
            )
