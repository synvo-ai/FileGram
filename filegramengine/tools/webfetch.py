"""WebFetch tool for fetching and processing web content."""

import re
from typing import Any
from urllib.parse import urlparse

import httpx

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from .base import BaseTool, ToolContext

# Load description from txt file
DESCRIPTION = load_tool_prompt("webfetch")


# Simple HTML to text conversion (can be enhanced with proper HTML parser)
def html_to_markdown(html: str) -> str:
    """Convert HTML to simplified markdown/text.

    This is a basic implementation. For production, consider using
    libraries like html2text or beautifulsoup4.
    """

    # Remove script and style elements
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Convert headers
    for i in range(1, 7):
        html = re.sub(
            rf"<h{i}[^>]*>(.*?)</h{i}>",
            rf"\n{'#' * i} \1\n",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )

    # Convert paragraphs
    html = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\1\n", html, flags=re.DOTALL | re.IGNORECASE)

    # Convert line breaks
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)

    # Convert links
    html = re.sub(
        r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>',
        r"[\2](\1)",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Convert bold
    html = re.sub(r"<(strong|b)[^>]*>(.*?)</\1>", r"**\2**", html, flags=re.DOTALL | re.IGNORECASE)

    # Convert italic
    html = re.sub(r"<(em|i)[^>]*>(.*?)</\1>", r"*\2*", html, flags=re.DOTALL | re.IGNORECASE)

    # Convert code blocks
    html = re.sub(
        r"<pre[^>]*><code[^>]*>(.*?)</code></pre>",
        r"\n```\n\1\n```\n",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    html = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", html, flags=re.DOTALL | re.IGNORECASE)

    # Convert lists
    html = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"</?[ou]l[^>]*>", "\n", html, flags=re.IGNORECASE)

    # Remove all remaining HTML tags
    html = re.sub(r"<[^>]+>", "", html)

    # Decode HTML entities
    html = html.replace("&nbsp;", " ")
    html = html.replace("&lt;", "<")
    html = html.replace("&gt;", ">")
    html = html.replace("&amp;", "&")
    html = html.replace("&quot;", '"')
    html = html.replace("&#39;", "'")

    # Clean up whitespace
    html = re.sub(r"\n\s*\n\s*\n", "\n\n", html)
    html = html.strip()

    return html


class WebFetchTool(BaseTool):
    """Tool for fetching web content and converting to markdown."""

    def __init__(self, max_content_size: int = 100000, timeout: int = 30):
        """Initialize the WebFetch tool.

        Args:
            max_content_size: Maximum content size in characters
            timeout: Request timeout in seconds
        """
        self._max_content_size = max_content_size
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "webfetch"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from",
                },
                "prompt": {
                    "type": "string",
                    "description": "Optional prompt to describe what information to extract (for context)",
                },
            },
            "required": ["url"],
        }

    def _validate_url(self, url: str) -> tuple[bool, str]:
        """Validate and normalize the URL.

        Returns:
            Tuple of (is_valid, normalized_url_or_error)
        """
        try:
            # Auto-upgrade HTTP to HTTPS
            if url.startswith("http://"):
                url = "https://" + url[7:]

            # Add https:// if no scheme
            if not url.startswith(("https://", "http://")):
                url = "https://" + url

            parsed = urlparse(url)

            if not parsed.netloc:
                return False, "Invalid URL: missing domain"

            if parsed.scheme not in ("http", "https"):
                return False, f"Invalid URL scheme: {parsed.scheme}"

            return True, url

        except Exception as e:
            return False, f"Invalid URL: {str(e)}"

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        url = arguments.get("url", "")
        prompt = arguments.get("prompt", "")

        if not url:
            return self._make_result(
                tool_use_id,
                "No URL provided",
                is_error=True,
            )

        # Validate URL
        is_valid, result = self._validate_url(url)
        if not is_valid:
            return self._make_result(
                tool_use_id,
                result,
                is_error=True,
            )

        url = result  # normalized URL

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "FileGram/1.0 (Web Fetch Tool)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                content = response.text

                # Check for redirect to different host
                if response.history:
                    original_host = urlparse(url).netloc
                    final_host = urlparse(str(response.url)).netloc
                    if original_host != final_host:
                        return self._make_result(
                            tool_use_id,
                            f"URL redirected to different host: {response.url}\n"
                            f"Please make a new request with the redirect URL.",
                            metadata={
                                "redirect_url": str(response.url),
                                "original_url": url,
                            },
                        )

                # Convert HTML to markdown
                if "text/html" in content_type or "application/xhtml" in content_type:
                    content = html_to_markdown(content)

                # Truncate if too large
                truncated = False
                if len(content) > self._max_content_size:
                    content = content[: self._max_content_size]
                    truncated = True

                # Build output
                output_parts = [f"# Content from {url}\n"]
                if prompt:
                    output_parts.append(f"Requested info: {prompt}\n")
                output_parts.append(content)
                if truncated:
                    output_parts.append("\n\n[Content truncated due to size limit]")

                output = "\n".join(output_parts)

                return self._make_result(
                    tool_use_id,
                    output,
                    metadata={
                        "url": url,
                        "content_type": content_type,
                        "content_length": len(content),
                        "truncated": truncated,
                    },
                )

        except httpx.TimeoutException:
            return self._make_result(
                tool_use_id,
                f"Request timed out after {self._timeout} seconds",
                is_error=True,
            )
        except httpx.HTTPStatusError as e:
            return self._make_result(
                tool_use_id,
                f"HTTP error {e.response.status_code}: {e.response.reason_phrase}",
                is_error=True,
            )
        except httpx.RequestError as e:
            return self._make_result(
                tool_use_id,
                f"Request failed: {str(e)}",
                is_error=True,
            )
        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Failed to fetch URL: {str(e)}",
                is_error=True,
            )
