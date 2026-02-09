"""Token counting utilities."""

from dataclasses import dataclass
from typing import Any

# Try to import tiktoken for accurate counting, fall back to estimation
try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


@dataclass
class TokenUsage:
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens + self.cache_read_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage instances."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


class TokenCounter:
    """
    Token counter for managing context window.

    Uses tiktoken if available, otherwise falls back to estimation.
    """

    # Average characters per token for estimation (varies by language)
    CHARS_PER_TOKEN = 4

    def __init__(self, model: str = "gpt-4"):
        """
        Initialize token counter.

        Args:
            model: Model name for tiktoken encoding selection
        """
        self.model = model
        self._encoding = None

        if HAS_TIKTOKEN:
            try:
                self._encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fall back to cl100k_base for unknown models
                self._encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens in

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        if self._encoding:
            return len(self._encoding.encode(text))

        # Fallback: estimate based on character count
        return self.estimate(text)

    def estimate(self, text: str) -> int:
        """
        Estimate token count (fast, less accurate).

        Uses heuristics based on character count and word boundaries.
        """
        if not text:
            return 0

        # Basic estimation: characters / 4
        char_estimate = len(text) // self.CHARS_PER_TOKEN

        # Adjust for whitespace and punctuation
        words = len(text.split())
        word_estimate = words * 1.3  # Average tokens per word

        # Return the higher estimate for safety
        return int(max(char_estimate, word_estimate))

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """
        Count tokens in a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Total token count
        """
        total = 0

        for msg in messages:
            # Count role (roughly 1-2 tokens per role)
            total += 2

            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                # Handle multi-part content
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            total += self.count(part.get("text", ""))
                        elif part.get("type") == "tool_use":
                            total += self.count(str(part.get("input", {})))
                        elif part.get("type") == "tool_result":
                            total += self.count(str(part.get("content", "")))

            # Tool calls
            tool_calls = msg.get("tool_calls", [])
            for tool_call in tool_calls:
                total += self.count(tool_call.get("function", {}).get("name", ""))
                total += self.count(tool_call.get("function", {}).get("arguments", ""))

        # Add overhead for message formatting
        total += len(messages) * 3

        return total

    def count_tool_definitions(self, tools: list[dict[str, Any]]) -> int:
        """
        Count tokens in tool definitions.

        Args:
            tools: List of tool definition dicts

        Returns:
            Total token count
        """
        total = 0
        for tool in tools:
            func = tool.get("function", tool)
            total += self.count(func.get("name", ""))
            total += self.count(func.get("description", ""))
            total += self.count(str(func.get("parameters", {})))
        return total

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within max_tokens.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated text
        """
        if self.count(text) <= max_tokens:
            return text

        if self._encoding:
            tokens = self._encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated = self._encoding.decode(tokens[:max_tokens])
            return truncated + "..."

        # Fallback: estimate and truncate by characters
        max_chars = max_tokens * self.CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."
