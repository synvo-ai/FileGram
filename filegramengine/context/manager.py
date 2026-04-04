"""Context window management with automatic compaction."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .token import TokenCounter, TokenUsage


class ContextOverflowError(Exception):
    """Raised when context window is exceeded and cannot be compacted."""

    def __init__(self, current_tokens: int, max_tokens: int):
        self.current_tokens = current_tokens
        self.max_tokens = max_tokens
        super().__init__(f"Context overflow: {current_tokens} tokens exceeds limit of {max_tokens}")


@dataclass
class MessageRecord:
    """Record of a message with metadata."""

    id: str
    role: str
    content: Any
    tokens: int
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    is_compacted: bool = False
    is_summary: bool = False


@dataclass
class ContextManager:
    """
    Manages conversation context with automatic compaction.

    Features:
    - Token counting and tracking
    - Automatic pruning of old tool outputs
    - Compaction (summarization) when approaching context limit
    - Protection of recent messages
    """

    max_context_tokens: int = 128000  # Default for GPT-4-turbo
    max_output_tokens: int = 8192
    prune_threshold: float = 0.75  # Start pruning at 75% of context
    prune_protect_tokens: int = 40000  # Protect this many tokens of recent content
    prune_minimum_savings: int = 20000  # Minimum tokens to save when pruning

    token_counter: TokenCounter = field(default_factory=TokenCounter)
    messages: list[MessageRecord] = field(default_factory=list)
    total_usage: TokenUsage = field(default_factory=TokenUsage)

    # Callback for compaction (should be set by AgentLoop)
    compaction_callback: Callable[[list[dict]], str] | None = None

    def __post_init__(self):
        """Initialize after dataclass creation."""
        if self.token_counter is None:
            self.token_counter = TokenCounter()

    @property
    def current_tokens(self) -> int:
        """Current total tokens in context."""
        return sum(msg.tokens for msg in self.messages if not msg.is_compacted)

    @property
    def usable_tokens(self) -> int:
        """Usable tokens (context - reserved output)."""
        return self.max_context_tokens - self.max_output_tokens

    def is_overflow(self, additional_tokens: int = 0) -> bool:
        """Check if context would overflow with additional tokens."""
        return (self.current_tokens + additional_tokens) > self.usable_tokens

    def should_prune(self) -> bool:
        """Check if we should start pruning."""
        return self.current_tokens > (self.usable_tokens * self.prune_threshold)

    def add_message(
        self,
        role: str,
        content: Any,
        message_id: str | None = None,
        tool_calls: list[dict] | None = None,
        tool_results: list[dict] | None = None,
    ) -> MessageRecord:
        """
        Add a message to the context.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            message_id: Optional message ID
            tool_calls: Optional tool calls
            tool_results: Optional tool results

        Returns:
            MessageRecord for the added message
        """
        if message_id is None:
            message_id = f"msg_{len(self.messages)}"

        # Calculate tokens
        tokens = self._calculate_message_tokens(role, content, tool_calls, tool_results)

        record = MessageRecord(
            id=message_id,
            role=role,
            content=content,
            tokens=tokens,
            tool_calls=tool_calls or [],
            tool_results=tool_results or [],
        )

        self.messages.append(record)
        return record

    def _calculate_message_tokens(
        self,
        role: str,
        content: Any,
        tool_calls: list[dict] | None,
        tool_results: list[dict] | None,
    ) -> int:
        """Calculate tokens for a message."""
        tokens = 2  # Role overhead

        if isinstance(content, str):
            tokens += self.token_counter.count(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    tokens += self.token_counter.count(str(part))
                else:
                    tokens += self.token_counter.count(str(part))

        if tool_calls:
            for tc in tool_calls:
                tokens += self.token_counter.count(str(tc))

        if tool_results:
            for tr in tool_results:
                tokens += self.token_counter.count(str(tr))

        return tokens

    def prune_old_tool_outputs(self) -> int:
        """
        Prune old tool outputs to save tokens.

        Keeps recent tool outputs intact, marks older ones as compacted.
        Returns number of tokens saved.
        """
        if not self.should_prune():
            return 0

        total_saved = 0
        protected_tokens = 0
        to_prune: list[MessageRecord] = []

        # Scan backwards to find tool results to prune
        for msg in reversed(self.messages):
            if msg.is_compacted or msg.is_summary:
                continue

            if protected_tokens < self.prune_protect_tokens:
                protected_tokens += msg.tokens
                continue

            # Only prune tool results
            if msg.role == "tool" or (msg.tool_results and not msg.is_compacted):
                to_prune.append(msg)
                total_saved += msg.tokens

        # Only prune if we save enough
        if total_saved < self.prune_minimum_savings:
            return 0

        for msg in to_prune:
            msg.is_compacted = True

        return total_saved

    async def compact_if_needed(self) -> bool:
        """
        Perform compaction if context is overflowing.

        Returns True if compaction was performed.
        """
        if not self.is_overflow():
            return False

        # First try pruning
        saved = self.prune_old_tool_outputs()
        if saved > 0 and not self.is_overflow():
            return True

        # If still overflowing and we have a compaction callback, use it
        if self.compaction_callback and self.is_overflow():
            # Get non-compacted messages for summarization
            messages_to_summarize = [
                {"role": msg.role, "content": msg.content}
                for msg in self.messages
                if not msg.is_compacted and not msg.is_summary
            ]

            if len(messages_to_summarize) > 2:
                summary = await self.compaction_callback(messages_to_summarize)

                # Mark old messages as compacted
                for msg in self.messages:
                    if not msg.is_summary:
                        msg.is_compacted = True

                # Add summary as new message
                self.add_message(
                    role="assistant",
                    content=summary,
                    message_id=f"summary_{len(self.messages)}",
                )
                self.messages[-1].is_summary = True
                return True

        return False

    def get_messages_for_api(self) -> list[dict[str, Any]]:
        """
        Get messages formatted for API call.

        Excludes compacted messages (except summaries).
        """
        result = []

        for msg in self.messages:
            if msg.is_compacted and not msg.is_summary:
                continue

            message_dict: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }

            if msg.tool_calls:
                message_dict["tool_calls"] = msg.tool_calls

            result.append(message_dict)

        return result

    def update_usage(self, usage: TokenUsage) -> None:
        """Update total usage statistics."""
        self.total_usage = self.total_usage + usage

    def get_stats(self) -> dict[str, Any]:
        """Get context statistics."""
        return {
            "current_tokens": self.current_tokens,
            "max_tokens": self.max_context_tokens,
            "usable_tokens": self.usable_tokens,
            "utilization": self.current_tokens / self.usable_tokens if self.usable_tokens > 0 else 0,
            "message_count": len(self.messages),
            "compacted_count": sum(1 for m in self.messages if m.is_compacted),
            "total_usage": {
                "input": self.total_usage.input_tokens,
                "output": self.total_usage.output_tokens,
                "cache_read": self.total_usage.cache_read_tokens,
                "total": self.total_usage.total,
            },
        }

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()

    def reset_usage(self) -> None:
        """Reset usage statistics."""
        self.total_usage = TokenUsage()
