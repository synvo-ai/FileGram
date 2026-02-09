"""Conversation compaction (summarization) with intelligent pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..context.token import TokenCounter
    from ..models.message import Message, ToolPart


# ============== Constants ==============

# Prune settings (matching OpenCode)
PRUNE_MINIMUM = 20_000  # Only prune if we save at least 20K tokens
PRUNE_PROTECT = 40_000  # Protect last 40K tokens of tool outputs

# Protected tools that should never be pruned
PRUNE_PROTECTED_TOOLS = {"skill", "task"}


# ============== Prompts ==============

COMPACTION_SYSTEM_PROMPT = """You are a helpful AI assistant tasked with summarizing conversations.

When asked to summarize, provide a detailed but concise summary of the conversation.
Focus on information that would be helpful for continuing the conversation, including:
- What was done
- What is currently being worked on
- Which files are being modified
- What needs to be done next
- Key user requests, constraints, or preferences that should persist
- Important technical decisions and why they were made

Your summary should be comprehensive enough to provide context but concise enough to be quickly understood.
"""

COMPACTION_USER_PROMPT = (
    "Provide a detailed prompt for continuing our conversation above. Focus on information "
    "that would be helpful for continuing the conversation, including what we did, what we're "
    "doing, which files we're working on, and what we're going to do next considering new session "
    "will not have access to our conversation."
)


# ============== LLM Protocol ==============


class LLMClient(Protocol):
    """Protocol for LLM client used by Compactor."""

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Make a chat completion request."""
        ...


# ============== Results ==============


@dataclass
class CompactionResult:
    """Result of a compaction operation."""

    summary: str
    original_tokens: int
    summary_tokens: int
    tokens_saved: int


@dataclass
class PruneResult:
    """Result of a prune operation."""

    pruned_count: int
    tokens_saved: int
    protected_tokens: int


# ============== Smart Pruner ==============


class SmartPruner:
    """
    Intelligent pruning of old tool outputs.

    Strategy (matching OpenCode):
    1. Scan backwards through messages
    2. Skip the last 2 user turns (recent context)
    3. Stop at any summary message (already compacted)
    4. Protect last 40K tokens of tool outputs
    5. Only prune if we save at least 20K tokens
    6. Mark pruned parts with compacted timestamp (soft delete)
    7. Never prune protected tools (skill, task)
    """

    def __init__(
        self,
        token_counter: TokenCounter,
        prune_minimum: int = PRUNE_MINIMUM,
        prune_protect: int = PRUNE_PROTECT,
        protected_tools: set[str] = PRUNE_PROTECTED_TOOLS,
    ):
        self.token_counter = token_counter
        self.prune_minimum = prune_minimum
        self.prune_protect = prune_protect
        self.protected_tools = protected_tools

    def prune(self, messages: list[Message]) -> PruneResult:
        """
        Prune old tool outputs from messages.

        Args:
            messages: List of messages to prune

        Returns:
            PruneResult with statistics
        """
        from ..models.message import ToolPart, ToolStatus

        total_tokens = 0
        protected_tokens = 0
        pruned_tokens = 0
        to_prune: list[ToolPart] = []
        user_turns = 0

        # Scan backwards
        for msg_idx in range(len(messages) - 1, -1, -1):
            msg = messages[msg_idx]

            # Count user turns
            if msg.role.value == "user":
                user_turns += 1

            # Skip last 2 user turns
            if user_turns < 2:
                continue

            # Stop at summary messages
            if hasattr(msg, "info") and msg.info and hasattr(msg.info, "summary") and msg.info.summary:
                break

            # Process parts
            for part in msg.content:
                if not isinstance(part, ToolPart):
                    continue

                # Skip already compacted
                if part.is_compacted:
                    break  # Stop at compaction boundary

                # Skip protected tools
                if part.tool in self.protected_tools:
                    continue

                # Skip non-completed tools
                if part.state.status != ToolStatus.COMPLETED:
                    continue

                # Estimate output tokens
                output_tokens = self.token_counter.estimate(part.state.output)
                total_tokens += output_tokens

                # Protect recent outputs
                if total_tokens <= self.prune_protect:
                    protected_tokens += output_tokens
                    continue

                # Mark for pruning
                to_prune.append(part)
                pruned_tokens += output_tokens

        # Only prune if we save enough
        if pruned_tokens < self.prune_minimum:
            return PruneResult(
                pruned_count=0,
                tokens_saved=0,
                protected_tokens=protected_tokens,
            )

        # Mark parts as compacted
        for part in to_prune:
            part.mark_compacted()

        return PruneResult(
            pruned_count=len(to_prune),
            tokens_saved=pruned_tokens,
            protected_tokens=protected_tokens,
        )


# ============== Compactor ==============


class Compactor:
    """
    Handles conversation compaction (summarization).

    When the context window is getting full, this class can summarize
    the conversation to free up space while preserving important context.
    """

    def __init__(self, llm_client: Any):
        """
        Initialize compactor.

        Args:
            llm_client: LLM client for generating summaries
        """
        self.llm_client = llm_client

    async def compact(
        self,
        messages: list[dict[str, Any]],
        token_counter: TokenCounter | None = None,
    ) -> CompactionResult:
        """
        Compact a conversation by summarizing it.

        Args:
            messages: List of messages to summarize
            token_counter: Optional token counter for statistics

        Returns:
            CompactionResult with summary and statistics
        """
        # Calculate original tokens if counter provided
        original_tokens = 0
        if token_counter:
            original_tokens = token_counter.count_messages(messages)

        # Build messages for summarization
        summarization_messages = [
            {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
            *messages,
            {"role": "user", "content": COMPACTION_USER_PROMPT},
        ]

        # Get summary from LLM
        summary, _ = self.llm_client.chat_completion(
            messages=summarization_messages,
            tools=None,
        )

        # Calculate summary tokens
        summary_tokens = 0
        if token_counter:
            summary_tokens = token_counter.count(summary)

        return CompactionResult(
            summary=summary,
            original_tokens=original_tokens,
            summary_tokens=summary_tokens,
            tokens_saved=original_tokens - summary_tokens,
        )

    def create_summary_message(self, summary: str) -> dict[str, Any]:
        """
        Create a message containing the conversation summary.

        This message can be used to replace the original conversation.
        """
        return {
            "role": "assistant",
            "content": f"""[Conversation Summary]

{summary}

---
The above is a summary of our previous conversation. We can continue from here.""",
        }


# ============== Auto Compactor ==============


class AutoCompactor:
    """
    Automatic compaction manager that integrates with the agent loop.

    Monitors token usage and automatically triggers:
    1. Pruning: Remove old tool outputs
    2. Compaction: Summarize conversation when approaching limits
    """

    def __init__(
        self,
        compactor: Compactor,
        token_counter: TokenCounter,
        max_context_tokens: int = 128000,
        max_output_tokens: int = 8192,
        compact_threshold: float = 0.8,
        min_messages_to_compact: int = 4,
    ):
        """
        Initialize auto compactor.

        Args:
            compactor: Compactor instance
            token_counter: Token counter instance
            max_context_tokens: Maximum context window size
            max_output_tokens: Reserved tokens for output
            compact_threshold: Trigger compaction at this percentage
            min_messages_to_compact: Minimum messages before compaction is allowed
        """
        self.compactor = compactor
        self.token_counter = token_counter
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.compact_threshold = compact_threshold
        self.min_messages_to_compact = min_messages_to_compact

        # Create pruner
        self.pruner = SmartPruner(token_counter)

    @property
    def usable_tokens(self) -> int:
        """Tokens available for context (excluding output reserve)."""
        return self.max_context_tokens - self.max_output_tokens

    def is_overflow(self, current_tokens: int) -> bool:
        """Check if we're over the context limit."""
        return current_tokens > self.usable_tokens

    def should_compact(self, current_tokens: int) -> bool:
        """Check if we should trigger compaction."""
        return current_tokens > (self.usable_tokens * self.compact_threshold)

    async def process(
        self,
        messages: list[Message],
    ) -> tuple[list[Message], CompactionResult | None, PruneResult | None]:
        """
        Process messages: prune and compact if needed.

        Args:
            messages: Current messages (Message objects)

        Returns:
            Tuple of (processed messages, compaction result, prune result)
        """

        # First try pruning
        prune_result = self.pruner.prune(messages)

        # Recalculate tokens after pruning
        message_dicts = [msg.to_dict() for msg in messages]
        current_tokens = self.token_counter.count_messages(message_dicts)

        # Check if compaction is needed
        if not self.should_compact(current_tokens):
            return messages, None, prune_result

        # Check if we have enough messages
        if len(messages) < self.min_messages_to_compact:
            return messages, None, prune_result

        # Find a good split point
        system_messages = [m for m in messages if m.role.value == "system"]
        non_system = [m for m in messages if m.role.value != "system"]

        # Keep at least the last 2 exchanges
        if len(non_system) <= 4:
            return messages, None, prune_result

        messages_to_compact = non_system[:-4]
        messages_to_keep = non_system[-4:]

        # Convert to dicts for compaction
        compact_dicts = [m.to_dict() for m in messages_to_compact]

        # Compact
        result = await self.compactor.compact(compact_dicts, self.token_counter)

        # Build new message list
        summary_dict = self.compactor.create_summary_message(result.summary)

        from ..models.message import Message

        new_messages = list(system_messages)
        new_messages.append(Message.from_dict(summary_dict))
        # Mark it as a summary
        if hasattr(new_messages[-1], "info") and new_messages[-1].info:
            new_messages[-1].info.summary = True
        new_messages.extend(messages_to_keep)

        return new_messages, result, prune_result


# ============== Export ==============

__all__ = [
    "Compactor",
    "CompactionResult",
    "AutoCompactor",
    "SmartPruner",
    "PruneResult",
    "PRUNE_MINIMUM",
    "PRUNE_PROTECT",
]
