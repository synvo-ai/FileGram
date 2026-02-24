"""Azure OpenAI client for CodeAgent with full streaming support."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from openai import AzureOpenAI
from openai.types.chat import ChatCompletionChunk

from ..config import AzureOpenAIConfig
from ..models.message import (
    Message,
    Part,
    ReasoningPart,
    Role,
    TextPart,
    TokenUsage,
    ToolPart,
    ToolState,
    ToolStatus,
    ToolTime,
)
from ..models.tool import ToolDefinition

# ============== Stream Event Types ==============


class StreamEventType(str, Enum):
    """Types of streaming events."""

    START = "start"
    TEXT_DELTA = "text-delta"
    TEXT_DONE = "text-done"
    REASONING_START = "reasoning-start"
    REASONING_DELTA = "reasoning-delta"
    REASONING_DONE = "reasoning-done"
    TOOL_INPUT_START = "tool-input-start"
    TOOL_INPUT_DELTA = "tool-input-delta"
    TOOL_CALL = "tool-call"
    TOOL_RESULT = "tool-result"
    TOOL_ERROR = "tool-error"
    FINISH = "finish"
    ERROR = "error"
    USAGE = "usage"


@dataclass
class StreamEvent:
    """Event from streaming response."""

    type: StreamEventType
    id: str = ""
    text: str = ""
    tool_name: str = ""
    tool_call_id: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] | None = None
    error: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    metadata: dict[str, Any] | None = None


class StreamCallback(Protocol):
    """Protocol for stream callbacks."""

    def __call__(self, event: StreamEvent) -> None: ...


# ============== Stream Processor ==============


class StreamProcessor:
    """
    Processes streaming response and emits structured events.

    Handles:
    - Text streaming
    - Reasoning token streaming (Claude thinking, o1 reasoning)
    - Tool call streaming
    - Usage tracking
    """

    def __init__(
        self,
        on_event: StreamCallback | None = None,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ):
        self.on_event = on_event
        self.on_text = on_text
        self.on_reasoning = on_reasoning

        self.collected_text = ""
        self.collected_reasoning = ""
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason = "stop"
        self.usage = TokenUsage()
        self.parts: list[Part] = []

        # Current state
        self._reasoning_id: str | None = None
        self._text_started = False

    def _emit(self, event: StreamEvent) -> None:
        """Emit an event."""
        if self.on_event:
            self.on_event(event)

    def process_chunk(self, chunk: ChatCompletionChunk) -> None:
        """Process a single chunk from the stream."""
        if not chunk.choices:
            # Check for usage in final chunk
            if hasattr(chunk, "usage") and chunk.usage:
                self.usage = TokenUsage(
                    input=chunk.usage.prompt_tokens or 0,
                    output=chunk.usage.completion_tokens or 0,
                )
                self._emit(
                    StreamEvent(
                        type=StreamEventType.USAGE,
                        usage=self.usage,
                    )
                )
            return

        choice = chunk.choices[0]

        # Handle finish reason
        if choice.finish_reason:
            self.finish_reason = choice.finish_reason
            self._emit(
                StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=choice.finish_reason,
                )
            )
            return

        delta = choice.delta

        # Handle reasoning tokens (for o1 models)
        # OpenAI o1 uses a special format for reasoning
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            if not self._reasoning_id:
                self._reasoning_id = f"reasoning_{datetime.now().timestamp()}"
                self._emit(
                    StreamEvent(
                        type=StreamEventType.REASONING_START,
                        id=self._reasoning_id,
                    )
                )

            self.collected_reasoning += delta.reasoning_content
            if self.on_reasoning:
                self.on_reasoning(delta.reasoning_content)
            self._emit(
                StreamEvent(
                    type=StreamEventType.REASONING_DELTA,
                    id=self._reasoning_id,
                    text=delta.reasoning_content,
                )
            )

        # Handle regular text content
        if delta.content:
            if not self._text_started:
                self._text_started = True
                # End reasoning if it was active
                if self._reasoning_id:
                    self._emit(
                        StreamEvent(
                            type=StreamEventType.REASONING_DONE,
                            id=self._reasoning_id,
                            text=self.collected_reasoning,
                        )
                    )
                    self._reasoning_id = None

            self.collected_text += delta.content
            if self.on_text:
                self.on_text(delta.content)
            self._emit(
                StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text=delta.content,
                )
            )

        # Handle tool calls
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index

                # New tool call
                if idx not in self.tool_calls:
                    self.tool_calls[idx] = {
                        "id": tc.id or "",
                        "name": tc.function.name if tc.function else "",
                        "arguments": "",
                        "started": False,
                    }

                if tc.id:
                    self.tool_calls[idx]["id"] = tc.id

                if tc.function:
                    if tc.function.name:
                        self.tool_calls[idx]["name"] = tc.function.name
                        # Emit tool input start
                        if not self.tool_calls[idx]["started"]:
                            self.tool_calls[idx]["started"] = True
                            self._emit(
                                StreamEvent(
                                    type=StreamEventType.TOOL_INPUT_START,
                                    id=self.tool_calls[idx]["id"],
                                    tool_name=tc.function.name,
                                )
                            )

                    if tc.function.arguments:
                        self.tool_calls[idx]["arguments"] += tc.function.arguments
                        self._emit(
                            StreamEvent(
                                type=StreamEventType.TOOL_INPUT_DELTA,
                                id=self.tool_calls[idx]["id"],
                                tool_name=self.tool_calls[idx]["name"],
                                text=tc.function.arguments,
                            )
                        )

    def finalize(self) -> tuple[list[Part], str]:
        """Finalize processing and return collected parts."""
        # End any active reasoning
        if self._reasoning_id:
            self._emit(
                StreamEvent(
                    type=StreamEventType.REASONING_DONE,
                    id=self._reasoning_id,
                    text=self.collected_reasoning,
                )
            )

        # Add reasoning part if present
        if self.collected_reasoning:
            self.parts.append(
                ReasoningPart(
                    id=self._reasoning_id or f"reasoning_{datetime.now().timestamp()}",
                    text=self.collected_reasoning,
                    time={"start": datetime.now().timestamp()},
                )
            )

        # Add text part if present
        if self.collected_text:
            self.parts.append(TextPart(text=self.collected_text))

        # Add tool call parts
        for idx in sorted(self.tool_calls.keys()):
            tc = self.tool_calls[idx]
            try:
                arguments = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                arguments = {}

            # Emit tool call event
            self._emit(
                StreamEvent(
                    type=StreamEventType.TOOL_CALL,
                    id=tc["id"],
                    tool_call_id=tc["id"],
                    tool_name=tc["name"],
                    input=arguments,
                )
            )

            # Add ToolPart with full state
            self.parts.append(
                ToolPart(
                    id=tc["id"],
                    tool=tc["name"],
                    call_id=tc["id"],
                    state=ToolState(
                        status=ToolStatus.PENDING,
                        input=arguments,
                        time=ToolTime(start=datetime.now().timestamp()),
                    ),
                )
            )

        return self.parts, self.finish_reason


# ============== Azure OpenAI Client ==============


class AzureOpenAIClient:
    """Client for Azure OpenAI API with full streaming support."""

    def __init__(self, config: AzureOpenAIConfig):
        self.config = config
        import httpx

        self.client = AzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
            timeout=httpx.Timeout(timeout=300.0, connect=10.0),
        )

    def _messages_to_openai_format(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        openai_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                openai_messages.append(
                    {
                        "role": "system",
                        "content": msg.get_text(),
                    }
                )
            elif msg.role == Role.USER:
                openai_messages.append(
                    {
                        "role": "user",
                        "content": msg.get_text(),
                    }
                )
            elif msg.role == Role.ASSISTANT:
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                text_parts = [p for p in msg.content if isinstance(p, TextPart) and not getattr(p, "ignored", False)]
                tool_uses = msg.get_tool_uses()

                if text_parts:
                    assistant_msg["content"] = "\n".join(p.text for p in text_parts)
                else:
                    assistant_msg["content"] = None

                if tool_uses:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tu.tool_use_id,
                            "type": "function",
                            "function": {
                                "name": tu.name,
                                "arguments": json.dumps(tu.arguments),
                            },
                        }
                        for tu in tool_uses
                    ]

                openai_messages.append(assistant_msg)
            elif msg.role == Role.TOOL:
                for part in msg.content:
                    if hasattr(part, "tool_use_id"):
                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": part.tool_use_id,
                                "content": part.content if hasattr(part, "content") else str(part),
                            }
                        )

        return openai_messages

    def chat_completion_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        on_event: StreamCallback | None = None,
    ) -> tuple[list[Part], str]:
        """
        Stream chat completion and return collected parts and finish reason.

        Args:
            messages: Conversation messages
            tools: Available tools
            on_text: Callback for text deltas
            on_reasoning: Callback for reasoning deltas (o1/Claude thinking)
            on_event: Callback for all stream events

        Returns:
            Tuple of (collected parts, finish_reason)
        """
        openai_messages = self._messages_to_openai_format(messages)

        kwargs: dict[str, Any] = {
            "model": self.config.deployment,
            "messages": openai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},  # Request usage in final chunk
        }

        if tools:
            kwargs["tools"] = [t.to_openai_format() for t in tools]

        response = self.client.chat.completions.create(**kwargs)

        # Create processor
        processor = StreamProcessor(
            on_event=on_event,
            on_text=on_text,
            on_reasoning=on_reasoning,
        )

        # Emit start event
        if on_event:
            on_event(StreamEvent(type=StreamEventType.START))

        # Process stream
        for chunk in response:
            processor.process_chunk(chunk)

        # Finalize and return
        return processor.finalize()

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Non-streaming chat completion (for compaction, etc.).

        Args:
            messages: Messages in OpenAI format
            tools: Tool definitions (optional)

        Returns:
            Tuple of (response text, metadata)
        """
        kwargs: dict[str, Any] = {
            "model": self.config.deployment,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)

        text = response.choices[0].message.content or ""
        metadata = {
            "usage": {
                "input": response.usage.prompt_tokens if response.usage else 0,
                "output": response.usage.completion_tokens if response.usage else 0,
            },
            "finish_reason": response.choices[0].finish_reason,
        }

        return text, metadata
