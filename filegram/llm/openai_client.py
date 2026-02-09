"""Standard OpenAI client for FileGram."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from ..config import OpenAIConfig
from ..models.message import (
    Message,
    Part,
    Role,
    TextPart,
)
from ..models.tool import ToolDefinition
from .client import StreamCallback, StreamEvent, StreamEventType, StreamProcessor


class OpenAIClient:
    """Client for standard OpenAI API (non-Azure)."""

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key)

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
        Stream chat completion.

        Uses the same StreamProcessor as AzureOpenAIClient for consistency.
        """
        openai_messages = self._messages_to_openai_format(messages)

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": openai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if tools:
            kwargs["tools"] = [t.to_openai_format() for t in tools]

        response = self.client.chat.completions.create(**kwargs)

        processor = StreamProcessor(
            on_event=on_event,
            on_text=on_text,
            on_reasoning=on_reasoning,
        )

        if on_event:
            on_event(StreamEvent(type=StreamEventType.START))

        for chunk in response:
            processor.process_chunk(chunk)

        return processor.finalize()

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Non-streaming chat completion (for compaction, etc.)."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
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
