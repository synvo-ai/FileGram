"""Standard OpenAI client for FileGram."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

import httpx
from openai import OpenAI

from ..config import OpenAIConfig
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
from .client import StreamCallback, StreamEvent, StreamEventType, StreamProcessor

CODEX_API_BASE = "https://chatgpt.com/backend-api/codex"


class _CodexTransport(httpx.BaseTransport):
    """Custom httpx transport that rewrites URLs and injects OAuth auth for the Codex endpoint."""

    def __init__(self, oauth_token: str, account_id: str | None = None):
        self._inner = httpx.HTTPTransport()
        self._oauth_token = oauth_token
        self._account_id = account_id

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        # Replace auth header with OAuth Bearer token
        request.headers["authorization"] = f"Bearer {self._oauth_token}"

        # Add account ID header if available
        if self._account_id:
            request.headers["chatgpt-account-id"] = self._account_id

        return self._inner.handle_request(request)

    def close(self):
        self._inner.close()


class OpenAIClient:
    """Client for standard OpenAI API (non-Azure).

    Supports both API key and OAuth (ChatGPT Pro/Plus) authentication.
    When using OAuth, API calls use the Responses API via the Codex endpoint.
    """

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self._is_oauth = config.oauth_token is not None

        if self._is_oauth:
            # OAuth mode: use Codex endpoint with custom transport for auth
            transport = _CodexTransport(config.oauth_token, config.account_id)
            self.client = OpenAI(
                api_key="codex-oauth",  # Placeholder, transport overrides auth  # pragma: allowlist secret
                base_url=CODEX_API_BASE,
                http_client=httpx.Client(transport=transport),
            )
        else:
            self.client = OpenAI(api_key=config.api_key)

    def _messages_to_openai_format(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal messages to OpenAI Chat Completions format."""
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

    def _messages_to_responses_format(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Convert internal messages to OpenAI Responses API format.

        Returns:
            Tuple of (instructions, input_items)
        """
        instructions = ""
        input_items: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # System message becomes instructions
                instructions = msg.get_text()
            elif msg.role == Role.USER:
                input_items.append(
                    {
                        "role": "user",
                        "content": msg.get_text(),
                    }
                )
            elif msg.role == Role.ASSISTANT:
                text_parts = [p for p in msg.content if isinstance(p, TextPart) and not getattr(p, "ignored", False)]
                tool_uses = msg.get_tool_uses()

                # Add text content
                if text_parts:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "\n".join(p.text for p in text_parts)}],
                        }
                    )

                # Add tool calls as function_call items
                for tu in tool_uses:
                    input_items.append(
                        {
                            "type": "function_call",
                            "id": tu.tool_use_id,
                            "call_id": tu.tool_use_id,
                            "name": tu.name,
                            "arguments": json.dumps(tu.arguments),
                        }
                    )

            elif msg.role == Role.TOOL:
                for part in msg.content:
                    if hasattr(part, "tool_use_id"):
                        content = part.content if hasattr(part, "content") else str(part)
                        input_items.append(
                            {
                                "type": "function_call_output",
                                "call_id": part.tool_use_id,
                                "output": content,
                            }
                        )

        if not instructions:
            instructions = "You are a helpful assistant."

        return instructions, input_items

    def _tools_to_responses_format(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert tool definitions to Responses API format."""
        responses_tools = []
        for t in tools:
            openai_fmt = t.to_openai_format()
            # Responses API uses a slightly different tool format
            responses_tools.append(
                {
                    "type": "function",
                    "name": openai_fmt["function"]["name"],
                    "description": openai_fmt["function"].get("description", ""),
                    "parameters": openai_fmt["function"].get("parameters", {}),
                }
            )
        return responses_tools

    def chat_completion_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        on_event: StreamCallback | None = None,
    ) -> tuple[list[Part], str, dict[str, int]]:
        """
        Stream completion (Chat Completions API or Responses API for OAuth/Codex).
        """
        if self._is_oauth:
            return self._responses_stream(messages, tools, on_text, on_reasoning, on_event)
        else:
            return self._chat_completions_stream(messages, tools, on_text, on_reasoning, on_event)

    def _chat_completions_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        on_event: StreamCallback | None = None,
    ) -> tuple[list[Part], str, dict[str, int]]:
        """Standard Chat Completions streaming."""
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

    def _responses_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        on_event: StreamCallback | None = None,
    ) -> tuple[list[Part], str, dict[str, int]]:
        """Responses API streaming for Codex/OAuth mode."""
        instructions, input_items = self._messages_to_responses_format(messages)

        # Codex models need explicit tool-use instruction
        if tools:
            instructions += (
                "\n\nIMPORTANT: You MUST use the provided tools to complete tasks. "
                "Do NOT just describe what you would do — actually call the tools to create files, "
                "run commands, and perform actions. Start working immediately by calling tools."
            )

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "instructions": instructions,
            "input": input_items,
            "stream": True,
            "store": False,
        }

        if tools:
            kwargs["tools"] = self._tools_to_responses_format(tools)

        if on_event:
            on_event(StreamEvent(type=StreamEventType.START))

        # Use the Responses API streaming
        collected_text = ""
        collected_reasoning = ""
        tool_calls: dict[str, dict[str, Any]] = {}  # keyed by call_id
        finish_reason = "stop"
        usage = TokenUsage()
        parts: list[Part] = []
        text_started = False
        reasoning_id: str | None = None

        with self.client.responses.create(**kwargs) as stream:
            for event in stream:
                event_type = event.type

                # Text output delta
                if event_type == "response.output_text.delta":
                    delta = event.delta
                    if not text_started:
                        text_started = True
                        if reasoning_id and on_event:
                            on_event(
                                StreamEvent(
                                    type=StreamEventType.REASONING_DONE,
                                    id=reasoning_id,
                                    text=collected_reasoning,
                                )
                            )
                            reasoning_id = None

                    collected_text += delta
                    if on_text:
                        on_text(delta)
                    if on_event:
                        on_event(StreamEvent(type=StreamEventType.TEXT_DELTA, text=delta))

                # Reasoning/thinking delta
                elif event_type == "response.reasoning.delta":
                    delta = getattr(event, "delta", "")
                    if not reasoning_id:
                        reasoning_id = f"reasoning_{datetime.now().timestamp()}"
                        if on_event:
                            on_event(
                                StreamEvent(
                                    type=StreamEventType.REASONING_START,
                                    id=reasoning_id,
                                )
                            )
                    collected_reasoning += delta
                    if on_reasoning:
                        on_reasoning(delta)
                    if on_event:
                        on_event(
                            StreamEvent(
                                type=StreamEventType.REASONING_DELTA,
                                id=reasoning_id,
                                text=delta,
                            )
                        )

                # Function call arguments delta
                elif event_type == "response.function_call_arguments.delta":
                    call_id = getattr(event, "item_id", "") or getattr(event, "call_id", "")
                    delta = event.delta
                    if call_id not in tool_calls:
                        tool_calls[call_id] = {
                            "id": call_id,
                            "name": "",
                            "arguments": "",
                        }
                    tool_calls[call_id]["arguments"] += delta
                    if on_event:
                        on_event(
                            StreamEvent(
                                type=StreamEventType.TOOL_INPUT_DELTA,
                                id=call_id,
                                tool_name=tool_calls[call_id]["name"],
                                text=delta,
                            )
                        )

                # Function call arguments done
                elif event_type == "response.function_call_arguments.done":
                    call_id = getattr(event, "item_id", "") or getattr(event, "call_id", "")
                    if call_id in tool_calls:
                        # Arguments are already accumulated
                        pass

                # Output item added (start of function_call or message)
                elif event_type == "response.output_item.added":
                    item = event.item
                    if getattr(item, "type", None) == "function_call":
                        call_id = item.id or getattr(item, "call_id", "")
                        name = getattr(item, "name", "")
                        tool_calls[call_id] = {
                            "id": call_id,
                            "name": name,
                            "arguments": "",
                        }
                        if on_event:
                            on_event(
                                StreamEvent(
                                    type=StreamEventType.TOOL_INPUT_START,
                                    id=call_id,
                                    tool_name=name,
                                )
                            )

                # Response completed
                elif event_type == "response.completed":
                    resp = event.response
                    if hasattr(resp, "usage") and resp.usage:
                        usage = TokenUsage(
                            input=getattr(resp.usage, "input_tokens", 0) or 0,
                            output=getattr(resp.usage, "output_tokens", 0) or 0,
                        )
                        if on_event:
                            on_event(StreamEvent(type=StreamEventType.USAGE, usage=usage))

                    status = getattr(resp, "status", "completed")
                    if status == "completed":
                        finish_reason = "stop"
                    elif status == "incomplete":
                        finish_reason = "length"
                    else:
                        finish_reason = status

        # Finalize parts

        # End reasoning if active
        if reasoning_id and on_event:
            on_event(
                StreamEvent(
                    type=StreamEventType.REASONING_DONE,
                    id=reasoning_id,
                    text=collected_reasoning,
                )
            )

        if collected_reasoning:
            parts.append(
                ReasoningPart(
                    id=reasoning_id or f"reasoning_{datetime.now().timestamp()}",
                    text=collected_reasoning,
                    time={"start": datetime.now().timestamp()},
                )
            )

        if collected_text:
            parts.append(TextPart(text=collected_text))

        # Add tool call parts
        for call_id, tc in tool_calls.items():
            try:
                arguments = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                arguments = {}

            if on_event:
                on_event(
                    StreamEvent(
                        type=StreamEventType.TOOL_CALL,
                        id=tc["id"],
                        tool_call_id=tc["id"],
                        tool_name=tc["name"],
                        input=arguments,
                    )
                )

            parts.append(
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

        # Responses API returns "stop" even with tool calls — override for agent loop
        if tool_calls:
            finish_reason = "tool_calls"

        if on_event:
            on_event(StreamEvent(type=StreamEventType.FINISH, finish_reason=finish_reason))

        usage_dict = {
            "input_tokens": usage.input_tokens if hasattr(usage, "input_tokens") else getattr(usage, "input", 0),
            "output_tokens": usage.output_tokens if hasattr(usage, "output_tokens") else getattr(usage, "output", 0),
        }
        return parts, finish_reason, usage_dict

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Non-streaming completion (for compaction, etc.).

        For OAuth/Codex mode, uses Responses API.
        """
        if self._is_oauth:
            return self._responses_non_stream(messages, tools)

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

    def _responses_non_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Non-streaming Responses API call."""
        # Convert raw message dicts to instructions + input
        instructions = ""
        input_items = []
        for msg in messages:
            if msg.get("role") == "system":
                instructions = msg.get("content", "")
            else:
                input_items.append(msg)

        if not instructions:
            instructions = "You are a helpful assistant."

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "instructions": instructions,
            "input": input_items,
            "store": False,
        }

        if tools:
            kwargs["tools"] = tools

        response = self.client.responses.create(**kwargs)

        # Extract text from response output
        text = ""
        for item in response.output:
            if getattr(item, "type", None) == "message":
                for content in item.content:
                    if getattr(content, "type", None) == "output_text":
                        text += content.text

        metadata = {
            "usage": {
                "input": getattr(response.usage, "input_tokens", 0) if response.usage else 0,
                "output": getattr(response.usage, "output_tokens", 0) if response.usage else 0,
            },
            "finish_reason": response.status,
        }

        return text, metadata
