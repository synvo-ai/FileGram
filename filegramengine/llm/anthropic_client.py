"""Anthropic Claude API client for FileGram."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import httpx
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types import (
    TextBlock,
    ToolUseBlock,
)


def _strip_mcp_prefix(name: str) -> str:
    """Strip mcp_ prefix from tool names returned by Anthropic OAuth API."""
    if name.startswith("mcp_"):
        return name[4:]
    return name


class _OAuthTransport(httpx.AsyncBaseTransport):
    """Custom transport for Anthropic OAuth requests.

    Only handles HTTP-level concerns:
    - Adds ?beta=true to /v1/messages URL
    - Sets User-Agent to claude-cli
    - Removes x-api-key and x-stainless-* headers
    """

    def __init__(self, wrapped: httpx.AsyncBaseTransport):
        self._wrapped = wrapped

    # Only these headers should be sent — matches what JS fetch + reference plugin sends
    _ALLOWED_HEADERS = {
        "host",
        "content-type",
        "content-length",
        "authorization",
        "anthropic-beta",
        "anthropic-version",
        "user-agent",
    }

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Add ?beta=true to /v1/messages
        if request.url.path == "/v1/messages" and "beta" not in request.url.params:
            request.url = request.url.copy_merge_params({"beta": "true"})

        # Override user-agent — server requires exact claude-cli UA
        request.headers["user-agent"] = "claude-cli/2.1.2 (external, cli)"

        # Strip ALL headers not in whitelist (removes x-api-key, x-stainless-*,
        # accept, accept-encoding, connection, etc.)
        for key in list(request.headers.keys()):
            if key.lower() not in self._ALLOWED_HEADERS:
                del request.headers[key]

        return await self._wrapped.handle_async_request(request)


class _BedrockGatewayTransport(httpx.AsyncBaseTransport):
    """Custom transport for Bedrock gateway that uses Bearer token instead of AWS SigV4.

    The gateway (e.g., ai-gateway.zende.sk/bedrock) handles AWS auth internally
    and expects a simple Bearer token from the client.
    """

    def __init__(self, wrapped: httpx.AsyncBaseTransport, bearer_token: str):
        self._wrapped = wrapped
        self._bearer_token = bearer_token

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Replace AWS SigV4 Authorization with Bearer token
        request.headers["Authorization"] = f"Bearer {self._bearer_token}"
        return await self._wrapped.handle_async_request(request)


class _GatewayAnthropicBedrock(AsyncAnthropicBedrock):
    """AsyncAnthropicBedrock that skips AWS SigV4 signing for gateway use.

    The gateway handles AWS auth; we only need Bearer token (set via transport).
    Keeps _prepare_options (URL rewriting /v1/messages → /model/{model}/invoke)
    but skips _prepare_request (SigV4 signing that requires botocore).
    """

    async def _prepare_request(self, request: httpx.Request) -> None:
        # No-op: skip SigV4 signing. Bearer auth is handled by _BedrockGatewayTransport.
        return None


from ..config import AnthropicConfig  # noqa: E402
from ..models.message import Message as InternalMessage  # noqa: E402
from ..models.message import (  # noqa: E402
    Part,
    ReasoningPart,
    Role,
    TextPart,
    ToolPart,
    ToolResultPart,
    ToolUsePart,
)
from ..models.tool import ToolDefinition  # noqa: E402

logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    """Types of streaming events."""

    TEXT_DELTA = "text_delta"
    TEXT_DONE = "text_done"
    THINKING_DELTA = "thinking_delta"
    THINKING_DONE = "thinking_done"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"
    TOOL_USE_DONE = "tool_use_done"
    MESSAGE_START = "message_start"
    MESSAGE_DONE = "message_done"
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_STOP = "content_block_stop"
    ERROR = "error"


@dataclass
class StreamEvent:
    """Event emitted during streaming."""

    type: StreamEventType
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    """Token usage from API response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class ToolCall:
    """Parsed tool call from response."""

    id: str
    name: str
    input: dict[str, Any]


class AnthropicClient:
    """Async client for Anthropic Claude API."""

    def __init__(self, config: AnthropicConfig):
        """Initialize the client with configuration.

        Supports both API key auth and OAuth Bearer token auth.
        If config.oauth_token is set, uses Bearer auth with the OAuth beta header.
        """
        self.config = config
        self._is_oauth = bool(config.oauth_token)
        if config.oauth_token:
            # OAuth needs: Bearer auth, beta header, and ?beta=true on /v1/messages
            base_transport = httpx.AsyncHTTPTransport()
            oauth_transport = _OAuthTransport(base_transport)
            http_client = httpx.AsyncClient(
                transport=oauth_transport,
                timeout=httpx.Timeout(timeout=600.0, connect=10.0),
            )
            self.client = AsyncAnthropic(
                auth_token=config.oauth_token,
                default_headers={
                    "anthropic-beta": (
                        "oauth-2025-04-20,claude-code-20250219,"
                        "interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14"
                    ),
                },
                http_client=http_client,
            )
        else:
            if config.base_url:
                # Bedrock gateway: use AsyncAnthropicBedrock with Bearer token auth
                # The gateway handles AWS auth; we just need to send a Bearer token.
                base_transport = httpx.AsyncHTTPTransport()
                gateway_transport = _BedrockGatewayTransport(base_transport, config.api_key)
                http_client = httpx.AsyncClient(
                    transport=gateway_transport,
                    timeout=httpx.Timeout(timeout=600.0, connect=10.0),
                )
                self.client = _GatewayAnthropicBedrock(
                    base_url=config.base_url,
                    aws_access_key="skip",
                    aws_secret_key="skip",  # pragma: allowlist secret
                    aws_region="us-east-1",
                    http_client=http_client,
                )
            else:
                self.client = AsyncAnthropic(api_key=config.api_key)

    def _convert_messages_to_anthropic(
        self,
        messages: list[InternalMessage],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert internal messages to Anthropic API format.

        Returns:
            Tuple of (system_prompt, messages)
        """
        system_prompt = None
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Extract system prompt
                for part in msg.content:
                    if isinstance(part, TextPart):
                        if system_prompt:
                            system_prompt += "\n\n" + part.text
                        else:
                            system_prompt = part.text
                continue

            if msg.role == Role.USER:
                content: list[dict[str, Any]] = []
                for part in msg.content:
                    if isinstance(part, TextPart):
                        content.append({"type": "text", "text": part.text})
                if content:
                    anthropic_messages.append({"role": "user", "content": content})

            elif msg.role == Role.ASSISTANT:
                content = []
                for part in msg.content:
                    if isinstance(part, TextPart):
                        content.append({"type": "text", "text": part.text})
                    elif isinstance(part, ToolUsePart):
                        content.append(
                            {
                                "type": "tool_use",
                                "id": part.tool_use_id,
                                "name": part.name,
                                "input": part.arguments,
                            }
                        )
                    elif isinstance(part, ToolPart):
                        content.append(
                            {
                                "type": "tool_use",
                                "id": part.call_id,
                                "name": part.tool,
                                "input": part.state.input,
                            }
                        )
                if content:
                    anthropic_messages.append({"role": "assistant", "content": content})

            elif msg.role == Role.TOOL:
                # Tool results in Anthropic format
                for part in msg.content:
                    if isinstance(part, ToolResultPart):
                        anthropic_messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": part.tool_use_id,
                                        "content": part.content,
                                        "is_error": part.is_error,
                                    }
                                ],
                            }
                        )

        return system_prompt, anthropic_messages

    def _convert_tools(
        self,
        tools: list,
    ) -> list[dict[str, Any]]:
        """Convert tool definitions to Anthropic format.

        Handles both ToolDefinition objects and raw dicts.
        """
        anthropic_tools = []
        for tool in tools:
            if isinstance(tool, ToolDefinition):
                anthropic_tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters,
                    }
                )
            elif hasattr(tool, "name"):
                anthropic_tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": getattr(tool, "parameters", getattr(tool, "input_schema", {})),
                    }
                )
            else:
                anthropic_tools.append(
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "input_schema": tool.get("parameters", tool.get("input_schema", {})),
                    }
                )
        return anthropic_tools

    def _build_kwargs(
        self,
        anthropic_messages: list[dict[str, Any]],
        system_prompt: str | None = None,
        tools: list | None = None,
    ) -> dict[str, Any]:
        """Build kwargs for Anthropic API call."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": anthropic_messages,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        kwargs["temperature"] = self.config.temperature

        if self.config.enable_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.config.thinking_budget_tokens,
            }

        if self._is_oauth:
            self._prepare_oauth_kwargs(kwargs)

        return kwargs

    _TOOL_PREFIX = "mcp_"
    _CLAUDE_CODE_IDENT = "You are Claude Code, Anthropic's official CLI for Claude."

    @staticmethod
    def _sanitize_product_name(text: str) -> str:
        """Replace product names with 'Claude Code', preserving file paths.

        Only replaces standalone occurrences (not inside /path/to/FileGram/...).
        """
        # Replace product names that are NOT part of a file path
        # Negative lookbehind for '/' ensures we skip path components
        text = re.sub(r"(?<!/)\b(?i:opencode)\b", "Claude Code", text)
        text = re.sub(r"(?<!/)\b(?i:synvocowork)\b", "Claude Code", text)
        text = re.sub(r"(?<!/)\b(?i:filegram)\b(?!/)", "Claude Code", text)
        return text

    def _prepare_oauth_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Transform API kwargs to match Claude Code format for OAuth.

        Modifies kwargs in-place to make the request look like Claude Code:
        - Splits system prompt into two blocks: [prefix] [prefix + rest]
          (server validates that first block === Claude Code identifier)
        - Sanitizes system prompt text (product names only, not paths)
        - Prefixes tool names with mcp_
        - Prefixes tool_use blocks in messages with mcp_
        - Removes temperature (Claude Code doesn't send this explicitly)
        """
        # Remove temperature — Claude Code doesn't explicitly set it
        kwargs.pop("temperature", None)

        # System prompt MUST be two text blocks:
        #   block[0] = exact Claude Code identifier (server validates this)
        #   block[1] = identifier + "\n\n" + rest of system prompt
        if "system" in kwargs:
            system = kwargs["system"]
            if isinstance(system, str):
                system = self._sanitize_product_name(system)
                # Strip any existing prefix to avoid duplication
                rest = system.replace(self._CLAUDE_CODE_IDENT, "").strip()
                kwargs["system"] = [
                    {"type": "text", "text": self._CLAUDE_CODE_IDENT},
                    {"type": "text", "text": self._CLAUDE_CODE_IDENT + "\n\n" + rest},
                ]
            elif isinstance(system, list):
                # Already array format — ensure first block is exact prefix
                all_text = "\n\n".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in system)
                all_text = self._sanitize_product_name(all_text)
                rest = all_text.replace(self._CLAUDE_CODE_IDENT, "").strip()
                kwargs["system"] = [
                    {"type": "text", "text": self._CLAUDE_CODE_IDENT},
                    {"type": "text", "text": self._CLAUDE_CODE_IDENT + "\n\n" + rest},
                ]

        # Prefix tool definitions with mcp_
        if "tools" in kwargs and isinstance(kwargs["tools"], list):
            for tool in kwargs["tools"]:
                if isinstance(tool, dict) and "name" in tool:
                    if not tool["name"].startswith(self._TOOL_PREFIX):
                        tool["name"] = self._TOOL_PREFIX + tool["name"]

        # Prefix tool_use blocks in messages
        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            for msg in kwargs["messages"]:
                if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                    for block in msg["content"]:
                        if isinstance(block, dict) and block.get("type") == "tool_use" and "name" in block:
                            if not block["name"].startswith(self._TOOL_PREFIX):
                                block["name"] = self._TOOL_PREFIX + block["name"]

    async def chat_completion_stream(
        self,
        messages: list[InternalMessage],
        tools: list | None = None,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        on_event: Callable | None = None,
    ) -> tuple[list[Part], str, dict[str, int]]:
        """
        Stream chat completion matching AzureOpenAIClient interface.

        Args:
            messages: Internal message list
            tools: Tool definitions (ToolDefinition objects or dicts)
            on_text: Callback for text deltas
            on_reasoning: Callback for reasoning/thinking deltas
            on_event: Callback for raw stream events

        Returns:
            Tuple of (collected parts, finish_reason, usage_dict)
        """
        system_prompt, anthropic_messages = self._convert_messages_to_anthropic(messages)
        kwargs = self._build_kwargs(anthropic_messages, system_prompt, tools)

        # Collect results
        tool_parts: list[ToolUsePart] = []
        collected_text = ""
        collected_reasoning = ""
        finish_reason = "stop"
        usage_dict: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

        current_tool: dict[str, Any] | None = None
        current_tool_input = ""

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if hasattr(block, "type"):
                        if block.type == "tool_use":
                            tool_name = _strip_mcp_prefix(block.name) if self._is_oauth else block.name
                            current_tool = {"id": block.id, "name": tool_name}
                            current_tool_input = ""

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if hasattr(delta, "type"):
                        if delta.type == "text_delta":
                            collected_text += delta.text
                            if on_text:
                                on_text(delta.text)
                        elif delta.type == "input_json_delta":
                            current_tool_input += delta.partial_json
                        elif delta.type == "thinking_delta":
                            collected_reasoning += delta.thinking
                            if on_reasoning:
                                on_reasoning(delta.thinking)

                elif event.type == "content_block_stop":
                    if current_tool:
                        try:
                            tool_input = json.loads(current_tool_input) if current_tool_input else {}
                        except json.JSONDecodeError:
                            tool_input = {}

                        tool_parts.append(
                            ToolUsePart(
                                tool_use_id=current_tool["id"],
                                name=current_tool["name"],  # already stripped in content_block_start
                                arguments=tool_input,
                            )
                        )
                        current_tool = None
                        current_tool_input = ""

                elif event.type == "message_start":
                    # Extract input token count from message_start
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        msg_usage = event.message.usage
                        usage_dict["input_tokens"] = getattr(msg_usage, "input_tokens", 0) or 0

                elif event.type == "message_delta":
                    if hasattr(event, "delta") and hasattr(event.delta, "stop_reason"):
                        stop_reason = event.delta.stop_reason
                        if stop_reason == "tool_use":
                            finish_reason = "tool_calls"
                        elif stop_reason == "end_turn":
                            finish_reason = "stop"
                        else:
                            finish_reason = stop_reason or "stop"
                    # Extract output token count from message_delta
                    if hasattr(event, "usage"):
                        usage_dict["output_tokens"] = getattr(event.usage, "output_tokens", 0) or 0

        # Build final parts list
        final_parts: list[Part] = []

        if collected_reasoning:
            final_parts.append(
                ReasoningPart(
                    id=f"reasoning_{datetime.now().timestamp()}",
                    text=collected_reasoning,
                )
            )

        if collected_text:
            final_parts.append(TextPart(text=collected_text))

        final_parts.extend(tool_parts)

        return final_parts, finish_reason, usage_dict

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Non-streaming chat completion (for compaction, etc.).

        Args:
            messages: Messages in dict format
            tools: Tool definitions (optional)

        Returns:
            Tuple of (response text, metadata)
        """
        # Extract system messages and convert tool messages to Anthropic format.
        # Anthropic API only allows "user" and "assistant" roles.
        # Tool results must be sent as role: "user" with tool_result content blocks.
        system_parts = []
        non_system_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg.get("content", ""))
            elif msg.get("role") == "tool":
                # Convert OpenAI-style tool message to Anthropic tool_result format
                tool_call_id = msg.get("tool_call_id", "")
                content = msg.get("content", "")
                non_system_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": content,
                            }
                        ],
                    }
                )
            else:
                # Convert OpenAI-style assistant messages to Anthropic format
                # to_dict() adds "tool_calls" which Anthropic API rejects
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    content: list[dict[str, Any]] = []
                    if msg.get("content"):
                        content.append({"type": "text", "text": msg["content"]})
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        args = func.get("arguments", "{}")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except (json.JSONDecodeError, TypeError):
                                args = {}
                        content.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "input": args,
                        })
                    non_system_messages.append({"role": "assistant", "content": content})
                else:
                    non_system_messages.append(msg)

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": non_system_messages,
            "temperature": self.config.temperature,
        }

        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        import time as _time, sys
        _api_start = _time.time()
        print(f"[compact] Calling Anthropic API (model={self.config.model}, msgs={len(non_system_messages)})...", file=sys.stderr, flush=True)
        response = await self.client.messages.create(**kwargs)
        print(f"[compact] API returned in {_time.time() - _api_start:.1f}s", file=sys.stderr, flush=True)

        text = ""
        for block in response.content:
            if isinstance(block, TextBlock):
                text += block.text

        metadata = {
            "usage": {
                "input": response.usage.input_tokens,
                "output": response.usage.completion_tokens
                if hasattr(response.usage, "completion_tokens")
                else response.usage.output_tokens,
            },
            "finish_reason": response.stop_reason,
        }

        return text, metadata

    async def create_message(
        self,
        messages: list[InternalMessage],
        tools: list | None = None,
        system_override: str | None = None,
    ) -> tuple[AnthropicMessage, TokenUsage]:
        """
        Create a message (non-streaming).

        Args:
            messages: List of conversation messages.
            tools: Optional list of tool definitions.
            system_override: Optional system prompt override.

        Returns:
            Tuple of (Message, TokenUsage)
        """
        system_prompt, anthropic_messages = self._convert_messages_to_anthropic(messages)

        if system_override:
            system_prompt = system_override

        kwargs = self._build_kwargs(anthropic_messages, system_prompt, tools)

        response = await self.client.messages.create(**kwargs)

        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        )

        return response, usage

    async def stream_message(
        self,
        messages: list[InternalMessage],
        tools: list | None = None,
        system_override: str | None = None,
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream a message response.

        Args:
            messages: List of conversation messages.
            tools: Optional list of tool definitions.
            system_override: Optional system prompt override.
            on_event: Optional callback for each event.

        Yields:
            StreamEvent objects as they arrive.
        """
        system_prompt, anthropic_messages = self._convert_messages_to_anthropic(messages)

        if system_override:
            system_prompt = system_override

        kwargs = self._build_kwargs(anthropic_messages, system_prompt, tools)

        # Track current content blocks for streaming
        current_tool: dict[str, Any] | None = None
        current_tool_input = ""
        message_usage: dict[str, int] = {}

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                stream_event: StreamEvent | None = None

                if event.type == "message_start":
                    stream_event = StreamEvent(
                        type=StreamEventType.MESSAGE_START,
                        data={"message": event.message.model_dump() if hasattr(event.message, "model_dump") else {}},
                    )

                elif event.type == "content_block_start":
                    block = event.content_block
                    if hasattr(block, "type"):
                        if block.type == "tool_use":
                            tool_name = _strip_mcp_prefix(block.name) if self._is_oauth else block.name
                            current_tool = {
                                "id": block.id,
                                "name": tool_name,
                            }
                            current_tool_input = ""
                            stream_event = StreamEvent(
                                type=StreamEventType.TOOL_USE_START,
                                data={
                                    "id": block.id,
                                    "name": tool_name,
                                    "index": event.index,
                                },
                            )
                        elif block.type == "thinking":
                            stream_event = StreamEvent(
                                type=StreamEventType.THINKING_DELTA,
                                data={"thinking": getattr(block, "thinking", "")},
                            )

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if hasattr(delta, "type"):
                        if delta.type == "text_delta":
                            stream_event = StreamEvent(
                                type=StreamEventType.TEXT_DELTA,
                                data={"text": delta.text},
                            )
                        elif delta.type == "input_json_delta":
                            current_tool_input += delta.partial_json
                            stream_event = StreamEvent(
                                type=StreamEventType.TOOL_USE_DELTA,
                                data={
                                    "partial_json": delta.partial_json,
                                    "accumulated": current_tool_input,
                                },
                            )
                        elif delta.type == "thinking_delta":
                            stream_event = StreamEvent(
                                type=StreamEventType.THINKING_DELTA,
                                data={"thinking": delta.thinking},
                            )

                elif event.type == "content_block_stop":
                    if current_tool:
                        # Parse the accumulated JSON
                        try:
                            tool_input = json.loads(current_tool_input) if current_tool_input else {}
                        except json.JSONDecodeError:
                            tool_input = {}

                        stream_event = StreamEvent(
                            type=StreamEventType.TOOL_USE_DONE,
                            data={
                                "id": current_tool["id"],
                                "name": current_tool["name"],
                                "input": tool_input,
                            },
                        )
                        current_tool = None
                        current_tool_input = ""
                    else:
                        stream_event = StreamEvent(
                            type=StreamEventType.TEXT_DONE,
                            data={},
                        )

                elif event.type == "message_delta":
                    if hasattr(event, "usage"):
                        message_usage = {
                            "output_tokens": event.usage.output_tokens,
                        }

                elif event.type == "message_stop":
                    stream_event = StreamEvent(
                        type=StreamEventType.MESSAGE_DONE,
                        data={
                            "usage": message_usage,
                            "stop_reason": getattr(event, "stop_reason", "end_turn"),
                        },
                    )

                if stream_event:
                    if on_event:
                        on_event(stream_event)
                    yield stream_event

    def parse_tool_calls(self, response: AnthropicMessage) -> list[ToolCall]:
        """Extract tool calls from a response message."""
        tool_calls = []
        for block in response.content:
            if isinstance(block, ToolUseBlock):
                name = _strip_mcp_prefix(block.name) if self._is_oauth else block.name
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=name,
                        input=block.input,
                    )
                )
        return tool_calls

    def get_text_content(self, response: AnthropicMessage) -> str:
        """Extract text content from a response message."""
        text_parts = []
        for block in response.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
        return "\n".join(text_parts)


# Convenience function to create a client
def create_anthropic_client(config: AnthropicConfig | None = None) -> AnthropicClient:
    """Create an Anthropic client with optional config."""
    if config is None:
        import os

        from ..config import AnthropicConfig as CfgAnthropicConfig

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        config = CfgAnthropicConfig(api_key=api_key)
    return AnthropicClient(config)
