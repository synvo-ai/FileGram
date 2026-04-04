"""Google Gemini client for FileGram."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..config import GoogleConfig
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
    ToolUsePart,
)
from ..models.tool import ToolDefinition
from .client import StreamCallback, StreamEvent, StreamEventType


class GeminiClient:
    """Client for Google Gemini API with function calling support.

    Uses the google-genai SDK. Supports streaming and non-streaming modes.
    """

    def __init__(self, config: GoogleConfig):
        from google import genai

        self.config = config
        self.client = genai.Client(api_key=config.api_key)

    @staticmethod
    def _clean_schema(schema: Any) -> Any:
        """Recursively remove fields unsupported by Gemini's function declaration schema."""
        if isinstance(schema, dict):
            # Fields that Gemini does not accept in function declaration parameters
            unsupported = {"additionalProperties", "additional_properties", "$schema", "title"}
            return {k: GeminiClient._clean_schema(v) for k, v in schema.items() if k not in unsupported}
        if isinstance(schema, list):
            return [GeminiClient._clean_schema(item) for item in schema]
        return schema

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert ToolDefinition list to Gemini function declarations."""
        declarations = []
        for t in tools:
            declarations.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": self._clean_schema(t.parameters),
                }
            )
        return declarations

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert internal messages to Gemini format.

        Returns:
            Tuple of (system_instruction, contents)
        """
        system_instruction = None
        contents: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.get_text()

            elif msg.role == Role.USER:
                contents.append(
                    {
                        "role": "user",
                        "parts": [{"text": msg.get_text()}],
                    }
                )

            elif msg.role == Role.ASSISTANT:
                parts = []
                text_parts = [p for p in msg.content if isinstance(p, TextPart) and not getattr(p, "ignored", False)]

                if text_parts:
                    parts.append({"text": "\n".join(p.text for p in text_parts)})

                # Build function_call parts with thought_signature from ToolPart metadata
                for p in msg.content:
                    if isinstance(p, ToolPart):
                        fc_part: dict[str, Any] = {
                            "function_call": {
                                "name": p.tool,
                                "args": p.state.input,
                            }
                        }
                        # Include thought_signature for Gemini 3.x round-tripping
                        if p.metadata and p.metadata.get("thought_signature"):
                            fc_part["thought_signature"] = p.metadata["thought_signature"]
                        parts.append(fc_part)
                    elif isinstance(p, ToolUsePart):
                        parts.append(
                            {
                                "function_call": {
                                    "name": p.name,
                                    "args": p.arguments,
                                }
                            }
                        )

                if parts:
                    contents.append({"role": "model", "parts": parts})

            elif msg.role == Role.TOOL:
                parts = []
                for part in msg.content:
                    if hasattr(part, "tool_use_id") and hasattr(part, "name"):
                        content = part.content if hasattr(part, "content") else str(part)
                        # Parse JSON content if possible
                        try:
                            response_val = json.loads(content) if isinstance(content, str) else content
                        except (json.JSONDecodeError, TypeError):
                            response_val = {"result": content}
                        parts.append(
                            {
                                "function_response": {
                                    "name": part.name,
                                    "response": response_val,
                                }
                            }
                        )
                if parts:
                    contents.append({"role": "user", "parts": parts})

        return system_instruction, contents

    def chat_completion_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        on_event: StreamCallback | None = None,
    ) -> tuple[list[Part], str, dict[str, int]]:
        """Non-streaming completion from Gemini API.

        Despite the method name (kept for interface compatibility), this uses
        the non-streaming API to avoid google-genai SDK streaming bugs that
        cause 100% CPU hangs (list_extend infinite loop after CLOSE_WAIT).

        Returns:
            Tuple of (collected parts, finish_reason)
        """
        from google.genai import types

        system_instruction, contents = self._convert_messages(messages)

        # Build config
        gen_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )

        if system_instruction:
            gen_config.system_instruction = system_instruction

        if tools:
            declarations = self._convert_tools(tools)
            gen_config.tools = [types.Tool(function_declarations=declarations)]

        if on_event:
            on_event(StreamEvent(type=StreamEventType.START))

        # Non-streaming request
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=contents,
            config=gen_config,
        )

        # Parse usage metadata
        usage = TokenUsage()
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = TokenUsage(
                input=getattr(um, "prompt_token_count", 0) or 0,
                output=getattr(um, "candidates_token_count", 0) or 0,
            )

        # Parse response
        finish_reason = "stop"
        parts: list[Part] = []
        tool_calls: list[dict[str, Any]] = []
        collected_text_parts: list[str] = []
        collected_reasoning_parts: list[str] = []
        reasoning_id: str | None = None

        if response.candidates:
            candidate = response.candidates[0]

            # Check finish reason
            fr = getattr(candidate, "finish_reason", None)
            if fr is not None:
                fr_str = str(fr).lower() if fr else "stop"
                if "stop" in fr_str:
                    finish_reason = "stop"
                elif "max_tokens" in fr_str or "length" in fr_str:
                    finish_reason = "length"

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    # Handle thinking/reasoning
                    if hasattr(part, "thought") and part.thought:
                        text = part.text or ""
                        if not reasoning_id:
                            reasoning_id = f"reasoning_{datetime.now().timestamp()}"
                            if on_event:
                                on_event(StreamEvent(type=StreamEventType.REASONING_START, id=reasoning_id))
                        collected_reasoning_parts.append(text)
                        if on_reasoning:
                            on_reasoning(text)
                        if on_event:
                            on_event(StreamEvent(type=StreamEventType.REASONING_DELTA, id=reasoning_id, text=text))
                        continue

                    # Handle text
                    if hasattr(part, "text") and part.text and not getattr(part, "function_call", None):
                        text = part.text
                        # Close reasoning if transitioning to text
                        if reasoning_id and on_event:
                            on_event(
                                StreamEvent(
                                    type=StreamEventType.REASONING_DONE,
                                    id=reasoning_id,
                                    text="".join(collected_reasoning_parts),
                                )
                            )
                            reasoning_id = None

                        collected_text_parts.append(text)
                        if on_text:
                            on_text(text)
                        if on_event:
                            on_event(StreamEvent(type=StreamEventType.TEXT_DELTA, text=text))

                    # Handle function calls
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        call_id = f"call_{uuid.uuid4().hex[:12]}"
                        name = fc.name or ""
                        args = dict(fc.args) if fc.args else {}
                        thought_sig = getattr(part, "thought_signature", None)

                        tool_calls.append(
                            {
                                "id": call_id,
                                "name": name,
                                "arguments": args,
                                "thought_signature": thought_sig,
                            }
                        )

                        if on_event:
                            on_event(
                                StreamEvent(
                                    type=StreamEventType.TOOL_INPUT_START,
                                    id=call_id,
                                    tool_name=name,
                                )
                            )
                            on_event(
                                StreamEvent(
                                    type=StreamEventType.TOOL_CALL,
                                    id=call_id,
                                    tool_call_id=call_id,
                                    tool_name=name,
                                    input=args,
                                )
                            )

        # Finalize collected text
        collected_text = "".join(collected_text_parts)
        collected_reasoning = "".join(collected_reasoning_parts)

        # End reasoning if still active
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
        for tc in tool_calls:
            meta = None
            if tc.get("thought_signature"):
                meta = {"thought_signature": tc["thought_signature"]}

            parts.append(
                ToolPart(
                    id=tc["id"],
                    tool=tc["name"],
                    call_id=tc["id"],
                    state=ToolState(
                        status=ToolStatus.PENDING,
                        input=tc["arguments"],
                        time=ToolTime(start=datetime.now().timestamp()),
                    ),
                    metadata=meta,
                )
            )

        if tool_calls:
            finish_reason = "tool_calls"

        # Emit usage and finish
        if on_event:
            on_event(StreamEvent(type=StreamEventType.USAGE, usage=usage))
            on_event(StreamEvent(type=StreamEventType.FINISH, finish_reason=finish_reason))

        usage_dict = {
            "input_tokens": usage.input if usage else 0,
            "output_tokens": usage.output if usage else 0,
        }
        return parts, finish_reason, usage_dict

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Non-streaming completion (for compaction, etc.)."""
        from google.genai import types

        # Convert raw message dicts to Gemini format
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})

        gen_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )

        if system_instruction:
            gen_config.system_instruction = system_instruction

        if tools:
            declarations = []
            for t in tools:
                func = t.get("function", t)
                declarations.append(
                    {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": self._clean_schema(func.get("parameters", {})),
                    }
                )
            gen_config.tools = [types.Tool(function_declarations=declarations)]

        response = self.client.models.generate_content(
            model=self.config.model,
            contents=contents,
            config=gen_config,
        )

        text = response.text or ""
        um = response.usage_metadata
        metadata = {
            "usage": {
                "input": getattr(um, "prompt_token_count", 0) if um else 0,
                "output": getattr(um, "candidates_token_count", 0) if um else 0,
            },
            "finish_reason": "stop",
        }

        return text, metadata
