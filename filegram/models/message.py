"""Message and Part models for CodeAgent - V2 Complete Implementation."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolStatus(str, Enum):
    """Tool execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


# ============== Part Types ==============


class PartBase(BaseModel):
    """Base class for all message parts."""

    id: str = ""
    session_id: str = ""
    message_id: str = ""


class TextPart(PartBase):
    """Text content part."""

    type: Literal["text"] = "text"
    text: str
    synthetic: bool = False  # System-generated, not user input
    ignored: bool = False  # Should be ignored in context
    time: dict[str, float] | None = None  # {start, end}
    metadata: dict[str, Any] | None = None


class ReasoningPart(PartBase):
    """Reasoning/thinking content part (for Claude thinking, o1 reasoning)."""

    type: Literal["reasoning"] = "reasoning"
    text: str
    time: dict[str, float] = Field(default_factory=lambda: {"start": datetime.now().timestamp()})
    metadata: dict[str, Any] | None = None


class ToolTime(BaseModel):
    """Time tracking for tool execution."""

    start: float = 0
    end: float | None = None
    compacted: float | None = None  # When this tool output was pruned


class ToolState(BaseModel):
    """Tool execution state with full state machine."""

    status: ToolStatus = ToolStatus.PENDING
    input: dict[str, Any] = Field(default_factory=dict)
    raw: str = ""  # Raw input string before parsing
    output: str = ""
    title: str = ""
    error: str | None = None
    metadata: dict[str, Any] | None = None
    time: ToolTime = Field(default_factory=ToolTime)
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class ToolPart(PartBase):
    """Tool call part with complete state machine."""

    type: Literal["tool"] = "tool"
    tool: str  # Tool name
    call_id: str  # Unique call ID from LLM
    state: ToolState = Field(default_factory=ToolState)
    metadata: dict[str, Any] | None = None

    @property
    def is_compacted(self) -> bool:
        """Check if this tool output has been pruned."""
        return self.state.time.compacted is not None

    def mark_compacted(self) -> None:
        """Mark this tool output as pruned."""
        self.state.time.compacted = datetime.now().timestamp()


class FilePart(PartBase):
    """File attachment part (images, PDFs, etc.)."""

    type: Literal["file"] = "file"
    mime: str
    filename: str | None = None
    url: str  # File URL or data URL
    source: dict[str, Any] | None = None  # Source information


class SnapshotPart(PartBase):
    """Snapshot reference part."""

    type: Literal["snapshot"] = "snapshot"
    snapshot: str  # Snapshot ID


class PatchPart(PartBase):
    """Code patch part."""

    type: Literal["patch"] = "patch"
    hash: str
    files: list[str] = Field(default_factory=list)


class CompactionPart(PartBase):
    """Compaction marker part."""

    type: Literal["compaction"] = "compaction"
    auto: bool = False  # Whether this was automatic compaction


class AgentPart(PartBase):
    """Agent invocation part (@agent)."""

    type: Literal["agent"] = "agent"
    name: str  # Agent name
    source: dict[str, Any] | None = None


class SubtaskPart(PartBase):
    """Subtask part for task tool."""

    type: Literal["subtask"] = "subtask"
    agent: str
    description: str
    command: str | None = None


class StepStartPart(PartBase):
    """Step start marker."""

    type: Literal["step-start"] = "step-start"
    snapshot: str | None = None


class StepFinishPart(PartBase):
    """Step finish marker."""

    type: Literal["step-finish"] = "step-finish"
    snapshot: str | None = None


# ============== Legacy Part Types (for backward compatibility) ==============


class ToolUsePart(PartBase):
    """Tool use part (from assistant) - Legacy format."""

    type: Literal["tool_use"] = "tool_use"
    tool_use_id: str
    name: str
    arguments: dict[str, Any]

    def to_tool_part(self, session_id: str = "", message_id: str = "") -> ToolPart:
        """Convert to new ToolPart format."""
        return ToolPart(
            id=self.tool_use_id,
            session_id=session_id,
            message_id=message_id,
            tool=self.name,
            call_id=self.tool_use_id,
            state=ToolState(
                status=ToolStatus.RUNNING,
                input=self.arguments,
                time=ToolTime(start=datetime.now().timestamp()),
            ),
        )


class ToolResultPart(PartBase):
    """Tool result part - Legacy format."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


# Union of all part types (including legacy types for backward compatibility)
Part = Union[  # noqa: UP007
    TextPart,
    ReasoningPart,
    ToolPart,
    FilePart,
    SnapshotPart,
    PatchPart,
    CompactionPart,
    AgentPart,
    SubtaskPart,
    StepStartPart,
    StepFinishPart,
    ToolUsePart,
    ToolResultPart,
]


# ============== Token Tracking ==============


class TokenUsage(BaseModel):
    """Token usage for a message."""

    input: int = 0
    output: int = 0
    reasoning: int = 0  # Reasoning/thinking tokens
    cache_read: int = 0
    cache_write: int = 0

    @property
    def total(self) -> int:
        """Total tokens used."""
        return self.input + self.output + self.reasoning + self.cache_read


# ============== Message Classes ==============


class MessageInfo(BaseModel):
    """Base message information."""

    id: str
    session_id: str
    role: Role
    time: dict[str, float] = Field(default_factory=lambda: {"created": datetime.now().timestamp()})


class UserMessageInfo(MessageInfo):
    """User message information."""

    role: Literal[Role.USER] = Role.USER
    agent: str = "build"  # Which agent to use
    model: dict[str, str] | None = None  # {provider_id, model_id}
    system: str | None = None  # Custom system prompt
    tools: dict[str, bool] | None = None  # Tool overrides
    variant: str | None = None  # Model variant
    summary: dict[str, Any] | None = None  # Message summary


class AssistantMessageInfo(MessageInfo):
    """Assistant message information."""

    role: Literal[Role.ASSISTANT] = Role.ASSISTANT
    parent_id: str = ""  # Parent user message ID
    agent: str = "build"
    mode: str = "build"  # Agent mode
    model_id: str = ""
    provider_id: str = ""
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    cost: float = 0
    finish: str | None = None  # Finish reason
    error: dict[str, Any] | None = None
    summary: bool = False  # Is this a summary/compaction message
    path: dict[str, str] | None = None  # {cwd, root}


class Message(BaseModel):
    """A message in the conversation with V2 format support."""

    role: Role
    content: list[Part] = Field(default_factory=list)

    # V2 fields
    info: UserMessageInfo | AssistantMessageInfo | MessageInfo | None = None

    # Token tracking (for assistant messages)
    tokens: TokenUsage | None = None

    @classmethod
    def system(cls, text: str) -> Message:
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=[TextPart(text=text)])

    @classmethod
    def user(cls, text: str, agent: str = "build") -> Message:
        """Create a user message."""
        msg = cls(role=Role.USER, content=[TextPart(text=text)])
        return msg

    @classmethod
    def assistant(cls, parts: list[Part]) -> Message:
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=parts, tokens=TokenUsage())

    @classmethod
    def tool_result(cls, tool_use_id: str, content: str, is_error: bool = False) -> Message:
        """Create a tool result message (legacy format)."""
        return cls(
            role=Role.TOOL,
            content=[ToolResultPart(tool_use_id=tool_use_id, content=content, is_error=is_error)],
        )

    def get_text(self) -> str:
        """Get concatenated text from all text parts."""
        texts = []
        for part in self.content:
            if isinstance(part, TextPart):
                texts.append(part.text)
        return "\n".join(texts)

    def get_reasoning(self) -> str:
        """Get concatenated reasoning from all reasoning parts."""
        texts = []
        for part in self.content:
            if isinstance(part, ReasoningPart):
                texts.append(part.text)
        return "\n".join(texts)

    def get_tool_uses(self) -> list[ToolUsePart]:
        """Get all tool use parts (legacy format)."""
        result = []
        for p in self.content:
            if isinstance(p, ToolUsePart):
                result.append(p)
            elif isinstance(p, ToolPart):
                # Convert new format to legacy
                result.append(
                    ToolUsePart(
                        tool_use_id=p.call_id,
                        name=p.tool,
                        arguments=p.state.input,
                    )
                )
        return result

    def get_tool_parts(self) -> list[ToolPart]:
        """Get all tool parts (V2 format)."""
        return [p for p in self.content if isinstance(p, ToolPart)]

    def add_reasoning(self, text: str, reasoning_id: str = "") -> ReasoningPart:
        """Add a reasoning part to this message."""
        part = ReasoningPart(
            id=reasoning_id,
            text=text,
            time={"start": datetime.now().timestamp()},
        )
        self.content.append(part)
        return part

    def add_tool_call(self, tool_name: str, call_id: str, arguments: dict[str, Any]) -> ToolPart:
        """Add a tool call part to this message."""
        part = ToolPart(
            id=call_id,
            tool=tool_name,
            call_id=call_id,
            state=ToolState(
                status=ToolStatus.PENDING,
                input=arguments,
                time=ToolTime(start=datetime.now().timestamp()),
            ),
        )
        self.content.append(part)
        return part

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary for API calls."""
        result: dict[str, Any] = {"role": self.role.value}

        if self.role == Role.TOOL:
            # Tool messages have special format
            for part in self.content:
                if isinstance(part, ToolResultPart):
                    result["tool_call_id"] = part.tool_use_id
                    result["content"] = part.content
                    break
        elif self.role == Role.ASSISTANT:
            # Assistant may have text, reasoning, and/or tool calls
            text_parts = [p for p in self.content if isinstance(p, TextPart) and not p.ignored]
            tool_parts = [p for p in self.content if isinstance(p, ToolPart | ToolUsePart)]

            if text_parts:
                result["content"] = "\n".join(p.text for p in text_parts)
            else:
                result["content"] = ""

            if tool_parts:
                result["tool_calls"] = []
                for tp in tool_parts:
                    if isinstance(tp, ToolPart):
                        result["tool_calls"].append(
                            {
                                "id": tp.call_id,
                                "type": "function",
                                "function": {
                                    "name": tp.tool,
                                    "arguments": json.dumps(tp.state.input),
                                },
                            }
                        )
                    elif isinstance(tp, ToolUsePart):
                        result["tool_calls"].append(
                            {
                                "id": tp.tool_use_id,
                                "type": "function",
                                "function": {
                                    "name": tp.name,
                                    "arguments": json.dumps(tp.arguments),
                                },
                            }
                        )
        else:
            # System and user messages
            text_parts = [p for p in self.content if isinstance(p, TextPart) and not p.ignored]
            result["content"] = "\n".join(p.text for p in text_parts) if text_parts else self.get_text()

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Create message from dictionary."""
        role = Role(data["role"])
        parts: list[Part] = []

        if role == Role.TOOL:
            parts.append(
                ToolResultPart(
                    tool_use_id=data.get("tool_call_id", ""),
                    content=data.get("content", ""),
                    is_error=data.get("is_error", False),
                )
            )
        elif role == Role.ASSISTANT:
            if data.get("content"):
                parts.append(TextPart(text=data["content"]))

            for tc in data.get("tool_calls", []):
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                parts.append(
                    ToolPart(
                        id=tc.get("id", ""),
                        tool=func.get("name", ""),
                        call_id=tc.get("id", ""),
                        state=ToolState(
                            status=ToolStatus.RUNNING,
                            input=args,
                            time=ToolTime(start=datetime.now().timestamp()),
                        ),
                    )
                )
        else:
            parts.append(TextPart(text=data.get("content", "")))

        return cls(role=role, content=parts)


class MessageWithParts(BaseModel):
    """Message with its parts, matching OpenCode's WithParts type."""

    info: UserMessageInfo | AssistantMessageInfo | MessageInfo
    parts: list[Part] = Field(default_factory=list)

    def to_message(self) -> Message:
        """Convert to Message format."""
        return Message(
            role=self.info.role,
            content=self.parts,
            info=self.info,
        )
