"""Message models for session management."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


class MessageRole(str, Enum):
    """Role of a message in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessagePartType(str, Enum):
    """Type of message part."""

    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    REASONING = "reasoning"
    IMAGE = "image"


class MessagePart(BaseModel):
    """A part of a message."""

    id: str
    message_id: str
    session_id: str
    type: MessagePartType
    content: Any
    metadata: dict[str, Any] = {}

    class Config:
        use_enum_values = True


class MessageInfo(BaseModel):
    """Information about a message."""

    id: str
    session_id: str
    role: MessageRole
    parent_id: str | None = None
    time: dict[str, int] = {}
    metadata: dict[str, Any] = {}

    class Config:
        use_enum_values = True

    @classmethod
    def create(
        cls,
        id: str,
        session_id: str,
        role: MessageRole,
        parent_id: str | None = None,
    ) -> "MessageInfo":
        """Create a new message info."""
        now = int(datetime.now().timestamp() * 1000)
        return cls(
            id=id,
            session_id=session_id,
            role=role,
            parent_id=parent_id,
            time={"created": now, "updated": now},
        )


@dataclass
class MessageWithParts:
    """A message with its parts."""

    info: MessageInfo
    parts: list[MessagePart] = field(default_factory=list)

    def get_text(self) -> str:
        """Get combined text content."""
        text_parts = []
        for part in self.parts:
            if part.type == MessagePartType.TEXT:
                text_parts.append(str(part.content))
        return "\n".join(text_parts)

    def get_tool_uses(self) -> list[MessagePart]:
        """Get tool use parts."""
        return [p for p in self.parts if p.type == MessagePartType.TOOL_USE]
