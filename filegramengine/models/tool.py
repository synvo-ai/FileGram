"""Tool-related models for CodeAgent."""

from enum import Enum
from typing import Any

from pydantic import BaseModel


class ToolState(str, Enum):
    """Tool execution state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class ToolResult(BaseModel):
    """Result of a tool execution."""

    tool_use_id: str
    name: str
    output: str
    is_error: bool = False
    metadata: dict[str, Any] | None = None

    def to_content(self) -> str:
        """Convert to content string for the LLM."""
        if self.is_error:
            return f"Error: {self.output}"
        return self.output


class ToolDefinition(BaseModel):
    """Tool definition for OpenAI API."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
