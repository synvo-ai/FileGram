"""Todo tools for task management.

Provides todoread and todowrite tools for managing structured task lists.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from .base import BaseTool, ToolContext

# Load descriptions from txt files (like OpenCode's static import)
TODOREAD_DESCRIPTION = load_tool_prompt("todoread")
TODOWRITE_DESCRIPTION = load_tool_prompt("todowrite")


class TodoStatus(str, Enum):
    """Todo item status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TodoPriority(str, Enum):
    """Todo item priority."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TodoItem:
    """A single todo item."""

    content: str
    status: str = TodoStatus.PENDING.value
    priority: str = TodoPriority.MEDIUM.value
    id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TodoItem:
        return cls(
            content=data.get("content", ""),
            status=data.get("status", TodoStatus.PENDING.value),
            priority=data.get("priority", TodoPriority.MEDIUM.value),
            id=data.get("id", ""),
        )


class TodoStorage:
    """Manages todo persistence for sessions."""

    def __init__(self, storage_dir: Path | None = None):
        if storage_dir is None:
            storage_dir = Path.home() / ".synvocode" / "storage" / "todo"
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}.json"

    def read(self, session_id: str) -> list[TodoItem]:
        """Read todos for a session."""
        path = self._get_path(session_id)
        if not path.exists():
            return []

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return [TodoItem.from_dict(item) for item in data]
        except (json.JSONDecodeError, OSError):
            return []

    def write(self, session_id: str, todos: list[TodoItem]) -> None:
        """Write todos for a session."""
        path = self._get_path(session_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump([item.to_dict() for item in todos], f, indent=2)
        except OSError as e:
            raise RuntimeError(f"Failed to save todos: {e}")

    def clear(self, session_id: str) -> None:
        """Clear todos for a session."""
        path = self._get_path(session_id)
        if path.exists():
            path.unlink()


# Global storage instance
_storage: TodoStorage | None = None


def get_storage() -> TodoStorage:
    """Get or create the global todo storage."""
    global _storage
    if _storage is None:
        _storage = TodoStorage()
    return _storage


# ============== TodoRead Tool ==============


class TodoReadTool(BaseTool):
    """Tool for reading the current todo list."""

    @property
    def name(self) -> str:
        return "todoread"

    @property
    def description(self) -> str:
        return TODOREAD_DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        try:
            storage = get_storage()
            todos = storage.read(context.session_id)

            if not todos:
                return self._make_result(
                    tool_use_id,
                    "No todos in the list.",
                    metadata={"todos": [], "count": 0},
                )

            # Format output
            pending_count = sum(1 for t in todos if t.status == TodoStatus.PENDING.value)
            in_progress_count = sum(1 for t in todos if t.status == TodoStatus.IN_PROGRESS.value)
            completed_count = sum(1 for t in todos if t.status == TodoStatus.COMPLETED.value)

            lines = [
                f"Todo List ({len(todos)} items):",
                f"  Pending: {pending_count}, In Progress: {in_progress_count}, Completed: {completed_count}",
                "",
            ]

            for i, todo in enumerate(todos, 1):
                status_icon = {
                    TodoStatus.PENDING.value: "[ ]",
                    TodoStatus.IN_PROGRESS.value: "[•]",
                    TodoStatus.COMPLETED.value: "[✓]",
                    TodoStatus.CANCELLED.value: "[✗]",
                }.get(todo.status, "[ ]")

                priority_marker = ""
                if todo.priority == TodoPriority.HIGH.value:
                    priority_marker = " (!)"
                elif todo.priority == TodoPriority.LOW.value:
                    priority_marker = " (~)"

                lines.append(f"  {i}. {status_icon} {todo.content}{priority_marker}")

            output = "\n".join(lines)

            return self._make_result(
                tool_use_id,
                output,
                metadata={
                    "todos": [t.to_dict() for t in todos],
                    "count": len(todos),
                    "pending": pending_count,
                    "in_progress": in_progress_count,
                    "completed": completed_count,
                },
            )

        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Failed to read todos: {str(e)}",
                is_error=True,
            )


# ============== TodoWrite Tool ==============


class TodoWriteTool(BaseTool):
    """Tool for writing/updating the todo list."""

    @property
    def name(self) -> str:
        return "todowrite"

    @property
    def description(self) -> str:
        return TODOWRITE_DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "The complete updated todo list",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Brief description of the task (1-2 sentences)",
                            },
                            "status": {
                                "type": "string",
                                "enum": [
                                    "pending",
                                    "in_progress",
                                    "completed",
                                    "cancelled",
                                ],
                                "description": "Current status of the task",
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "Priority level (default: medium)",
                            },
                            "id": {
                                "type": "string",
                                "description": "Optional unique identifier",
                            },
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["todos"],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        try:
            todos_data = arguments.get("todos", [])

            if not isinstance(todos_data, list):
                return self._make_result(
                    tool_use_id,
                    "todos must be an array",
                    is_error=True,
                )

            # Parse and validate todos
            todos: list[TodoItem] = []
            in_progress_count = 0

            for i, item in enumerate(todos_data):
                if not isinstance(item, dict):
                    return self._make_result(
                        tool_use_id,
                        f"Todo item {i} must be an object",
                        is_error=True,
                    )

                content = item.get("content", "")
                if not content:
                    return self._make_result(
                        tool_use_id,
                        f"Todo item {i} must have content",
                        is_error=True,
                    )

                status = item.get("status", TodoStatus.PENDING.value)
                if status == TodoStatus.IN_PROGRESS.value:
                    in_progress_count += 1

                todo = TodoItem(
                    content=content,
                    status=status,
                    priority=item.get("priority", TodoPriority.MEDIUM.value),
                    id=item.get("id", f"todo_{i + 1}"),
                )
                todos.append(todo)

            # Warn if multiple in_progress
            warning = ""
            if in_progress_count > 1:
                warning = (
                    f"\n\nWarning: {in_progress_count} tasks marked as in_progress. Best practice is to have only one."
                )

            # Save todos
            storage = get_storage()
            storage.write(context.session_id, todos)

            # Calculate stats
            pending = sum(1 for t in todos if t.status == TodoStatus.PENDING.value)
            completed = sum(1 for t in todos if t.status == TodoStatus.COMPLETED.value)

            output = (
                f"Todo list updated ({len(todos)} items):\n"
                f"  Pending: {pending}, In Progress: {in_progress_count}, Completed: {completed}"
                f"{warning}"
            )

            return self._make_result(
                tool_use_id,
                output,
                metadata={
                    "todos": [t.to_dict() for t in todos],
                    "count": len(todos),
                    "pending": pending,
                    "in_progress": in_progress_count,
                    "completed": completed,
                },
            )

        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Failed to write todos: {str(e)}",
                is_error=True,
            )


__all__ = [
    "TodoItem",
    "TodoStatus",
    "TodoPriority",
    "TodoStorage",
    "TodoReadTool",
    "TodoWriteTool",
    "get_storage",
]
