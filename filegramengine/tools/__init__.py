"""Tool implementations."""

from .apply_patch import ApplyPatchTool
from .base import BaseTool, ToolContext
from .bash import BashTool
from .batch import BatchTool
from .codesearch import CodeSearchTool
from .edit import EditTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .ls import ListTool
from .lsp_tool import LspTool
from .multiedit import MultiEditTool
from .plan import (
    PlanEnterTool,
    PlanExitTool,
    get_plan_state,
    is_operation_allowed_in_plan_mode,
)
from .question import QuestionTool
from .read import ReadTool
from .registry import ToolRegistry, create_default_registry, create_plan_mode_registry
from .replacer import REPLACERS, replace, trim_diff
from .skill import DynamicSkillTool, SkillTool
from .task import TaskTool
from .todo import TodoItem, TodoPriority, TodoReadTool, TodoStatus, TodoWriteTool
from .webfetch import WebFetchTool
from .websearch import WebSearchTool
from .write import WriteTool

__all__ = [
    # Base
    "BaseTool",
    "ToolContext",
    # Registry
    "ToolRegistry",
    "create_default_registry",
    "create_plan_mode_registry",
    # Core tools
    "BashTool",
    "ReadTool",
    "WriteTool",
    "EditTool",
    "GrepTool",
    "GlobTool",
    "TaskTool",
    # Web tools
    "WebFetchTool",
    "WebSearchTool",
    # Multi-edit tool
    "MultiEditTool",
    # Question tool
    "QuestionTool",
    # New tools (OpenCode parity)
    "ListTool",
    "BatchTool",
    "CodeSearchTool",
    "ApplyPatchTool",
    "LspTool",
    # Todo tools
    "TodoReadTool",
    "TodoWriteTool",
    "TodoItem",
    "TodoStatus",
    "TodoPriority",
    # Plan tools
    "PlanEnterTool",
    "PlanExitTool",
    "get_plan_state",
    "is_operation_allowed_in_plan_mode",
    # Skill tool
    "SkillTool",
    "DynamicSkillTool",
    # Replacer
    "replace",
    "trim_diff",
    "REPLACERS",
]
