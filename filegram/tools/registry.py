"""Tool registry for managing available tools."""

from typing import Any

from ..models.tool import ToolDefinition, ToolResult
from .base import BaseTool, ToolContext


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        if name in self._tools:
            del self._tools[name]

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_definitions(self) -> list[ToolDefinition]:
        """Get definitions for all registered tools."""
        return [tool.get_definition() for tool in self._tools.values()]

    async def execute(
        self,
        name: str,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                tool_use_id=tool_use_id,
                name=name,
                output=f"Unknown tool: {name}",
                is_error=True,
            )

        try:
            return await tool.execute(tool_use_id, arguments, context)
        except Exception as e:
            return ToolResult(
                tool_use_id=tool_use_id,
                name=name,
                output=f"Tool execution error: {str(e)}",
                is_error=True,
            )

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())


def create_default_registry(
    include_task: bool = True,
    include_todo: bool = True,
    include_plan: bool = True,
    include_skill: bool = True,
    include_web: bool = True,
    include_question: bool = True,
    include_mcp: bool = True,
    project_dir: str | None = None,
    skill_custom_paths: list[str] | None = None,
    disable_claude_skills: bool = False,
    question_callback=None,
    mcp_tools: list[BaseTool] | None = None,
) -> ToolRegistry:
    """
    Create a registry with all default tools.

    Args:
        include_task: Whether to include the Task tool (for sub-agent spawning)
        include_todo: Whether to include Todo tools (todoread/todowrite)
        include_plan: Whether to include Plan tools (plan_enter/plan_exit)
        include_skill: Whether to include the Skill tool
        include_web: Whether to include web tools (webfetch/websearch)
        include_question: Whether to include the Question tool
        include_mcp: Whether to include MCP tools
        project_dir: Project directory for skill scanning
        skill_custom_paths: Additional paths to scan for skills
        disable_claude_skills: Whether to disable Claude Code skills
        question_callback: Callback function for the Question tool
        mcp_tools: List of MCP tools to register (from MCPClientManager)
    """
    from .apply_patch import ApplyPatchTool
    from .bash import BashTool
    from .batch import BatchTool
    from .codesearch import CodeSearchTool
    from .edit import EditTool
    from .glob_tool import GlobTool
    from .grep import GrepTool
    from .ls import ListTool
    from .lsp_tool import LspTool
    from .multiedit import MultiEditTool
    from .plan import PlanEnterTool, PlanExitTool
    from .question import QuestionTool
    from .read import ReadTool
    from .skill import SkillTool
    from .task import TaskTool
    from .todo import TodoReadTool, TodoWriteTool
    from .webfetch import WebFetchTool
    from .websearch import WebSearchTool
    from .write import WriteTool

    registry = ToolRegistry()

    # Core tools
    registry.register(BashTool())
    registry.register(ReadTool())
    registry.register(WriteTool())
    registry.register(EditTool())
    registry.register(MultiEditTool())
    registry.register(GrepTool())
    registry.register(GlobTool())
    registry.register(ListTool())
    registry.register(ApplyPatchTool())
    registry.register(LspTool())
    registry.register(CodeSearchTool())

    # Batch tool (needs registry reference)
    batch_tool = BatchTool()
    registry.register(batch_tool)
    batch_tool.set_registry(registry)

    # Task tool (sub-agent spawning)
    if include_task:
        registry.register(TaskTool())

    # Todo tools
    if include_todo:
        registry.register(TodoReadTool())
        registry.register(TodoWriteTool())

    # Plan tools
    if include_plan:
        registry.register(PlanEnterTool())
        registry.register(PlanExitTool())

    # Skill tool
    if include_skill:
        registry.register(
            SkillTool(
                project_dir=project_dir,
                custom_paths=skill_custom_paths,
                disable_claude_skills=disable_claude_skills,
            )
        )

    # Web tools
    if include_web:
        registry.register(WebFetchTool())
        registry.register(WebSearchTool())

    # Question tool
    if include_question:
        registry.register(QuestionTool(question_callback=question_callback))

    # MCP tools
    if include_mcp and mcp_tools:
        for tool in mcp_tools:
            registry.register(tool)

    return registry


def create_plan_mode_registry() -> ToolRegistry:
    """
    Create a registry with only read-only tools for plan mode.

    In plan mode, only these tools are available:
    - read: Read file contents
    - grep: Search for patterns
    - glob: Find files
    - bash: Limited to read-only commands
    - todoread: Read todo list
    - webfetch: Fetch web content (read-only)
    - websearch: Search the web (read-only)
    - plan_exit: Exit plan mode
    """
    from .bash import BashTool
    from .glob_tool import GlobTool
    from .grep import GrepTool
    from .plan import PlanExitTool
    from .read import ReadTool
    from .todo import TodoReadTool
    from .webfetch import WebFetchTool
    from .websearch import WebSearchTool

    registry = ToolRegistry()

    # Read-only tools
    registry.register(ReadTool())
    registry.register(GrepTool())
    registry.register(GlobTool())
    registry.register(BashTool())  # Will be filtered to read-only in execution
    registry.register(TodoReadTool())

    # Web tools (read-only)
    registry.register(WebFetchTool())
    registry.register(WebSearchTool())

    # Plan control
    registry.register(PlanExitTool())

    return registry
