"""Question tool for asking user questions during execution."""

from typing import Any

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from .base import BaseTool, ToolContext

# Load description from txt file
DESCRIPTION = load_tool_prompt("question")


class QuestionTool(BaseTool):
    """Tool for asking questions to the user.

    This tool is used when the agent needs clarification or input
    from the user to proceed with a task.
    """

    def __init__(self, question_callback=None):
        """Initialize the Question tool.

        Args:
            question_callback: Optional async callback function that will be called
                             with the question. If None, returns question for display.
        """
        self._question_callback = question_callback

    @property
    def name(self) -> str:
        return "question"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user",
                },
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "Short label for the option",
                            },
                            "description": {
                                "type": "string",
                                "description": "Longer description of what this option means",
                            },
                        },
                        "required": ["label"],
                    },
                    "description": "Optional list of predefined options for the user to choose from",
                },
                "allow_free_text": {
                    "type": "boolean",
                    "description": "Whether to allow free-form text input in addition to options (default: true)",
                },
            },
            "required": ["question"],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        question = arguments.get("question", "")
        options = arguments.get("options", [])
        allow_free_text = arguments.get("allow_free_text", True)

        if not question:
            return self._make_result(
                tool_use_id,
                "No question provided",
                is_error=True,
            )

        # Format the question for display
        output_parts = [f"Question: {question}"]

        if options:
            output_parts.append("\nOptions:")
            for i, opt in enumerate(options, 1):
                label = opt.get("label", f"Option {i}")
                description = opt.get("description", "")
                if description:
                    output_parts.append(f"  {i}. {label} - {description}")
                else:
                    output_parts.append(f"  {i}. {label}")

        if allow_free_text:
            output_parts.append("\n(You can also provide a custom response)")

        output = "\n".join(output_parts)

        # If we have a callback, use it
        if self._question_callback:
            try:
                response = await self._question_callback(question, options, allow_free_text)
                return self._make_result(
                    tool_use_id,
                    f"User response: {response}",
                    metadata={
                        "question": question,
                        "response": response,
                        "had_options": len(options) > 0,
                    },
                )
            except Exception as e:
                return self._make_result(
                    tool_use_id,
                    f"Failed to get user response: {str(e)}",
                    is_error=True,
                )

        # Return a special result that indicates user input is needed
        return self._make_result(
            tool_use_id,
            output,
            metadata={
                "requires_user_input": True,
                "question": question,
                "options": options,
                "allow_free_text": allow_free_text,
            },
        )
