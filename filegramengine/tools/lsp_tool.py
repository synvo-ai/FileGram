"""LSP tool for interacting with Language Server Protocol servers."""

import json
import subprocess
from pathlib import Path
from typing import Any

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from .base import BaseTool, ToolContext

# Load description from txt file
DESCRIPTION = load_tool_prompt("lsp")

# Supported LSP operations
OPERATIONS = [
    "goToDefinition",
    "findReferences",
    "hover",
    "documentSymbol",
    "workspaceSymbol",
    "goToImplementation",
    "prepareCallHierarchy",
    "incomingCalls",
    "outgoingCalls",
]


class LspTool(BaseTool):
    """Tool for interacting with Language Server Protocol servers.

    Provides code intelligence features like:
    - Go to definition
    - Find references
    - Hover information
    - Document symbols
    - Workspace symbols
    - Go to implementation
    - Call hierarchy
    """

    @property
    def name(self) -> str:
        return "lsp"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": OPERATIONS,
                    "description": "The LSP operation to perform",
                },
                "file_path": {
                    "type": "string",
                    "description": "The absolute or relative path to the file",
                },
                "line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "The line number (1-based, as shown in editors)",
                },
                "character": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "The character offset (1-based, as shown in editors)",
                },
            },
            "required": ["operation", "file_path", "line", "character"],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        operation = arguments.get("operation", "")
        file_path_str = arguments.get("file_path", "")
        line = arguments.get("line", 1)
        character = arguments.get("character", 1)

        if operation not in OPERATIONS:
            return self._make_result(
                tool_use_id,
                f"Invalid operation: {operation}. Valid operations: {', '.join(OPERATIONS)}",
                is_error=True,
            )

        # Resolve file path
        file_path = Path(file_path_str)
        if not file_path.is_absolute():
            file_path = context.target_directory / file_path

        if not file_path.exists():
            return self._make_result(
                tool_use_id,
                f"File not found: {file_path}",
                is_error=True,
            )

        # Convert to 0-based indices for LSP
        lsp_line = line - 1
        lsp_character = character - 1

        try:
            # Try to use available LSP tools
            result = await self._execute_lsp_operation(
                operation,
                str(file_path),
                lsp_line,
                lsp_character,
                context,
            )

            if result is None:
                return self._make_result(
                    tool_use_id,
                    "No LSP server available for this file type. "
                    "LSP operations require a language server to be configured.",
                    is_error=True,
                )

            if not result:
                return self._make_result(
                    tool_use_id,
                    f"No results found for {operation}",
                    metadata={"operation": operation, "file": str(file_path)},
                )

            output = json.dumps(result, indent=2)

            return self._make_result(
                tool_use_id,
                output,
                metadata={
                    "operation": operation,
                    "file": str(file_path),
                    "line": line,
                    "character": character,
                    "result_count": len(result) if isinstance(result, list) else 1,
                },
            )

        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"LSP operation failed: {str(e)}",
                is_error=True,
            )

    async def _execute_lsp_operation(
        self,
        operation: str,
        file_path: str,
        line: int,
        character: int,
        context: ToolContext,
    ) -> list[Any] | None:
        """Execute an LSP operation.

        This is a simplified implementation that tries common approaches:
        1. Check if there's a running LSP server via MCP
        2. Use ctags/cscope as fallback for basic operations

        A full implementation would manage LSP server connections.
        """
        # Try using ctags for basic operations
        if operation in ["goToDefinition", "findReferences", "documentSymbol"]:
            return await self._use_ctags_fallback(operation, file_path, line, character, context)

        # For other operations, we need a real LSP server
        # This would require a more complex implementation
        return None

    async def _use_ctags_fallback(
        self,
        operation: str,
        file_path: str,
        line: int,
        character: int,
        context: ToolContext,
    ) -> list[Any] | None:
        """Use ctags as a fallback for basic LSP operations."""
        try:
            # Check if ctags is available
            result = subprocess.run(
                ["which", "ctags"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return None

            if operation == "documentSymbol":
                # Get symbols in file
                result = subprocess.run(
                    ["ctags", "-x", "--output-format=json", file_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    symbols = []
                    for line in result.stdout.strip().split("\n"):
                        if line:
                            try:
                                symbol = json.loads(line)
                                symbols.append(
                                    {
                                        "name": symbol.get("name", ""),
                                        "kind": symbol.get("kind", ""),
                                        "line": symbol.get("line", 0),
                                    }
                                )
                            except json.JSONDecodeError:
                                # Parse plain ctags output
                                parts = line.split()
                                if len(parts) >= 4:
                                    symbols.append(
                                        {
                                            "name": parts[0],
                                            "kind": parts[1],
                                            "line": int(parts[2]) if parts[2].isdigit() else 0,
                                        }
                                    )
                    return symbols

            elif operation == "goToDefinition":
                # Read the file to get the symbol at position
                with open(file_path) as f:
                    lines = f.readlines()

                if line < len(lines):
                    line_content = lines[line]
                    # Extract word at character position
                    word = ""
                    start = character
                    while start > 0 and line_content[start - 1].isalnum() or line_content[start - 1] == "_":
                        start -= 1
                    end = character
                    while end < len(line_content) and (line_content[end].isalnum() or line_content[end] == "_"):
                        end += 1
                    word = line_content[start:end]

                    if word:
                        # Search for definition using grep
                        result = subprocess.run(
                            [
                                "grep",
                                "-rn",
                                f"def {word}\\|class {word}\\|function {word}\\|{word} =",
                                str(context.target_directory),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )

                        if result.returncode == 0:
                            definitions = []
                            for line in result.stdout.strip().split("\n")[:10]:
                                if ":" in line:
                                    parts = line.split(":", 2)
                                    if len(parts) >= 2:
                                        definitions.append(
                                            {
                                                "file": parts[0],
                                                "line": int(parts[1]) if parts[1].isdigit() else 0,
                                                "content": parts[2] if len(parts) > 2 else "",
                                            }
                                        )
                            return definitions

            return None

        except Exception:
            return None
