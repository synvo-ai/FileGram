"""List tool for listing files and directories."""

import os
import subprocess
from typing import Any

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from .base import BaseTool, ToolContext

# Load description from txt file
DESCRIPTION = load_tool_prompt("ls")

# Default ignore patterns
IGNORE_PATTERNS = [
    "node_modules/",
    "__pycache__/",
    ".git/",
    "dist/",
    "build/",
    "target/",
    "vendor/",
    "bin/",
    "obj/",
    ".idea/",
    ".vscode/",
    ".zig-cache/",
    "zig-out",
    ".coverage",
    "coverage/",
    "tmp/",
    "temp/",
    ".cache/",
    "cache/",
    "logs/",
    ".venv/",
    "venv/",
    "env/",
]

LIMIT = 100


class ListTool(BaseTool):
    """Tool for listing files and directories."""

    @property
    def name(self) -> str:
        return "list"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute path to the directory to list (must be absolute, not relative)",
                },
                "ignore": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of glob patterns to ignore",
                },
            },
            "required": [],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        path_arg = arguments.get("path", ".")
        custom_ignore = arguments.get("ignore", [])

        # Resolve path with security check
        try:
            search_path = str(context.resolve_path(path_arg))
        except ValueError as e:
            return self._make_result(
                tool_use_id,
                str(e),
                is_error=True,
            )

        if not os.path.exists(search_path):
            return self._make_result(
                tool_use_id,
                f"Path does not exist: {search_path}",
                is_error=True,
            )

        if not os.path.isdir(search_path):
            return self._make_result(
                tool_use_id,
                f"Path is not a directory: {search_path}",
                is_error=True,
            )

        try:
            # Build ignore globs
            ignore_globs = IGNORE_PATTERNS + custom_ignore

            # Use ripgrep if available, otherwise fall back to os.walk
            files = await self._list_files_rg(search_path, ignore_globs)
            if files is None:
                files = self._list_files_walk(search_path, ignore_globs)

            # Limit results
            truncated = len(files) >= LIMIT
            files = files[:LIMIT]

            # Build directory structure
            dirs: set[str] = set()
            files_by_dir: dict[str, list[str]] = {}

            for file in files:
                dir_path = os.path.dirname(file)
                # Normalize empty string to "." for root directory
                if dir_path == "":
                    dir_path = "."
                parts = dir_path.split(os.sep) if dir_path != "." else []

                # Add all parent directories
                for i in range(len(parts) + 1):
                    d = os.sep.join(parts[:i]) if i > 0 else "."
                    dirs.add(d)

                # Add file to its directory
                if dir_path not in files_by_dir:
                    files_by_dir[dir_path] = []
                files_by_dir[dir_path].append(os.path.basename(file))

            # Render tree
            output = self._render_tree(search_path, dirs, files_by_dir)

            return self._make_result(
                tool_use_id,
                output,
                metadata={
                    "count": len(files),
                    "truncated": truncated,
                    "path": search_path,
                },
            )

        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Failed to list directory: {str(e)}",
                is_error=True,
            )

    async def _list_files_rg(self, search_path: str, ignore_globs: list[str]) -> list[str] | None:
        """List files using ripgrep."""
        try:
            # Check if rg is available
            result = subprocess.run(
                ["which", "rg"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return None

            # Build rg command
            cmd = ["rg", "--files"]
            for pattern in ignore_globs:
                cmd.extend(["-g", f"!{pattern}"])

            result = subprocess.run(
                cmd,
                cwd=search_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return None

            files = [f for f in result.stdout.strip().split("\n") if f]
            return files

        except Exception:
            return None

    def _list_files_walk(self, search_path: str, ignore_globs: list[str]) -> list[str]:
        """List files using os.walk (fallback)."""
        files = []

        for root, dirnames, filenames in os.walk(search_path):
            # Filter directories
            dirnames[:] = [
                d
                for d in dirnames
                if not any(d.startswith(p.rstrip("/")) for p in ignore_globs if "/" in p) and not d.startswith(".")
            ]

            rel_root = os.path.relpath(root, search_path)
            if rel_root == ".":
                rel_root = ""

            for filename in filenames:
                if filename.startswith("."):
                    continue
                rel_path = os.path.join(rel_root, filename) if rel_root else filename
                files.append(rel_path)

                if len(files) >= LIMIT:
                    return files

        return files

    def _render_tree(
        self,
        base_path: str,
        dirs: set[str],
        files_by_dir: dict[str, list[str]],
    ) -> str:
        """Render directory tree."""

        def render_dir(dir_path: str, depth: int) -> str:
            indent = "  " * depth
            output = ""

            if depth > 0:
                output += f"{indent}{os.path.basename(dir_path)}/\n"

            child_indent = "  " * (depth + 1)

            # Get child directories
            children = sorted([d for d in dirs if os.path.dirname(d) == dir_path and d != dir_path])

            # Render subdirectories first
            for child in children:
                output += render_dir(child, depth + 1)

            # Render files
            dir_files = files_by_dir.get(dir_path, [])
            for file in sorted(dir_files):
                output += f"{child_indent}{file}\n"

            return output

        output = f"{base_path}/\n" + render_dir(".", 0)
        return output
