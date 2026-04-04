"""Bash tool for executing shell commands."""

import asyncio
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from ..utils.truncate import truncate_output
from .base import BaseTool, ToolContext

# Load description from txt file (like OpenCode's static import)
DESCRIPTION = load_tool_prompt("bash")

# Patterns for filesystem command detection — applied per sub-command
_LS_PATTERN = re.compile(r"^\s*(?:ls|dir)\b(.*)$")
_MV_PATTERN = re.compile(r"^\s*mv\s+(.+)$")
_MKDIR_PATTERN = re.compile(r"^\s*mkdir\s+(.+)$")
_RM_PATTERN = re.compile(r"^\s*rm\s+(.+)$")
_CP_PATTERN = re.compile(r"^\s*cp\s+(.+)$")

_TEMP_EXTENSIONS = {".tmp", ".temp", ".bak", ".swp", ".swo", ".pyc", ".o", ".log"}
_BACKUP_EXTENSIONS = {".bak", ".backup", ".orig", ".old", ".save"}

# Shell keywords and builtins that should never be recorded as paths
_SHELL_KEYWORDS = frozenset(
    {
        "&&",
        "||",
        ";",
        "|",
        ">",
        ">>",
        "<",
        "<<",
        "echo",
        "printf",
        "find",
        "mkdir",
        "rm",
        "cp",
        "mv",
        "ls",
        "cat",
        "grep",
        "awk",
        "sed",
        "cd",
        "pwd",
        "true",
        "false",
        "test",
        "xargs",
        "sort",
        "uniq",
        "head",
        "tail",
        "wc",
        "tee",
        "touch",
        "chmod",
        "chown",
    }
)


def _safe_shlex_split(args_str: str) -> list[str]:
    """Split command arguments, falling back on whitespace split."""
    try:
        return shlex.split(args_str)
    except ValueError:
        return args_str.split()


def _extract_paths(args_str: str) -> list[str]:
    """Extract non-flag arguments (file paths) from an argument string.

    Filters out flags (starting with -) and shell keywords/builtins.
    """
    parts = _safe_shlex_split(args_str)
    return [p for p in parts if not p.startswith("-") and p not in _SHELL_KEYWORDS]


def _split_compound_commands(command: str) -> list[str]:
    """Split compound bash command into individual sub-commands.

    Splits by &&, ||, ; while respecting quoted strings.
    """
    sub_commands = []
    current = []
    in_single_quote = False
    in_double_quote = False
    i = 0
    chars = command

    while i < len(chars):
        c = chars[i]

        # Track quote state
        if c == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            current.append(c)
            i += 1
        elif c == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            current.append(c)
            i += 1
        elif in_single_quote or in_double_quote:
            current.append(c)
            i += 1
        # Split on && || ;
        elif c == "&" and i + 1 < len(chars) and chars[i + 1] == "&":
            sub_commands.append("".join(current))
            current = []
            i += 2
        elif c == "|" and i + 1 < len(chars) and chars[i + 1] == "|":
            sub_commands.append("".join(current))
            current = []
            i += 2
        elif c == ";":
            sub_commands.append("".join(current))
            current = []
            i += 1
        else:
            current.append(c)
            i += 1

    if current:
        sub_commands.append("".join(current))

    return [s.strip() for s in sub_commands if s.strip()]


def _resolve_path(path: str, context: ToolContext) -> str:
    """Resolve relative path to absolute using target_directory."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(context.target_directory) / p
    return str(p)


def _record_fs_events(command: str, context: ToolContext, output: str) -> None:
    """Parse a bash command and record filesystem behavioral events."""
    collector = context.behavior_collector
    if collector is None or not collector.enabled:
        return

    for sub_cmd in _split_compound_commands(command):
        sub_cmd = sub_cmd.strip()
        if not sub_cmd:
            continue

        # ls / dir → FILE_BROWSE
        m = _LS_PATTERN.match(sub_cmd)
        if m:
            args = m.group(1).strip()
            paths = _extract_paths(args)
            dir_path = _resolve_path(paths[0] if paths else ".", context)
            lines = [line for line in output.split("\n") if line.strip()] if output else []
            collector.record_file_browse(
                directory_path=dir_path,
                files_listed=len(lines),
            )

        # mv → FILE_RENAME or FILE_MOVE
        m = _MV_PATTERN.match(sub_cmd)
        if m:
            paths = _extract_paths(m.group(1))
            if len(paths) >= 2:
                old_path = _resolve_path(paths[-2], context)
                new_path = _resolve_path(paths[-1], context)
                old_dir = str(Path(old_path).parent)
                new_dir = str(Path(new_path).parent)
                if old_dir == new_dir or new_dir == ".":
                    collector.record_file_rename(old_path=old_path, new_path=new_path)
                else:
                    collector.record_file_move(old_path=old_path, new_path=new_path)

        # mkdir → DIR_CREATE
        m = _MKDIR_PATTERN.match(sub_cmd)
        if m:
            paths = _extract_paths(m.group(1))
            for dir_path in paths:
                collector.record_dir_create(dir_path=_resolve_path(dir_path, context))

        # rm → FILE_DELETE
        m = _RM_PATTERN.match(sub_cmd)
        if m:
            paths = _extract_paths(m.group(1))
            for file_path in paths:
                resolved = _resolve_path(file_path, context)
                ext = Path(resolved).suffix.lower()
                was_temporary = ext in _TEMP_EXTENSIONS
                collector.record_file_delete(
                    file_path=resolved,
                    was_temporary=was_temporary,
                )

        # cp → FILE_COPY
        m = _CP_PATTERN.match(sub_cmd)
        if m:
            paths = _extract_paths(m.group(1))
            if len(paths) >= 2:
                source = _resolve_path(paths[-2], context)
                dest = _resolve_path(paths[-1], context)
                dest_ext = Path(dest).suffix.lower()
                is_backup = dest_ext in _BACKUP_EXTENSIONS
                collector.record_file_copy(
                    source_path=source,
                    dest_path=dest,
                    is_backup=is_backup,
                )


class BashTool(BaseTool):
    """Tool for executing shell commands."""

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default: 120000)",
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory for the command (optional)",
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        command = arguments.get("command", "")
        timeout_ms = arguments.get("timeout", context.default_timeout)
        workdir = arguments.get("workdir", str(context.target_directory))

        # Auto-correct workdir if it points to a non-existent path
        # (e.g., LLM generates "Claude Code" instead of "FileGram")
        workdir_path = Path(workdir)
        if not workdir_path.exists():
            corrected = context._auto_correct_path(workdir_path, context.target_directory.resolve())
            if corrected is not None and corrected.exists():
                workdir = str(corrected)

        if not command:
            return self._make_result(
                tool_use_id,
                "No command provided",
                is_error=True,
            )

        timeout_seconds = timeout_ms / 1000

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=workdir,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                return self._make_result(
                    tool_use_id,
                    f"Command timed out after {timeout_seconds}s",
                    is_error=True,
                )

            output_parts = []
            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                output_parts.append(f"stderr:\n{stderr.decode('utf-8', errors='replace')}")

            output = "\n".join(output_parts) if output_parts else "(no output)"

            if process.returncode != 0:
                output = f"Exit code: {process.returncode}\n{output}"

            truncated_output = truncate_output(output, context.max_output_chars)

            # Record filesystem events on successful commands
            if process.returncode == 0:
                try:
                    _record_fs_events(command, context, output)
                except Exception:
                    pass  # Best-effort: never fail the tool due to event recording

            return self._make_result(
                tool_use_id,
                truncated_output,
                is_error=process.returncode != 0,
                metadata={"exit_code": process.returncode},
            )

        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Failed to execute command: {str(e)}",
                is_error=True,
            )
