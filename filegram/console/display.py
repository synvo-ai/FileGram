"""Output formatting for console display.

Replicates OpenCode's run.ts compact tool output style and printEvent formatting.
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

from .logo import print_logo


class Display:
    """Core output formatting class.

    Replicates OpenCode's run.ts printEvent output style with compact
    single-line tool output, markdown rendering, and status display.
    """

    # Tool display name and Rich style mapping
    # Replicates OpenCode's TOOL constant from run.ts:
    #   todowrite: ["Todo", TEXT_WARNING_BOLD]     -> yellow bold
    #   bash:      ["Bash", TEXT_DANGER_BOLD]       -> red bold
    #   edit:      ["Edit", TEXT_SUCCESS_BOLD]      -> green bold
    #   glob:      ["Glob", TEXT_INFO_BOLD]         -> blue bold
    #   grep:      ["Grep", TEXT_INFO_BOLD]         -> blue bold
    #   list:      ["List", TEXT_INFO_BOLD]         -> blue bold
    #   read:      ["Read", TEXT_HIGHLIGHT_BOLD]    -> cyan bold
    #   write:     ["Write", TEXT_SUCCESS_BOLD]     -> green bold
    #   websearch: ["Search", TEXT_DIM_BOLD]        -> dim bold
    TOOL_STYLES: dict[str, tuple[str, str]] = {
        "bash": ("Bash", "red bold"),
        "edit": ("Edit", "green bold"),
        "read": ("Read", "cyan bold"),
        "write": ("Write", "green bold"),
        "glob": ("Glob", "blue bold"),
        "glob_tool": ("Glob", "blue bold"),
        "grep": ("Grep", "blue bold"),
        "ls": ("List", "blue bold"),
        "todowrite": ("Todo", "yellow bold"),
        "todoread": ("Todo", "yellow bold"),
        "websearch": ("Search", "dim bold"),
        "webfetch": ("Fetch", "dim bold"),
        "task": ("Task", "magenta bold"),
        "question": ("Ask", "yellow bold"),
        "plan_enter": ("Plan", "cyan bold"),
        "plan_exit": ("Plan", "cyan bold"),
        "skill": ("Skill", "magenta bold"),
    }

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def print_tool_compact(self, tool_name: str, title: str) -> None:
        """Print compact tool output in OpenCode style.

        Replicates OpenCode's printEvent(color, type, title):
            | Read    src/file.ts

        Args:
            tool_name: The tool name (e.g., "read", "bash")
            title: The title/description (e.g., file path, command)
        """
        display_name, style = self.TOOL_STYLES.get(tool_name, (tool_name.capitalize(), "dim bold"))
        # Format: " | Read    src/file.ts"
        # The pipe character gets the tool color, name is padded to 8 chars
        bar = f"[{style}]|[/{style}]"
        name_padded = f"[{style}]{display_name:<8}[/{style}]"
        self.console.print(f" {bar} {name_padded}{title}")

    def get_tool_title(self, tool_name: str, args: dict[str, Any]) -> str:
        """Generate a title string for a tool call.

        Replicates OpenCode's: part.state.title || JSON.stringify(part.state.input)

        Args:
            tool_name: Tool name
            args: Tool arguments dict

        Returns:
            Human-readable title string
        """
        if tool_name in ("read", "write", "edit"):
            return args.get("file_path", "")
        elif tool_name == "bash":
            cmd = args.get("command", "")
            return cmd[:80] + ("..." if len(cmd) > 80 else "")
        elif tool_name in ("glob", "glob_tool"):
            return args.get("pattern", "")
        elif tool_name == "grep":
            pattern = args.get("pattern", "")
            path = args.get("path", "")
            if path:
                return f"{pattern} ({path})"
            return pattern
        elif tool_name == "ls":
            return args.get("path", ".")
        elif tool_name == "task":
            return args.get("description", args.get("prompt", "")[:60])
        elif tool_name == "question":
            questions = args.get("questions", [])
            if questions and isinstance(questions, list):
                return questions[0].get("question", "")[:60] if questions else ""
            return ""
        elif tool_name == "websearch":
            return args.get("query", "")
        elif tool_name == "webfetch":
            return args.get("url", "")
        elif tool_name == "todowrite":
            todos = args.get("todos", [])
            if todos and isinstance(todos, list):
                return f"{len(todos)} items"
            return ""
        else:
            if args:
                return json.dumps(args, ensure_ascii=False)[:60]
            return ""

    def print_tool_use(self, tool_name: str, args: dict[str, Any]) -> None:
        """Print a tool call in compact format.

        Combines get_tool_title and print_tool_compact.

        Args:
            tool_name: Tool name
            args: Tool arguments
        """
        title = self.get_tool_title(tool_name, args)
        self.print_tool_compact(tool_name, title)

    def print_tool_result_bash(self, output: str) -> None:
        """Print bash tool stdout output.

        Replicates OpenCode's special handling for bash:
            if (part.tool === "bash" && part.state.output?.trim()) {
                UI.println(part.state.output)
            }

        Args:
            output: The bash command output
        """
        stripped = output.strip()
        if stripped:
            self.console.print()
            self.console.print(stripped)

    def print_tool_result(self, name: str, output: str, is_error: bool) -> None:
        """Print tool result (errors only shown explicitly).

        In compact mode, successful results are silent (only the compact
        tool call line is shown). Errors are displayed.

        Args:
            name: Tool name
            output: Tool output
            is_error: Whether this is an error result
        """
        if is_error:
            self.console.print(f" [red bold]![/red bold] [red]{name}: {output[:200]}[/red]")

    def print_text(self, text: str) -> None:
        """Print assistant text output with Markdown rendering.

        Args:
            text: Markdown text to render
        """
        self.console.print()
        self.console.print(Markdown(text))
        self.console.print()

    def print_thinking(self, text: str) -> None:
        """Print thinking/reasoning text in dim style.

        Args:
            text: Reasoning text (streamed delta)
        """
        self.console.print(f"[dim italic]{text}[/dim italic]", end="")

    def print_status_bar(
        self,
        model: str,
        tokens: int = 0,
        directory: str = "",
        profile: str = "",
    ) -> None:
        """Print a status bar with model, token, and directory info.

        Args:
            model: Model display name
            tokens: Total tokens used
            directory: Working directory
            profile: Active profile name
        """
        parts = []
        if model:
            parts.append(f"[bold]{model}[/bold]")
        if profile:
            parts.append(f"[dim]profile:[/dim] {profile}")
        if directory:
            parts.append(f"[dim]dir:[/dim] {directory}")
        if tokens > 0:
            parts.append(f"[dim]tokens:[/dim] {tokens:,}")

        status_line = "  ".join(parts)
        self.console.print(f"[dim]{status_line}[/dim]")

    def print_error(self, msg: str) -> None:
        """Print an error message.

        Args:
            msg: Error message text
        """
        self.console.print(f"[red bold]Error:[/red bold] {msg}")

    def print_welcome(
        self,
        directory: str,
        model: str,
        profile: str = "",
        session_id: str = "",
    ) -> None:
        """Print welcome information with logo.

        Replaces the old Panel-based welcome with OpenCode-style output.

        Args:
            directory: Working directory
            model: Model display name
            profile: Active profile name
            session_id: Current session ID or slug (unused, kept for compatibility)
        """
        print_logo(self.console)
        self.console.print()
        self.print_status_bar(model=model, directory=directory, profile=profile)
        if session_id:
            self.console.print(f"  [dim]session:[/dim] {session_id}")
        self.console.print()

    def print_session_info(self, session_id: str, title: str, slug: str = "") -> None:
        """Print current session information.

        Args:
            session_id: Session ID
            title: Session title
            slug: Session slug
        """
        self.console.print(f"  [bold]Session:[/bold] {title}")
        if slug:
            self.console.print(f"  [dim]Slug:[/dim] {slug}")
        self.console.print(f"  [dim]ID:[/dim] {session_id}")

    def print_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Print token usage after a response (dim, compact).

        Args:
            input_tokens: Input tokens used
            output_tokens: Output tokens used
        """
        total = input_tokens + output_tokens
        self.console.print(f"[dim]  tokens: {total:,} (in: {input_tokens:,}, out: {output_tokens:,})[/dim]")

    def print_sessions_list(self, sessions: list[dict[str, Any]], current_id: str = "") -> None:
        """Print a list of sessions.

        Args:
            sessions: List of session dicts with id, title, slug, time
            current_id: Currently active session ID (will be highlighted)
        """
        if not sessions:
            self.console.print("[dim]No sessions found.[/dim]")
            return

        self.console.print("[bold]Recent Sessions:[/bold]")
        for i, s in enumerate(sessions[:10]):
            marker = "[green]>[/green] " if s.get("id") == current_id else "  "
            title = s.get("title", "Untitled")
            slug = s.get("slug", "")

            # Format time
            time_info = s.get("time", {})
            updated = time_info.get("updated", 0)
            if updated:
                from datetime import datetime

                dt = datetime.fromtimestamp(updated / 1000)
                time_str = dt.strftime("%m/%d %H:%M")
            else:
                time_str = ""

            idx_str = f"[dim]{i + 1}.[/dim]"
            self.console.print(f"{marker}{idx_str} [bold]{title}[/bold]  [dim]{slug}[/dim]  [dim]{time_str}[/dim]")
