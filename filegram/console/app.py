"""Console application - enhanced interactive loop with session management.

Replaces the basic run_interactive() with a full console experience
including logo display, session persistence, and slash commands.
"""

from __future__ import annotations

import hashlib

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console

from ..agent.loop import AgentLoop
from ..config import Config
from ..profile import get_current_profile, list_profiles
from ..session import Session, SessionInfo
from .display import Display

# ============== Command Completer ==============


class SlashCommandCompleter(Completer):
    """Auto-completer for slash commands."""

    # All available commands with descriptions
    COMMANDS = {
        "/help": "Show available commands",
        "/stats": "Show context statistics",
        "/reset": "Reset conversation",
        "/model": "Show or switch model (provider/model)",
        "/profile": "Show or switch agent profile",
        "/auth": "Manage credentials (login|logout|status)",
        "/diff": "Show files changed in this session",
        "/revert": "Revert changes (last|all|<message_id>)",
        "/new": "Create a new session",
        "/sessions": "List recent sessions",
        "/continue": "Continue last or specified session",
        "/session": "Show current session info",
    }

    # Subcommands for specific commands
    SUBCOMMANDS = {
        "/model": ["anthropic/", "openai/", "azure_openai/", "azure/"],
        "/revert": ["last", "all"],
        "/auth": ["login", "logout", "status"],
    }

    def __init__(self, get_profiles_fn=None):
        """Initialize completer with optional dynamic profile loader."""
        self.get_profiles_fn = get_profiles_fn or list_profiles

    def get_completions(self, document: Document, complete_event):
        """Generate completions for the current input."""
        text = document.text_before_cursor

        # Only complete if starts with /
        if not text.startswith("/"):
            return

        # Check if we're completing a subcommand
        parts = text.split()
        if len(parts) >= 1:
            cmd = parts[0]

            # Completing the command itself
            if len(parts) == 1 and not text.endswith(" "):
                for command, description in self.COMMANDS.items():
                    if command.startswith(text):
                        yield Completion(
                            command,
                            start_position=-len(text),
                            display=command,
                            display_meta=description,
                        )

            # Completing subcommands
            elif cmd in self.SUBCOMMANDS:
                subcommand_text = parts[1] if len(parts) > 1 else ""
                for sub in self.SUBCOMMANDS[cmd]:
                    if sub.startswith(subcommand_text):
                        yield Completion(
                            sub,
                            start_position=-len(subcommand_text),
                            display=sub,
                        )

            # Special handling for /profile - show available profiles
            elif cmd == "/profile":
                profile_text = parts[1] if len(parts) > 1 else ""
                try:
                    profiles = self.get_profiles_fn()
                    for profile in profiles:
                        if profile.startswith(profile_text):
                            yield Completion(
                                profile,
                                start_position=-len(profile_text),
                                display=profile,
                            )
                except Exception:
                    pass


class ConsoleApp:
    """Enhanced console application with session management.

    Provides the main interactive loop with:
    - Brand logo display on startup
    - Session create/continue/list commands
    - Compact tool output via Display
    - Slash command handling
    """

    def __init__(
        self,
        config: Config,
        agent_loop: AgentLoop,
        console: Console | None = None,
        continue_session: bool = False,
        session_id: str | None = None,
    ):
        """Initialize ConsoleApp.

        Args:
            config: Application configuration
            agent_loop: The agent loop instance
            console: Rich console (shares with agent_loop if not provided)
            continue_session: Whether to continue the last session
            session_id: Specific session ID to continue
        """
        self.config = config
        self.agent = agent_loop
        self.console = console or agent_loop.console
        self.display = Display(self.console)
        self.current_session: SessionInfo | None = None
        self._continue_session = continue_session
        self._session_id = session_id

        # Inject display into agent loop
        self.agent.display = self.display

    def _get_project_id(self) -> str:
        """Get a project ID based on the working directory hash.

        Returns:
            A stable project ID derived from the target directory path.
        """
        dir_str = str(self.config.target_directory)
        return hashlib.sha256(dir_str.encode()).hexdigest()[:16]

    async def _new_session(self) -> SessionInfo:
        """Create a new session.

        Returns:
            The newly created SessionInfo
        """
        project_id = self._get_project_id()
        session = await Session.create(
            project_id=project_id,
            directory=str(self.config.target_directory),
        )
        self.current_session = session
        self.agent.session_id = session.id
        return session

    async def _continue_last_session(self, session_id: str | None = None) -> SessionInfo | None:
        """Continue the last session or a specific session.

        Args:
            session_id: Optional specific session ID to continue

        Returns:
            The session to continue, or None if not found
        """
        project_id = self._get_project_id()

        if session_id:
            session = await Session.get(project_id, session_id)
            if session:
                self.current_session = session
                self.agent.session_id = session.id
                return session
            self.display.print_error(f"Session not found: {session_id}")
            return None

        # Find the most recent session
        sessions: list[SessionInfo] = []
        async for s in Session.list_sessions(project_id):
            sessions.append(s)
            if len(sessions) >= 1:
                break

        if sessions:
            session = sessions[0]
            self.current_session = session
            self.agent.session_id = session.id
            return session

        return None

    async def _list_sessions(self) -> list[SessionInfo]:
        """List recent sessions.

        Returns:
            List of recent SessionInfo objects
        """
        project_id = self._get_project_id()
        sessions: list[SessionInfo] = []
        async for s in Session.list_sessions(project_id):
            sessions.append(s)
            if len(sessions) >= 10:
                break
        return sessions

    async def _handle_session_command(self, user_input: str) -> tuple[bool, str]:
        """Handle session-related slash commands.

        Args:
            user_input: The user's input string

        Returns:
            Tuple of (was_handled, response_text)
        """
        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if command == "/new":
            session = await self._new_session()
            self.agent.messages.clear()
            self.agent.doom_detector.reset()
            self.agent._project_instructions = None
            self.console.print(f"[green]New session created:[/green] {session.slug}")
            return True, f"New session: {session.slug}"

        if command == "/sessions":
            sessions = await self._list_sessions()
            current_id = self.current_session.id if self.current_session else ""
            session_dicts = [
                {
                    "id": s.id,
                    "title": s.title,
                    "slug": s.slug,
                    "time": s.time,
                }
                for s in sessions
            ]
            self.display.print_sessions_list(session_dicts, current_id)
            return True, f"Listed {len(sessions)} sessions"

        if command == "/continue":
            sid = args if args else None
            session = await self._continue_last_session(sid)
            if session:
                self.console.print(f"[green]Continuing session:[/green] {session.title} ({session.slug})")
                return True, f"Continuing: {session.slug}"
            else:
                self.console.print("[dim]No previous session found. Starting new session.[/dim]")
                await self._new_session()
                return True, "No previous session, created new"

        if command == "/session":
            if self.current_session:
                self.display.print_session_info(
                    session_id=self.current_session.id,
                    title=self.current_session.title,
                    slug=self.current_session.slug,
                )
            else:
                self.console.print("[dim]No active session.[/dim]")
            return True, "Session info displayed"

        if command == "/auth":
            from ..auth.commands import (
                auth_login_interactive,
                auth_logout_interactive,
                auth_status,
            )

            action = args.lower() if args else "login"

            if action == "login":
                success = await auth_login_interactive(self.console)
                if success:
                    # Hot-reload config so current session picks up new creds
                    try:
                        new_llm = type(self.config.llm).from_env()
                        self.config.llm = new_llm
                        self.console.print("[dim]Config reloaded with new credentials.[/dim]")
                    except Exception:
                        pass
                return True, "Auth login"
            elif action == "logout":
                await auth_logout_interactive(self.console)
                return True, "Auth logout"
            elif action == "status":
                await auth_status(self.console)
                return True, "Auth status"
            else:
                self.console.print("[dim]Usage: /auth login|logout|status[/dim]")
                return True, "Auth usage"

        return False, ""

    async def _handle_input(self, user_input: str) -> bool:
        """Handle user input, including slash commands.

        Args:
            user_input: The user's input string

        Returns:
            True if the app should continue, False to exit
        """
        stripped = user_input.strip()
        if not stripped:
            return True

        if stripped.lower() in ("exit", "quit"):
            # Finalize behavioral signals before exit
            summary = self.agent.finalize_behavior()
            if summary:
                self.console.print(
                    f"[dim]Behavioral data saved to data/behavior/sessions/{self.agent.session_id}/[/dim]"
                )
            self.console.print("[dim]Goodbye![/dim]")
            return False

        # Try session commands first
        if stripped.startswith("/"):
            handled, _ = await self._handle_session_command(stripped)
            if handled:
                return True

            # Try agent's slash commands
            handled, _ = await self.agent.handle_slash_command(stripped)
            if handled:
                return True

            # Unknown command
            self.console.print(f"[dim]Unknown command: {stripped.split()[0]}[/dim]")
            return True

        # Regular message - send to agent
        await self.agent.run(stripped)

        # Update session title if it's still default
        if self.current_session and Session.is_default_title(self.current_session.title):
            # Use first user message as title (truncated)
            new_title = stripped[:50] + ("..." if len(stripped) > 50 else "")
            try:
                project_id = self._get_project_id()
                self.current_session = await Session.update(
                    project_id,
                    self.current_session.id,
                    lambda data: data.update({"title": new_title}),
                )
            except Exception:
                pass  # Non-critical, ignore

        return True

    async def run(self) -> None:
        """Run the main interactive console loop."""
        # Load project instructions
        project_instructions = await self.agent._load_project_instructions()

        # Initialize or continue session
        if self._continue_session or self._session_id:
            session = await self._continue_last_session(self._session_id)
            if not session:
                session = await self._new_session()
        else:
            session = await self._new_session()

        # Display welcome
        profile = get_current_profile()
        profile_name = profile.basic.name if profile else ""
        model_display = self.config.get_model_display()

        self.display.print_welcome(
            directory=str(self.config.target_directory),
            model=model_display,
            profile=profile_name,
            session_id=session.slug if session else "",
        )

        if project_instructions:
            self.console.print("[dim]  Loaded project instructions from AGENTS.md[/dim]\n")

        # Help hint
        self.console.print("[dim]  Type /help for commands, /new for new session, exit to quit[/dim]\n")

        # Setup prompt_toolkit session with autocomplete and Unicode support
        pt_style = PTStyle.from_dict(
            {
                "prompt": "ansigreen bold",
                "completion-menu.completion": "bg:#333333 #ffffff",
                "completion-menu.completion.current": "bg:#00aa00 #ffffff",
                "completion-menu.meta.completion": "bg:#444444 #aaaaaa",
                "completion-menu.meta.completion.current": "bg:#00aa00 #ffffff",
            }
        )
        completer = SlashCommandCompleter(get_profiles_fn=list_profiles)
        prompt_session: PromptSession[str] = PromptSession(
            style=pt_style,
            completer=completer,
            complete_while_typing=True,
        )

        while True:
            try:
                user_input = await prompt_session.prompt_async("\n> ")
                should_continue = await self._handle_input(user_input)
                if not should_continue:
                    break
            except KeyboardInterrupt:
                self.console.print("\n[dim]Use 'exit' to quit[/dim]")
            except EOFError:
                # Finalize behavioral signals before exit
                summary = self.agent.finalize_behavior()
                if summary:
                    self.console.print(
                        f"\n[dim]Behavioral data saved to data/behavior/sessions/{self.agent.session_id}/[/dim]"
                    )
                self.console.print("[dim]Goodbye![/dim]")
                break
            except Exception as e:
                self.display.print_error(str(e))
