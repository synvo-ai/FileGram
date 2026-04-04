"""CLI entry point for CodeAgent."""

import argparse
import asyncio
import sys

from rich.console import Console

from .agent.loop import AgentLoop
from .config import Config
from .console import ConsoleApp
from .profile import list_profiles, load_profile, set_current_profile


def _is_auth_command() -> bool:
    """Check if the CLI invocation is an auth subcommand."""
    # Look for 'auth' as the first positional argument (skip flags)
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            continue
        return arg == "auth"
    return False


def _handle_auth_command() -> int:
    """Handle `filegram auth login|logout|status` commands.

    These run before Config.from_env() since credentials may not exist yet.
    """
    parser = argparse.ArgumentParser(
        prog="filegram auth",
        description="Manage authentication credentials",
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="login",
        choices=["login", "logout", "status"],
        help="Auth action (default: login)",
    )

    # Parse only the args after 'auth'
    auth_idx = sys.argv.index("auth")
    args = parser.parse_args(sys.argv[auth_idx + 1 :])

    console = Console()

    from .auth.commands import (
        auth_login_interactive,
        auth_logout_interactive,
        auth_status,
    )

    async def _run() -> int:
        if args.action == "login":
            success = await auth_login_interactive(console)
            return 0 if success else 1
        elif args.action == "logout":
            success = await auth_logout_interactive(console)
            return 0 if success else 1
        elif args.action == "status":
            await auth_status(console)
            return 0
        return 1

    return asyncio.run(_run())


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="filegramengine",
        description="A Python code agent powered by LLM",
    )

    parser.add_argument(
        "-d",
        "--target-dir",
        type=str,
        default=None,
        help="Target directory for operations (default: current directory)",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model to use in format provider/model (e.g., anthropic/claude-sonnet-4-20250514)",
    )

    parser.add_argument(
        "-1",
        "--one-shot",
        action="store_true",
        help="Run single task and exit (no interactive follow-up)",
    )

    parser.add_argument(
        "-p",
        "--profile",
        type=str,
        default=None,
        help="Agent profile to load (e.g., p1_methodical, p3_efficient_executor)",
    )

    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available agent profiles and exit",
    )

    parser.add_argument(
        "-c",
        "--continue",
        action="store_true",
        dest="continue_session",
        help="Continue the last session",
    )

    parser.add_argument(
        "-s",
        "--session",
        type=str,
        default=None,
        dest="session_id",
        help="Session ID to continue",
    )

    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="Disable sandbox mode (allow access to files outside target directory)",
    )

    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Autonomous mode: skip all permission checks, auto-continue on confirmation requests, never ask user",
    )

    parser.add_argument(
        "task",
        nargs="?",
        type=str,
        default=None,
        help="Initial task to execute (will enter interactive mode after unless --one-shot)",
    )

    return parser.parse_args()


async def run_agent(
    config: Config,
    task: str | None,
    one_shot: bool = False,
    continue_session: bool = False,
    session_id: str | None = None,
    autonomous: bool = False,
) -> int:
    """Run the agent with given configuration."""
    console = Console()

    try:
        agent = AgentLoop(config=config, console=console, autonomous_mode=autonomous)

        # If a task is provided, run it first
        if task is not None:
            await agent.run(task)

        # In one-shot mode, finalize behavioral signals before exiting
        if one_shot:
            summary = agent.finalize_behavior()
            if summary:
                console.print(f"[dim]Behavioral data saved to data/behavior/sessions/{agent.session_id}/[/dim]")
            return 0

        # Enter interactive mode after initial task
        if not one_shot:
            # Use ConsoleApp for enhanced interactive experience
            app = ConsoleApp(
                config=config,
                agent_loop=agent,
                console=console,
                continue_session=continue_session,
                session_id=session_id,
            )
            await app.run()

        return 0

    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted[/dim]")
        return 130
    except Exception as e:
        from rich.markup import escape

        console.print(f"[red]Error: {escape(str(e))}[/red]")
        return 1


def main() -> int:
    """Main entry point."""
    # Handle auth subcommand early (before Config.from_env)
    if _is_auth_command():
        return _handle_auth_command()

    args = parse_args()
    console = Console()

    # Handle --list-profiles
    if args.list_profiles:
        profiles = list_profiles()
        if profiles:
            console.print("[bold]Available profiles:[/bold]")
            for p in profiles:
                console.print(f"  - {p}")
        else:
            console.print("[dim]No profiles found.[/dim]")
        return 0

    try:
        config = Config.from_env(
            target_directory=args.target_dir,
            sandbox_enabled=not args.no_sandbox,
        )
    except ValueError as e:
        from rich.markup import escape

        console.print(f"[red]Configuration error: {escape(str(e))}[/red]")
        console.print("[dim]Set AZURE_OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY.[/dim]")
        console.print("[dim]Or run: filegram auth login[/dim]")
        return 1

    # Show sandbox status
    if args.target_dir:
        sandbox_status = (
            "[green]sandbox enabled[/green]" if config.sandbox_enabled else "[yellow]sandbox disabled[/yellow]"
        )
        console.print(f"[dim]Target directory: {config.target_directory} ({sandbox_status})[/dim]")

    # Apply model override if specified
    if args.model:
        try:
            config.apply_model_override(args.model)
            console.print(f"[dim]Using model: {config.get_model_display()}[/dim]")
        except ValueError as e:
            from rich.markup import escape

            console.print(f"[red]Model error: {escape(str(e))}[/red]")
            return 1

    # Load profile if specified
    profile = None
    if args.profile:
        try:
            profile = load_profile(args.profile)
            set_current_profile(profile)
            console.print(f"[green]Loaded profile: {profile.basic.name}[/green]")
            if profile.greeting:
                console.print(f"\n[italic]{profile.greeting}[/italic]\n")
        except FileNotFoundError:
            console.print(f"[red]Profile not found: {args.profile}[/red]")
            console.print(f"[dim]Available profiles: {', '.join(list_profiles())}[/dim]")
            return 1

    return asyncio.run(
        run_agent(
            config,
            args.task,
            one_shot=args.one_shot,
            continue_session=args.continue_session,
            session_id=args.session_id,
            autonomous=args.autonomous,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
