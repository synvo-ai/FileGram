"""Main agent loop for CodeAgent with full feature support."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from ..behavior import BehaviorCollector
from ..compaction import AutoCompactor, Compactor
from ..config import Config, LLMProvider
from ..context import TokenCounter
from ..instruction import InstructionLoader
from ..llm.client import AzureOpenAIClient
from ..models.message import (
    Message,
    Part,
    Role,
    TokenUsage,
    ToolUsePart,
)

# Import modules
from ..permission import Permission, PermissionAction, PermissionError
from ..profile import (
    get_current_profile,
    list_profiles,
    load_profile,
    set_current_profile,
)
from ..prompts import load_session_prompt
from ..session.revert import SessionRevert
from ..snapshot import SnapshotManager
from ..tools.base import ToolContext
from ..tools.plan import get_plan_state, is_operation_allowed_in_plan_mode
from ..tools.registry import ToolRegistry, create_default_registry
from ..tools.task import TaskTool
from .types import AgentInfo

# ============== Constants ==============

DOOM_LOOP_THRESHOLD = 3  # Detect after 3 identical tool calls

RETRY_INITIAL_DELAY = 2.0  # seconds
RETRY_BACKOFF_FACTOR = 2
RETRY_MAX_ATTEMPTS = 3
RETRY_MAX_DELAY = 30.0  # seconds


# ============== System Prompt ==============

# Template for adding dynamic context to system prompt
SYSTEM_PROMPT_CONTEXT = """
Current working directory: {target_directory}

{project_instructions}
"""


def get_system_prompt_base(provider: LLMProvider) -> str:
    """Get the base system prompt for a specific provider.

    Tries to load a provider-specific prompt first, falls back to generic.
    """
    provider_map = {
        LLMProvider.ANTHROPIC: "anthropic",
        LLMProvider.OPENAI: "openai",
        LLMProvider.AZURE_OPENAI: "openai",  # Azure uses OpenAI prompts
    }
    provider_name = provider_map.get(provider)
    return load_session_prompt("system", provider_name)


# ============== Doom Loop Detection ==============


@dataclass
class ToolCallRecord:
    """Record of a tool call for doom loop detection."""

    tool_name: str
    input_hash: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


class DoomLoopDetector:
    """
    Detects when the agent is stuck in a loop making identical tool calls.

    A doom loop is detected when the same tool is called with identical
    arguments DOOM_LOOP_THRESHOLD times in a row.
    """

    def __init__(self, threshold: int = DOOM_LOOP_THRESHOLD):
        self.threshold = threshold
        self.recent_calls: list[ToolCallRecord] = []

    def _hash_input(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Create a hash of tool call for comparison."""
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"

    def record(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Record a tool call."""
        input_hash = self._hash_input(tool_name, arguments)
        self.recent_calls.append(
            ToolCallRecord(
                tool_name=tool_name,
                input_hash=input_hash,
            )
        )

        # Keep only recent calls
        if len(self.recent_calls) > self.threshold * 2:
            self.recent_calls = self.recent_calls[-self.threshold * 2 :]

    def is_doom_loop(self) -> bool:
        """Check if we're in a doom loop."""
        if len(self.recent_calls) < self.threshold:
            return False

        # Check if last N calls are identical
        last_calls = self.recent_calls[-self.threshold :]
        first_hash = last_calls[0].input_hash

        return all(call.input_hash == first_hash for call in last_calls)

    def get_repeated_call(self) -> tuple[str, str] | None:
        """Get the repeated tool name and input hash if in doom loop."""
        if not self.is_doom_loop():
            return None
        last_call = self.recent_calls[-1]
        return (last_call.tool_name, last_call.input_hash)

    def reset(self) -> None:
        """Reset the detector."""
        self.recent_calls.clear()


# ============== Retry Handler ==============


class RetryHandler:
    """
    Handles retry logic for API calls with exponential backoff.

    Retryable errors:
    - Rate limits (429)
    - Server errors (5xx)
    - Overloaded errors
    """

    def __init__(
        self,
        max_attempts: int = RETRY_MAX_ATTEMPTS,
        initial_delay: float = RETRY_INITIAL_DELAY,
        backoff_factor: float = RETRY_BACKOFF_FACTOR,
        max_delay: float = RETRY_MAX_DELAY,
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay

    def get_delay(self, attempt: int, error: Exception | None = None) -> float:
        """Calculate delay for a given attempt."""
        # Check for Retry-After header in error
        if error and hasattr(error, "response"):
            response = error.response
            if hasattr(response, "headers"):
                retry_after = response.headers.get("retry-after")
                if retry_after:
                    try:
                        return float(retry_after)
                    except ValueError:
                        pass

        # Exponential backoff
        delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
        return min(delay, self.max_delay)

    def is_retryable(self, error: Exception) -> bool:
        """Check if an error is retryable."""
        error_str = str(error).lower()

        # Rate limit errors
        if "rate" in error_str or "429" in error_str:
            return True

        # Server errors
        if "500" in error_str or "502" in error_str or "503" in error_str:
            return True

        # Overloaded
        if "overloaded" in error_str:
            return True

        # OpenAI specific
        if "too many requests" in error_str:
            return True

        return False

    async def execute_with_retry(
        self,
        func,
        *args,
        on_retry: callable | None = None,
        **kwargs,
    ):
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute (sync or async)
            on_retry: Optional callback called before each retry
            *args, **kwargs: Arguments to pass to func

        Returns:
            Result of func

        Raises:
            Last exception if all retries fail
        """
        last_error = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                # Handle if func is a coroutine
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except Exception as e:
                last_error = e

                if not self.is_retryable(e) or attempt >= self.max_attempts:
                    raise

                delay = self.get_delay(attempt, e)

                if on_retry:
                    on_retry(attempt, delay, e)

                await asyncio.sleep(delay)

        raise last_error


# ============== Agent Loop ==============


class AgentLoop:
    """Main agent loop that coordinates LLM and tool execution."""

    def __init__(
        self,
        config: Config,
        registry: ToolRegistry | None = None,
        console: Console | None = None,
        permission: Permission | None = None,
        enable_compaction: bool = True,
        max_context_tokens: int = 128000,
        enable_doom_loop_detection: bool = True,
        enable_retry: bool = True,
        enable_behavior_collection: bool = True,
    ):
        self.config = config
        self.client = self._create_client(config)
        self.console = console or Console()
        self.session_id = str(uuid.uuid4())
        self.messages: list[Message] = []

        # Display module (set by ConsoleApp, or lazy-initialized)
        if TYPE_CHECKING:
            from ..console.display import Display

        self.display: Display | None = None

        # Initialize permission system with ask callback
        self.permission = permission or Permission.create_default(ask_callback=self._ask_permission)

        # Initialize instruction loader
        self.instruction_loader = InstructionLoader.create(
            target_directory=config.target_directory,
        )

        # Initialize token counter
        self.token_counter = TokenCounter()

        # Initialize compaction
        self.enable_compaction = enable_compaction
        if enable_compaction:
            self.compactor = Compactor(self.client)
            self.auto_compactor = AutoCompactor(
                compactor=self.compactor,
                token_counter=self.token_counter,
                max_context_tokens=max_context_tokens,
            )
        else:
            self.compactor = None
            self.auto_compactor = None

        # Initialize tool registry
        self.registry = registry or create_default_registry(include_task=True)

        # Setup task tool executor
        self._setup_task_tool()

        # Initialize behavioral signal collector
        self.enable_behavior_collection = enable_behavior_collection
        if enable_behavior_collection:
            from ..profile import get_current_profile

            profile = get_current_profile()
            profile_id = profile.id if profile else "default"
            provider_name = config.llm.provider.value if config.llm.provider else "unknown"
            model_name = config.get_model_display()

            self.behavior_collector: BehaviorCollector | None = BehaviorCollector(
                session_id=self.session_id,
                profile_id=profile_id,
                model_provider=provider_name,
                model_name=model_name,
                target_directory=config.target_directory,
                enabled=True,
            )
        else:
            self.behavior_collector = None

        self.tool_context = ToolContext(
            session_id=self.session_id,
            target_directory=config.target_directory,
            max_output_chars=config.max_output_chars,
            default_timeout=config.default_timeout,
            sandbox_enabled=config.sandbox_enabled,
            behavior_collector=self.behavior_collector,
        )

        # Doom loop detection
        self.enable_doom_loop_detection = enable_doom_loop_detection
        self.doom_detector = DoomLoopDetector()

        # Retry handler
        self.enable_retry = enable_retry
        self.retry_handler = RetryHandler()

        # Project instructions cache
        self._project_instructions: str | None = None

        # Token usage tracking
        self.total_usage = TokenUsage()

        # Snapshot and revert system
        self.snapshot_manager = SnapshotManager(
            project_dir=config.target_directory,
            enabled=True,
        )
        self.session_revert = SessionRevert(self.snapshot_manager)
        self._snapshot_initialized = False

        # LLM timing tracking for behavior collection
        self._llm_call_start_time: float = 0

    @staticmethod
    def _create_client(config: Config):
        """Create LLM client based on provider configuration."""
        provider = config.llm.provider

        if provider == LLMProvider.ANTHROPIC:
            from ..llm.anthropic_client import AnthropicClient

            if config.llm.anthropic is None:
                raise ValueError("Anthropic configuration not available. Set ANTHROPIC_API_KEY.")
            return AnthropicClient(config.llm.anthropic)
        elif provider == LLMProvider.OPENAI:
            from ..llm.openai_client import OpenAIClient

            if config.llm.openai is None:
                raise ValueError("OpenAI configuration not available. Set OPENAI_API_KEY.")
            return OpenAIClient(config.llm.openai)
        else:
            # Default: Azure OpenAI
            return AzureOpenAIClient(config.azure_openai)

    def _setup_task_tool(self) -> None:
        """Setup the task tool with sub-agent executor."""
        task_tool = self.registry.get("task")
        if isinstance(task_tool, TaskTool):
            task_tool.set_executor(self._execute_subagent)

    async def _execute_subagent(
        self,
        prompt: str,
        description: str,
        agent_info: AgentInfo,
        context: ToolContext,
    ) -> str:
        """Execute a sub-agent task."""
        self.console.print(f"\n[dim]Spawning sub-agent: {agent_info.name}[/dim]")
        self.console.print(f"[dim]Task: {description}[/dim]\n")

        # Create a new agent loop for the sub-agent
        sub_registry = create_default_registry(include_task=False)

        # Filter tools based on agent permissions
        if agent_info.tools_denied:
            for tool_name in agent_info.tools_denied:
                if tool_name in sub_registry._tools:
                    del sub_registry._tools[tool_name]

        sub_loop = AgentLoop(
            config=self.config,
            registry=sub_registry,
            console=self.console,
            permission=Permission.create_default(),
            enable_compaction=False,
            enable_doom_loop_detection=False,  # Sub-agents don't need doom detection
            enable_behavior_collection=False,  # Sub-agents share parent's collector
        )

        # Build sub-agent system prompt
        base_prompt = get_system_prompt_base(self.config.llm.provider)
        sub_system_prompt = base_prompt + SYSTEM_PROMPT_CONTEXT.format(
            target_directory=self.config.target_directory,
            project_instructions="",
        )

        if agent_info.prompt:
            sub_system_prompt = agent_info.prompt + "\n\n" + sub_system_prompt

        # Run sub-agent
        sub_loop.messages.append(Message.system(sub_system_prompt))
        result = await sub_loop.run(
            user_input=prompt,
            max_iterations=agent_info.max_steps or 20,
            show_welcome=False,
        )

        return result

    def _ask_permission(
        self,
        tool: str,
        target: str,
        metadata: dict,
    ) -> PermissionAction:
        """Callback for permission system to ask user."""
        self.console.print(
            Panel(
                f"Tool: [bold]{tool}[/bold]\nTarget: {target}\nMetadata: {metadata}",
                title="[yellow]Permission Required[/yellow]",
                border_style="yellow",
            )
        )

        if Confirm.ask("Allow this operation?"):
            if Confirm.ask("Always allow this?"):
                self.permission.add_session_approval(tool, target)
                return PermissionAction.ALLOW
            return PermissionAction.ALLOW
        return PermissionAction.DENY

    def _ask_doom_loop(self, tool_name: str) -> bool:
        """Ask user if they want to continue despite doom loop."""
        self.console.print(
            Panel(
                f"The agent has called [bold]{tool_name}[/bold] with identical "
                f"arguments {DOOM_LOOP_THRESHOLD} times in a row.\n\n"
                "This may indicate the agent is stuck in a loop.",
                title="[yellow]Doom Loop Detected[/yellow]",
                border_style="yellow",
            )
        )
        return Confirm.ask("Continue anyway?")

    async def _load_project_instructions(self) -> str:
        """Load and cache project instructions."""
        if self._project_instructions is None:
            self._project_instructions = await self.instruction_loader.get_system_prompt_addition()
        return self._project_instructions

    def _get_system_prompt(self, project_instructions: str = "") -> str:
        """Generate system prompt with context and profile.

        Uses provider-specific prompts when available.
        """
        # Get provider-specific base prompt
        base_prompt = get_system_prompt_base(self.config.llm.provider)

        prompt = base_prompt + SYSTEM_PROMPT_CONTEXT.format(
            target_directory=self.config.target_directory,
            project_instructions=project_instructions,
        )

        # Add profile if active
        profile = get_current_profile()
        if profile:
            prompt += "\n\n" + profile.to_system_prompt()

        return prompt

    def _get_display(self):
        """Get or lazy-initialize the Display module."""
        if self.display is None:
            from ..console.display import Display

            self.display = Display(self.console)
        return self.display

    def _get_available_models(self) -> list[tuple[str, str, bool, bool]]:
        """Get list of available models with their status.

        Returns:
            List of (model_spec, display_name, is_current, is_configured) tuples
        """
        models = []
        current = self.config.get_model_display()

        # Anthropic models
        anthropic_configured = self.config.llm.anthropic is not None
        anthropic_models = [
            ("anthropic/claude-opus-4-6", "Claude Opus 4.6"),
            ("anthropic/claude-sonnet-4-5-20250929", "Claude Sonnet 4.5"),
            ("anthropic/claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
            ("anthropic/claude-sonnet-4-20250514", "Claude Sonnet 4"),
            ("anthropic/claude-opus-4-20250514", "Claude Opus 4"),
            ("anthropic/claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
            ("anthropic/claude-3-5-haiku-20241022", "Claude 3.5 Haiku"),
        ]
        for spec, name in anthropic_models:
            is_current = current == spec
            models.append((spec, f"  {name}", is_current, anthropic_configured))

        # OpenAI models
        openai_configured = self.config.llm.openai is not None
        openai_models = [
            ("openai/gpt-4o", "GPT-4o"),
            ("openai/gpt-4o-mini", "GPT-4o Mini"),
            ("openai/gpt-4-turbo", "GPT-4 Turbo"),
            ("openai/o1", "O1"),
            ("openai/o1-mini", "O1 Mini"),
            ("openai/o3-mini", "O3 Mini"),
        ]
        for spec, name in openai_models:
            is_current = current == spec
            models.append((spec, f"  {name}", is_current, openai_configured))

        # Azure OpenAI models
        azure_configured = self.config.llm.azure_openai is not None
        if azure_configured:
            azure_models = [
                (
                    f"azure/{self.config.llm.azure_openai.deployment}",
                    f"Azure: {self.config.llm.azure_openai.deployment}",
                ),
            ]
            for spec, name in azure_models:
                is_current = current == spec
                models.append((spec, f"  {name}", is_current, True))

        return models

    async def _show_model_selector(self) -> str | None:
        """Show interactive model selection menu.

        Returns:
            Selected model spec or None if cancelled
        """
        from prompt_toolkit import PromptSession

        models = self._get_available_models()
        if not models:
            self.console.print("[red]No models available.[/red]")
            return None

        # Display available models
        current = self.config.get_model_display()
        self.console.print(f"\n[bold]Current model:[/bold] {current}\n")

        # Group by provider
        providers: dict[str, list[tuple[str, str, bool, bool]]] = {}
        for spec, name, is_current, is_configured in models:
            provider = spec.split("/")[0]
            if provider not in providers:
                providers[provider] = []
            providers[provider].append((spec, name, is_current, is_configured))

        # Display with numbers
        model_list: list[tuple[int, str, bool]] = []  # (idx, spec, is_configured)
        idx = 1
        for provider, provider_models in providers.items():
            provider_display = {
                "anthropic": "Anthropic (Claude)",
                "openai": "OpenAI",
                "azure": "Azure OpenAI",
            }.get(provider, provider.title())

            # Check if provider is configured
            is_provider_configured = any(cfg for _, _, _, cfg in provider_models)
            config_hint = "" if is_provider_configured else " [dim](not configured)[/dim]"
            self.console.print(f"[bold cyan]── {provider_display}{config_hint} ──[/bold cyan]")

            for spec, name, is_current, is_configured in provider_models:
                marker = " [green]✓[/green]" if is_current else ""
                if is_configured:
                    self.console.print(f"  [yellow]{idx}[/yellow]. {name.strip()}{marker}")
                else:
                    self.console.print(f"  [dim]{idx}. {name.strip()} (需要API Key)[/dim]")
                model_list.append((idx, spec, is_configured))
                idx += 1

        # Show API key hints
        unconfigured = []
        if not self.config.llm.anthropic:
            unconfigured.append("ANTHROPIC_API_KEY")
        if not self.config.llm.openai:
            unconfigured.append("OPENAI_API_KEY")
        if not self.config.llm.azure_openai:
            unconfigured.append("AZURE_OPENAI_API_KEY")

        if unconfigured:
            self.console.print(f"\n[dim]配置更多模型: 在 .env 中设置 {', '.join(unconfigured)}[/dim]")

        self.console.print(f"\n[dim]Enter number (1-{len(model_list)}) or 'q' to cancel:[/dim]")

        # Simple number input
        try:
            session: PromptSession[str] = PromptSession()
            choice = await session.prompt_async("Select: ")

            if choice.lower() in ("q", "quit", "cancel", ""):
                self.console.print("[dim]Cancelled[/dim]")
                return None

            try:
                num = int(choice)
                if 1 <= num <= len(model_list):
                    _, selected_spec, is_configured = model_list[num - 1]

                    if not is_configured:
                        provider = selected_spec.split("/")[0]
                        key_name = {
                            "anthropic": "ANTHROPIC_API_KEY",
                            "openai": "OPENAI_API_KEY",
                            "azure": "AZURE_OPENAI_API_KEY",
                        }.get(provider, f"{provider.upper()}_API_KEY")
                        self.console.print(f"[red]未配置 {key_name}，请在 .env 中设置后重试[/red]")
                        return None

                    self.config.apply_model_override(selected_spec)
                    self.client = self._create_client(self.config)
                    if self.enable_compaction:
                        self.compactor = Compactor(self.client)
                        self.auto_compactor = AutoCompactor(
                            compactor=self.compactor,
                            token_counter=self.token_counter,
                            max_context_tokens=128000,
                        )
                    model_display = self.config.get_model_display()
                    # Update behavior collector with new model
                    if self.behavior_collector:
                        provider_name = self.config.llm.provider.value if self.config.llm.provider else "unknown"
                        self.behavior_collector.update_model(provider_name, model_display)
                    self.console.print(f"[green]Switched to: {model_display}[/green]")
                    return model_display
                else:
                    self.console.print(f"[red]Invalid choice. Enter 1-{len(model_list)}[/red]")
                    return None
            except ValueError:
                self.console.print("[red]Invalid input. Enter a number.[/red]")
                return None
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[dim]Cancelled[/dim]")
            return None

    async def _show_profile_selector(self) -> str | None:
        """Show interactive profile selection menu.

        Returns:
            Selected profile name or None if cancelled
        """
        from prompt_toolkit import PromptSession

        profiles = list_profiles()
        current = get_current_profile()
        current_name = current.basic.name.lower() if current else None

        # Display current profile
        if current:
            self.console.print(f"\n[bold]Current profile:[/bold] {current.basic.name}")
        else:
            self.console.print("\n[bold]Current profile:[/bold] [dim]None (default)[/dim]")

        # Display options
        self.console.print("\n[bold cyan]── Available Profiles ──[/bold cyan]")
        self.console.print("  [yellow]0[/yellow]. Default (no profile)")

        profile_list: list[tuple[int, str]] = [(0, "none")]

        for idx, name in enumerate(profiles, 1):
            try:
                profile = load_profile(name)
                is_current = name.lower() == current_name if current_name else False
                marker = " [green]✓[/green]" if is_current else ""
                self.console.print(f"  [yellow]{idx}[/yellow]. {profile.basic.name} - {profile.basic.role}{marker}")
                profile_list.append((idx, name))
            except Exception:
                self.console.print(f"  [yellow]{idx}[/yellow]. {name}")
                profile_list.append((idx, name))

        self.console.print(f"\n[dim]Enter number (0-{len(profiles)}) or 'q' to cancel:[/dim]")

        try:
            session: PromptSession[str] = PromptSession()
            choice = await session.prompt_async("Select: ")

            if choice.lower() in ("q", "quit", "cancel", ""):
                self.console.print("[dim]Cancelled[/dim]")
                return None

            try:
                num = int(choice)
                if num == 0:
                    set_current_profile(None)
                    project_instructions = await self._load_project_instructions()
                    if self.messages and self.messages[0].role == Role.SYSTEM:
                        self.messages[0] = Message.system(self._get_system_prompt(project_instructions))
                    self.console.print("[green]Profile cleared. Using default.[/green]")
                    return "default"
                elif 1 <= num <= len(profiles):
                    selected_name = profile_list[num][1]
                    profile = load_profile(selected_name)
                    set_current_profile(profile)
                    # Update behavior collector with new profile
                    if self.behavior_collector:
                        self.behavior_collector.update_profile(profile.id)
                    self.console.print(f"[green]Switched to profile: {profile.basic.name}[/green]")
                    if profile.greeting:
                        self.console.print(f"\n[italic]{profile.greeting}[/italic]")
                    project_instructions = await self._load_project_instructions()
                    if self.messages and self.messages[0].role == Role.SYSTEM:
                        self.messages[0] = Message.system(self._get_system_prompt(project_instructions))
                    return profile.basic.name
                else:
                    self.console.print(f"[red]Invalid choice. Enter 0-{len(profiles)}[/red]")
                    return None
            except ValueError:
                self.console.print("[red]Invalid input. Enter a number.[/red]")
                return None
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[dim]Cancelled[/dim]")
            return None

    def _print_text(self, text: str) -> None:
        """Print text output from assistant."""
        self._get_display().print_text(text)

    def _print_reasoning(self, text: str) -> None:
        """Print reasoning output from assistant."""
        self._get_display().print_thinking(text)

    def _print_tool_use(self, tool_use: ToolUsePart) -> None:
        """Print tool use in compact OpenCode format."""
        display = self._get_display()
        display.print_tool_use(tool_use.name, tool_use.arguments)

    def _print_tool_result(self, name: str, output: str, is_error: bool) -> None:
        """Print tool execution result.

        In compact mode: errors shown inline, bash stdout shown,
        other successful results are silent (compact line already shown).
        """
        display = self._get_display()
        # Show errors
        display.print_tool_result(name, output, is_error)
        # Show bash stdout (replicates OpenCode's special bash handling)
        if not is_error and name == "bash":
            display.print_tool_result_bash(output)

    def _check_permission(self, tool_name: str, arguments: dict) -> None:
        """Check permission for a tool call."""
        if tool_name in ("read", "write", "edit"):
            target = arguments.get("file_path", "*")
        elif tool_name == "bash":
            target = arguments.get("command", "*")
        elif tool_name == "grep":
            target = arguments.get("pattern", "*")
        else:
            target = "*"

        self.permission.check(tool_name, target, arguments)

    async def _execute_tools(self, tool_uses: list[ToolUsePart]) -> list[Message]:
        """Execute tool calls and return result messages."""
        results: list[Message] = []

        for tool_use in tool_uses:
            self._print_tool_use(tool_use)

            # Record for doom loop detection
            if self.enable_doom_loop_detection:
                self.doom_detector.record(tool_use.name, tool_use.arguments)

                # Check for doom loop
                if self.doom_detector.is_doom_loop():
                    if not self._ask_doom_loop(tool_use.name):
                        results.append(
                            Message.tool_result(
                                tool_use_id=tool_use.tool_use_id,
                                content="User stopped execution due to detected doom loop.",
                                is_error=True,
                            )
                        )
                        continue
                    # User wants to continue, reset detector
                    self.doom_detector.reset()

            # Check plan mode restrictions
            plan_state = get_plan_state(self.session_id)
            if plan_state.is_plan_mode:
                # Get target for plan mode check
                if tool_use.name in ("read", "write", "edit"):
                    target = tool_use.arguments.get("file_path", "")
                elif tool_use.name == "bash":
                    target = tool_use.arguments.get("command", "")
                else:
                    target = ""

                allowed, reason = is_operation_allowed_in_plan_mode(self.session_id, tool_use.name, target)
                if not allowed:
                    self._print_tool_result(tool_use.name, f"[Plan Mode] {reason}", is_error=True)
                    results.append(
                        Message.tool_result(
                            tool_use_id=tool_use.tool_use_id,
                            content=f"[Plan Mode] {reason}",
                            is_error=True,
                        )
                    )
                    continue

            # Check permission
            try:
                self._check_permission(tool_use.name, tool_use.arguments)
            except PermissionError as e:
                self._print_tool_result(tool_use.name, str(e), is_error=True)
                results.append(
                    Message.tool_result(
                        tool_use_id=tool_use.tool_use_id,
                        content=str(e),
                        is_error=True,
                    )
                )
                continue

            # Create revert point before file-modifying tools
            modifying_tools = {"write", "edit", "multiedit", "apply_patch", "bash"}
            if tool_use.name in modifying_tools:
                await self.session_revert.create_revert_point(
                    message_id=tool_use.tool_use_id,
                    description=f"{tool_use.name}: {str(tool_use.arguments)[:50]}...",
                )

            # Record tool execution timing for behavior tracking
            tool_start_time = time.time() * 1000

            result = await self.registry.execute(
                name=tool_use.name,
                tool_use_id=tool_use.tool_use_id,
                arguments=tool_use.arguments,
                context=self.tool_context,
            )

            tool_end_time = time.time() * 1000

            # Record tool call for behavior tracking
            if self.behavior_collector:
                self.behavior_collector.record_tool_call(
                    tool_name=tool_use.name,
                    tool_parameters=tool_use.arguments,
                    execution_time_ms=int(tool_end_time - tool_start_time),
                    success=not result.is_error,
                    error_type="tool_error" if result.is_error else None,
                    error_message=result.output[:200] if result.is_error else None,
                )

                # Record error/recovery events
                if result.is_error:
                    file_path = tool_use.arguments.get("file_path") or tool_use.arguments.get("path")
                    self.behavior_collector.record_error_encounter(
                        error_type="tool_error",
                        context=result.output[:200],
                        severity="medium",
                        tool_name=tool_use.name,
                        file_path=file_path,
                    )
                else:
                    self.behavior_collector.check_error_recovery(
                        tool_name=tool_use.name,
                        success=True,
                    )

            # Handle plan mode state changes
            if tool_use.name == "plan_enter" and not result.is_error:
                self.console.print("\n[bold cyan]📝 Entered PLAN MODE[/bold cyan]\n")
            elif tool_use.name == "plan_exit" and not result.is_error:
                self.console.print("\n[bold green]✓ Exited PLAN MODE - Ready to implement[/bold green]\n")

            self._print_tool_result(result.name, result.output, result.is_error)

            results.append(
                Message.tool_result(
                    tool_use_id=result.tool_use_id,
                    content=result.to_content(),
                    is_error=result.is_error,
                )
            )

        return results

    async def _maybe_compact(self) -> None:
        """Perform compaction if needed."""
        if not self.enable_compaction or not self.auto_compactor:
            return

        messages_before = len(self.messages)
        new_messages, compact_result, prune_result = await self.auto_compactor.process(
            self.messages,
        )

        if prune_result and prune_result.pruned_count > 0:
            self.console.print(
                f"\n[dim]Pruned {prune_result.pruned_count} old tool outputs "
                f"(saved ~{prune_result.tokens_saved} tokens)[/dim]\n"
            )

        if compact_result:
            self.console.print(f"\n[dim]Context compacted: saved {compact_result.tokens_saved} tokens[/dim]\n")
            self.messages = new_messages

            # Record compaction for behavior tracking
            if self.behavior_collector:
                self.behavior_collector.record_compaction(
                    reason="context_overflow",
                    messages_before=messages_before,
                    messages_after=len(new_messages),
                    tokens_saved=compact_result.tokens_saved,
                )

    def _on_retry(self, attempt: int, delay: float, error: Exception) -> None:
        """Callback for retry attempts."""
        self.console.print(f"[yellow]API error (attempt {attempt}/{RETRY_MAX_ATTEMPTS}): {error}[/yellow]")
        self.console.print(f"[dim]Retrying in {delay:.1f}s...[/dim]")

    async def _call_llm(
        self,
        tool_definitions: list,
        on_text: callable,
        on_reasoning: callable | None = None,
    ) -> tuple[list[Part], str]:
        """Call LLM with optional retry."""
        if self.enable_retry:
            return await self.retry_handler.execute_with_retry(
                self._call_llm_inner,
                tool_definitions,
                on_text,
                on_reasoning,
                on_retry=self._on_retry,
            )
        else:
            return await self._call_llm_inner(tool_definitions, on_text, on_reasoning)

    async def _call_llm_inner(
        self,
        tool_definitions: list,
        on_text: callable,
        on_reasoning: callable | None = None,
    ) -> tuple[list[Part], str]:
        """Inner LLM call without retry. Handles both sync and async clients."""
        result = self.client.chat_completion_stream(
            messages=self.messages,
            tools=tool_definitions,
            on_text=on_text,
            on_reasoning=on_reasoning,
        )
        # Handle async clients (e.g., Anthropic)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def handle_slash_command(self, user_input: str) -> tuple[bool, str]:
        """Handle slash commands from CLI or frontend.

        Returns:
            Tuple of (was_handled, response_text)
        """
        if not user_input.startswith("/"):
            return False, ""

        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if command == "/stats":
            message_dicts = [msg.to_dict() for msg in self.messages]
            current_tokens = self.token_counter.count_messages(message_dicts)
            usable = self.auto_compactor.usable_tokens if self.auto_compactor else 120000
            utilization = current_tokens / usable if usable > 0 else 0

            response = (
                f"Messages: {len(self.messages)}\n"
                f"Tokens: ~{current_tokens:,} / {usable:,}\n"
                f"Utilization: {utilization:.1%}\n"
                f"Model: {self.config.get_model_display()}\n"
                f"Total usage: {self.total_usage.total:,} tokens"
            )
            self.console.print(Panel(response, title="Context Statistics", border_style="cyan"))
            return True, response

        if command == "/reset":
            self.messages.clear()
            self.doom_detector.reset()
            self._project_instructions = None
            await self._load_project_instructions()
            self.console.print("[dim]Conversation reset.[/dim]")
            return True, "Conversation reset."

        if command == "/model":
            if not args:
                # Show interactive model selection
                selected = await self._show_model_selector()
                if selected:
                    return True, f"Switched to: {selected}"
                return True, "Model selection cancelled"
            else:
                try:
                    self.config.apply_model_override(args)
                    self.client = self._create_client(self.config)
                    # Update compactor client if needed
                    if self.enable_compaction:
                        self.compactor = Compactor(self.client)
                        self.auto_compactor = AutoCompactor(
                            compactor=self.compactor,
                            token_counter=self.token_counter,
                            max_context_tokens=128000,
                        )
                    model_display = self.config.get_model_display()
                    self.console.print(f"[green]Switched to: {model_display}[/green]")
                    return True, f"Switched to: {model_display}"
                except ValueError as e:
                    self.console.print(f"[red]{e}[/red]")
                    return True, str(e)

        if command == "/profile":
            project_instructions = await self._load_project_instructions()
            if not args:
                # Show interactive profile selection
                selected = await self._show_profile_selector()
                if selected:
                    return True, f"Switched to profile: {selected}"
                return True, "Profile selection cancelled"
            else:
                profile_name = args
                try:
                    profile = load_profile(profile_name)
                    set_current_profile(profile)
                    self.console.print(f"[green]Switched to profile: {profile.basic.name}[/green]")
                    if profile.greeting:
                        self.console.print(f"\n[italic]{profile.greeting}[/italic]")
                    # Reset system message with new profile
                    if self.messages and self.messages[0].role == Role.SYSTEM:
                        self.messages[0] = Message.system(self._get_system_prompt(project_instructions))
                    return True, f"Switched to profile: {profile.basic.name}"
                except FileNotFoundError:
                    self.console.print(f"[red]Profile not found: {profile_name}[/red]")
                    self.console.print(f"[dim]Available: {', '.join(list_profiles())}[/dim]")
                    return True, f"Profile not found: {profile_name}"

        if command == "/diff":
            # Show files changed in this session
            files = await self.session_revert.get_diff()
            if files:
                file_list = "\n".join(f"  - {f}" for f in files[:20])
                if len(files) > 20:
                    file_list += f"\n  ... and {len(files) - 20} more files"
                response = f"Changed files ({len(files)}):\n{file_list}"
            else:
                response = "No files changed in this session."
            self.console.print(Panel(response, title="Session Diff", border_style="cyan"))
            return True, response

        if command == "/revert":
            if not args:
                # Show available revert points
                points = self.session_revert.get_revert_points()
                if points:
                    point_list = "\n".join(
                        f"  {i + 1}. {p.message_id[:12]}... - {p.description or 'No description'}"
                        for i, p in enumerate(points[-10:])
                    )
                    response = (
                        f"Available revert points:\n{point_list}\n\n"
                        "Usage: /revert last | /revert all | /revert <message_id>"
                    )
                else:
                    response = "No revert points available yet."
                self.console.print(Panel(response, title="Revert Points", border_style="yellow"))
                return True, response
            elif args == "last":
                success = await self.session_revert.revert_last()
                if success:
                    self.console.print("[green]Reverted to previous state.[/green]")
                    return True, "Reverted to previous state."
                else:
                    self.console.print("[red]Failed to revert.[/red]")
                    return True, "Failed to revert."
            elif args == "all":
                success = await self.session_revert.revert_all()
                if success:
                    self.console.print("[green]Reverted all changes to initial state.[/green]")
                    return True, "Reverted all changes."
                else:
                    self.console.print("[red]Failed to revert all changes.[/red]")
                    return True, "Failed to revert all changes."
            else:
                # Try to revert to specific message
                success = await self.session_revert.revert_to_message(args)
                if success:
                    self.console.print(f"[green]Reverted to message: {args}[/green]")
                    return True, f"Reverted to message: {args}"
                else:
                    self.console.print(f"[red]Failed to revert to message: {args}[/red]")
                    return True, f"Failed to revert to message: {args}"

        if command == "/help":
            help_text = (
                "Available commands:\n"
                "  /stats              - Show context statistics\n"
                "  /reset              - Reset conversation\n"
                "  /model [p/m]        - Show or switch model (provider/model)\n"
                "  /profile [name]     - Show or switch agent profile\n"
                "  /diff               - Show files changed in this session\n"
                "  /revert [target]    - Revert changes (last|all|<message_id>)\n"
                "  /new                - Create a new session\n"
                "  /sessions           - List recent sessions\n"
                "  /continue [id]      - Continue last or specified session\n"
                "  /session            - Show current session info\n"
                "  /help               - Show this help"
            )
            self.console.print(Panel(help_text, title="Commands", border_style="cyan"))
            return True, help_text

        # Unknown slash command - not handled
        return False, ""

    async def run(
        self,
        user_input: str,
        max_iterations: int = 50,
        show_welcome: bool = True,
    ) -> str:
        """
        Run the agent loop with user input.

        Args:
            user_input: The user's task or question
            max_iterations: Maximum number of LLM calls to prevent infinite loops
            show_welcome: Whether to show user input (False for sub-agents)

        Returns:
            Final assistant response text
        """
        # Handle slash commands (works for both CLI and frontend)
        handled, response = await self.handle_slash_command(user_input)
        if handled:
            return response

        # Initialize snapshot system on first run
        if not self._snapshot_initialized:
            await self.session_revert.init()
            self._snapshot_initialized = True

        # Load project instructions on first run
        project_instructions = await self._load_project_instructions()

        if not self.messages:
            self.messages.append(Message.system(self._get_system_prompt(project_instructions)))

        self.messages.append(Message.user(user_input))

        if show_welcome:
            self.console.print(f"\n[bold blue]User:[/bold blue] {user_input}\n")

        tool_definitions = self.registry.get_definitions()
        final_response = ""
        iterations = 0

        # Generate message ID for behavior tracking
        current_message_id = str(uuid.uuid4())
        if self.behavior_collector:
            self.behavior_collector.set_message_id(current_message_id)

        while iterations < max_iterations:
            iterations += 1

            # Record iteration start for behavior tracking
            if self.behavior_collector:
                self.behavior_collector.record_iteration_start()

            # Check for compaction before each iteration
            await self._maybe_compact()

            # Only show iteration count after the first iteration (reduce noise)
            if iterations > 1:
                self.console.print(f"[dim]  ... (iteration {iterations})[/dim]")

            text_buffer = []
            reasoning_buffer = []
            reasoning_header_shown = False

            def on_text(delta: str) -> None:
                text_buffer.append(delta)

            def on_reasoning(delta: str) -> None:
                nonlocal reasoning_header_shown
                if not reasoning_header_shown:
                    self.console.print("[cyan bold]💭 Thinking[/cyan bold]")
                    reasoning_header_shown = True
                reasoning_buffer.append(delta)
                self.console.print(f"[cyan]{delta}[/cyan]", end="")

            llm_start_time = time.time() * 1000
            try:
                parts, finish_reason = await self._call_llm(
                    tool_definitions,
                    on_text,
                    on_reasoning,
                )
            except Exception as e:
                self.console.print(f"\n[red]LLM Error: {e}[/red]")
                # Record iteration end on error
                if self.behavior_collector:
                    self.behavior_collector.record_iteration_end()
                break
            llm_end_time = time.time() * 1000

            # Close thinking block, then print response all at once
            if reasoning_buffer:
                self.console.print()
                self.console.print("[cyan bold]───[/cyan bold]")

            if text_buffer:
                full_text = "".join(text_buffer)
                self._get_display().print_text(full_text)

            # Record LLM response for behavior tracking
            if self.behavior_collector:
                has_reasoning = len(reasoning_buffer) > 0
                # Estimate token usage (actual values would come from API response)
                input_text = "".join(str(m.to_dict()) for m in self.messages[-3:]) if len(self.messages) >= 3 else ""
                output_text = "".join(text_buffer)
                self.behavior_collector.record_llm_response(
                    response_time_ms=int(llm_end_time - llm_start_time),
                    input_tokens=len(input_text) // 4,  # Rough estimate
                    output_tokens=len(output_text) // 4,  # Rough estimate
                    has_reasoning=has_reasoning,
                    stop_reason=finish_reason or "unknown",
                )

            assistant_message = Message.assistant(parts)
            self.messages.append(assistant_message)

            text_content = assistant_message.get_text()
            tool_uses = assistant_message.get_tool_uses()

            if text_content:
                final_response = text_content

            if finish_reason == "tool_calls" and tool_uses:
                tool_results = await self._execute_tools(tool_uses)
                self.messages.extend(tool_results)

                # Record iteration end for behavior tracking
                if self.behavior_collector:
                    self.behavior_collector.record_iteration_end()
                continue

            # Record iteration end for behavior tracking
            if self.behavior_collector:
                self.behavior_collector.record_iteration_end()
            break

        if iterations >= max_iterations:
            self.console.print(f"\n[yellow]Warning: Reached maximum iterations ({max_iterations})[/yellow]")

        return final_response

    def finalize_behavior(self) -> dict[str, Any] | None:
        """Finalize and export behavioral signals.

        Should be called when the session ends to generate the summary.

        Returns:
            Session summary dictionary, or None if behavior collection is disabled
        """
        if self.behavior_collector:
            return self.behavior_collector.finalize()
        return None

    async def run_interactive(self) -> None:
        """Run an interactive session.

        Note: When using ConsoleApp, this method is not called directly.
        ConsoleApp provides enhanced session management on top of this.
        This method remains as a simpler fallback.
        """
        # Load project instructions
        project_instructions = await self._load_project_instructions()

        # Use Display for welcome
        display = self._get_display()
        profile = get_current_profile()
        profile_name = profile.basic.name if profile else ""
        model_display = self.config.get_model_display()

        display.print_welcome(
            directory=str(self.config.target_directory),
            model=model_display,
            profile=profile_name,
        )

        if project_instructions:
            self.console.print("[dim]  Loaded project instructions from AGENTS.md[/dim]\n")

        self.console.print("[dim]  Type /help for commands, exit to quit[/dim]\n")

        # Setup prompt_toolkit session for better Unicode support (Chinese input)
        pt_style = PTStyle.from_dict({"prompt": "ansigreen bold"})
        session: PromptSession[str] = PromptSession(style=pt_style)

        while True:
            try:
                # Use prompt_toolkit async for better Unicode support
                user_input = await session.prompt_async("\n> ")
                user_input = user_input.strip()

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit"):
                    # Finalize behavioral signals before exit
                    summary = self.finalize_behavior()
                    if summary:
                        self.console.print(
                            f"[dim]Behavioral data saved to data/behavior/sessions/{self.session_id}/[/dim]"
                        )
                    self.console.print("[dim]Goodbye![/dim]")
                    break

                # Handle slash commands via unified handler
                handled, _ = await self.handle_slash_command(user_input)
                if handled:
                    continue

                await self.run(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[dim]Use 'exit' to quit[/dim]")
            except EOFError:
                # Finalize behavioral signals before exit
                summary = self.finalize_behavior()
                if summary:
                    self.console.print(
                        f"\n[dim]Behavioral data saved to data/behavior/sessions/{self.session_id}/[/dim]"
                    )
                self.console.print("[dim]Goodbye![/dim]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
