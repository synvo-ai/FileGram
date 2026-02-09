"""Prompt loader for loading prompt templates from text files.

This module provides functionality similar to OpenCode's TypeScript static imports
for .txt files. In Python, we use importlib.resources to load package resources.

Usage:
    from filegram.prompts import load_tool_prompt, load_agent_prompt

    # Load tool description
    bash_description = load_tool_prompt("bash")

    # Load agent prompt
    explore_prompt = load_agent_prompt("explore")

    # Load session prompt
    system_prompt = load_session_prompt("system")
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path


class PromptLoader:
    """Loader for prompt templates from text files.

    This class provides methods to load prompt templates from the prompts directory.
    It mirrors the behavior of OpenCode's TypeScript static imports.
    """

    def __init__(self, base_package: str = "filegram.prompts"):
        """Initialize the prompt loader.

        Args:
            base_package: The base package path for prompts.
        """
        self.base_package = base_package
        self._cache: dict[str, str] = {}

    def _load_resource(self, subpackage: str, filename: str) -> str:
        """Load a text resource from a subpackage.

        Args:
            subpackage: The subpackage name (e.g., "tools", "agents", "session")
            filename: The filename to load (e.g., "bash.txt")

        Returns:
            The content of the text file.

        Raises:
            FileNotFoundError: If the resource doesn't exist.
        """
        cache_key = f"{subpackage}/{filename}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        package = f"{self.base_package}.{subpackage}"

        try:
            # Python 3.9+ approach using importlib.resources.files
            files = importlib.resources.files(package)
            resource = files.joinpath(filename)
            content = resource.read_text(encoding="utf-8")
        except (ModuleNotFoundError, FileNotFoundError, TypeError):
            # Fallback: try loading from filesystem directly
            try:
                base_path = Path(__file__).parent
                file_path = base_path / subpackage / filename
                if file_path.exists():
                    content = file_path.read_text(encoding="utf-8")
                else:
                    raise FileNotFoundError(f"Prompt file not found: {subpackage}/{filename}")
            except Exception as e:
                raise FileNotFoundError(f"Failed to load prompt: {subpackage}/{filename}: {e}")

        self._cache[cache_key] = content
        return content

    def load_tool(self, tool_name: str) -> str:
        """Load a tool description prompt.

        Args:
            tool_name: The name of the tool (e.g., "bash", "read", "edit")

        Returns:
            The tool description text.
        """
        # Handle tool names with underscores or hyphens
        filename = f"{tool_name.replace('-', '_')}.txt"
        return self._load_resource("tools", filename)

    def load_agent(self, agent_name: str) -> str:
        """Load an agent prompt.

        Args:
            agent_name: The name of the agent (e.g., "explore", "compaction")

        Returns:
            The agent prompt text.
        """
        filename = f"{agent_name}.txt"
        return self._load_resource("agents", filename)

    def load_session(self, name: str, provider: str | None = None) -> str:
        """Load a session prompt, optionally with provider-specific variant.

        Args:
            name: The name of the session prompt (e.g., "system", "plan_mode")
            provider: Optional provider name (e.g., "anthropic", "openai", "gemini")
                     If specified, tries to load provider-specific prompt first.

        Returns:
            The session prompt text.
        """
        # Try provider-specific prompt first
        if provider:
            provider_filename = f"{name}_{provider}.txt"
            try:
                return self._load_resource("session", provider_filename)
            except FileNotFoundError:
                pass  # Fall back to generic prompt

        filename = f"{name}.txt"
        return self._load_resource("session", filename)

    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._cache.clear()


# Global singleton instance
_loader: PromptLoader | None = None


def get_prompt_loader() -> PromptLoader:
    """Get the global prompt loader instance."""
    global _loader
    if _loader is None:
        _loader = PromptLoader()
    return _loader


# Convenience functions for common use cases
def load_tool_prompt(tool_name: str) -> str:
    """Load a tool description prompt.

    Args:
        tool_name: The name of the tool (e.g., "bash", "read", "edit")

    Returns:
        The tool description text.

    Example:
        >>> description = load_tool_prompt("bash")
        >>> print(description[:50])
        'Execute a bash command in the shell...'
    """
    return get_prompt_loader().load_tool(tool_name)


def load_agent_prompt(agent_name: str) -> str:
    """Load an agent prompt.

    Args:
        agent_name: The name of the agent (e.g., "explore", "compaction")

    Returns:
        The agent prompt text.

    Example:
        >>> prompt = load_agent_prompt("explore")
        >>> print(prompt[:50])
        'You are a file search specialist...'
    """
    return get_prompt_loader().load_agent(agent_name)


def load_session_prompt(name: str, provider: str | None = None) -> str:
    """Load a session prompt, optionally with provider-specific variant.

    Args:
        name: The name of the session prompt (e.g., "system", "plan_mode")
        provider: Optional provider name (e.g., "anthropic", "openai", "gemini")
                 If specified, tries to load provider-specific prompt first.

    Returns:
        The session prompt text.

    Example:
        >>> prompt = load_session_prompt("system")
        >>> print(prompt[:50])
        'You are FileGram...'

        >>> prompt = load_session_prompt("system", "anthropic")
        >>> # Loads system_anthropic.txt if exists, else system.txt
    """
    return get_prompt_loader().load_session(name, provider)


# Pre-load commonly used prompts for faster access
# These will be cached after first use
TOOL_NAMES = [
    "bash",
    "read",
    "write",
    "edit",
    "glob",
    "grep",
    "task",
    "todoread",
    "todowrite",
    "plan_enter",
    "plan_exit",
]

AGENT_NAMES = ["explore", "compaction"]

SESSION_NAMES = ["system", "plan_mode"]


def preload_all() -> None:
    """Preload all prompts into cache for faster access."""
    loader = get_prompt_loader()

    for name in TOOL_NAMES:
        try:
            loader.load_tool(name)
        except FileNotFoundError:
            pass  # Skip missing prompts

    for name in AGENT_NAMES:
        try:
            loader.load_agent(name)
        except FileNotFoundError:
            pass

    for name in SESSION_NAMES:
        try:
            loader.load_session(name)
        except FileNotFoundError:
            pass
