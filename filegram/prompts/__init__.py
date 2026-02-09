"""Prompts package for loading prompt templates.

This package provides functionality to load prompt templates from text files,
similar to how OpenCode handles prompts in TypeScript with static imports.
"""

from .loader import (
    PromptLoader,
    get_prompt_loader,
    load_agent_prompt,
    load_session_prompt,
    load_tool_prompt,
)

__all__ = [
    "PromptLoader",
    "load_tool_prompt",
    "load_agent_prompt",
    "load_session_prompt",
    "get_prompt_loader",
]
