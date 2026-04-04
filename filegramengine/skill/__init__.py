"""Skill module for loading and managing skills.

Skills are markdown files with YAML frontmatter that provide specialized
instructions for the agent. They are loaded from various directories:

- .filegramengine/skill/ (project-level)
- .claude/skills/ (project-level, Claude Code compatibility)
- ~/.filegramengine/skills/ (global)
- ~/.claude/skills/ (global)

Example SKILL.md format:
    ---
    name: skill-name
    description: Short description for the LLM
    ---

    Detailed skill instructions here...
"""

from .errors import (
    SkillError,
    SkillFrontmatterError,
    SkillInvalidError,
    SkillNameMismatchError,
    SkillNotFoundError,
    SkillParseError,
)
from .loader import (
    SkillLoader,
    get_default_loader,
    reset_default_loader,
)
from .models import (
    ParsedSkill,
    SkillFrontmatter,
    SkillInfo,
)
from .parser import (
    get_skill_content,
    parse_frontmatter,
    parse_skill_file,
    preprocess_frontmatter,
)

__all__ = [
    # Errors
    "SkillError",
    "SkillParseError",
    "SkillFrontmatterError",
    "SkillNotFoundError",
    "SkillInvalidError",
    "SkillNameMismatchError",
    # Models
    "SkillFrontmatter",
    "ParsedSkill",
    "SkillInfo",
    # Parser
    "preprocess_frontmatter",
    "parse_frontmatter",
    "parse_skill_file",
    "get_skill_content",
    # Loader
    "SkillLoader",
    "get_default_loader",
    "reset_default_loader",
]
