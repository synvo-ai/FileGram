"""Skill data models."""

from dataclasses import dataclass
from typing import Any


@dataclass
class SkillFrontmatter:
    """Frontmatter data from a SKILL.md file."""

    name: str
    description: str
    raw_data: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillFrontmatter":
        """Create from a dictionary (parsed YAML frontmatter)."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            raw_data=data,
        )

    def validate(self) -> list[str]:
        """Validate frontmatter and return list of issues."""
        issues = []
        if not self.name:
            issues.append("missing required field 'name'")
        if not self.description:
            issues.append("missing required field 'description'")
        return issues


@dataclass
class ParsedSkill:
    """Complete parsed SKILL.md file."""

    frontmatter: SkillFrontmatter
    content: str
    location: str

    @property
    def name(self) -> str:
        """Skill name from frontmatter."""
        return self.frontmatter.name

    @property
    def description(self) -> str:
        """Skill description from frontmatter."""
        return self.frontmatter.description


@dataclass
class SkillInfo:
    """Skill information for listing and lookup.

    This is the minimal information needed to display and load a skill.
    """

    name: str
    description: str
    location: str

    @classmethod
    def from_parsed(cls, parsed: ParsedSkill) -> "SkillInfo":
        """Create from a ParsedSkill."""
        return cls(
            name=parsed.name,
            description=parsed.description,
            location=parsed.location,
        )
