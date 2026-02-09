"""SKILL.md file parser with YAML frontmatter support."""

import re
from pathlib import Path
from typing import Any

import yaml

from .errors import SkillFrontmatterError, SkillInvalidError, SkillParseError
from .models import ParsedSkill, SkillFrontmatter

# Regex to match YAML frontmatter block
FRONTMATTER_PATTERN = re.compile(r"^---\r?\n([\s\S]*?)\r?\n---")


def preprocess_frontmatter(content: str) -> str:
    """
    Preprocess YAML frontmatter to handle special characters like colons.

    If a value contains a colon and isn't already quoted or using block scalar,
    convert it to block scalar format to avoid YAML parsing issues.

    Args:
        content: Raw markdown content with potential frontmatter

    Returns:
        Content with preprocessed frontmatter
    """
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return content

    frontmatter = match.group(1)
    lines = frontmatter.split("\n")
    result: list[str] = []

    for line in lines:
        # Skip comments and empty lines
        if line.strip().startswith("#") or line.strip() == "":
            result.append(line)
            continue

        # Skip lines that are continuations (indented)
        if re.match(r"^\s+", line):
            result.append(line)
            continue

        # Match key: value pattern
        kv_match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*)$", line)
        if not kv_match:
            result.append(line)
            continue

        key = kv_match.group(1)
        value = kv_match.group(2).strip()

        # Skip if value is empty, already quoted, or uses block scalar
        if value == "" or value in (">", "|") or value.startswith('"') or value.startswith("'"):
            result.append(line)
            continue

        # If value contains a colon, convert to block scalar
        if ":" in value:
            result.append(f"{key}: |")
            result.append(f"  {value}")
            continue

        result.append(line)

    processed = "\n".join(result)
    return content.replace(frontmatter, processed, 1)


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """
    Parse YAML frontmatter from markdown content.

    Args:
        content: Markdown content with potential frontmatter

    Returns:
        Tuple of (frontmatter_data, body_content)
    """
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return {}, content

    frontmatter_raw = match.group(1)
    body = content[match.end() :].strip()

    try:
        data = yaml.safe_load(frontmatter_raw)
        if data is None:
            data = {}
    except yaml.YAMLError as e:
        raise SkillFrontmatterError("", str(e))

    return data, body


def parse_skill_file(file_path: str | Path) -> ParsedSkill:
    """
    Parse a SKILL.md file.

    Args:
        file_path: Path to the SKILL.md file

    Returns:
        ParsedSkill with frontmatter and content

    Raises:
        SkillParseError: If file cannot be read
        SkillFrontmatterError: If frontmatter is invalid
        SkillInvalidError: If required fields are missing
    """
    path = Path(file_path)

    try:
        raw_content = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SkillParseError(str(path), str(e))

    # Preprocess to handle special characters
    preprocessed = preprocess_frontmatter(raw_content)

    try:
        data, content = parse_frontmatter(preprocessed)
    except SkillFrontmatterError as e:
        e.path = str(path)
        raise

    # Create frontmatter model
    frontmatter = SkillFrontmatter.from_dict(data)

    # Validate required fields
    issues = frontmatter.validate()
    if issues:
        raise SkillInvalidError(str(path), issues)

    return ParsedSkill(
        frontmatter=frontmatter,
        content=content,
        location=str(path.resolve()),
    )


def get_skill_content(skill_path: str | Path) -> str:
    """
    Get the full formatted content of a skill.

    Args:
        skill_path: Path to the SKILL.md file

    Returns:
        Formatted skill content with metadata
    """
    parsed = parse_skill_file(skill_path)
    skill_dir = Path(parsed.location).parent

    return "\n".join(
        [
            f"## Skill: {parsed.name}",
            "",
            f"**Base directory**: {skill_dir}",
            "",
            parsed.content.strip(),
        ]
    )
