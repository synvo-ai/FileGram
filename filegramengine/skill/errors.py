"""Custom exceptions for skill module."""


class SkillError(Exception):
    """Base exception for skill-related errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class SkillParseError(SkillError):
    """Error parsing a SKILL.md file."""

    def __init__(self, path: str, message: str):
        self.path = path
        super().__init__(f"Failed to parse skill at '{path}': {message}")


class SkillFrontmatterError(SkillError):
    """Error parsing YAML frontmatter in SKILL.md."""

    def __init__(self, path: str, message: str):
        self.path = path
        super().__init__(f"Failed to parse frontmatter at '{path}': {message}")


class SkillNotFoundError(SkillError):
    """Skill not found error."""

    def __init__(self, name: str, available: list[str] | None = None):
        self.name = name
        self.available = available or []
        available_str = ", ".join(available) if available else "none"
        super().__init__(f"Skill '{name}' not found. Available skills: {available_str}")


class SkillInvalidError(SkillError):
    """Skill has invalid structure or missing required fields."""

    def __init__(self, path: str, issues: list[str]):
        self.path = path
        self.issues = issues
        issues_str = "; ".join(issues)
        super().__init__(f"Invalid skill at '{path}': {issues_str}")


class SkillNameMismatchError(SkillError):
    """Skill name doesn't match directory name."""

    def __init__(self, path: str, expected: str, actual: str):
        self.path = path
        self.expected = expected
        self.actual = actual
        super().__init__(f"Skill name mismatch at '{path}': expected '{expected}', got '{actual}'")
