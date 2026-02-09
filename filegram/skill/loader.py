"""Skill loader - scans directories and loads skills."""

import logging
import os
from collections.abc import Iterator
from pathlib import Path

from .errors import SkillError, SkillNotFoundError
from .models import ParsedSkill, SkillInfo
from .parser import get_skill_content, parse_skill_file

logger = logging.getLogger(__name__)


# Glob patterns for finding skills (note: Python glob doesn't support {a,b} syntax)
SYNVOCODE_SKILL_PATTERNS = ["skill/**/SKILL.md", "skills/**/SKILL.md"]
OPENCODE_SKILL_PATTERNS = ["skill/**/SKILL.md", "skills/**/SKILL.md"]
CLAUDE_SKILL_PATTERNS = ["skills/**/SKILL.md"]


class SkillLoader:
    """Loader for skill files.

    Scans multiple directories for SKILL.md files and provides access
    to loaded skills.

    Directory scan order (later entries override earlier ones):
    1. .synvocode/skill/ (project-level)
    2. .opencode/skill/ (project-level, OpenCode compatibility)
    3. .claude/skills/ (project-level, Claude Code compatibility)
    4. ~/.synvocode/skills/ (global)
    5. ~/.opencode/skills/ (global)
    6. ~/.claude/skills/ (global)
    7. Custom paths (via configuration)
    """

    def __init__(
        self,
        project_dir: str | Path | None = None,
        custom_paths: list[str | Path] | None = None,
        disable_claude_skills: bool = False,
    ):
        """
        Initialize the skill loader.

        Args:
            project_dir: Project directory to scan for skills.
                        Defaults to current working directory.
            custom_paths: Additional paths to scan for skills.
            disable_claude_skills: If True, skip .claude/skills/ directories.
        """
        self.project_dir = Path(project_dir).resolve() if project_dir else Path.cwd()
        self.custom_paths = [Path(p) for p in (custom_paths or [])]
        self.disable_claude_skills = disable_claude_skills
        self._skills: dict[str, SkillInfo] | None = None

    def _get_home_dir(self) -> Path:
        """Get home directory."""
        return Path.home()

    def _find_git_root(self, start: Path) -> Path | None:
        """Find git root directory from start path, going up."""
        current = start.resolve()
        while current != current.parent:
            if (current / ".git").exists():
                return current
            current = current.parent
        return None

    def _get_project_dirs(self) -> list[Path]:
        """Get list of project directories to scan (bottom-up from project_dir)."""
        dirs: list[Path] = []
        git_root = self._find_git_root(self.project_dir)
        stop_at = git_root.parent if git_root else self._get_home_dir()

        current = self.project_dir
        while current != stop_at and current != current.parent:
            dirs.append(current)
            current = current.parent

        # Ensure git_root is included if it wasn't already
        if git_root and git_root not in dirs:
            dirs.append(git_root)

        return dirs

    def _scan_directory(self, base_dir: Path, patterns: list[str]) -> Iterator[Path]:
        """Scan a directory for skill files matching patterns."""
        import glob as glob_module

        for pattern in patterns:
            # Expand glob pattern
            full_pattern = str(base_dir / pattern)
            for match in glob_module.glob(full_pattern, recursive=True):
                path = Path(match)
                if path.is_file() and path.name == "SKILL.md":
                    yield path

    def _load_skill_file(self, path: Path) -> SkillInfo | None:
        """Load a single skill file and return SkillInfo, or None on error."""
        try:
            parsed = parse_skill_file(path)
            return SkillInfo.from_parsed(parsed)
        except SkillError as e:
            logger.warning(f"Failed to load skill: {e}")
            return None

    def _scan_all_directories(self) -> dict[str, SkillInfo]:
        """Scan all directories and return skill map."""
        skills: dict[str, SkillInfo] = {}
        home = self._get_home_dir()

        # Helper to add skill (later additions override earlier ones)
        def add_skill(path: Path) -> None:
            skill_info = self._load_skill_file(path)
            if skill_info:
                if skill_info.name in skills:
                    logger.debug(
                        f"Skill '{skill_info.name}' overridden: "
                        f"{skills[skill_info.name].location} -> {skill_info.location}"
                    )
                skills[skill_info.name] = skill_info

        # 1. Scan global directories (lowest priority)
        # ~/.synvocode/skills/
        global_synvocode = home / ".synvocode"
        if global_synvocode.exists():
            for path in self._scan_directory(global_synvocode, SYNVOCODE_SKILL_PATTERNS):
                add_skill(path)

        # ~/.opencode/skills/
        global_opencode = home / ".opencode"
        if global_opencode.exists():
            for path in self._scan_directory(global_opencode, OPENCODE_SKILL_PATTERNS):
                add_skill(path)

        # ~/.claude/skills/
        if not self.disable_claude_skills:
            global_claude = home / ".claude"
            if global_claude.exists():
                for path in self._scan_directory(global_claude, CLAUDE_SKILL_PATTERNS):
                    add_skill(path)

        # 2. Scan project directories (bottom-up, so closest to project wins)
        project_dirs = list(reversed(self._get_project_dirs()))  # Start from root

        for project_dir in project_dirs:
            # .synvocode/
            synvocode_dir = project_dir / ".synvocode"
            if synvocode_dir.exists():
                for path in self._scan_directory(synvocode_dir, SYNVOCODE_SKILL_PATTERNS):
                    add_skill(path)

            # .opencode/
            opencode_dir = project_dir / ".opencode"
            if opencode_dir.exists():
                for path in self._scan_directory(opencode_dir, OPENCODE_SKILL_PATTERNS):
                    add_skill(path)

            # .claude/
            if not self.disable_claude_skills:
                claude_dir = project_dir / ".claude"
                if claude_dir.exists():
                    for path in self._scan_directory(claude_dir, CLAUDE_SKILL_PATTERNS):
                        add_skill(path)

        # 3. Scan custom paths (highest priority)
        for custom_path in self.custom_paths:
            if custom_path.exists():
                for path in self._scan_directory(custom_path, SYNVOCODE_SKILL_PATTERNS):
                    add_skill(path)

        return skills

    def load_all(self) -> dict[str, SkillInfo]:
        """
        Load all skills from configured directories.

        Returns:
            Dictionary mapping skill names to SkillInfo objects.
        """
        if self._skills is None:
            self._skills = self._scan_all_directories()
        return self._skills

    def reload(self) -> dict[str, SkillInfo]:
        """Force reload all skills."""
        self._skills = None
        return self.load_all()

    def get(self, name: str) -> SkillInfo | None:
        """
        Get a skill by name.

        Args:
            name: Skill name to look up.

        Returns:
            SkillInfo if found, None otherwise.
        """
        skills = self.load_all()
        return skills.get(name)

    def all(self) -> list[SkillInfo]:
        """
        Get all loaded skills.

        Returns:
            List of all SkillInfo objects.
        """
        return list(self.load_all().values())

    def get_content(self, name: str) -> str:
        """
        Get the full content of a skill.

        Args:
            name: Skill name to load.

        Returns:
            Formatted skill content.

        Raises:
            SkillNotFoundError: If skill is not found.
        """
        skill = self.get(name)
        if not skill:
            available = list(self.load_all().keys())
            raise SkillNotFoundError(name, available)

        return get_skill_content(skill.location)

    def get_parsed(self, name: str) -> ParsedSkill:
        """
        Get the parsed skill.

        Args:
            name: Skill name to load.

        Returns:
            ParsedSkill object.

        Raises:
            SkillNotFoundError: If skill is not found.
        """
        skill = self.get(name)
        if not skill:
            available = list(self.load_all().keys())
            raise SkillNotFoundError(name, available)

        return parse_skill_file(skill.location)


# Module-level loader instance (lazy initialization)
_default_loader: SkillLoader | None = None


def get_default_loader(
    project_dir: str | Path | None = None,
    custom_paths: list[str | Path] | None = None,
    disable_claude_skills: bool | None = None,
) -> SkillLoader:
    """
    Get the default skill loader instance.

    Creates a new loader if one doesn't exist or if parameters differ.

    Args:
        project_dir: Project directory to scan.
        custom_paths: Additional paths to scan.
        disable_claude_skills: Whether to disable Claude skills.

    Returns:
        SkillLoader instance.
    """
    global _default_loader

    # Check environment variables for configuration
    env_custom_paths = os.environ.get("SYNVOCODE_SKILL_PATHS")
    env_disable_claude = os.environ.get("SYNVOCODE_DISABLE_CLAUDE_SKILLS", "").lower() in ("1", "true", "yes")

    # Merge environment config with parameters
    if env_custom_paths and custom_paths is None:
        custom_paths = [p.strip() for p in env_custom_paths.split(":") if p.strip()]

    if disable_claude_skills is None:
        disable_claude_skills = env_disable_claude

    # Create new loader if needed
    if _default_loader is None:
        _default_loader = SkillLoader(
            project_dir=project_dir,
            custom_paths=custom_paths,
            disable_claude_skills=disable_claude_skills,
        )

    return _default_loader


def reset_default_loader() -> None:
    """Reset the default loader (useful for testing)."""
    global _default_loader
    _default_loader = None
