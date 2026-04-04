"""Profile loader for agent personalities and work habits.

Profiles define an agent's personality, communication style, and work preferences.
They can be loaded via CLI argument (--profile) or interactive command (/profile).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BasicInfo:
    """Basic information about the agent persona."""

    name: str = "Assistant"
    age: int | None = None
    role: str = "Assistant"
    nationality: str | None = None
    language: str = "English"  # Primary response language


@dataclass
class Personality:
    """Personality traits and communication style."""

    traits: list[str] = field(default_factory=list)  # e.g., ["friendly", "patient", "detail-oriented"]
    tone: str = "professional"  # professional, casual, friendly, formal
    humor_level: str = "moderate"  # none, low, moderate, high
    emoji_usage: str = "minimal"  # none, minimal, moderate, frequent
    verbosity: str = "balanced"  # concise, balanced, detailed


@dataclass
class WorkHabits:
    """Work habits and behavioral preferences."""

    working_style: str = "methodical"  # methodical, exploratory, pragmatic
    thoroughness: str = "balanced"  # exhaustive, balanced, minimal
    documentation: str = "moderate"  # comprehensive, moderate, minimal
    organization_style: str = "by_type"  # deeply_nested, flat, by_type, by_project, by_date
    naming_convention: str = "descriptive_long"  # descriptive_long, short_abbreviation, date_prefixed
    version_strategy: str = "keep_history"  # keep_history, archive_old, overwrite
    error_handling: str = "defensive"  # defensive, balanced, optimistic

    # Custom preferences
    preferences: list[str] = field(default_factory=list)
    # Things to avoid
    avoidances: list[str] = field(default_factory=list)


@dataclass
class Profile:
    """Complete agent profile with personality and work habits."""

    id: str  # Unique identifier (filename without extension)
    basic: BasicInfo = field(default_factory=BasicInfo)
    personality: Personality = field(default_factory=Personality)
    work_habits: WorkHabits = field(default_factory=WorkHabits)

    # Custom system prompt addition
    system_prompt_addition: str = ""

    # Greeting message when profile is loaded
    greeting: str = ""

    def to_system_prompt(self) -> str:
        """Generate system prompt section from profile."""
        lines = []

        # Strong identity override
        lines.append("=" * 60)
        lines.append("# IDENTITY OVERRIDE")
        lines.append(f"You are {self.basic.name}, NOT FileGram.")
        lines.append(f"When asked who you are, respond as {self.basic.name}.")
        lines.append(f"Maintain {self.basic.name}'s personality throughout the conversation.")
        lines.append("=" * 60)
        lines.append("")

        # Basic info
        lines.append(f"# Agent Profile: {self.basic.name}")
        if self.basic.role:
            lines.append(f"Role: {self.basic.role}")
        if self.basic.age:
            lines.append(f"Age: {self.basic.age}")
        if self.basic.nationality:
            lines.append(f"Nationality: {self.basic.nationality}")
        lines.append(f"Primary Language: {self.basic.language}")
        lines.append("")

        # Personality
        lines.append("## Personality & Communication Style")
        if self.personality.traits:
            lines.append(f"Traits: {', '.join(self.personality.traits)}")
        lines.append(f"Tone: {self.personality.tone}")
        lines.append(f"Humor: {self.personality.humor_level}")
        lines.append(f"Emoji Usage: {self.personality.emoji_usage}")
        lines.append(f"Verbosity: {self.personality.verbosity}")
        lines.append("")

        # Work habits
        lines.append("## Work Habits & Behavioral Preferences")
        lines.append(f"Working Style: {self.work_habits.working_style}")
        lines.append(f"Thoroughness: {self.work_habits.thoroughness}")
        lines.append(f"Documentation: {self.work_habits.documentation}")
        lines.append(f"Organization: {self.work_habits.organization_style}")
        lines.append(f"Naming Convention: {self.work_habits.naming_convention}")
        lines.append(f"Version Strategy: {self.work_habits.version_strategy}")
        lines.append(f"Error Handling: {self.work_habits.error_handling}")

        if self.work_habits.preferences:
            lines.append("")
            lines.append("### Preferences")
            for pref in self.work_habits.preferences:
                lines.append(f"- {pref}")

        if self.work_habits.avoidances:
            lines.append("")
            lines.append("### Avoid")
            for avoid in self.work_habits.avoidances:
                lines.append(f"- {avoid}")

        if self.system_prompt_addition:
            lines.append("")
            lines.append("## Additional Instructions")
            lines.append(self.system_prompt_addition)

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, profile_id: str, data: dict[str, Any]) -> Profile:
        """Create Profile from dictionary (parsed YAML)."""
        basic_data = data.get("basic", {})
        basic = BasicInfo(
            name=basic_data.get("name", "Assistant"),
            age=basic_data.get("age"),
            role=basic_data.get("role", "Assistant"),
            nationality=basic_data.get("nationality"),
            language=basic_data.get("language", "English"),
        )

        personality_data = data.get("personality", {})
        personality = Personality(
            traits=personality_data.get("traits", []),
            tone=personality_data.get("tone", "professional"),
            humor_level=personality_data.get("humor_level", "moderate"),
            emoji_usage=personality_data.get("emoji_usage", "minimal"),
            verbosity=personality_data.get("verbosity", "balanced"),
        )

        work_data = data.get("work_habits", {})
        work_habits = WorkHabits(
            working_style=work_data.get("working_style", "methodical"),
            thoroughness=work_data.get("thoroughness", "balanced"),
            documentation=work_data.get("documentation", "moderate"),
            organization_style=work_data.get("organization_style", "by_type"),
            naming_convention=work_data.get("naming_convention", "descriptive_long"),
            version_strategy=work_data.get("version_strategy", "keep_history"),
            error_handling=work_data.get("error_handling", "defensive"),
            preferences=work_data.get("preferences", []),
            avoidances=work_data.get("avoidances", []),
        )

        return cls(
            id=profile_id,
            basic=basic,
            personality=personality,
            work_habits=work_habits,
            system_prompt_addition=data.get("system_prompt_addition", ""),
            greeting=data.get("greeting", ""),
        )


class ProfileLoader:
    """Loader for agent profiles from YAML files."""

    def __init__(self, profiles_dir: Path | None = None):
        """Initialize profile loader.

        Args:
            profiles_dir: Directory containing profile YAML files.
                         Defaults to the built-in profiles directory.
        """
        if profiles_dir is None:
            # Use built-in profiles directory
            profiles_dir = Path(__file__).parent / "profiles"
        self.profiles_dir = profiles_dir
        self._cache: dict[str, Profile] = {}

    def list(self) -> list[str]:
        """List available profile IDs."""
        if not self.profiles_dir.exists():
            return []

        profiles = []
        for f in self.profiles_dir.glob("*.yaml"):
            profiles.append(f.stem)
        for f in self.profiles_dir.glob("*.yml"):
            profiles.append(f.stem)

        return sorted(set(profiles))

    def load(self, profile_id: str) -> Profile:
        """Load a profile by ID.

        Args:
            profile_id: The profile ID (filename without extension)

        Returns:
            The loaded Profile

        Raises:
            FileNotFoundError: If profile doesn't exist
        """
        if profile_id in self._cache:
            return self._cache[profile_id]

        # Try .yaml first, then .yml
        yaml_path = self.profiles_dir / f"{profile_id}.yaml"
        if not yaml_path.exists():
            yaml_path = self.profiles_dir / f"{profile_id}.yml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_id}")

        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        profile = Profile.from_dict(profile_id, data or {})
        self._cache[profile_id] = profile
        return profile

    def clear_cache(self) -> None:
        """Clear the profile cache."""
        self._cache.clear()


# Global state
_loader: ProfileLoader | None = None
_current_profile: Profile | None = None


def get_profile_loader() -> ProfileLoader:
    """Get the global profile loader."""
    global _loader
    if _loader is None:
        _loader = ProfileLoader()
    return _loader


def load_profile(profile_id: str) -> Profile:
    """Load a profile by ID."""
    return get_profile_loader().load(profile_id)


def list_profiles() -> list[str]:
    """List available profile IDs."""
    return get_profile_loader().list()


def get_current_profile() -> Profile | None:
    """Get the currently active profile."""
    return _current_profile


def set_current_profile(profile: Profile | None) -> None:
    """Set the currently active profile."""
    global _current_profile
    _current_profile = profile
