"""Load project instructions from AGENTS.md and similar files."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx


@dataclass
class LoadedInstruction:
    """A loaded instruction file."""

    path: str  # File path or URL
    content: str
    is_url: bool = False


@dataclass
class InstructionLoader:
    """
    Loader for project-specific instructions.

    Supports:
    - AGENTS.md (project root)
    - CLAUDE.md (project root, for compatibility)
    - Global config (~/.filegramengine/AGENTS.md)
    - Custom instruction files via config
    - HTTP/HTTPS URLs
    """

    # Files to search for (in order of preference)
    INSTRUCTION_FILES = [
        "AGENTS.md",
        "CLAUDE.md",
        "CONTEXT.md",  # deprecated but supported
    ]

    target_directory: Path
    config_dir: Path | None = None
    additional_instructions: list[str] = field(default_factory=list)
    _cache: dict[str, LoadedInstruction] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        target_directory: Path,
        config_dir: Path | None = None,
        instructions: list[str] | None = None,
    ) -> "InstructionLoader":
        """Create an InstructionLoader instance."""
        if config_dir is None:
            config_dir = Path.home() / ".filegramengine"

        return cls(
            target_directory=target_directory.resolve(),
            config_dir=config_dir,
            additional_instructions=instructions or [],
        )

    def _find_up(self, filename: str, start_dir: Path, stop_dir: Path | None = None) -> Path | None:
        """
        Search for a file starting from start_dir and moving up to parent directories.
        Stop at stop_dir if provided, otherwise stop at filesystem root.
        """
        current = start_dir.resolve()
        stop = stop_dir.resolve() if stop_dir else Path("/")

        while current >= stop:
            candidate = current / filename
            if candidate.exists() and candidate.is_file():
                return candidate
            if current == stop or current == current.parent:
                break
            current = current.parent

        return None

    def _get_project_instruction_path(self) -> Path | None:
        """Find project instruction file (AGENTS.md, CLAUDE.md, etc.)."""
        for filename in self.INSTRUCTION_FILES:
            path = self._find_up(filename, self.target_directory, self.target_directory)
            if path:
                return path
        return None

    def _get_global_instruction_path(self) -> Path | None:
        """Find global instruction file."""
        if self.config_dir:
            global_path = self.config_dir / "AGENTS.md"
            if global_path.exists():
                return global_path
        return None

    async def _load_url(self, url: str) -> str | None:
        """Load instruction from URL."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.text
        except Exception:
            pass
        return None

    def _load_file(self, path: Path) -> str | None:
        """Load instruction from file."""
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None

    async def load_all(self) -> list[LoadedInstruction]:
        """
        Load all instruction files.

        Returns list of LoadedInstruction in order:
        1. Project instruction (AGENTS.md in project root)
        2. Global instruction (~/.filegramengine/AGENTS.md)
        3. Additional instructions from config
        """
        instructions: list[LoadedInstruction] = []

        # 1. Load project instruction
        project_path = self._get_project_instruction_path()
        if project_path:
            content = self._load_file(project_path)
            if content:
                instructions.append(
                    LoadedInstruction(
                        path=str(project_path),
                        content=content,
                        is_url=False,
                    )
                )

        # 2. Load global instruction
        global_path = self._get_global_instruction_path()
        if global_path:
            content = self._load_file(global_path)
            if content:
                instructions.append(
                    LoadedInstruction(
                        path=str(global_path),
                        content=content,
                        is_url=False,
                    )
                )

        # 3. Load additional instructions
        for instruction in self.additional_instructions:
            if instruction.startswith("http://") or instruction.startswith("https://"):
                content = await self._load_url(instruction)
                if content:
                    instructions.append(
                        LoadedInstruction(
                            path=instruction,
                            content=content,
                            is_url=True,
                        )
                    )
            else:
                # Expand home directory
                if instruction.startswith("~/"):
                    instruction = os.path.expanduser(instruction)

                # Handle relative paths
                if not os.path.isabs(instruction):
                    path = self.target_directory / instruction
                else:
                    path = Path(instruction)

                if path.exists():
                    content = self._load_file(path)
                    if content:
                        instructions.append(
                            LoadedInstruction(
                                path=str(path),
                                content=content,
                                is_url=False,
                            )
                        )

        return instructions

    async def get_system_prompt_addition(self) -> str:
        """
        Get instruction content formatted for system prompt.

        Returns a string that can be appended to the system prompt.
        """
        instructions = await self.load_all()
        if not instructions:
            return ""

        parts = []
        for inst in instructions:
            parts.append(f"Instructions from: {inst.path}\n{inst.content}")

        return "\n\n".join(parts)

    def find_instruction_for_file(self, filepath: Path) -> LoadedInstruction | None:
        """
        Find instruction file relevant to a specific file being edited.

        This searches from the file's directory up to the project root
        for any AGENTS.md that might have file-specific instructions.
        """
        file_dir = filepath.parent if filepath.is_file() else filepath

        for filename in self.INSTRUCTION_FILES:
            path = self._find_up(filename, file_dir, self.target_directory)
            if path:
                # Skip if it's the same as the project root instruction
                project_path = self._get_project_instruction_path()
                if project_path and path == project_path:
                    continue

                content = self._load_file(path)
                if content:
                    return LoadedInstruction(
                        path=str(path),
                        content=content,
                        is_url=False,
                    )

        return None
