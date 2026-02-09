"""File ignore patterns for filtering files during operations."""

import fnmatch
from pathlib import Path


class FileIgnore:
    """File ignore patterns management."""

    # Folders to ignore
    FOLDERS = {
        "node_modules",
        "bower_components",
        ".pnpm-store",
        "vendor",
        ".npm",
        "dist",
        "build",
        "out",
        ".next",
        "target",
        "bin",
        "obj",
        ".git",
        ".svn",
        ".hg",
        ".vscode",
        ".idea",
        ".turbo",
        ".output",
        "desktop",
        ".sst",
        ".cache",
        ".webkit-cache",
        "__pycache__",
        ".pytest_cache",
        "mypy_cache",
        ".history",
        ".gradle",
    }

    # File patterns to ignore
    FILE_PATTERNS = [
        "**/*.swp",
        "**/*.swo",
        "**/*.pyc",
        # OS
        "**/.DS_Store",
        "**/Thumbs.db",
        # Logs & temp
        "**/logs/**",
        "**/tmp/**",
        "**/temp/**",
        "**/*.log",
        # Coverage/test outputs
        "**/coverage/**",
        "**/.nyc_output/**",
    ]

    @classmethod
    def get_patterns(cls) -> list[str]:
        """Get all ignore patterns."""
        return cls.FILE_PATTERNS + list(cls.FOLDERS)

    @classmethod
    def match(
        cls,
        filepath: str,
        extra: list[str] | None = None,
        whitelist: list[str] | None = None,
    ) -> bool:
        """
        Check if a filepath should be ignored.

        Args:
            filepath: The file path to check
            extra: Additional patterns to ignore
            whitelist: Patterns that should NOT be ignored (overrides ignore)

        Returns:
            True if the file should be ignored, False otherwise
        """
        # Check whitelist first
        if whitelist:
            for pattern in whitelist:
                if fnmatch.fnmatch(filepath, pattern):
                    return False

        # Check folder patterns
        parts = Path(filepath).parts
        for part in parts:
            if part in cls.FOLDERS:
                return True

        # Check file patterns
        all_patterns = cls.FILE_PATTERNS + (extra or [])
        for pattern in all_patterns:
            if fnmatch.fnmatch(filepath, pattern):
                return True

        return False

    @classmethod
    def filter_files(
        cls,
        files: list[str],
        extra: list[str] | None = None,
        whitelist: list[str] | None = None,
    ) -> list[str]:
        """
        Filter a list of files, removing ignored ones.

        Args:
            files: List of file paths
            extra: Additional patterns to ignore
            whitelist: Patterns that should NOT be ignored

        Returns:
            Filtered list of file paths
        """
        return [f for f in files if not cls.match(f, extra, whitelist)]
