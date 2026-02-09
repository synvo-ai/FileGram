"""Storage implementation for persistent data."""

from __future__ import annotations

import asyncio
import glob as globlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class NotFoundError(Exception):
    """Error raised when a resource is not found."""

    def __init__(self, message: str, path: str | None = None):
        super().__init__(message)
        self.path = path


class Storage:
    """File-based storage for persistent data.

    Storage uses a hierarchical key system where keys are lists of strings
    that form a path. Data is stored as JSON files.
    """

    _data_dir: Path | None = None
    _lock = asyncio.Lock()

    @classmethod
    def get_data_dir(cls) -> Path:
        """Get the data directory, creating if needed."""
        if cls._data_dir is None:
            # Default to ~/.synvocode/storage
            home = Path.home()
            cls._data_dir = home / ".synvocode" / "storage"
            cls._data_dir.mkdir(parents=True, exist_ok=True)
        return cls._data_dir

    @classmethod
    def set_data_dir(cls, path: str | Path) -> None:
        """Set the data directory."""
        cls._data_dir = Path(path)
        cls._data_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _key_to_path(cls, key: list[str]) -> Path:
        """Convert a key list to a file path."""
        return cls.get_data_dir() / ("/".join(key) + ".json")

    @classmethod
    async def read(cls, key: list[str]) -> Any:
        """Read data from storage.

        Args:
            key: List of strings forming the key path

        Returns:
            The stored data

        Raises:
            NotFoundError: If the key doesn't exist
        """
        path = cls._key_to_path(key)

        async with cls._lock:
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except FileNotFoundError:
                raise NotFoundError(f"Resource not found: {'/'.join(key)}", str(path))
            except json.JSONDecodeError as e:
                raise NotFoundError(f"Invalid JSON at {'/'.join(key)}: {e}", str(path))

    @classmethod
    async def write(cls, key: list[str], content: Any) -> None:
        """Write data to storage.

        Args:
            key: List of strings forming the key path
            content: Data to store (must be JSON serializable)
        """
        path = cls._key_to_path(key)

        async with cls._lock:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2)

    @classmethod
    async def update(cls, key: list[str], fn: Callable[[T], None]) -> T:
        """Update data in storage atomically.

        Args:
            key: List of strings forming the key path
            fn: Function to modify the data in place

        Returns:
            The updated data
        """
        path = cls._key_to_path(key)

        async with cls._lock:
            try:
                with open(path, encoding="utf-8") as f:
                    content = json.load(f)
            except FileNotFoundError:
                raise NotFoundError(f"Resource not found: {'/'.join(key)}", str(path))

            fn(content)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2)

            return content

    @classmethod
    async def remove(cls, key: list[str]) -> None:
        """Remove data from storage.

        Args:
            key: List of strings forming the key path
        """
        path = cls._key_to_path(key)

        async with cls._lock:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    @classmethod
    async def list(cls, prefix: list[str]) -> list[list[str]]:
        """List all keys under a prefix.

        Args:
            prefix: Key prefix to list under

        Returns:
            List of keys (as string lists)
        """
        base_dir = cls.get_data_dir() / "/".join(prefix)

        if not base_dir.exists():
            return []

        results = []
        pattern = str(base_dir / "**" / "*.json")

        for filepath in globlib.glob(pattern, recursive=True):
            # Convert path back to key
            rel_path = Path(filepath).relative_to(cls.get_data_dir())
            key_parts = str(rel_path).replace(".json", "").split(os.sep)
            results.append(key_parts)

        results.sort()
        return results

    @classmethod
    async def exists(cls, key: list[str]) -> bool:
        """Check if a key exists in storage.

        Args:
            key: List of strings forming the key path

        Returns:
            True if the key exists
        """
        path = cls._key_to_path(key)
        return path.exists()
