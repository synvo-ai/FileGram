"""Ripgrep utility functions for file search and listing."""

import asyncio
import json
import os
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RipgrepMatch:
    """A match result from ripgrep."""

    path: str
    line_number: int
    line_text: str
    absolute_offset: int
    submatches: list[dict[str, Any]]


class Ripgrep:
    """Ripgrep wrapper for file search operations."""

    @staticmethod
    def is_available() -> bool:
        """Check if ripgrep is available."""
        return shutil.which("rg") is not None

    @staticmethod
    def get_path() -> str | None:
        """Get the path to ripgrep executable."""
        return shutil.which("rg")

    @staticmethod
    async def files(
        cwd: str,
        glob: list[str] | None = None,
        hidden: bool = True,
        follow: bool = True,
        max_depth: int | None = None,
    ) -> AsyncIterator[str]:
        """
        List files using ripgrep.

        Args:
            cwd: Working directory to search in
            glob: Glob patterns to filter files
            hidden: Include hidden files
            follow: Follow symlinks
            max_depth: Maximum directory depth

        Yields:
            File paths relative to cwd
        """
        rg_path = Ripgrep.get_path()
        if not rg_path:
            # Fallback to os.walk if ripgrep not available
            async for f in Ripgrep._files_fallback(cwd, glob, hidden, max_depth):
                yield f
            return

        args = [rg_path, "--files", "--glob=!.git/*"]

        if follow:
            args.append("--follow")
        if hidden:
            args.append("--hidden")
        if max_depth is not None:
            args.append(f"--max-depth={max_depth}")
        if glob:
            for g in glob:
                args.append(f"--glob={g}")

        # Check if directory exists
        if not Path(cwd).is_dir():
            raise FileNotFoundError(f"No such directory: '{cwd}'")

        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        buffer = ""
        while True:
            chunk = await proc.stdout.read(8192)
            if not chunk:
                break

            buffer += chunk.decode("utf-8", errors="replace")
            lines = buffer.split("\n")
            buffer = lines.pop()

            for line in lines:
                if line:
                    yield line

        if buffer:
            yield buffer

        await proc.wait()

    @staticmethod
    async def _files_fallback(
        cwd: str,
        glob: list[str] | None = None,
        hidden: bool = True,
        max_depth: int | None = None,
    ) -> AsyncIterator[str]:
        """Fallback file listing using os.walk."""
        import fnmatch

        base_path = Path(cwd)

        for root, dirs, files in os.walk(cwd):
            rel_root = Path(root).relative_to(base_path)

            # Check depth
            if max_depth is not None:
                depth = len(rel_root.parts) if str(rel_root) != "." else 0
                if depth >= max_depth:
                    dirs.clear()
                    continue

            # Filter hidden directories
            if not hidden:
                dirs[:] = [d for d in dirs if not d.startswith(".")]

            # Skip .git
            if ".git" in dirs:
                dirs.remove(".git")

            for file in files:
                # Filter hidden files
                if not hidden and file.startswith("."):
                    continue

                rel_path = str(rel_root / file) if str(rel_root) != "." else file

                # Apply glob filter
                if glob:
                    matched = False
                    for pattern in glob:
                        if fnmatch.fnmatch(rel_path, pattern):
                            matched = True
                            break
                    if not matched:
                        continue

                yield rel_path

    @staticmethod
    async def tree(cwd: str, limit: int = 50) -> str:
        """
        Generate a tree view of the directory.

        Args:
            cwd: Working directory
            limit: Maximum number of entries to show

        Returns:
            Tree-formatted string
        """
        files = []
        async for f in Ripgrep.files(cwd=cwd):
            if ".filegramengine" in f:
                continue
            files.append(f)

        # Build tree structure
        class Node:
            def __init__(self, path: list[str]):
                self.path = path
                self.children: list[Node] = []

        def get_or_create(root: Node, parts: list[str]) -> Node:
            current = root
            for i, part in enumerate(parts):
                existing = None
                for child in current.children:
                    if child.path and child.path[-1] == part:
                        existing = child
                        break
                if not existing:
                    existing = Node(current.path + [part])
                    current.children.append(existing)
                current = existing
            return current

        root = Node([])
        for file in files:
            parts = file.split(os.sep)
            get_or_create(root, parts)

        # Sort children
        def sort_node(node: Node):
            node.children.sort(key=lambda x: (not x.children, x.path[-1] if x.path else ""))
            for child in node.children:
                sort_node(child)

        sort_node(root)

        # Render tree with limit
        lines: list[str] = []
        count = [0]

        def render(node: Node, depth: int):
            if count[0] >= limit:
                return
            if node.path:
                indent = "\t" * depth
                name = node.path[-1]
                suffix = "/" if node.children else ""
                lines.append(f"{indent}{name}{suffix}")
                count[0] += 1
            for child in node.children:
                if count[0] >= limit:
                    break
                render(child, depth + (1 if node.path else 0))

        render(root, 0)

        if len(files) > limit:
            lines.append(f"\n[{len(files) - limit} more files truncated]")

        return "\n".join(lines)

    @staticmethod
    async def search(
        cwd: str,
        pattern: str,
        glob: list[str] | None = None,
        limit: int | None = None,
        follow: bool = True,
    ) -> list[RipgrepMatch]:
        """
        Search for a pattern using ripgrep.

        Args:
            cwd: Working directory
            pattern: Search pattern (regex)
            glob: Glob patterns to filter files
            limit: Maximum number of matches
            follow: Follow symlinks

        Returns:
            List of matches
        """
        rg_path = Ripgrep.get_path()
        if not rg_path:
            # Fallback to grep
            return await Ripgrep._search_fallback(cwd, pattern, glob, limit)

        args = [rg_path, "--json", "--hidden", "--glob=!.git/*"]

        if follow:
            args.append("--follow")
        if glob:
            for g in glob:
                args.append(f"--glob={g}")
        if limit:
            args.append(f"--max-count={limit}")

        args.append("--")
        args.append(pattern)

        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        stdout, _ = await proc.communicate()

        if proc.returncode != 0:
            return []

        matches = []
        for line in stdout.decode("utf-8", errors="replace").strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("type") == "match":
                    match_data = data["data"]
                    matches.append(
                        RipgrepMatch(
                            path=match_data["path"]["text"],
                            line_number=match_data["line_number"],
                            line_text=match_data["lines"]["text"],
                            absolute_offset=match_data["absolute_offset"],
                            submatches=match_data.get("submatches", []),
                        )
                    )
            except json.JSONDecodeError:
                continue

        return matches

    @staticmethod
    async def _search_fallback(
        cwd: str,
        pattern: str,
        glob: list[str] | None = None,
        limit: int | None = None,
    ) -> list[RipgrepMatch]:
        """Fallback search using grep."""
        import re

        regex = re.compile(pattern)
        matches = []
        count = 0

        async for file in Ripgrep.files(cwd=cwd, glob=glob):
            if limit and count >= limit:
                break

            filepath = Path(cwd) / file
            try:
                with open(filepath, encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(
                                RipgrepMatch(
                                    path=file,
                                    line_number=line_num,
                                    line_text=line.rstrip("\n"),
                                    absolute_offset=0,
                                    submatches=[],
                                )
                            )
                            count += 1
                            if limit and count >= limit:
                                break
            except (OSError, UnicodeDecodeError):
                continue

        return matches
