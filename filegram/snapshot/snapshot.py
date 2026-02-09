"""Snapshot system for tracking file changes and enabling rollback.

Uses a shadow Git repository to track file states without affecting
the user's actual Git repository.
"""

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Patch:
    """Represents a set of file changes."""

    hash: str
    files: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class Snapshot:
    """A snapshot of the file system state."""

    hash: str
    timestamp: float
    files_changed: int = 0


class SnapshotManager:
    """Manages file snapshots using a shadow Git repository.

    Creates snapshots before tool executions and allows reverting
    to previous states when errors occur.
    """

    def __init__(self, project_dir: Path, enabled: bool = True):
        """Initialize snapshot manager.

        Args:
            project_dir: The project directory to track
            enabled: Whether snapshot tracking is enabled
        """
        self.project_dir = project_dir.resolve()
        self.enabled = enabled
        self._shadow_git_dir = self.project_dir / ".synvocode" / ".snapshot-git"
        self._initialized = False

    async def init(self) -> bool:
        """Initialize the shadow Git repository.

        Returns:
            True if initialization succeeded
        """
        if not self.enabled:
            return False

        # Check if we're in a git repo (required for snapshot to work)
        if not (self.project_dir / ".git").exists():
            logger.debug("Not a git repository, snapshots disabled")
            self.enabled = False
            return False

        try:
            # Create shadow git directory
            self._shadow_git_dir.mkdir(parents=True, exist_ok=True)

            # Check if already initialized
            if (self._shadow_git_dir / "HEAD").exists():
                self._initialized = True
                return True

            # Initialize shadow git repo
            result = await self._run_git(["init"], env_override=True)
            if result.returncode != 0:
                logger.warning(f"Failed to init shadow git: {result.stderr}")
                return False

            # Configure to not convert line endings
            await self._run_git(["config", "core.autocrlf", "false"], env_override=True)

            self._initialized = True
            logger.info(f"Snapshot system initialized at {self._shadow_git_dir}")
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize snapshot system: {e}")
            self.enabled = False
            return False

    async def _run_git(
        self,
        args: list[str],
        env_override: bool = False,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a git command.

        Args:
            args: Git command arguments
            env_override: If True, set GIT_DIR and GIT_WORK_TREE
            cwd: Working directory

        Returns:
            CompletedProcess result
        """
        cmd = ["git"]

        if env_override:
            cmd.extend(
                [
                    "--git-dir",
                    str(self._shadow_git_dir),
                    "--work-tree",
                    str(self.project_dir),
                ]
            )

        cmd.extend(args)

        env = os.environ.copy()
        if env_override:
            env["GIT_DIR"] = str(self._shadow_git_dir)
            env["GIT_WORK_TREE"] = str(self.project_dir)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or self.project_dir,
            env=env,
        )
        stdout, stderr = await proc.communicate()

        return subprocess.CompletedProcess(
            cmd,
            proc.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    async def track(self) -> str | None:
        """Create a snapshot of current file state.

        Returns:
            Snapshot hash if successful, None otherwise
        """
        if not self.enabled:
            return None

        if not self._initialized:
            if not await self.init():
                return None

        try:
            # Stage all files
            result = await self._run_git(["add", "."], env_override=True)
            if result.returncode != 0:
                logger.warning(f"Failed to stage files: {result.stderr}")
                return None

            # Create tree object (without committing)
            result = await self._run_git(["write-tree"], env_override=True)
            if result.returncode != 0:
                logger.warning(f"Failed to write tree: {result.stderr}")
                return None

            hash_value = result.stdout.strip()
            logger.debug(f"Created snapshot: {hash_value[:8]}")
            return hash_value

        except Exception as e:
            logger.warning(f"Failed to create snapshot: {e}")
            return None

    async def diff(self, from_hash: str) -> list[str]:
        """Get list of files changed since a snapshot.

        Args:
            from_hash: The snapshot hash to compare against

        Returns:
            List of changed file paths
        """
        if not self.enabled or not from_hash:
            return []

        try:
            # Stage current state
            await self._run_git(["add", "."], env_override=True)

            # Get diff
            result = await self._run_git(
                [
                    "-c",
                    "core.quotepath=false",
                    "diff",
                    "--no-ext-diff",
                    "--name-only",
                    from_hash,
                    "--",
                    ".",
                ],
                env_override=True,
            )

            if result.returncode != 0:
                logger.warning(f"Failed to get diff: {result.stderr}")
                return []

            files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            return files

        except Exception as e:
            logger.warning(f"Failed to get diff: {e}")
            return []

    async def patch(self, from_hash: str) -> Patch:
        """Create a patch from a snapshot hash.

        Args:
            from_hash: The snapshot hash

        Returns:
            Patch object with changed files
        """
        files = await self.diff(from_hash)
        return Patch(hash=from_hash, files=files)

    async def revert(self, patches: list[Patch]) -> bool:
        """Revert file changes from patches.

        Args:
            patches: List of patches to revert (applied in reverse order)

        Returns:
            True if revert succeeded
        """
        if not self.enabled or not patches:
            return False

        try:
            # Checkout files from the snapshot
            for patch in patches:
                for file_path in patch.files:
                    full_path = self.project_dir / file_path

                    # Try to restore from snapshot
                    result = await self._run_git(["checkout", patch.hash, "--", file_path], env_override=True)

                    if result.returncode != 0:
                        # File might not exist in snapshot (was created)
                        # Try to remove it
                        if full_path.exists():
                            try:
                                full_path.unlink()
                                logger.debug(f"Removed file: {file_path}")
                            except Exception as e:
                                logger.warning(f"Failed to remove {file_path}: {e}")
                    else:
                        logger.debug(f"Reverted file: {file_path}")

            logger.info(f"Reverted {sum(len(p.files) for p in patches)} files")
            return True

        except Exception as e:
            logger.warning(f"Failed to revert: {e}")
            return False

    async def revert_to(self, hash_value: str) -> bool:
        """Revert all files to a specific snapshot.

        Args:
            hash_value: The snapshot hash to revert to

        Returns:
            True if revert succeeded
        """
        if not self.enabled or not hash_value:
            return False

        try:
            # Get list of changed files
            files = await self.diff(hash_value)

            if not files:
                logger.debug("No files to revert")
                return True

            # Checkout all changed files from snapshot
            result = await self._run_git(["checkout", hash_value, "--", "."], env_override=True)

            if result.returncode != 0:
                logger.warning(f"Failed to revert: {result.stderr}")
                return False

            logger.info(f"Reverted to snapshot {hash_value[:8]}")
            return True

        except Exception as e:
            logger.warning(f"Failed to revert to snapshot: {e}")
            return False

    async def cleanup(self, max_age_days: int = 7) -> None:
        """Clean up old snapshot data.

        Args:
            max_age_days: Maximum age of snapshots to keep
        """
        if not self.enabled or not self._initialized:
            return

        try:
            result = await self._run_git(["gc", f"--prune={max_age_days}.days"], env_override=True)

            if result.returncode == 0:
                logger.debug(f"Cleaned up snapshots older than {max_age_days} days")

        except Exception as e:
            logger.warning(f"Failed to cleanup snapshots: {e}")
