"""Session revert functionality for rolling back changes.

Allows reverting file changes made during a session back to a previous state.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from ..snapshot import SnapshotManager

logger = logging.getLogger(__name__)


@dataclass
class RevertPoint:
    """A point in the session that can be reverted to."""

    message_id: str
    snapshot_hash: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    description: str = ""


@dataclass
class RevertState:
    """State tracking for session revert functionality."""

    # The snapshot hash before any changes in this session
    initial_snapshot: str | None = None
    # Stack of revert points (most recent last)
    revert_points: list[RevertPoint] = field(default_factory=list)
    # Current revert state (if we've reverted)
    reverted_to: RevertPoint | None = None
    # Files that have been changed since initial snapshot
    changed_files: list[str] = field(default_factory=list)


class SessionRevert:
    """Manages session revert functionality.

    Tracks file changes during a session and allows reverting to
    previous states.
    """

    def __init__(self, snapshot_manager: SnapshotManager):
        """Initialize session revert.

        Args:
            snapshot_manager: The snapshot manager to use
        """
        self.snapshot = snapshot_manager
        self.state = RevertState()

    async def init(self) -> bool:
        """Initialize revert tracking for this session.

        Creates an initial snapshot of the current state.

        Returns:
            True if initialization succeeded
        """
        if not self.snapshot.enabled:
            return False

        # Create initial snapshot
        initial_hash = await self.snapshot.track()
        if initial_hash:
            self.state.initial_snapshot = initial_hash
            logger.debug(f"Revert tracking initialized: {initial_hash[:8]}")
            return True

        return False

    async def create_revert_point(
        self,
        message_id: str,
        description: str = "",
    ) -> RevertPoint | None:
        """Create a revert point before executing tools.

        Args:
            message_id: The message ID for this point
            description: Optional description

        Returns:
            RevertPoint if created, None otherwise
        """
        if not self.snapshot.enabled:
            return None

        snapshot_hash = await self.snapshot.track()
        if not snapshot_hash:
            return None

        point = RevertPoint(
            message_id=message_id,
            snapshot_hash=snapshot_hash,
            description=description,
        )
        self.state.revert_points.append(point)
        logger.debug(f"Created revert point: {message_id} -> {snapshot_hash[:8]}")
        return point

    async def revert_to_message(self, message_id: str) -> bool:
        """Revert file changes back to a specific message.

        Args:
            message_id: The message ID to revert to

        Returns:
            True if revert succeeded
        """
        if not self.snapshot.enabled:
            return False

        # Find the revert point for this message
        target_point = None
        for point in self.state.revert_points:
            if point.message_id == message_id:
                target_point = point
                break

        if not target_point:
            logger.warning(f"No revert point found for message: {message_id}")
            return False

        # Revert to this snapshot
        success = await self.snapshot.revert_to(target_point.snapshot_hash)
        if success:
            self.state.reverted_to = target_point
            # Remove all revert points after this one
            idx = self.state.revert_points.index(target_point)
            self.state.revert_points = self.state.revert_points[: idx + 1]
            logger.info(f"Reverted to message: {message_id}")

        return success

    async def revert_last(self) -> bool:
        """Revert to the previous revert point.

        Returns:
            True if revert succeeded
        """
        if not self.state.revert_points:
            logger.warning("No revert points available")
            return False

        if len(self.state.revert_points) < 2:
            # Revert to initial state
            if self.state.initial_snapshot:
                success = await self.snapshot.revert_to(self.state.initial_snapshot)
                if success:
                    self.state.revert_points.clear()
                return success
            return False

        # Get the second-to-last revert point
        target_point = self.state.revert_points[-2]
        return await self.revert_to_message(target_point.message_id)

    async def revert_all(self) -> bool:
        """Revert all changes back to initial state.

        Returns:
            True if revert succeeded
        """
        if not self.state.initial_snapshot:
            logger.warning("No initial snapshot available")
            return False

        success = await self.snapshot.revert_to(self.state.initial_snapshot)
        if success:
            self.state.revert_points.clear()
            self.state.reverted_to = None
            logger.info("Reverted all changes to initial state")

        return success

    async def get_diff(self) -> list[str]:
        """Get list of files changed since initial snapshot.

        Returns:
            List of changed file paths
        """
        if not self.state.initial_snapshot:
            return []

        return await self.snapshot.diff(self.state.initial_snapshot)

    async def get_diff_from_point(self, message_id: str) -> list[str]:
        """Get list of files changed since a specific message.

        Args:
            message_id: The message ID to compare from

        Returns:
            List of changed file paths
        """
        for point in self.state.revert_points:
            if point.message_id == message_id:
                return await self.snapshot.diff(point.snapshot_hash)

        return []

    def get_revert_points(self) -> list[RevertPoint]:
        """Get all available revert points.

        Returns:
            List of revert points
        """
        return self.state.revert_points.copy()

    def clear(self) -> None:
        """Clear all revert state."""
        self.state = RevertState()
