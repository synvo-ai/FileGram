"""Snapshot module for file state tracking and rollback."""

from .snapshot import Patch, Snapshot, SnapshotManager

__all__ = ["Snapshot", "SnapshotManager", "Patch"]
