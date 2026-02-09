"""Storage module for persistent data storage."""

from .storage import NotFoundError, Storage

__all__ = [
    "Storage",
    "NotFoundError",
]
