"""Permission control module."""

from .permission import Permission, PermissionAction, PermissionError, PermissionRule

__all__ = ["Permission", "PermissionAction", "PermissionRule", "PermissionError"]
