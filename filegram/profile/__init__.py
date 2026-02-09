"""Profile module for agent personalities and work habits."""

from .loader import (
    Profile,
    ProfileLoader,
    get_current_profile,
    get_profile_loader,
    list_profiles,
    load_profile,
    set_current_profile,
)

__all__ = [
    "Profile",
    "ProfileLoader",
    "get_profile_loader",
    "load_profile",
    "list_profiles",
    "get_current_profile",
    "set_current_profile",
]
