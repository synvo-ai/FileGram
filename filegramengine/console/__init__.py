"""Console module - enhanced terminal UI for FileGram.

Provides OpenCode-style console experience with:
- ASCII art logo
- Compact tool output formatting
- Session management commands
- Status bar display
"""

from .app import ConsoleApp
from .display import Display
from .logo import print_logo, render_logo

__all__ = [
    "ConsoleApp",
    "Display",
    "print_logo",
    "render_logo",
]
