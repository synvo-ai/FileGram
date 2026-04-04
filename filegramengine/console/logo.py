"""ASCII art logo for FileGram."""

from rich.console import Console
from rich.text import Text

# Compact block-style logo
LOGO_LINES = [
    "███████╗██╗   ██╗███╗   ██╗██╗   ██╗ ██████╗ ",
    "██╔════╝╚██╗ ██╔╝████╗  ██║██║   ██║██╔═══██╗",
    "███████╗ ╚████╔╝ ██╔██╗ ██║██║   ██║██║   ██║",
    "╚════██║  ╚██╔╝  ██║╚██╗██║╚██╗ ██╔╝██║   ██║",
    "███████║   ██║   ██║ ╚████║ ╚████╔╝ ╚██████╔╝",
    "╚══════╝   ╚═╝   ╚═╝  ╚═══╝  ╚═══╝   ╚═════╝ ",
]


def render_logo(pad: str = "") -> str:
    """Render the FileGram logo.

    Returns the full logo string.
    """
    return "\n".join(LOGO_LINES)


def print_logo(console: Console, pad: str = "") -> None:
    """Print the FileGram ASCII art logo to the console.

    Args:
        console: Rich Console instance
        pad: Left padding string (unused, kept for compatibility)
    """
    # Create gradient effect with cyan/blue colors
    logo_str = render_logo()
    text = Text(logo_str)
    text.stylize("bold cyan")
    console.print(text)
