"""Output truncation utilities."""


def truncate_output(output: str, max_chars: int = 30000) -> str:
    """
    Truncate output to a maximum number of characters.

    If truncated, adds a message indicating the truncation.

    Args:
        output: The output string to truncate
        max_chars: Maximum number of characters (default: 30000)

    Returns:
        The truncated string with truncation notice if applicable
    """
    if len(output) <= max_chars:
        return output

    truncation_notice = f"\n\n[Output truncated: {len(output)} chars total, showing first {max_chars} chars]"
    available_chars = max_chars - len(truncation_notice)

    if available_chars <= 0:
        return f"[Output too large: {len(output)} chars, max {max_chars}]"

    return output[:available_chars] + truncation_notice
