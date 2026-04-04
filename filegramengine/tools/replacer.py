"""Advanced text replacement strategies for edit tool.

This module implements 9 different replacement strategies, from strict to flexible,
matching OpenCode's edit functionality.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Generator

# Type alias for replacer function
Replacer = Callable[[str, str], Generator[str, None, None]]


def levenshtein(a: str, b: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(a) < len(b):
        return levenshtein(b, a)

    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))

    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings (0.0 to 1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    max_len = max(len(a), len(b))
    distance = levenshtein(a, b)
    return 1.0 - (distance / max_len)


# ============== Replacer Strategies ==============


def simple_replacer(content: str, find: str) -> Generator[str, None, None]:
    """
    Strategy 1: Simple exact match replacer.

    The most basic replacer - returns the search string as-is.
    """
    yield find


def line_trimmed_replacer(content: str, find: str) -> Generator[str, None, None]:
    """
    Strategy 2: Line-trimmed replacer.

    Handles indentation changes in multi-line text.
    Compares trimmed lines but returns original text.
    """
    original_lines = content.split("\n")
    search_lines = find.split("\n")

    if len(search_lines) == 0:
        return

    for i in range(len(original_lines) - len(search_lines) + 1):
        matches = True
        for j, search_line in enumerate(search_lines):
            if i + j >= len(original_lines):
                matches = False
                break
            original_trimmed = original_lines[i + j].strip()
            search_trimmed = search_line.strip()
            if original_trimmed != search_trimmed:
                matches = False
                break

        if matches:
            # Calculate original text position
            start_idx = sum(len(line) + 1 for line in original_lines[:i])
            end_idx = start_idx + sum(len(line) + 1 for line in original_lines[i : i + len(search_lines)]) - 1
            yield content[start_idx:end_idx]


def block_anchor_replacer(content: str, find: str) -> Generator[str, None, None]:
    """
    Strategy 3: Block anchor replacer.

    Uses first and last lines as anchors, fuzzy matches middle content
    using Levenshtein distance.
    """
    single_candidate_threshold = 0.0
    multiple_candidates_threshold = 0.3

    search_lines = find.split("\n")
    if len(search_lines) < 3:
        return

    original_lines = content.split("\n")
    first_line = search_lines[0].strip()
    last_line = search_lines[-1].strip()

    # Find all potential blocks
    candidates: list[tuple[int, int, float]] = []  # (start, end, similarity)

    for i in range(len(original_lines)):
        if original_lines[i].strip() != first_line:
            continue

        # Find matching last line
        for j in range(i + len(search_lines) - 1, len(original_lines)):
            if original_lines[j].strip() != last_line:
                continue

            # Calculate similarity of middle lines
            middle_original = "\n".join(original_lines[i + 1 : j])
            middle_search = "\n".join(search_lines[1:-1])

            if not middle_search and not middle_original:
                sim = 1.0
            elif not middle_search or not middle_original:
                sim = 0.0
            else:
                sim = similarity(middle_original.strip(), middle_search.strip())

            candidates.append((i, j, sim))

    if len(candidates) == 0:
        return

    if len(candidates) == 1:
        start, end, sim = candidates[0]
        if sim >= single_candidate_threshold:
            start_idx = sum(len(line) + 1 for line in original_lines[:start])
            end_idx = start_idx + sum(len(line) + 1 for line in original_lines[start : end + 1]) - 1
            yield content[start_idx:end_idx]
    else:
        # Pick best candidate
        best = max(candidates, key=lambda x: x[2])
        if best[2] >= multiple_candidates_threshold:
            start, end, _ = best
            start_idx = sum(len(line) + 1 for line in original_lines[:start])
            end_idx = start_idx + sum(len(line) + 1 for line in original_lines[start : end + 1]) - 1
            yield content[start_idx:end_idx]


def whitespace_normalized_replacer(content: str, find: str) -> Generator[str, None, None]:
    """
    Strategy 4: Whitespace normalized replacer.

    Converts all consecutive whitespace to single spaces before matching.
    """

    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    normalized_find = normalize(find)

    # Try line-by-line matching
    original_lines = content.split("\n")
    search_lines = find.split("\n")

    for i in range(len(original_lines) - len(search_lines) + 1):
        matches = True
        for j, search_line in enumerate(search_lines):
            if normalize(original_lines[i + j]) != normalize(search_line):
                matches = False
                break

        if matches:
            start_idx = sum(len(line) + 1 for line in original_lines[:i])
            end_idx = start_idx + sum(len(line) + 1 for line in original_lines[i : i + len(search_lines)]) - 1
            yield content[start_idx:end_idx]
            return

    # Try substring matching within single line
    for i, line in enumerate(original_lines):
        normalized_line = normalize(line)
        if normalized_find in normalized_line:
            yield line


def indentation_flexible_replacer(content: str, find: str) -> Generator[str, None, None]:
    """
    Strategy 5: Indentation flexible replacer.

    Ignores overall indentation differences while preserving relative indentation.
    """

    def remove_min_indent(text: str) -> str:
        lines = text.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        if not non_empty_lines:
            return text

        min_indent = float("inf")
        for line in non_empty_lines:
            match = re.match(r"^(\s*)", line)
            if match:
                min_indent = min(min_indent, len(match.group(1)))

        if min_indent == float("inf") or min_indent == 0:
            return text

        return "\n".join(line[min_indent:] if line.strip() else line for line in lines)

    normalized_find = remove_min_indent(find)
    original_lines = content.split("\n")
    search_lines = normalized_find.split("\n")

    for i in range(len(original_lines) - len(search_lines) + 1):
        # Get the block and normalize it
        block_lines = original_lines[i : i + len(search_lines)]
        normalized_block = remove_min_indent("\n".join(block_lines))

        if normalized_block == normalized_find:
            start_idx = sum(len(line) + 1 for line in original_lines[:i])
            end_idx = start_idx + sum(len(line) + 1 for line in block_lines) - 1
            yield content[start_idx:end_idx]


def escape_normalized_replacer(content: str, find: str) -> Generator[str, None, None]:
    """
    Strategy 6: Escape normalized replacer.

    Handles escape sequence inconsistencies.
    """
    escape_map = {
        r"\n": "\n",
        r"\t": "\t",
        r"\r": "\r",
        r"\'": "'",
        r"\"": '"',
        r"\`": "`",
        r"\\": "\\",
        r"\$": "$",
    }

    def unescape(text: str) -> str:
        result = text
        for escaped, unescaped in escape_map.items():
            result = result.replace(escaped, unescaped)
        return result

    # Try unescaped version
    unescaped_find = unescape(find)
    if unescaped_find in content:
        yield unescaped_find

    # Try escaped version in content
    for escaped, unescaped in escape_map.items():
        if unescaped in find:
            escaped_find = find.replace(unescaped, escaped)
            if escaped_find in content:
                yield escaped_find


def multi_occurrence_replacer(content: str, find: str) -> Generator[str, None, None]:
    """
    Strategy 7: Multi-occurrence replacer.

    Yields all exact matches of the search string.
    Used with replace_all parameter.
    """
    start_index = 0
    while True:
        index = content.find(find, start_index)
        if index == -1:
            break
        yield find
        start_index = index + len(find)


def trimmed_boundary_replacer(content: str, find: str) -> Generator[str, None, None]:
    """
    Strategy 8: Trimmed boundary replacer.

    Handles extra whitespace at boundaries.
    """
    trimmed_find = find.strip()

    if trimmed_find != find and trimmed_find in content:
        yield trimmed_find


def context_aware_replacer(content: str, find: str) -> Generator[str, None, None]:
    """
    Strategy 9: Context-aware replacer.

    Uses first and last lines as context anchors.
    Requires at least 50% similarity in middle lines.
    """
    similarity_threshold = 0.5

    search_lines = find.split("\n")
    if len(search_lines) < 3:
        return

    original_lines = content.split("\n")
    first_line = search_lines[0].strip()
    last_line = search_lines[-1].strip()

    for i in range(len(original_lines)):
        if original_lines[i].strip() != first_line:
            continue

        # Look for last line
        for j in range(i + 2, len(original_lines)):
            if original_lines[j].strip() != last_line:
                continue

            # Check middle line similarity
            middle_original = original_lines[i + 1 : j]
            middle_search = search_lines[1:-1]

            if len(middle_search) == 0:
                # No middle lines to compare
                start_idx = sum(len(line) + 1 for line in original_lines[:i])
                end_idx = start_idx + sum(len(line) + 1 for line in original_lines[i : j + 1]) - 1
                yield content[start_idx:end_idx]
                return

            # Count matching lines
            total_non_empty = 0
            matching = 0

            for k, search_line in enumerate(middle_search):
                if not search_line.strip():
                    continue
                total_non_empty += 1

                if k < len(middle_original):
                    if middle_original[k].strip() == search_line.strip():
                        matching += 1

            if total_non_empty == 0 or (matching / total_non_empty) >= similarity_threshold:
                start_idx = sum(len(line) + 1 for line in original_lines[:i])
                end_idx = start_idx + sum(len(line) + 1 for line in original_lines[i : j + 1]) - 1
                yield content[start_idx:end_idx]
                return


# ============== Main Replace Function ==============

# Ordered list of replacers from strictest to most flexible
REPLACERS: list[Replacer] = [
    simple_replacer,
    line_trimmed_replacer,
    block_anchor_replacer,
    whitespace_normalized_replacer,
    indentation_flexible_replacer,
    escape_normalized_replacer,
    trimmed_boundary_replacer,
    context_aware_replacer,
    multi_occurrence_replacer,
]


def replace(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """
    Replace old_string with new_string in content using progressive matching strategies.

    Tries each replacer strategy in order from strictest to most flexible.

    Args:
        content: The original file content
        old_string: The text to find and replace
        new_string: The replacement text
        replace_all: If True, replace all occurrences

    Returns:
        Modified content with replacement(s) applied

    Raises:
        ValueError: If old_string not found or multiple ambiguous matches
    """
    not_found = True

    for replacer in REPLACERS:
        for search in replacer(content, old_string):
            index = content.find(search)
            if index == -1:
                continue

            not_found = False

            if replace_all:
                return content.replace(search, new_string)

            # Check for uniqueness
            last_index = content.rfind(search)
            if index != last_index:
                # Multiple matches - try next replacer for more specificity
                continue

            # Single unique match - perform replacement
            return content[:index] + new_string + content[index + len(search) :]

    if not_found:
        raise ValueError("old_string not found in content. Make sure you're using the exact text from the file.")

    raise ValueError(
        "Found multiple matches for old_string. Provide more surrounding lines for context to make the match unique."
    )


def trim_diff(diff: str) -> str:
    """
    Remove unnecessary indentation from diff output for better readability.
    """
    lines = diff.split("\n")

    # Extract content lines (+, -, space prefix)
    content_lines = [
        line
        for line in lines
        if (line.startswith("+") or line.startswith("-") or line.startswith(" "))
        and not line.startswith("---")
        and not line.startswith("+++")
    ]

    if not content_lines:
        return diff

    # Calculate minimum indentation
    min_indent = float("inf")
    for line in content_lines:
        content = line[1:]  # Skip prefix
        if content.strip():
            match = re.match(r"^(\s*)", content)
            if match:
                min_indent = min(min_indent, len(match.group(1)))

    if min_indent == float("inf") or min_indent == 0:
        return diff

    # Remove common indentation
    trimmed_lines = []
    for line in lines:
        if (
            (line.startswith("+") or line.startswith("-") or line.startswith(" "))
            and not line.startswith("---")
            and not line.startswith("+++")
        ):
            prefix = line[0]
            content = line[1:]
            trimmed_lines.append(prefix + content[min_indent:])
        else:
            trimmed_lines.append(line)

    return "\n".join(trimmed_lines)


__all__ = [
    "replace",
    "trim_diff",
    "levenshtein",
    "similarity",
    "REPLACERS",
    # Individual replacers
    "simple_replacer",
    "line_trimmed_replacer",
    "block_anchor_replacer",
    "whitespace_normalized_replacer",
    "indentation_flexible_replacer",
    "escape_normalized_replacer",
    "multi_occurrence_replacer",
    "trimmed_boundary_replacer",
    "context_aware_replacer",
]
