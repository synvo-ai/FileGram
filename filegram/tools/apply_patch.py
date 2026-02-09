"""ApplyPatch tool for applying patches to files."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from .base import BaseTool, ToolContext

# Load description from txt file
DESCRIPTION = load_tool_prompt("apply_patch")


@dataclass
class PatchHunk:
    """Represents a single file operation in a patch."""

    path: str
    type: str  # "add", "update", "delete"
    contents: str = ""
    move_path: str | None = None
    chunks: list[dict] | None = None


def parse_patch(patch_text: str) -> list[PatchHunk]:
    """Parse patch text into hunks.

    Patch format:
    *** Begin Patch
    *** Add File: <path>
    +content lines
    *** Update File: <path>
    *** Move to: <new_path>  (optional)
    @@ context line
    -removed line
    +added line
    *** Delete File: <path>
    *** End Patch
    """
    hunks = []

    # Normalize line endings
    patch_text = patch_text.replace("\r\n", "\n").replace("\r", "\n")

    # Check for patch markers
    if "*** Begin Patch" not in patch_text:
        raise ValueError("Patch must start with '*** Begin Patch'")
    if "*** End Patch" not in patch_text:
        raise ValueError("Patch must end with '*** End Patch'")

    # Extract content between markers
    start_idx = patch_text.index("*** Begin Patch") + len("*** Begin Patch")
    end_idx = patch_text.index("*** End Patch")
    content = patch_text[start_idx:end_idx].strip()

    if not content:
        return hunks

    # Parse file operations
    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Add File
        if line.startswith("*** Add File:"):
            path = line[len("*** Add File:") :].strip()
            contents = []
            i += 1

            while i < len(lines) and not lines[i].startswith("***"):
                if lines[i].startswith("+"):
                    contents.append(lines[i][1:])
                i += 1

            hunks.append(
                PatchHunk(
                    path=path,
                    type="add",
                    contents="\n".join(contents),
                )
            )
            continue

        # Delete File
        elif line.startswith("*** Delete File:"):
            path = line[len("*** Delete File:") :].strip()
            hunks.append(
                PatchHunk(
                    path=path,
                    type="delete",
                )
            )
            i += 1
            continue

        # Update File
        elif line.startswith("*** Update File:"):
            path = line[len("*** Update File:") :].strip()
            move_path = None
            chunks = []
            i += 1

            # Check for move
            if i < len(lines) and lines[i].startswith("*** Move to:"):
                move_path = lines[i][len("*** Move to:") :].strip()
                i += 1

            # Parse update chunks
            current_chunk = None

            while i < len(lines) and not lines[i].startswith("*** "):
                line = lines[i]

                # Context line (anchor)
                if line.startswith("@@"):
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = {
                        "context": line[2:].strip(),
                        "removals": [],
                        "additions": [],
                    }
                elif current_chunk is not None:
                    if line.startswith("-"):
                        current_chunk["removals"].append(line[1:])
                    elif line.startswith("+"):
                        current_chunk["additions"].append(line[1:])
                    elif line.startswith(" "):
                        # Context line within chunk
                        pass

                i += 1

            if current_chunk:
                chunks.append(current_chunk)

            hunks.append(
                PatchHunk(
                    path=path,
                    type="update",
                    move_path=move_path,
                    chunks=chunks,
                )
            )
            continue

        else:
            i += 1

    return hunks


def apply_update_chunks(content: str, chunks: list[dict]) -> str:
    """Apply update chunks to file content."""
    lines = content.split("\n")

    for chunk in chunks:
        context = chunk.get("context", "")
        removals = chunk.get("removals", [])
        additions = chunk.get("additions", [])

        # Find the context line
        context_idx = -1
        for i, line in enumerate(lines):
            if line.strip() == context.strip():
                context_idx = i
                break

        if context_idx == -1:
            # Try fuzzy match
            for i, line in enumerate(lines):
                if context.strip() in line or line in context:
                    context_idx = i
                    break

        if context_idx == -1:
            raise ValueError(f"Could not find context line: {context}")

        # Find and remove the lines to be removed
        remove_start = context_idx + 1
        for removal in removals:
            found = False
            for j in range(remove_start, len(lines)):
                if lines[j].strip() == removal.strip():
                    lines.pop(j)
                    found = True
                    break
            if not found:
                # Try to find it anywhere after context
                for j in range(context_idx, len(lines)):
                    if lines[j].strip() == removal.strip():
                        lines.pop(j)
                        found = True
                        break

        # Insert additions after context
        insert_idx = context_idx + 1
        for addition in additions:
            lines.insert(insert_idx, addition)
            insert_idx += 1

    return "\n".join(lines)


class ApplyPatchTool(BaseTool):
    """Tool for applying patches to files.

    Uses a simplified patch format that supports:
    - Adding new files
    - Updating existing files (with optional move/rename)
    - Deleting files
    """

    @property
    def name(self) -> str:
        return "apply_patch"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "patch_text": {
                    "type": "string",
                    "description": "The full patch text that describes all changes to be made",
                },
            },
            "required": ["patch_text"],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        patch_text = arguments.get("patch_text", "")

        if not patch_text:
            return self._make_result(
                tool_use_id,
                "No patch text provided",
                is_error=True,
            )

        try:
            # Parse the patch
            hunks = parse_patch(patch_text)

            if not hunks:
                return self._make_result(
                    tool_use_id,
                    "No valid hunks found in patch",
                    is_error=True,
                )

            # Apply changes
            changes = []

            for hunk in hunks:
                file_path = Path(hunk.path)
                if not file_path.is_absolute():
                    file_path = context.target_directory / file_path
                file_path = file_path.resolve()

                if hunk.type == "add":
                    # Create new file
                    file_path.parent.mkdir(parents=True, exist_ok=True)

                    content = hunk.contents
                    if content and not content.endswith("\n"):
                        content += "\n"

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)

                    changes.append(f"A {file_path.relative_to(context.target_directory)}")

                elif hunk.type == "delete":
                    # Delete file
                    if file_path.exists():
                        os.remove(file_path)
                        changes.append(f"D {file_path.relative_to(context.target_directory)}")
                    else:
                        return self._make_result(
                            tool_use_id,
                            f"File to delete not found: {file_path}",
                            is_error=True,
                        )

                elif hunk.type == "update":
                    # Update file
                    if not file_path.exists():
                        return self._make_result(
                            tool_use_id,
                            f"File to update not found: {file_path}",
                            is_error=True,
                        )

                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()

                    # Apply chunks
                    if hunk.chunks:
                        content = apply_update_chunks(content, hunk.chunks)

                    # Handle move
                    target_path = file_path
                    if hunk.move_path:
                        target_path = Path(hunk.move_path)
                        if not target_path.is_absolute():
                            target_path = context.target_directory / target_path
                        target_path = target_path.resolve()
                        target_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(content)

                    if hunk.move_path and target_path != file_path:
                        os.remove(file_path)
                        changes.append(
                            f"M {file_path.relative_to(context.target_directory)} -> "
                            f"{target_path.relative_to(context.target_directory)}"
                        )
                    else:
                        changes.append(f"M {file_path.relative_to(context.target_directory)}")

            output = "Success. Updated the following files:\n" + "\n".join(changes)

            # Record behavioral signals for each file change
            if context.behavior_collector:
                for change in changes:
                    if change.startswith("A "):
                        # File added
                        file_path_str = change[2:].split(" -> ")[0] if " -> " in change else change[2:]
                        context.behavior_collector.record_file_write(
                            file_path=str(context.target_directory / file_path_str),
                            operation="create",
                            content_length=0,  # Content length not tracked in patch
                        )
                    elif change.startswith("M "):
                        # File modified
                        file_path_str = change[2:].split(" -> ")[0] if " -> " in change else change[2:]
                        context.behavior_collector.record_file_edit(
                            file_path=str(context.target_directory / file_path_str),
                            edit_tool="apply_patch",
                            lines_added=0,
                            lines_deleted=0,
                            diff_summary="patch applied",
                        )
                    elif change.startswith("D "):
                        # File deleted - record as edit with special diff_summary
                        file_path_str = change[2:]
                        context.behavior_collector.record_file_edit(
                            file_path=str(context.target_directory / file_path_str),
                            edit_tool="apply_patch",
                            lines_added=0,
                            lines_deleted=0,
                            diff_summary="file deleted",
                        )

            return self._make_result(
                tool_use_id,
                output,
                metadata={
                    "files_changed": len(changes),
                    "changes": changes,
                },
            )

        except ValueError as e:
            return self._make_result(
                tool_use_id,
                f"Patch parsing error: {str(e)}",
                is_error=True,
            )
        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Failed to apply patch: {str(e)}",
                is_error=True,
            )
