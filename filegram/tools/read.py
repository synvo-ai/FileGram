"""Read tool for reading file contents."""

from pathlib import Path
from typing import Any

from ..models.tool import ToolResult
from ..prompts import load_tool_prompt
from ..utils.truncate import truncate_output
from .base import BaseTool, ToolContext

# Load description from txt file (like OpenCode's static import)
DESCRIPTION = load_tool_prompt("read")

# Supported file extensions
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def extract_pdf_text(file_path: Path) -> str:
    """Extract text content from a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text content
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        return "[Error: pymupdf not installed. Run: pip install pymupdf]"

    try:
        doc = fitz.open(file_path)
        text_parts = []

        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(f"--- Page {page_num} ---\n{page_text}")

        doc.close()

        if not text_parts:
            return "[PDF contains no extractable text (may be scanned/image-based)]"

        return "\n\n".join(text_parts)
    except Exception as e:
        return f"[Error reading PDF: {str(e)}]"


class ReadTool(BaseTool):
    """Tool for reading file contents."""

    @property
    def name(self) -> str:
        return "read"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed, default: 1)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: 2000)",
                },
            },
            "required": ["file_path"],
        }

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        file_path_str = arguments.get("file_path", "")
        offset = arguments.get("offset", 1)
        limit = arguments.get("limit", 2000)

        if not file_path_str:
            return self._make_result(
                tool_use_id,
                "No file path provided",
                is_error=True,
            )

        try:
            file_path = context.resolve_path(file_path_str)
        except ValueError as e:
            return self._make_result(
                tool_use_id,
                str(e),
                is_error=True,
            )

        if not file_path.exists():
            return self._make_result(
                tool_use_id,
                f"File not found: {file_path}",
                is_error=True,
            )

        if not file_path.is_file():
            return self._make_result(
                tool_use_id,
                f"Not a file: {file_path}",
                is_error=True,
            )

        file_ext = file_path.suffix.lower()

        # Handle PDF files
        if file_ext in PDF_EXTENSIONS:
            return self._read_pdf(tool_use_id, file_path, context)

        # Handle image files (return info, not content)
        if file_ext in IMAGE_EXTENSIONS:
            return self._read_image_info(tool_use_id, file_path)

        # Handle text files
        return self._read_text(tool_use_id, file_path, offset, limit, context)

    def _read_pdf(
        self,
        tool_use_id: str,
        file_path: Path,
        context: ToolContext,
    ) -> ToolResult:
        """Read a PDF file and extract text."""
        try:
            content = extract_pdf_text(file_path)

            # Truncate if needed
            truncated_content = truncate_output(content, context.max_output_chars)

            return self._make_result(
                tool_use_id,
                truncated_content,
                metadata={
                    "file_path": str(file_path),
                    "file_type": "pdf",
                },
            )
        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Failed to read PDF: {str(e)}",
                is_error=True,
            )

    def _read_image_info(
        self,
        tool_use_id: str,
        file_path: Path,
    ) -> ToolResult:
        """Return info about an image file (not the actual content)."""
        try:
            file_size = file_path.stat().st_size
            size_kb = file_size / 1024

            return self._make_result(
                tool_use_id,
                f"Image file: {file_path.name}\n"
                f"Size: {size_kb:.1f} KB\n"
                f"Type: {file_path.suffix.upper()}\n"
                f"[Image content cannot be displayed as text]",
                metadata={
                    "file_path": str(file_path),
                    "file_type": "image",
                    "size_bytes": file_size,
                },
            )
        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Failed to read image info: {str(e)}",
                is_error=True,
            )

    def _read_text(
        self,
        tool_use_id: str,
        file_path: Path,
        offset: int,
        limit: int,
        context: ToolContext,
    ) -> ToolResult:
        """Read a text file."""
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            start_idx = max(0, offset - 1)
            end_idx = min(total_lines, start_idx + limit)

            selected_lines = lines[start_idx:end_idx]

            output_lines = []
            for i, line in enumerate(selected_lines, start=start_idx + 1):
                line_content = line.rstrip("\n\r")
                if len(line_content) > 2000:
                    line_content = line_content[:2000] + "..."
                output_lines.append(f"{i:6}\t{line_content}")

            output = "\n".join(output_lines)

            if end_idx < total_lines:
                output += f"\n\n[Showing lines {start_idx + 1}-{end_idx} of {total_lines} total]"

            truncated_output = truncate_output(output, context.max_output_chars)

            # Record behavioral signal
            if context.behavior_collector:
                context.behavior_collector.record_file_read(
                    file_path=str(file_path),
                    view_range=(start_idx + 1, end_idx),
                    content_length=len(truncated_output),
                )

            return self._make_result(
                tool_use_id,
                truncated_output,
                metadata={
                    "file_path": str(file_path),
                    "file_type": "text",
                    "total_lines": total_lines,
                    "lines_shown": len(selected_lines),
                },
            )

        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"Failed to read file: {str(e)}",
                is_error=True,
            )
