"""Content parsers for semantic channel extraction.

TextParser handles markdown/txt/csv/json/yaml/xml.
MultimodalParser is a stub for future image/pdf/pptx support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .schema import MULTIMODAL_TYPES, TEXT_PARSEABLE_TYPES


class ContentParser(ABC):
    """Abstract base for content parsing."""

    @abstractmethod
    def can_parse(self, file_type: str) -> bool:
        """Return True if this parser handles the given file type."""

    @abstractmethod
    def parse(self, content: str, file_type: str) -> str:
        """Parse content and return a normalized text representation."""


class TextParser(ContentParser):
    """Handles text-based file types: md, txt, csv, json, yaml, xml, etc."""

    def can_parse(self, file_type: str) -> bool:
        return file_type.lower() in TEXT_PARSEABLE_TYPES

    def parse(self, content: str, file_type: str) -> str:
        return content


class MultimodalParser(ContentParser):
    """Stub parser for non-text files (images, PDFs, presentations).

    Returns a placeholder string. Will be replaced with VLM-based
    parsing in a future version.
    """

    def can_parse(self, file_type: str) -> bool:
        return file_type.lower() in MULTIMODAL_TYPES

    def parse(self, content: str, file_type: str) -> str:
        return f"[{file_type.upper()} file — multimodal parsing not yet implemented]"


class ParserRegistry:
    """Dispatches to the correct parser by file type."""

    def __init__(self):
        self._parsers: list[ContentParser] = [
            TextParser(),
            MultimodalParser(),
        ]

    def parse(self, content: str, file_type: str) -> str:
        for parser in self._parsers:
            if parser.can_parse(file_type):
                return parser.parse(content, file_type)
        # Fallback: return raw content (assume text)
        return content
