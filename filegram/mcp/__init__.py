"""MCP (Model Context Protocol) integration module.

This module provides support for connecting to MCP servers and using
their tools within FileGram.
"""

from .client import MCPClientManager, MCPConfig, MCPStatus, MCPToolWrapper
from .errors import MCPConnectionError, MCPError, MCPToolError

__all__ = [
    "MCPClientManager",
    "MCPConfig",
    "MCPStatus",
    "MCPToolWrapper",
    "MCPError",
    "MCPConnectionError",
    "MCPToolError",
]
