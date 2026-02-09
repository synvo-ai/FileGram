"""MCP-related errors."""


class MCPError(Exception):
    """Base exception for MCP-related errors."""

    pass


class MCPConnectionError(MCPError):
    """Error connecting to MCP server."""

    def __init__(self, server_name: str, message: str):
        self.server_name = server_name
        super().__init__(f"Failed to connect to MCP server '{server_name}': {message}")


class MCPToolError(MCPError):
    """Error executing MCP tool."""

    def __init__(self, server_name: str, tool_name: str, message: str):
        self.server_name = server_name
        self.tool_name = tool_name
        super().__init__(f"MCP tool '{tool_name}' from '{server_name}' failed: {message}")


class MCPTimeoutError(MCPError):
    """MCP operation timed out."""

    def __init__(self, operation: str, timeout: float):
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"MCP operation '{operation}' timed out after {timeout}s")
