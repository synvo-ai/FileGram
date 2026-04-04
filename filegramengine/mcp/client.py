"""MCP client implementation for FileGram."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..models.tool import ToolDefinition, ToolResult
from ..tools.base import BaseTool, ToolContext
from .errors import MCPConnectionError, MCPTimeoutError, MCPToolError

logger = logging.getLogger(__name__)

# Default timeout for MCP operations
DEFAULT_TIMEOUT = 30.0


class MCPServerType(Enum):
    """Type of MCP server."""

    LOCAL = "local"  # Local command (stdio transport)
    REMOTE = "remote"  # Remote HTTP/SSE server


@dataclass
class MCPConfig:
    """Configuration for an MCP server."""

    name: str
    type: MCPServerType
    # For local servers
    command: list[str] | None = None
    environment: dict[str, str] | None = None
    # For remote servers
    url: str | None = None
    headers: dict[str, str] | None = None
    # Common
    enabled: bool = True
    timeout: float = DEFAULT_TIMEOUT


@dataclass
class MCPStatus:
    """Status of an MCP connection."""

    class State(Enum):
        CONNECTED = "connected"
        DISABLED = "disabled"
        FAILED = "failed"
        CONNECTING = "connecting"

    state: State
    error: str | None = None


@dataclass
class MCPTool:
    """Tool definition from an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


class MCPToolWrapper(BaseTool):
    """Wrapper that exposes an MCP tool as a FileGram tool."""

    def __init__(self, mcp_tool: MCPTool, client_manager: MCPClientManager):
        self._mcp_tool = mcp_tool
        self._client_manager = client_manager
        # Create sanitized name: server_toolname
        sanitized_server = re.sub(r"[^a-zA-Z0-9_-]", "_", mcp_tool.server_name)
        sanitized_tool = re.sub(r"[^a-zA-Z0-9_-]", "_", mcp_tool.name)
        self._name = f"mcp__{sanitized_server}__{sanitized_tool}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"[MCP: {self._mcp_tool.server_name}] {self._mcp_tool.description}"

    @property
    def parameters(self) -> dict[str, Any]:
        schema = self._mcp_tool.input_schema.copy()
        # Ensure it's a valid JSON schema object
        if "type" not in schema:
            schema["type"] = "object"
        if "properties" not in schema:
            schema["properties"] = {}
        return schema

    async def execute(
        self,
        tool_use_id: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        try:
            result = await self._client_manager.call_tool(
                self._mcp_tool.server_name,
                self._mcp_tool.name,
                arguments,
            )
            return self._make_result(
                tool_use_id,
                result,
                metadata={
                    "mcp_server": self._mcp_tool.server_name,
                    "mcp_tool": self._mcp_tool.name,
                },
            )
        except Exception as e:
            return self._make_result(
                tool_use_id,
                f"MCP tool error: {str(e)}",
                is_error=True,
            )


class MCPClientManager:
    """Manages connections to MCP servers.

    This class handles:
    - Loading MCP configuration
    - Connecting to MCP servers (local via stdio, remote via HTTP)
    - Listing available tools
    - Calling tools on connected servers
    """

    def __init__(self, project_dir: str | None = None):
        self._project_dir = Path(project_dir) if project_dir else Path.cwd()
        self._configs: dict[str, MCPConfig] = {}
        self._status: dict[str, MCPStatus] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._tools: dict[str, list[MCPTool]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the MCP manager by loading configs and connecting."""
        if self._initialized:
            return

        # Load configuration
        await self._load_config()

        # Connect to enabled servers
        for name, config in self._configs.items():
            if config.enabled:
                await self.connect(name)

        self._initialized = True

    async def _load_config(self) -> None:
        """Load MCP configuration from config files.

        Looks for configuration in:
        1. .filegramengine/mcp.json (project level)
        3. Environment variable FILEGRAMENGINE_MCP_CONFIG
        """
        config_paths = [
            self._project_dir / ".filegramengine" / "mcp.json",
            Path.home() / ".filegramengine" / "mcp.json",
        ]

        # Check environment variable
        env_config = os.environ.get("FILEGRAMENGINE_MCP_CONFIG")
        if env_config:
            config_paths.insert(0, Path(env_config))

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        data = json.load(f)
                    self._parse_config(data)
                    logger.info(f"Loaded MCP config from {config_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load MCP config from {config_path}: {e}")

    def _parse_config(self, data: dict[str, Any]) -> None:
        """Parse MCP configuration data."""
        mcp_config = data.get("mcp", data.get("mcpServers", {}))

        for name, server_config in mcp_config.items():
            if isinstance(server_config, dict):
                server_type = server_config.get("type", "local")

                if server_type == "local" or "command" in server_config:
                    # Local server with command
                    command = server_config.get("command", [])
                    if isinstance(command, str):
                        command = [command]

                    self._configs[name] = MCPConfig(
                        name=name,
                        type=MCPServerType.LOCAL,
                        command=command,
                        environment=server_config.get("environment", {}),
                        enabled=server_config.get("enabled", True),
                        timeout=server_config.get("timeout", DEFAULT_TIMEOUT),
                    )
                elif server_type == "remote" or "url" in server_config:
                    # Remote server
                    self._configs[name] = MCPConfig(
                        name=name,
                        type=MCPServerType.REMOTE,
                        url=server_config.get("url"),
                        headers=server_config.get("headers", {}),
                        enabled=server_config.get("enabled", True),
                        timeout=server_config.get("timeout", DEFAULT_TIMEOUT),
                    )

    async def connect(self, name: str) -> MCPStatus:
        """Connect to an MCP server."""
        config = self._configs.get(name)
        if not config:
            status = MCPStatus(
                state=MCPStatus.State.FAILED,
                error=f"No configuration found for server '{name}'",
            )
            self._status[name] = status
            return status

        if not config.enabled:
            status = MCPStatus(state=MCPStatus.State.DISABLED)
            self._status[name] = status
            return status

        self._status[name] = MCPStatus(state=MCPStatus.State.CONNECTING)

        try:
            if config.type == MCPServerType.LOCAL:
                return await self._connect_local(name, config)
            else:
                return await self._connect_remote(name, config)
        except Exception as e:
            status = MCPStatus(
                state=MCPStatus.State.FAILED,
                error=str(e),
            )
            self._status[name] = status
            return status

    async def _connect_local(self, name: str, config: MCPConfig) -> MCPStatus:
        """Connect to a local MCP server via stdio."""
        if not config.command:
            raise MCPConnectionError(name, "No command specified for local server")

        try:
            # Build environment
            env = os.environ.copy()
            if config.environment:
                env.update(config.environment)

            # Start the process
            process = subprocess.Popen(
                config.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=str(self._project_dir),
            )

            self._processes[name] = process

            # Initialize the MCP connection
            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "0.1.0",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "filegramengine",
                        "version": "0.1.0",
                    },
                },
            }

            response = await self._send_request(name, init_request, config.timeout)

            if "error" in response:
                raise MCPConnectionError(name, response["error"].get("message", "Unknown error"))

            # Send initialized notification
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            await self._send_notification(name, initialized_notification)

            # List tools
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
            }

            tools_response = await self._send_request(name, tools_request, config.timeout)

            if "result" in tools_response:
                tools_data = tools_response["result"].get("tools", [])
                self._tools[name] = [
                    MCPTool(
                        name=t["name"],
                        description=t.get("description", ""),
                        input_schema=t.get("inputSchema", {}),
                        server_name=name,
                    )
                    for t in tools_data
                ]
                logger.info(f"Connected to MCP server '{name}' with {len(self._tools[name])} tools")

            status = MCPStatus(state=MCPStatus.State.CONNECTED)
            self._status[name] = status
            return status

        except Exception as e:
            # Clean up on failure
            if name in self._processes:
                self._processes[name].terminate()
                del self._processes[name]
            raise MCPConnectionError(name, str(e))

    async def _connect_remote(self, name: str, config: MCPConfig) -> MCPStatus:
        """Connect to a remote MCP server via HTTP/SSE."""
        # For now, remote servers are not fully implemented
        # This would require httpx and SSE handling
        logger.warning(f"Remote MCP servers not yet fully implemented: {name}")
        status = MCPStatus(
            state=MCPStatus.State.FAILED,
            error="Remote MCP servers not yet implemented",
        )
        self._status[name] = status
        return status

    async def _send_request(self, name: str, request: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Send a JSON-RPC request to an MCP server and wait for response."""
        process = self._processes.get(name)
        if not process or process.stdin is None or process.stdout is None:
            raise MCPConnectionError(name, "Server process not available")

        # Send request
        request_str = json.dumps(request) + "\n"
        process.stdin.write(request_str.encode())
        process.stdin.flush()

        # Read response with timeout
        try:
            response_line = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, process.stdout.readline),
                timeout=timeout,
            )
            if not response_line:
                raise MCPConnectionError(name, "No response from server")

            return json.loads(response_line.decode())
        except asyncio.TimeoutError:
            raise MCPTimeoutError(f"request to {name}", timeout)

    async def _send_notification(self, name: str, notification: dict[str, Any]) -> None:
        """Send a JSON-RPC notification to an MCP server (no response expected)."""
        process = self._processes.get(name)
        if not process or process.stdin is None:
            raise MCPConnectionError(name, "Server process not available")

        notification_str = json.dumps(notification) + "\n"
        process.stdin.write(notification_str.encode())
        process.stdin.flush()

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on an MCP server."""
        config = self._configs.get(server_name)
        if not config:
            raise MCPToolError(server_name, tool_name, "Server not configured")

        status = self._status.get(server_name)
        if not status or status.state != MCPStatus.State.CONNECTED:
            raise MCPToolError(server_name, tool_name, "Server not connected")

        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }

        response = await self._send_request(server_name, request, config.timeout)

        if "error" in response:
            raise MCPToolError(
                server_name,
                tool_name,
                response["error"].get("message", "Unknown error"),
            )

        result = response.get("result", {})
        content = result.get("content", [])

        # Extract text content
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)

        return "\n".join(text_parts) if text_parts else json.dumps(result)

    def get_status(self) -> dict[str, MCPStatus]:
        """Get status of all configured MCP servers."""
        return self._status.copy()

    def get_tools(self) -> list[MCPToolWrapper]:
        """Get all available MCP tools as wrapped BaseTool instances."""
        tools = []
        for server_tools in self._tools.values():
            for mcp_tool in server_tools:
                tools.append(MCPToolWrapper(mcp_tool, self))
        return tools

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Get tool definitions for all MCP tools."""
        definitions = []
        for tool_wrapper in self.get_tools():
            definitions.append(tool_wrapper.get_definition())
        return definitions

    async def disconnect(self, name: str) -> None:
        """Disconnect from an MCP server."""
        if name in self._processes:
            process = self._processes[name]
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            del self._processes[name]

        if name in self._tools:
            del self._tools[name]

        self._status[name] = MCPStatus(state=MCPStatus.State.DISABLED)

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for name in list(self._processes.keys()):
            await self.disconnect(name)

    async def __aenter__(self) -> MCPClientManager:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect_all()
