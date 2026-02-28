"""
MCP Client — Connect to MCP Servers and Use Their Tools
========================================================

WHAT THIS DOES:
  The MCP client connects to MCP servers (like the ones in ../servers/)
  and provides a clean Python API to:
  - Discover available tools (list tools)
  - Call tools (invoke functions on the server)
  - Handle responses and errors

HOW MCP COMMUNICATION WORKS:
  MCP uses JSON-RPC 2.0 over different transports:

  1. **stdio** — Server runs as a subprocess, communicates via stdin/stdout
     - Best for: local tools, CLI integrations
     - How: Client spawns the server process, sends JSON via stdin

  2. **SSE (Server-Sent Events)** — HTTP-based streaming
     - Best for: web apps, remote servers
     - How: Client sends HTTP requests, receives streaming responses

  3. **Streamable HTTP** — Standard HTTP with streaming support
     - Best for: modern web deployments

  ┌──────────┐  stdin/stdout  ┌──────────────┐
  │  Client   │ ◄────────────► │  MCP Server  │
  └──────────┘   JSON-RPC     └──────────────┘
       │                              │
       │ "list tools"                 │ returns tool schemas
       │ "call tool X"               │ executes & returns result
       │                              │

LEARNING RESOURCES:
  - MCP Client SDK: https://github.com/modelcontextprotocol/python-sdk
  - JSON-RPC 2.0 spec: https://www.jsonrpc.org/specification
  - VIDEO: "MCP Clients Explained" — https://www.youtube.com/watch?v=kQmXtrmQ5Zg
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentexplorr.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolInfo:
    """Metadata about an MCP tool.

    Attributes:
        name: Tool name (used to call it).
        description: What the tool does (shown to the LLM).
        input_schema: JSON Schema for the tool's parameters.
    """

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result from calling an MCP tool.

    Attributes:
        content: The tool's output (usually text).
        is_error: Whether the tool call failed.
    """

    content: str
    is_error: bool = False


class MCPClientManager:
    """Manage connections to multiple MCP servers.

    WHY A MANAGER?
      In real applications, you might connect to multiple MCP servers:
      - A filesystem server for file operations
      - A database server for SQL queries
      - An API server for external data

      The manager handles the lifecycle of all these connections
      and provides a unified interface to discover and call tools.

    Example:
        >>> manager = MCPClientManager()
        >>> # In production, you'd connect to real MCP servers:
        >>> # await manager.connect_stdio("filesystem", "python", ["-m", "agentexplorr.mcp.servers.filesystem_server"])
        >>>
        >>> # For now, register tools manually:
        >>> manager.register_tool(ToolInfo(
        ...     name="read_file",
        ...     description="Read file contents",
        ...     input_schema={"type": "object", "properties": {"path": {"type": "string"}}}
        ... ))
        >>> tools = manager.list_tools()
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolInfo] = {}
        self._tool_handlers: dict[str, Any] = {}

    def register_tool(
        self,
        tool_info: ToolInfo,
        handler: Any | None = None,
    ) -> None:
        """Register a tool with the manager.

        In a full MCP implementation, tools are discovered automatically
        when connecting to a server. This method allows manual registration
        for testing and local tool definitions.

        Args:
            tool_info: Tool metadata.
            handler: Callable that implements the tool (for local tools).
        """
        self._tools[tool_info.name] = tool_info
        if handler is not None:
            self._tool_handlers[tool_info.name] = handler
        logger.info("tool_registered", name=tool_info.name)

    def list_tools(self) -> list[ToolInfo]:
        """List all available tools across all connected servers.

        Returns:
            List of ToolInfo for every registered tool.
        """
        return list(self._tools.values())

    def get_tool(self, name: str) -> ToolInfo | None:
        """Get a specific tool's info by name.

        Args:
            name: Tool name.

        Returns:
            ToolInfo if found, None otherwise.
        """
        return self._tools.get(name)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Call a registered tool with the given arguments.

        HOW TOOL CALLING WORKS:
          1. LLM decides to use a tool → outputs tool name + arguments
          2. Client finds the tool handler
          3. Client calls the handler with the arguments
          4. Client returns the result to the LLM

        Args:
            name: Tool name.
            arguments: Tool input parameters.

        Returns:
            ToolResult with the tool's output.
        """
        if name not in self._tools:
            return ToolResult(
                content=f"Error: Unknown tool '{name}'. Available: {list(self._tools.keys())}",
                is_error=True,
            )

        handler = self._tool_handlers.get(name)
        if handler is None:
            return ToolResult(
                content=f"Error: No handler registered for tool '{name}'",
                is_error=True,
            )

        try:
            logger.info("calling_tool", name=name, args=list(arguments.keys()))
            result = handler(**arguments)
            return ToolResult(content=str(result))
        except Exception as e:
            logger.error("tool_call_failed", name=name, error=str(e))
            return ToolResult(content=f"Error calling {name}: {e}", is_error=True)

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        """Format tools as a list of dicts suitable for LLM tool-calling.

        This output format matches the common convention used by
        LangChain, OpenAI, and other frameworks for describing tools.

        Returns:
            List of tool descriptions with name, description, and parameters.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in self._tools.values()
        ]
