"""
Tests for the MCP Client Manager
==================================

Tests verify tool registration, discovery, and calling.
"""

from __future__ import annotations

from agentexplorr.mcp.clients.mcp_client import MCPClientManager, ToolInfo, ToolResult


class TestMCPClientManager:
    """Tests for MCPClientManager."""

    def setup_method(self) -> None:
        """Create a fresh manager for each test."""
        self.manager = MCPClientManager()

    def test_register_and_list_tools(self) -> None:
        """Test registering and listing tools."""
        tool = ToolInfo(name="test_tool", description="A test tool")
        self.manager.register_tool(tool)

        tools = self.manager.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    def test_get_tool(self) -> None:
        """Test getting a specific tool by name."""
        tool = ToolInfo(name="my_tool", description="My tool")
        self.manager.register_tool(tool)

        found = self.manager.get_tool("my_tool")
        assert found is not None
        assert found.name == "my_tool"

        not_found = self.manager.get_tool("nonexistent")
        assert not_found is None

    def test_call_tool_with_handler(self) -> None:
        """Test calling a tool that has a handler."""
        def add(a: int, b: int) -> int:
            return a + b

        tool = ToolInfo(name="add", description="Add two numbers")
        self.manager.register_tool(tool, handler=add)

        result = self.manager.call_tool("add", {"a": 2, "b": 3})
        assert result.content == "5"
        assert not result.is_error

    def test_call_unknown_tool(self) -> None:
        """Test calling a tool that doesn't exist."""
        result = self.manager.call_tool("unknown", {})
        assert result.is_error
        assert "Unknown tool" in result.content

    def test_call_tool_without_handler(self) -> None:
        """Test calling a tool with no handler registered."""
        tool = ToolInfo(name="no_handler", description="No handler")
        self.manager.register_tool(tool)

        result = self.manager.call_tool("no_handler", {})
        assert result.is_error

    def test_get_tools_for_llm(self) -> None:
        """Test LLM-formatted tool descriptions."""
        tool = ToolInfo(
            name="search",
            description="Search the web",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        self.manager.register_tool(tool)

        llm_tools = self.manager.get_tools_for_llm()
        assert len(llm_tools) == 1
        assert llm_tools[0]["name"] == "search"
        assert "query" in llm_tools[0]["parameters"]["properties"]
