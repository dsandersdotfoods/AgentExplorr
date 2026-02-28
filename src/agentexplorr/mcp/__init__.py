"""
Model Context Protocol (MCP) Module
====================================

WHAT IS MCP?
  The Model Context Protocol is an open standard (created by Anthropic) that
  defines how LLMs communicate with external tools and data sources. Think of
  it as a "USB-C for AI" — a universal interface that lets any LLM connect
  to any tool.

WHY MCP MATTERS:
  Before MCP, every LLM provider had its own tool-calling format. This meant:
  - Tools built for OpenAI didn't work with Anthropic
  - Each integration was a custom one-off
  - No standardization = no ecosystem

  MCP solves this with a standard protocol:
  - **MCP Servers** expose tools (functions the LLM can call)
  - **MCP Clients** connect to servers and make tools available to the LLM
  - **Transport** handles communication (stdio, SSE, HTTP)

THE ARCHITECTURE:
  ┌──────────┐     MCP Protocol     ┌──────────────┐
  │  LLM +   │ ◄──────────────────► │  MCP Server  │
  │  Client   │   (JSON-RPC 2.0)   │  (your tools)│
  └──────────┘                      └──────────────┘
       │                                   │
       │                            ┌──────┴──────┐
       │                            │  Database   │
       ▼                            │  APIs       │
    User sees                       │  Files      │
    tool results                    └─────────────┘

THIS MODULE COVERS:
  1. **Servers** — Build MCP servers that expose tools
     - Filesystem operations (read/write/list files)
     - Database queries (SQLite)
     - External API wrappers (weather data)

  2. **Clients** — Connect to MCP servers from your code
     - Discover available tools
     - Call tools and get results

  3. **Custom Tools** — Reusable tool implementations
     - SQL query execution
     - Sandboxed code execution

LEARNING RESOURCES:
  - MCP Specification: https://spec.modelcontextprotocol.io/
  - MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
  - MCP Official Docs: https://modelcontextprotocol.io/
  - VIDEO: "MCP Explained in 5 Minutes" — https://www.youtube.com/watch?v=kQmXtrmQ5Zg
  - VIDEO: "Build Your Own MCP Server" — https://www.youtube.com/watch?v=23mkvV2RLWU
"""
