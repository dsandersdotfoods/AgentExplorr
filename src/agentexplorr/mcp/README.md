# Model Context Protocol (MCP)

> "MCP is the USB-C for AI — a universal interface that lets any LLM connect to any tool."

## What Is MCP?

The **Model Context Protocol** is an open standard that defines how LLMs communicate with external tools and data sources. Created by Anthropic and adopted across the industry, MCP replaces custom tool-calling implementations with a single, standardized protocol.

```
┌──────────────┐    JSON-RPC 2.0    ┌────────────────┐
│              │ ◄────────────────► │                │
│   LLM +      │     (MCP)         │  MCP Server    │
│   MCP Client │                   │  (your tools)  │
│              │                   │                │
└──────────────┘                   └───────┬────────┘
                                           │
                                    ┌──────┴──────┐
                                    │  Database   │
                                    │  APIs       │
                                    │  Files      │
                                    └─────────────┘
```

## What You'll Learn

| File | Concept | Difficulty |
|------|---------|-----------|
| `servers/filesystem_server.py` | File operations as MCP tools | Beginner |
| `servers/database_server.py` | SQL queries as MCP tools (Text-to-SQL) | Intermediate |
| `servers/api_server.py` | External API wrapping (Open-Meteo weather) | Intermediate |
| `clients/mcp_client.py` | Connecting to and using MCP servers | Intermediate |
| `tools/sql_tool.py` | Safe SQL execution with parameterized queries | Intermediate |
| `tools/code_executor.py` | Sandboxed Python code execution | Advanced |

## Quick Start

### Running an MCP Server

```bash
# Run the filesystem server
python -m agentexplorr.mcp.servers.filesystem_server

# Run the database server
python -m agentexplorr.mcp.servers.database_server

# Run the weather API server
python -m agentexplorr.mcp.servers.api_server
```

### Using the SQL Tool

```python
from agentexplorr.mcp.tools.sql_tool import SQLTool

tool = SQLTool(db_path="data/sample.db")
result = tool.execute("SELECT name, price FROM products WHERE price > ?", params=(20.0,))
print(result.to_markdown_table())
```

### Using the Code Executor

```python
from agentexplorr.mcp.tools.code_executor import CodeExecutor

executor = CodeExecutor()
result = executor.execute("""
import math
for i in range(1, 6):
    print(f"sqrt({i}) = {math.sqrt(i):.4f}")
""")
print(result.stdout)
```

## Security Considerations

| Server | Safety Measure |
|--------|---------------|
| Filesystem | Path traversal prevention, base directory restriction |
| Database | Read-only queries, SQL injection prevention via parameterized queries |
| Code Executor | Restricted builtins, import whitelist, output capture |

## Learning Resources

### Official Docs
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Official Site](https://modelcontextprotocol.io/)

### Videos
- [MCP Explained in 5 Minutes](https://www.youtube.com/watch?v=kQmXtrmQ5Zg)
- [Build Your Own MCP Server](https://www.youtube.com/watch?v=23mkvV2RLWU)

### Papers & Articles
- [Anthropic MCP Announcement](https://www.anthropic.com/news/model-context-protocol)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
