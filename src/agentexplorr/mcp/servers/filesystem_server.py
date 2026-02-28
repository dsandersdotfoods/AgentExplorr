"""
Filesystem MCP Server — Expose File Operations as LLM Tools
=============================================================

WHAT THIS DOES:
  This MCP server lets an LLM read, write, and list files on the local
  filesystem. The LLM can explore project structures, read configuration
  files, and create new files — all through the standardized MCP protocol.

HOW MCP SERVERS WORK:
  1. You define "tools" — functions the LLM can call
  2. Each tool has a name, description, and input schema (JSON Schema)
  3. The MCP SDK handles serialization, transport, and error handling
  4. The LLM sees your tools and decides when to call them

SECURITY CONSIDERATIONS:
  ⚠️  NEVER give an LLM unrestricted filesystem access in production!
  This server restricts operations to a configurable base directory.
  All paths are resolved relative to this base, and path traversal
  attacks (../../etc/passwd) are blocked.

LEARNING RESOURCES:
  - MCP Server SDK: https://github.com/modelcontextprotocol/python-sdk
  - MCP Tool Definition: https://spec.modelcontextprotocol.io/specification/server/tools/
  - VIDEO: "Build MCP Servers" — https://www.youtube.com/watch?v=23mkvV2RLWU
  - pathlib docs: https://docs.python.org/3/library/pathlib.html
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from agentexplorr.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Server Setup
# ---------------------------------------------------------------------------
# FastMCP is the high-level API for creating MCP servers.
# It handles all the protocol details — you just define tools as functions.

server = FastMCP(
    name="filesystem",
    instructions="File system operations. Read, write, and list files within the allowed directory.",
)

# Default base directory — all operations are restricted to this path.
# In production, this should be configurable and tightly scoped.
BASE_DIR = Path.cwd()


def _resolve_safe_path(relative_path: str) -> Path:
    """Resolve a path safely, preventing directory traversal attacks.

    SECURITY: Path traversal is a classic attack where an attacker uses
    "../" sequences to escape the intended directory:
      "../../etc/passwd" → reads system password file!

    We prevent this by:
      1. Resolving the full path (following symlinks)
      2. Checking that the resolved path starts with our base directory

    Args:
        relative_path: Path relative to BASE_DIR.

    Returns:
        Resolved absolute Path.

    Raises:
        ValueError: If the path would escape BASE_DIR.
    """
    resolved = (BASE_DIR / relative_path).resolve()

    # The is_relative_to check ensures the resolved path is INSIDE base_dir
    if not resolved.is_relative_to(BASE_DIR.resolve()):
        msg = f"Path traversal blocked: {relative_path!r} resolves outside base directory"
        raise ValueError(msg)

    return resolved


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------
# Each @server.tool() decorated function becomes an MCP tool.
# The docstring becomes the tool description (shown to the LLM).
# Type hints become the JSON Schema for the tool's input.


@server.tool()
def read_file(path: str) -> str:
    """Read the contents of a file.

    Args:
        path: Relative path to the file (from the base directory).

    Returns:
        The file contents as a string.
    """
    file_path = _resolve_safe_path(path)

    if not file_path.exists():
        return f"Error: File not found: {path}"

    if not file_path.is_file():
        return f"Error: Not a file: {path}"

    # Limit file size to prevent memory issues
    max_size = 1_000_000  # 1MB
    if file_path.stat().st_size > max_size:
        return f"Error: File too large (>{max_size} bytes): {path}"

    logger.info("reading_file", path=str(file_path))
    return file_path.read_text(encoding="utf-8")


@server.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates the file if it doesn't exist.

    Args:
        path: Relative path for the file.
        content: The text content to write.

    Returns:
        Confirmation message.
    """
    file_path = _resolve_safe_path(path)

    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("writing_file", path=str(file_path), size=len(content))
    file_path.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} characters to {path}"


@server.tool()
def list_directory(path: str = ".") -> str:
    """List files and directories at the given path.

    Args:
        path: Relative path to list (defaults to base directory).

    Returns:
        Formatted listing of directory contents.
    """
    dir_path = _resolve_safe_path(path)

    if not dir_path.exists():
        return f"Error: Directory not found: {path}"

    if not dir_path.is_dir():
        return f"Error: Not a directory: {path}"

    logger.info("listing_directory", path=str(dir_path))

    entries = sorted(dir_path.iterdir())
    lines = []
    for entry in entries:
        # Show type indicator: [DIR] or [FILE]
        prefix = "[DIR] " if entry.is_dir() else "[FILE]"
        # Show file size for files
        size = f" ({entry.stat().st_size:,} bytes)" if entry.is_file() else ""
        lines.append(f"  {prefix} {entry.name}{size}")

    header = f"Contents of {path}/ ({len(entries)} items):"
    return header + "\n" + "\n".join(lines)


@server.tool()
def search_files(pattern: str, path: str = ".") -> str:
    """Search for files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., "*.py", "**/*.json").
        path: Directory to search in (relative to base).

    Returns:
        List of matching file paths.
    """
    dir_path = _resolve_safe_path(path)

    if not dir_path.is_dir():
        return f"Error: Not a directory: {path}"

    matches = list(dir_path.glob(pattern))
    # Limit results to prevent overwhelming the LLM
    max_results = 100
    if len(matches) > max_results:
        matches = matches[:max_results]
        truncated = " (truncated to 100 results)"
    else:
        truncated = ""

    logger.info("searching_files", pattern=pattern, matches=len(matches))

    if not matches:
        return f"No files matching '{pattern}' in {path}/"

    lines = [str(m.relative_to(BASE_DIR)) for m in sorted(matches)]
    return f"Found {len(matches)} matches{truncated}:\n" + "\n".join(f"  {line}" for line in lines)


# ---------------------------------------------------------------------------
# Running the Server
# ---------------------------------------------------------------------------
# To run this MCP server:
#   python -m agentexplorr.mcp.servers.filesystem_server
#
# This starts the server using stdio transport (communicates via stdin/stdout).
# An MCP client can then connect to it and use the tools.

if __name__ == "__main__":
    server.run()
