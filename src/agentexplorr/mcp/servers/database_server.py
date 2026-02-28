"""
Database MCP Server — Expose SQL Queries as LLM Tools
======================================================

WHAT THIS DOES:
  This MCP server lets an LLM query a SQLite database. The LLM can:
  - List available tables
  - Describe table schemas
  - Run SELECT queries (read-only for safety)

WHY SQLITE?
  SQLite is perfect for learning because:
  - No server setup needed (it's a file)
  - Comes built into Python (no installation)
  - Supports standard SQL syntax
  - Used by more applications than any other database

TEXT-TO-SQL:
  This server is the foundation for "Text-to-SQL" — one of the most
  practical LLM applications. The LLM:
  1. Understands the user's question in natural language
  2. Inspects the database schema
  3. Writes and executes a SQL query
  4. Returns the results in human-readable form

SECURITY:
  ⚠️  Only SELECT queries are allowed. This prevents:
  - Data modification (INSERT, UPDATE, DELETE)
  - Schema changes (DROP TABLE, ALTER TABLE)
  - SQL injection attacks that could damage data

LEARNING RESOURCES:
  - SQLite docs: https://www.sqlite.org/docs.html
  - Python sqlite3 module: https://docs.python.org/3/library/sqlite3.html
  - Text-to-SQL overview: https://arxiv.org/abs/2406.11434
  - VIDEO: "SQLite Tutorial" — https://www.youtube.com/watch?v=byHcYRpMgI4
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from agentexplorr.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Server Setup
# ---------------------------------------------------------------------------

server = FastMCP(
    name="database",
    instructions="SQLite database operations. List tables, describe schemas, and run read-only SQL queries.",
)

# Default database path — override via environment or constructor
DEFAULT_DB_PATH = Path("data/sample.db")


def _get_connection(db_path: str | None = None, *, read_only: bool = False) -> sqlite3.Connection:
    """Get a SQLite connection with safety settings.

    SAFETY SETTINGS EXPLAINED:
      - row_factory = sqlite3.Row → results are dict-like (access by column name)
      - timeout = 5 → don't hang forever on locked databases
      - read_only mode → opens database as read-only via URI, preventing any writes

    Args:
        db_path: Path to SQLite database file. Uses default if None.
        read_only: If True, open in read-only mode (blocks all writes at DB level).

    Returns:
        sqlite3.Connection with safety settings applied.
    """
    path = Path(db_path) if db_path else DEFAULT_DB_PATH

    if not path.exists():
        # Create the database and a sample table for learning
        logger.info("creating_sample_database", path=str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        _create_sample_data(conn)
        conn.close()

    if read_only:
        # Open in read-only mode via URI — the database engine itself blocks writes
        uri = f"file:{path.resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
    else:
        conn = sqlite3.connect(str(path), timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _get_valid_tables(conn: sqlite3.Connection) -> set[str]:
    """Get the set of actual table names from the database.

    Used to validate table names before interpolation into SQL,
    preventing SQL injection via crafted table names.
    """
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {row["name"] for row in cursor.fetchall()}


def _create_sample_data(conn: sqlite3.Connection) -> None:
    """Create sample tables for learning and experimentation.

    This gives you something to query right away without needing
    to set up a real database.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            city TEXT
        );

        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            product_id INTEGER REFERENCES products(id),
            quantity INTEGER NOT NULL,
            order_date TEXT NOT NULL
        );

        INSERT OR IGNORE INTO products (id, name, category, price, stock) VALUES
            (1, 'Neural Network Textbook', 'Books', 49.99, 100),
            (2, 'GPU Computing Card', 'Hardware', 599.99, 25),
            (3, 'Python Sticker Pack', 'Merch', 9.99, 500),
            (4, 'ML Conference Ticket', 'Events', 299.99, 50),
            (5, 'Data Science Poster', 'Merch', 14.99, 200);

        INSERT OR IGNORE INTO customers (id, name, email, city) VALUES
            (1, 'Alice Zhang', 'alice@example.com', 'San Francisco'),
            (2, 'Bob Smith', 'bob@example.com', 'New York'),
            (3, 'Carol Johnson', 'carol@example.com', 'Seattle');

        INSERT OR IGNORE INTO orders (id, customer_id, product_id, quantity, order_date) VALUES
            (1, 1, 1, 2, '2025-01-15'),
            (2, 1, 3, 10, '2025-01-16'),
            (3, 2, 2, 1, '2025-02-01'),
            (4, 3, 4, 2, '2025-02-10'),
            (5, 2, 5, 3, '2025-02-15');
    """)
    conn.commit()


def _is_read_only(query: str) -> bool:
    """Check if a SQL query is read-only (SELECT only).

    WHY THIS MATTERS:
      We ONLY allow SELECT queries. This prevents the LLM from:
      - Dropping tables (SQL injection via DROP TABLE)
      - Modifying data (INSERT, UPDATE, DELETE)
      - Changing schema (ALTER, CREATE)

    This is a SIMPLE check — in production, use a proper SQL parser
    or database-level permissions (GRANT SELECT ONLY).
    """
    # Normalize: strip whitespace, remove comments, uppercase
    stripped = query.strip().upper()

    # Block dangerous operations
    dangerous_keywords = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "REPLACE",
        "GRANT",
        "REVOKE",
        "ATTACH",
    ]

    # Check if query starts with SELECT or WITH (CTEs)
    if not (stripped.startswith("SELECT") or stripped.startswith("WITH")):
        return False

    # Double-check: no dangerous keywords after a semicolon
    # (prevents "SELECT 1; DROP TABLE users")
    if ";" in query:
        # Only allow the last semicolon to be trailing whitespace
        parts = [p.strip() for p in query.split(";") if p.strip()]
        if len(parts) > 1:
            return False

    # Check for dangerous keywords using regex word boundaries
    return all(not re.search(rf"\b{keyword}\b", stripped) for keyword in dangerous_keywords)


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------


@server.tool()
def list_tables(db_path: str | None = None) -> str:
    """List all tables in the database.

    Returns:
        Names and row counts of all tables.
    """
    conn = _get_connection(db_path, read_only=True)
    try:
        tables = sorted(_get_valid_tables(conn))

        if not tables:
            return "No tables found in the database."

        lines = ["Tables in database:"]
        for table in tables:
            count = conn.execute(f"SELECT COUNT(*) as c FROM [{table}]").fetchone()["c"]
            lines.append(f"  - {table} ({count:,} rows)")

        return "\n".join(lines)
    finally:
        conn.close()


@server.tool()
def describe_table(table_name: str, db_path: str | None = None) -> str:
    """Get the schema (columns, types) of a table.

    Args:
        table_name: Name of the table to describe.

    Returns:
        Table schema with column names, types, and constraints.
    """
    conn = _get_connection(db_path, read_only=True)
    try:
        # Validate table_name against actual tables to prevent SQL injection
        valid_tables = _get_valid_tables(conn)
        if table_name not in valid_tables:
            return f"Table '{table_name}' not found."

        # PRAGMA table_info returns column metadata
        cursor = conn.execute(f"PRAGMA table_info([{table_name}])")
        columns = cursor.fetchall()

        if not columns:
            return f"Table '{table_name}' has no columns."

        lines = [f"Schema for '{table_name}':"]
        lines.append(f"  {'Column':<20} {'Type':<15} {'Nullable':<10} {'PK'}")
        lines.append("  " + "-" * 55)

        for col in columns:
            nullable = "YES" if not col["notnull"] else "NO"
            pk = "✓" if col["pk"] else ""
            lines.append(f"  {col['name']:<20} {col['type']:<15} {nullable:<10} {pk}")

        return "\n".join(lines)
    finally:
        conn.close()


@server.tool()
def run_query(query: str, db_path: str | None = None) -> str:
    """Run a read-only SQL query and return results.

    Only SELECT queries are allowed for safety.

    Args:
        query: SQL SELECT query to execute.

    Returns:
        Query results formatted as a table.
    """
    # Security check: only allow SELECT queries
    if not _is_read_only(query):
        return (
            "Error: Only SELECT queries are allowed. Modification queries are blocked for safety."
        )

    # Defense-in-depth: open in read-only mode so even if _is_read_only()
    # is bypassed, the database engine itself blocks writes
    conn = _get_connection(db_path, read_only=True)
    try:
        logger.info("executing_query", query=query[:200])
        cursor = conn.execute(query)
        rows = cursor.fetchall()

        if not rows:
            return "Query returned no results."

        # Get column names from cursor description
        columns = [desc[0] for desc in cursor.description]

        # Format as a simple table
        # Calculate column widths
        widths = [len(col) for col in columns]
        str_rows = []
        for row in rows[:100]:  # Limit to 100 rows
            str_row = [str(row[col]) for col in columns]
            str_rows.append(str_row)
            for i, val in enumerate(str_row):
                widths[i] = max(widths[i], len(val))

        # Build output
        header = " | ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
        separator = "-+-".join("-" * w for w in widths)
        data_lines = [
            " | ".join(val.ljust(widths[i]) for i, val in enumerate(row)) for row in str_rows
        ]

        result = f"{header}\n{separator}\n" + "\n".join(data_lines)

        if len(rows) > 100:
            result += f"\n\n... ({len(rows) - 100} more rows not shown)"

        return f"({len(rows)} rows)\n\n{result}"
    except sqlite3.Error as e:
        return f"SQL Error: {e}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Run the server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server.run()
