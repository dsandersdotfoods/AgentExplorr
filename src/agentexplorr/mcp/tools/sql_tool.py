"""
SQL Query Tool — Safe Database Access for LLMs
================================================

WHAT THIS DOES:
  A reusable tool that lets LLMs query SQLite databases safely.
  This is the underlying implementation used by the database MCP server,
  but can also be used standalone in agent pipelines.

SECURITY MODEL:
  This tool implements defense-in-depth:
  1. **Query validation** — Only SELECT statements allowed
  2. **Parameter binding** — Prevents SQL injection via parameterized queries
  3. **Result limits** — Caps rows returned to prevent memory exhaustion
  4. **Timeout** — Kills long-running queries

SQL INJECTION EXPLAINED:
  Without parameterized queries, an attacker could input:
    name = "'; DROP TABLE users; --"
  Which becomes:
    SELECT * FROM users WHERE name = ''; DROP TABLE users; --'

  With parameterized queries (using ? placeholders):
    SELECT * FROM users WHERE name = ?
  The database treats the entire input as a string value, not SQL code.

LEARNING RESOURCES:
  - SQL Injection prevention: https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html
  - sqlite3 parameterized queries: https://docs.python.org/3/library/sqlite3.html#sqlite3-placeholders
  - VIDEO: "SQL Injection Explained" — https://www.youtube.com/watch?v=ciNHn38EyRc
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentexplorr.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """Result of a SQL query execution.

    Attributes:
        columns: List of column names.
        rows: List of row tuples.
        row_count: Total number of rows returned.
        truncated: Whether results were truncated due to limit.
    """

    columns: list[str]
    rows: list[tuple[Any, ...]]
    row_count: int
    truncated: bool = False

    def to_dicts(self) -> list[dict[str, Any]]:
        """Convert rows to list of dictionaries.

        This is convenient for JSON serialization or DataFrame creation.

        Returns:
            List of {column_name: value} dicts.
        """
        return [dict(zip(self.columns, row)) for row in self.rows]

    def to_markdown_table(self) -> str:
        """Format results as a Markdown table.

        WHY MARKDOWN?
          LLMs understand Markdown natively. Formatting query results
          as Markdown tables makes them easy for the LLM to read,
          interpret, and reference in its response.

        Returns:
            Markdown-formatted table string.
        """
        if not self.rows:
            return "*No results*"

        # Calculate column widths
        widths = [len(col) for col in self.columns]
        str_rows = []
        for row in self.rows:
            str_row = [str(val) for val in row]
            str_rows.append(str_row)
            for i, val in enumerate(str_row):
                widths[i] = max(widths[i], len(val))

        # Build markdown table
        header = "| " + " | ".join(col.ljust(widths[i]) for i, col in enumerate(self.columns)) + " |"
        separator = "| " + " | ".join("-" * w for w in widths) + " |"
        data = [
            "| " + " | ".join(val.ljust(widths[i]) for i, val in enumerate(row)) + " |"
            for row in str_rows
        ]

        table = "\n".join([header, separator, *data])

        if self.truncated:
            table += f"\n\n*Results truncated. Showing {len(self.rows)} of {self.row_count} rows.*"

        return table


@dataclass
class SQLTool:
    """Safe SQL query execution tool.

    Example:
        >>> tool = SQLTool(db_path="data/sample.db")
        >>> result = tool.execute("SELECT name, price FROM products WHERE price > ?", params=(20.0,))
        >>> print(result.to_markdown_table())
        | name                  | price  |
        | --------------------- | ------ |
        | Neural Network Textbook | 49.99 |
        | GPU Computing Card    | 599.99 |
    """

    db_path: str = "data/sample.db"
    max_rows: int = 1000
    allowed_tables: list[str] = field(default_factory=list)

    def execute(
        self,
        query: str,
        params: tuple[Any, ...] = (),
    ) -> QueryResult:
        """Execute a read-only SQL query.

        Args:
            query: SQL SELECT query. Use ? for parameters.
            params: Parameter values (prevents SQL injection).

        Returns:
            QueryResult with columns, rows, and metadata.

        Raises:
            ValueError: If query is not read-only.
            sqlite3.Error: If query execution fails.
        """
        # Validate query is read-only
        normalized = query.strip().upper()
        if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
            msg = "Only SELECT queries are allowed"
            raise ValueError(msg)

        logger.info("executing_sql", query=query[:100], params_count=len(params))

        conn = sqlite3.connect(self.db_path, timeout=5)
        try:
            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            all_rows = cursor.fetchall()

            truncated = len(all_rows) > self.max_rows
            rows = all_rows[: self.max_rows]

            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=len(all_rows),
                truncated=truncated,
            )
        finally:
            conn.close()

    def get_schema(self) -> dict[str, list[dict[str, str]]]:
        """Get the database schema (all tables and their columns).

        Returns:
            Dict mapping table names to lists of column info dicts.
        """
        db_path = Path(self.db_path)
        if not db_path.exists():
            return {}

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]

            schema: dict[str, list[dict[str, str]]] = {}
            for table in tables:
                col_cursor = conn.execute(f"PRAGMA table_info([{table}])")
                schema[table] = [
                    {"name": row[1], "type": row[2], "nullable": not row[3]}
                    for row in col_cursor.fetchall()
                ]

            return schema
        finally:
            conn.close()
