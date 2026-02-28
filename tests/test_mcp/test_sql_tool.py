"""
Tests for the SQL Tool
=======================

Tests verify SQL query execution, safety checks, and result formatting.
Uses an in-memory SQLite database — no files needed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agentexplorr.mcp.tools.sql_tool import QueryResult, SQLTool


@pytest.fixture
def sql_tool(tmp_dir: Path) -> SQLTool:
    """Create a SQLTool with a test database."""
    db_path = tmp_dir / "test.db"

    # Create a test database with sample data
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        INSERT INTO users VALUES (1, 'Alice', 30);
        INSERT INTO users VALUES (2, 'Bob', 25);
        INSERT INTO users VALUES (3, 'Carol', 35);
    """)
    conn.commit()
    conn.close()

    return SQLTool(db_path=str(db_path))


class TestSQLTool:
    """Tests for SQLTool query execution."""

    def test_select_all(self, sql_tool: SQLTool) -> None:
        """Test basic SELECT query."""
        result = sql_tool.execute("SELECT * FROM users")
        assert result.row_count == 3
        assert "name" in result.columns
        assert "age" in result.columns

    def test_select_with_where(self, sql_tool: SQLTool) -> None:
        """Test SELECT with WHERE clause."""
        result = sql_tool.execute("SELECT name FROM users WHERE age > ?", params=(28,))
        assert result.row_count == 2
        names = [row[0] for row in result.rows]
        assert "Alice" in names
        assert "Carol" in names

    def test_rejects_insert(self, sql_tool: SQLTool) -> None:
        """Test that INSERT queries are rejected."""
        with pytest.raises(ValueError, match="Only SELECT"):
            sql_tool.execute("INSERT INTO users VALUES (4, 'Dave', 40)")

    def test_rejects_drop(self, sql_tool: SQLTool) -> None:
        """Test that DROP queries are rejected."""
        with pytest.raises(ValueError, match="Only SELECT"):
            sql_tool.execute("DROP TABLE users")

    def test_to_dicts(self, sql_tool: SQLTool) -> None:
        """Test converting results to dictionaries."""
        result = sql_tool.execute("SELECT name, age FROM users WHERE id = 1")
        dicts = result.to_dicts()
        assert len(dicts) == 1
        assert dicts[0]["name"] == "Alice"
        assert dicts[0]["age"] == 30

    def test_to_markdown_table(self, sql_tool: SQLTool) -> None:
        """Test Markdown table formatting."""
        result = sql_tool.execute("SELECT name FROM users LIMIT 1")
        md = result.to_markdown_table()
        assert "Alice" in md
        assert "|" in md

    def test_empty_result(self, sql_tool: SQLTool) -> None:
        """Test handling of empty result sets."""
        result = sql_tool.execute("SELECT * FROM users WHERE age > 100")
        assert result.row_count == 0
        assert result.to_markdown_table() == "*No results*"

    def test_get_schema(self, sql_tool: SQLTool) -> None:
        """Test schema introspection."""
        schema = sql_tool.get_schema()
        assert "users" in schema
        column_names = [col["name"] for col in schema["users"]]
        assert "id" in column_names
        assert "name" in column_names


class TestQueryResult:
    """Tests for QueryResult formatting."""

    def test_to_dicts_empty(self) -> None:
        """Test to_dicts with no rows."""
        result = QueryResult(columns=["a", "b"], rows=[], row_count=0)
        assert result.to_dicts() == []

    def test_truncated_flag(self) -> None:
        """Test truncation indicator in markdown output."""
        result = QueryResult(
            columns=["x"],
            rows=[(1,), (2,)],
            row_count=100,
            truncated=True,
        )
        md = result.to_markdown_table()
        assert "truncated" in md.lower()
