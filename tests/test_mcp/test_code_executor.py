"""
Tests for the Code Executor
=============================

Verify that the sandbox properly restricts dangerous operations
while allowing safe computation.
"""

from __future__ import annotations

from agentexplorr.mcp.tools.code_executor import CodeExecutor


class TestCodeExecutor:
    """Tests for sandboxed code execution."""

    def setup_method(self) -> None:
        """Create a fresh executor for each test."""
        self.executor = CodeExecutor()

    def test_basic_print(self) -> None:
        """Test simple print statement."""
        result = self.executor.execute("print('hello world')")
        assert result.success
        assert "hello world" in result.stdout

    def test_math_operations(self) -> None:
        """Test math computations."""
        result = self.executor.execute("""
import math
print(f"pi = {math.pi:.4f}")
print(f"sqrt(2) = {math.sqrt(2):.4f}")
""")
        assert result.success
        assert "3.1416" in result.stdout
        assert "1.4142" in result.stdout

    def test_blocked_import_os(self) -> None:
        """Test that 'import os' is blocked."""
        result = self.executor.execute("import os")
        assert not result.success
        assert "not allowed" in str(result.error).lower()

    def test_blocked_import_subprocess(self) -> None:
        """Test that 'import subprocess' is blocked."""
        result = self.executor.execute("import subprocess")
        assert not result.success

    def test_syntax_error(self) -> None:
        """Test handling of syntax errors in user code."""
        result = self.executor.execute("def foo(:")
        assert not result.success
        assert "Syntax Error" in str(result.error)

    def test_runtime_error(self) -> None:
        """Test handling of runtime errors."""
        result = self.executor.execute("x = 1 / 0")
        assert not result.success
        assert "ZeroDivisionError" in str(result.error)

    def test_allowed_modules(self) -> None:
        """Test that safe modules can be imported."""
        result = self.executor.execute("""
import json
import statistics
data = [1, 2, 3, 4, 5]
print(f"mean = {statistics.mean(data)}")
print(json.dumps({"result": "ok"}))
""")
        assert result.success
        assert "mean = 3" in result.stdout

    def test_list_comprehension(self) -> None:
        """Test Python features work in sandbox."""
        result = self.executor.execute("""
squares = [x**2 for x in range(5)]
print(squares)
""")
        assert result.success
        assert "[0, 1, 4, 9, 16]" in result.stdout
