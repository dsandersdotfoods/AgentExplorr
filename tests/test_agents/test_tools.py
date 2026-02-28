"""
Tests for Agent Tools
======================

These tests verify that agent tools work correctly WITHOUT making
real API calls or network requests. Everything is mocked.

TESTING PHILOSOPHY:
  - Test the LOGIC, not the external services
  - Mock all network calls
  - Verify input validation and error handling
  - Keep tests fast (< 1 second each)
"""

from __future__ import annotations

from agentexplorr.agents.tools.calculator import safe_calculate


class TestCalculator:
    """Tests for the safe calculator tool."""

    def test_basic_arithmetic(self) -> None:
        """Test simple math operations."""
        assert safe_calculate("2 + 3") == "5"
        assert safe_calculate("10 - 4") == "6"
        assert safe_calculate("6 * 7") == "42"

    def test_division(self) -> None:
        """Test division including floating point."""
        result = safe_calculate("10 / 3")
        assert result.startswith("3.33")

    def test_exponentiation(self) -> None:
        """Test power operations."""
        assert safe_calculate("2 ** 10") == "1024"

    def test_invalid_expression(self) -> None:
        """Test that invalid expressions return error messages."""
        result = safe_calculate("import os")
        assert "error" in result.lower() or "Error" in result

    def test_empty_input(self) -> None:
        """Test empty input handling."""
        result = safe_calculate("")
        assert "error" in result.lower() or "Error" in result
