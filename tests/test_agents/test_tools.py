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

import math

import pytest

from agentexplorr.agents.tools.calculator import (
    SafeExpressionError,
    calculator,
    safe_evaluate,
)


class TestSafeEvaluate:
    """Tests for the safe_evaluate function (returns Python types)."""

    def test_basic_arithmetic(self) -> None:
        assert safe_evaluate("2 + 3") == 5
        assert safe_evaluate("10 - 4") == 6
        assert safe_evaluate("6 * 7") == 42

    def test_operator_precedence(self) -> None:
        assert safe_evaluate("2 + 3 * 4") == 14
        assert safe_evaluate("(2 + 3) * 4") == 20

    def test_division(self) -> None:
        assert safe_evaluate("10 / 4") == 2.5
        assert safe_evaluate("10 // 3") == 3

    def test_modulo(self) -> None:
        assert safe_evaluate("10 % 3") == 1

    def test_exponentiation(self) -> None:
        assert safe_evaluate("2 ** 10") == 1024

    def test_unary_operators(self) -> None:
        assert safe_evaluate("-5") == -5
        assert safe_evaluate("--5") == 5

    def test_math_functions(self) -> None:
        assert safe_evaluate("sqrt(144)") == 12.0
        assert safe_evaluate("abs(-42)") == 42
        assert safe_evaluate("factorial(5)") == 120

    def test_trig_functions(self) -> None:
        result = safe_evaluate("sin(pi / 2)")
        assert abs(result - 1.0) < 1e-10

    def test_log_functions(self) -> None:
        assert safe_evaluate("log10(1000)") == pytest.approx(3.0)
        assert safe_evaluate("log2(8)") == pytest.approx(3.0)

    def test_constants(self) -> None:
        assert safe_evaluate("pi") == pytest.approx(math.pi)
        assert safe_evaluate("e") == pytest.approx(math.e)

    def test_comparisons(self) -> None:
        assert safe_evaluate("3 > 2") is True
        assert safe_evaluate("1 == 2") is False

    def test_list_aggregation(self) -> None:
        assert safe_evaluate("min([3, 1, 4, 1, 5])") == 1
        assert safe_evaluate("max([3, 1, 4, 1, 5])") == 5
        assert safe_evaluate("sum([1, 2, 3, 4])") == 10

    def test_division_by_zero(self) -> None:
        with pytest.raises(SafeExpressionError, match="Division by zero"):
            safe_evaluate("1 / 0")

    def test_empty_expression(self) -> None:
        with pytest.raises(SafeExpressionError, match="Empty expression"):
            safe_evaluate("")

    def test_rejects_import(self) -> None:
        with pytest.raises(SafeExpressionError):
            safe_evaluate("__import__('os')")

    def test_rejects_attribute_access(self) -> None:
        with pytest.raises(SafeExpressionError):
            safe_evaluate("os.system('echo hi')")

    def test_rejects_string_literals(self) -> None:
        with pytest.raises(SafeExpressionError):
            safe_evaluate("'hello'")

    def test_exponent_too_large(self) -> None:
        with pytest.raises(SafeExpressionError, match="Exponent too large"):
            safe_evaluate("10 ** 5000")

    def test_expression_too_long(self) -> None:
        with pytest.raises(SafeExpressionError, match="too long"):
            safe_evaluate("1 + " * 500 + "1")


class TestCalculatorTool:
    """Tests for the calculator LangChain tool (returns strings)."""

    def test_basic_arithmetic(self) -> None:
        assert calculator.invoke({"expression": "2 + 3"}) == "5"
        assert calculator.invoke({"expression": "6 * 7"}) == "42"

    def test_float_result(self) -> None:
        result = calculator.invoke({"expression": "10 / 3"})
        assert result.startswith("3.33")

    def test_whole_number_float(self) -> None:
        """Floats that are whole numbers display as integers."""
        assert calculator.invoke({"expression": "sqrt(144)"}) == "12"

    def test_error_returns_string(self) -> None:
        result = calculator.invoke({"expression": ""})
        assert "error" in result.lower()

    def test_invalid_expression_returns_string(self) -> None:
        result = calculator.invoke({"expression": "import os"})
        assert "error" in result.lower()
