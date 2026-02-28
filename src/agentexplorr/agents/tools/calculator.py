"""
Safe Math Calculator Tool -- Expression Evaluation Without eval()
=================================================================

WHY NOT JUST USE eval()?
  ``eval()`` executes **arbitrary** Python code. If an LLM generates a
  malicious expression like ``__import__('os').system('rm -rf /')``, eval()
  will happily execute it. This is a critical security vulnerability known
  as **code injection**.

  Instead, we use Python's ``ast`` (Abstract Syntax Tree) module to parse
  the expression into a tree, validate that every node is a safe math
  operation, and then evaluate only the whitelisted operations. This is
  the same approach used by production systems that need to evaluate
  user-supplied expressions safely.

HOW IT WORKS:
  1. Parse the expression string into an AST using ``ast.parse()``.
  2. Walk every node in the AST tree.
  3. If any node is NOT in our whitelist (numbers, basic operators, math
     functions), we reject the expression immediately.
  4. If all nodes are safe, evaluate the expression using a restricted
     namespace that only contains math functions.

SUPPORTED OPERATIONS:
  - Arithmetic: +, -, *, /, //, %, **
  - Comparison: <, >, <=, >=, ==, !=
  - Unary: +x, -x
  - Math functions: sin, cos, tan, sqrt, log, log10, abs, round, min, max,
                    pi, e, ceil, floor, factorial
  - Parentheses for grouping

SECURITY MODEL:
  We whitelist AST node types rather than blacklisting. This is the safer
  approach because new Python features won't accidentally become exploitable.
  If a node type isn't in our whitelist, it's rejected -- period.

LEARNING RESOURCES:
  - Python ast module: https://docs.python.org/3/library/ast.html
  - Why eval() is dangerous: https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
  - LangChain custom tools: https://python.langchain.com/docs/how_to/custom_tools/
  - VIDEO: "Python AST Module Explained" -- https://www.youtube.com/watch?v=Yq3wTWkoaYY
  - VIDEO: "Building Safe Eval in Python" -- https://www.youtube.com/watch?v=bqMYsh_32wY
  - PAPER: "Tool-Augmented Language Models" -- https://arxiv.org/abs/2302.04761
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from langchain_core.tools import tool

from agentexplorr.core import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Safe math namespace -- the ONLY functions/constants the calculator can use
# ---------------------------------------------------------------------------

# This dict maps string names to actual Python callables/values.
# When the evaluator encounters a function call like "sqrt(16)", it looks
# up "sqrt" in this dict and calls math.sqrt(16).
SAFE_MATH_NAMESPACE: dict[str, Any] = {
    # Trigonometric functions
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    # Exponential and logarithmic
    "exp": math.exp,
    "log": math.log,      # Natural log (base e)
    "log2": math.log2,
    "log10": math.log10,
    "sqrt": math.sqrt,
    # Rounding and absolute value
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    # Combinatorics
    "factorial": math.factorial,
    # Aggregation (useful for lists of numbers)
    "min": min,
    "max": max,
    "sum": sum,
    # Constants
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,       # 2 * pi
    "inf": math.inf,
    # Power (alternative to ** operator)
    "pow": pow,
}

# ---------------------------------------------------------------------------
# Supported binary operators -- maps AST operator types to callables
# ---------------------------------------------------------------------------

SAFE_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,          # +
    ast.Sub: operator.sub,          # -
    ast.Mult: operator.mul,         # *
    ast.Div: operator.truediv,      # /
    ast.FloorDiv: operator.floordiv,  # //
    ast.Mod: operator.mod,          # %
    ast.Pow: operator.pow,          # **
}

SAFE_UNARY_OPERATORS: dict[type, Any] = {
    ast.UAdd: operator.pos,   # +x
    ast.USub: operator.neg,   # -x
}

SAFE_COMPARISON_OPERATORS: dict[type, Any] = {
    ast.Eq: operator.eq,       # ==
    ast.NotEq: operator.ne,   # !=
    ast.Lt: operator.lt,       # <
    ast.LtE: operator.le,     # <=
    ast.Gt: operator.gt,       # >
    ast.GtE: operator.ge,     # >=
}


# ---------------------------------------------------------------------------
# AST-based safe evaluator
# ---------------------------------------------------------------------------

class SafeExpressionError(Exception):
    """Raised when an expression contains disallowed operations."""
    pass


def _safe_eval_node(node: ast.AST) -> int | float | bool | list[Any]:
    """Recursively evaluate an AST node, allowing only safe operations.

    This is a recursive descent evaluator that walks the AST tree.
    At each node, we check what kind of node it is and handle it:
      - Numbers: return the value directly
      - BinOp (a + b): evaluate left and right, then apply the operator
      - Call (sqrt(x)): look up the function in our safe namespace
      - etc.

    If we encounter any node type not in our whitelist, we raise
    SafeExpressionError. This is the core security mechanism.

    Args:
        node: An AST node to evaluate.

    Returns:
        The computed numeric result.

    Raises:
        SafeExpressionError: If the expression contains unsafe operations.
    """
    # --- Numeric literal: 42, 3.14, etc. ---
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise SafeExpressionError(
            f"Unsupported constant type: {type(node.value).__name__}. "
            "Only int and float are allowed."
        )

    # --- Binary operation: a + b, x * y, etc. ---
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise SafeExpressionError(
                f"Unsupported operator: {op_type.__name__}"
            )
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)

        # Safety check: prevent absurdly large exponentiation
        # e.g., 10 ** 10000 would consume huge memory
        if op_type is ast.Pow:
            if isinstance(right, (int, float)) and abs(right) > 1000:
                raise SafeExpressionError(
                    f"Exponent too large: {right}. Maximum allowed is 1000."
                )

        try:
            return SAFE_OPERATORS[op_type](left, right)
        except ZeroDivisionError:
            raise SafeExpressionError("Division by zero")
        except OverflowError:
            raise SafeExpressionError("Result too large (overflow)")

    # --- Unary operation: -x, +x ---
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_UNARY_OPERATORS:
            raise SafeExpressionError(
                f"Unsupported unary operator: {op_type.__name__}"
            )
        operand = _safe_eval_node(node.operand)
        return SAFE_UNARY_OPERATORS[op_type](operand)

    # --- Comparison: a < b, x == y ---
    # Python AST represents chained comparisons (a < b < c) as a single
    # Compare node with multiple operators and comparators.
    if isinstance(node, ast.Compare):
        left = _safe_eval_node(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in SAFE_COMPARISON_OPERATORS:
                raise SafeExpressionError(
                    f"Unsupported comparison: {op_type.__name__}"
                )
            right = _safe_eval_node(comparator)
            if not SAFE_COMPARISON_OPERATORS[op_type](left, right):
                return False
            left = right
        return True

    # --- Function call: sqrt(16), log(100, 10) ---
    if isinstance(node, ast.Call):
        # We only allow simple function calls like sqrt(x), not method
        # calls like obj.method() or chained calls like f()(x).
        if not isinstance(node.func, ast.Name):
            raise SafeExpressionError(
                "Only simple function calls are allowed (e.g., sqrt(x)). "
                "Method calls and attribute access are not permitted."
            )

        func_name = node.func.id
        if func_name not in SAFE_MATH_NAMESPACE:
            raise SafeExpressionError(
                f"Unknown function: '{func_name}'. "
                f"Allowed functions: {', '.join(sorted(SAFE_MATH_NAMESPACE.keys()))}"
            )

        func = SAFE_MATH_NAMESPACE[func_name]
        if not callable(func):
            raise SafeExpressionError(
                f"'{func_name}' is a constant, not a function. "
                f"Use it without parentheses: {func_name}"
            )

        # Evaluate all arguments
        args = [_safe_eval_node(arg) for arg in node.args]

        # No keyword arguments allowed (keeps things simple and safe)
        if node.keywords:
            raise SafeExpressionError(
                "Keyword arguments are not supported in calculator functions."
            )

        try:
            return func(*args)
        except (ValueError, TypeError, OverflowError) as e:
            raise SafeExpressionError(
                f"Error calling {func_name}({', '.join(str(a) for a in args)}): {e}"
            )

    # --- Variable/constant name: pi, e ---
    if isinstance(node, ast.Name):
        name = node.id
        if name not in SAFE_MATH_NAMESPACE:
            raise SafeExpressionError(
                f"Unknown variable: '{name}'. "
                f"Allowed names: {', '.join(sorted(SAFE_MATH_NAMESPACE.keys()))}"
            )
        return SAFE_MATH_NAMESPACE[name]

    # --- List literal: [1, 2, 3] (useful for min/max/sum) ---
    if isinstance(node, ast.List):
        return [_safe_eval_node(elt) for elt in node.elts]

    # --- Tuple literal: (1, 2, 3) ---
    if isinstance(node, ast.Tuple):
        return [_safe_eval_node(elt) for elt in node.elts]

    # --- Expression wrapper (top-level) ---
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    # --- Anything else is NOT ALLOWED ---
    # This is the security boundary. Any AST node type we haven't
    # explicitly handled above is rejected. This includes:
    # - Import statements
    # - Function definitions
    # - Class definitions
    # - Attribute access (obj.attr)
    # - Subscripts (list[0])
    # - Assignments
    # - etc.
    raise SafeExpressionError(
        f"Disallowed expression type: {type(node).__name__}. "
        "Only arithmetic expressions, math functions, and comparisons "
        "are allowed."
    )


def safe_evaluate(expression: str) -> int | float | bool:
    """Parse and safely evaluate a mathematical expression.

    This is the main entry point for the safe evaluator. It:
      1. Strips whitespace and validates the expression isn't empty
      2. Parses it into an AST (catching syntax errors)
      3. Evaluates the AST using our safe recursive evaluator

    Args:
        expression: A mathematical expression string.
                   Examples: "2 + 3 * 4", "sqrt(16) + pi", "log(100, 10)"

    Returns:
        The numeric result of the expression.

    Raises:
        SafeExpressionError: If the expression is unsafe or invalid.

    Examples:
        >>> safe_evaluate("2 + 3 * 4")
        14
        >>> safe_evaluate("sqrt(16)")
        4.0
        >>> safe_evaluate("pi * 2")
        6.283185307179586
        >>> safe_evaluate("log(1000, 10)")
        2.9999999999999996
    """
    expression = expression.strip()
    if not expression:
        raise SafeExpressionError("Empty expression")

    # Limit expression length to prevent denial-of-service via absurdly
    # long expressions that would consume too much memory during parsing.
    if len(expression) > 1000:
        raise SafeExpressionError(
            f"Expression too long ({len(expression)} chars). Maximum is 1000."
        )

    try:
        # ast.parse in "eval" mode expects a single expression (not statements).
        # This already prevents things like "import os; os.system('...')"
        # because import is a statement, not an expression.
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise SafeExpressionError(f"Invalid syntax: {e}") from e

    result = _safe_eval_node(tree)

    logger.debug(
        "calculator_evaluated",
        expression=expression,
        result=result,
    )

    return result


# ---------------------------------------------------------------------------
# LangChain Tool (used by agents)
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Use this tool for arithmetic, math functions, and numeric computations.
    The calculator supports standard math operations and common functions.

    Supported operations:
      - Arithmetic: +, -, *, /, //, %, **
      - Functions: sqrt, sin, cos, tan, log, log10, exp, abs, round,
                   ceil, floor, factorial, min, max, sum, pow
      - Constants: pi, e, tau, inf
      - Grouping: parentheses ()

    Args:
        expression: A mathematical expression to evaluate.
                   Examples: "2 + 3 * 4", "sqrt(144)", "pi * 5**2",
                            "log(1000, 10)", "min([3, 1, 4, 1, 5])"

    Returns:
        The result as a string, or an error message if evaluation fails.
    """
    try:
        result = safe_evaluate(expression)

        # Format the result nicely:
        # - Integers should not have decimal points (42, not 42.0)
        # - Floats should have reasonable precision
        if isinstance(result, bool):
            return str(result)
        elif isinstance(result, float):
            # If the float is actually a whole number, display as int
            if result == int(result) and not math.isinf(result):
                return str(int(result))
            # Otherwise, round to 10 decimal places to avoid floating
            # point noise like 0.30000000000000004
            return str(round(result, 10))
        else:
            return str(result)

    except SafeExpressionError as e:
        return f"Calculator error: {e}"
    except Exception as e:
        # Catch-all for truly unexpected errors
        logger.error("calculator_unexpected_error", expression=expression, error=str(e))
        return f"Unexpected calculator error: {e}"


# ---------------------------------------------------------------------------
# Convenience function for direct use (outside agent context)
# ---------------------------------------------------------------------------

def calculate(expression: str) -> int | float | bool:
    """Public API for evaluating math expressions outside of an agent context.

    Unlike the ``calculator`` tool (which returns strings for LLM consumption),
    this function returns actual Python numeric types.

    Example:
        >>> from agentexplorr.agents.tools.calculator import calculate
        >>> calculate("sqrt(144) + pi")
        15.141592653589793
        >>> calculate("2 ** 10")
        1024

    Args:
        expression: Mathematical expression string.

    Returns:
        The numeric result (int, float, or bool).

    Raises:
        SafeExpressionError: If the expression is unsafe or invalid.
    """
    return safe_evaluate(expression)
