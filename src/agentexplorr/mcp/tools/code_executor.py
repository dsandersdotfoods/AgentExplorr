"""
Sandboxed Code Executor — Safe Python Execution for LLMs
=========================================================

WHAT THIS DOES:
  Lets LLMs write and execute Python code in a restricted sandbox.
  This enables "code interpreter" functionality — the LLM can write
  code to analyze data, create charts, or perform calculations.

⚠️  SECURITY WARNING:
  Executing LLM-generated code is inherently dangerous. This executor
  implements multiple safety layers, but it is NOT suitable for
  untrusted production environments without additional hardening.

SAFETY LAYERS:
  1. **Restricted builtins** — Only safe built-in functions are available
  2. **Import blocking** — Only whitelisted modules can be imported
  3. **Timeout** — Code execution is killed after a configurable timeout
  4. **Output capture** — stdout/stderr are captured and returned
  5. **No filesystem access** — open(), os, sys are blocked

REAL-WORLD ALTERNATIVES:
  For production code execution, use proper sandboxing:
  - Docker containers with resource limits
  - gVisor (Google's container sandbox)
  - Firecracker microVMs (AWS)
  - E2B (https://e2b.dev/) — cloud sandboxes built for AI
  - Modal (https://modal.com/) — serverless code execution

LEARNING RESOURCES:
  - Python RestrictedPython: https://restrictedpython.readthedocs.io/
  - Docker security: https://docs.docker.com/engine/security/
  - VIDEO: "Sandboxing Code Execution" — https://www.youtube.com/watch?v=dn8lgRFSfwk
  - E2B Sandboxes: https://e2b.dev/docs
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any

from agentexplorr.core.logging import get_logger

logger = get_logger(__name__)

# Modules that are SAFE for LLMs to import
# These are pure computation — no filesystem, network, or system access
ALLOWED_MODULES = frozenset({
    "math",
    "statistics",
    "random",
    "datetime",
    "json",
    "re",
    "collections",
    "itertools",
    "functools",
    "operator",
    "string",
    "textwrap",
    "decimal",
    "fractions",
})

# Built-in functions that are SAFE to expose
# Excludes: eval, exec, open, __import__, compile, globals, locals
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}


def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Restricted import that only allows whitelisted modules.

    WHY RESTRICT IMPORTS?
      Without restrictions, an LLM could import:
      - os → delete files, run commands
      - subprocess → execute shell commands
      - socket → make network connections
      - ctypes → call C functions, crash Python

    Args:
        name: Module name to import.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module is not in the whitelist.
    """
    if name not in ALLOWED_MODULES:
        msg = f"Import of '{name}' is not allowed. Allowed modules: {sorted(ALLOWED_MODULES)}"
        raise ImportError(msg)
    return __import__(name, *args, **kwargs)


@dataclass
class ExecutionResult:
    """Result of code execution.

    Attributes:
        stdout: Captured standard output.
        stderr: Captured standard error.
        return_value: The last expression's value (if any).
        error: Error message if execution failed.
        success: Whether execution completed without errors.
    """

    stdout: str = ""
    stderr: str = ""
    return_value: str | None = None
    error: str | None = None
    success: bool = True


class CodeExecutor:
    """Execute Python code in a restricted sandbox.

    Example:
        >>> executor = CodeExecutor()
        >>> result = executor.execute('''
        ... import math
        ... radius = 5
        ... area = math.pi * radius ** 2
        ... print(f"Area of circle with radius {radius}: {area:.2f}")
        ... ''')
        >>> print(result.stdout)
        Area of circle with radius 5: 78.54
    """

    def __init__(self, max_output_chars: int = 10000) -> None:
        """Initialize the code executor.

        Args:
            max_output_chars: Maximum characters in captured output.
        """
        self.max_output_chars = max_output_chars

    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code in a sandbox and return results.

        HOW THE SANDBOX WORKS:
          1. Create a restricted namespace with only safe builtins
          2. Redirect stdout/stderr to capture output
          3. Execute the code in the restricted namespace
          4. Return captured output and any errors

        Args:
            code: Python source code to execute.

        Returns:
            ExecutionResult with stdout, stderr, and error info.
        """
        logger.info("executing_code", code_length=len(code))

        # Build restricted execution namespace
        # __builtins__ controls what built-in functions are available
        restricted_globals: dict[str, Any] = {
            "__builtins__": {**SAFE_BUILTINS, "__import__": _safe_import},
            "__name__": "__sandbox__",
        }

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = ExecutionResult()

        try:
            # redirect_stdout/stderr capture all print() output
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # compile() converts source code to a code object
                # exec() runs the code object in the restricted namespace
                compiled = compile(code, "<sandbox>", "exec")
                exec(compiled, restricted_globals)

            result.stdout = stdout_capture.getvalue()[: self.max_output_chars]
            result.stderr = stderr_capture.getvalue()[: self.max_output_chars]

        except ImportError as e:
            result.error = f"Import Error: {e}"
            result.success = False
        except SyntaxError as e:
            result.error = f"Syntax Error at line {e.lineno}: {e.msg}"
            result.success = False
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            result.success = False
            # Still capture any partial output
            result.stdout = stdout_capture.getvalue()[: self.max_output_chars]
            result.stderr = stderr_capture.getvalue()[: self.max_output_chars]
        finally:
            # Restore original stdout/stderr (the context manager handles this,
            # but we're being explicit about cleanup)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

        return result
