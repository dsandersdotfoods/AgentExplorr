"""
Agent Tools -- The Capabilities That Make Agents Useful
========================================================

WHY DO AGENTS NEED TOOLS?
  A "naked" LLM can only generate text based on its training data. It can't:
    - Look up current information (training data has a cutoff date)
    - Perform precise calculations (LLMs are bad at math)
    - Interact with external systems (APIs, databases, files)

  **Tools** bridge this gap. They give the agent callable functions that
  extend its capabilities beyond pure text generation. The agent decides
  WHEN to call a tool and with WHAT arguments; the tool handles the actual
  execution and returns results.

TOOLS IN THIS MODULE:
  1. ``web_search`` / ``web_search_detailed`` -- Search the web via DuckDuckGo
     (no API key required). Returns titles, URLs, and snippets.

  2. ``calculator`` -- Safely evaluate math expressions using AST parsing.
     Supports arithmetic, trig, logarithms, and common math functions.
     Uses NO eval() -- all expressions are validated before execution.

  3. ``web_scrape`` -- Fetch and extract readable text from a URL using
     httpx + BeautifulSoup. Strips HTML noise, returns clean content.

HOW LANGCHAIN TOOLS WORK:
  LangChain tools are Python functions decorated with ``@tool``. The
  decorator extracts the function's name, docstring, and type hints to
  create a structured tool definition that the LLM can understand.

  When an agent receives a user query, the LLM sees the tool definitions
  (name + description + args schema) and decides whether to call a tool.
  The agent framework handles:
    1. Sending tool definitions to the LLM
    2. Parsing the LLM's tool-call request
    3. Executing the tool function
    4. Returning the result to the LLM

TOOL DESIGN PRINCIPLES:
  - **Clear names**: The LLM reads the tool name to decide when to use it.
    "web_search" is better than "search_v2_final".
  - **Descriptive docstrings**: The docstring IS the tool description. The
    LLM uses it to understand WHEN and HOW to call the tool.
  - **Typed arguments**: Type hints become the argument schema. The LLM
    needs to know what types to provide.
  - **String returns**: Tools should return strings (not complex objects)
    because the LLM needs to read and reason about the results.
  - **Error handling**: Tools should catch exceptions and return error
    messages as strings, not raise exceptions that crash the agent loop.

LEARNING RESOURCES:
  - LangChain Tools Conceptual Guide: https://python.langchain.com/docs/concepts/tools/
  - LangChain Custom Tools: https://python.langchain.com/docs/how_to/custom_tools/
  - Tool Use design patterns: https://www.anthropic.com/research/tool-use-patterns
  - VIDEO: "LangChain Tools Deep Dive" -- https://www.youtube.com/watch?v=q-HNphrWsDE
  - VIDEO: "Building AI Agents with Tools" -- https://www.youtube.com/watch?v=cN9S6CYjmi8
  - PAPER: "Toolformer" (Schick et al., 2023) -- https://arxiv.org/abs/2302.04761
"""

from __future__ import annotations

# Import tool functions so they can be accessed as:
#   from agentexplorr.agents.tools import web_search, calculator, web_scrape
from agentexplorr.agents.tools.calculator import calculate, calculator
from agentexplorr.agents.tools.search import search, web_search, web_search_detailed
from agentexplorr.agents.tools.web_scraper import scrape_url, web_scrape

__all__ = [
    # LangChain @tool decorated functions (for agents)
    "web_search",
    "web_search_detailed",
    "calculator",
    "web_scrape",
    # Plain Python functions (for direct use in scripts/notebooks)
    "search",
    "calculate",
    "scrape_url",
]
