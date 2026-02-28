"""
DuckDuckGo Search Tool -- Web Search Without API Keys
======================================================

WHY DUCKDUCKGO?
  Most search tools require API keys (Google, Bing, Serper). DuckDuckGo's
  search is completely free, requires NO API key, and respects user privacy.
  This makes it perfect for open-source projects and local development.

HOW IT WORKS:
  1. The ``duckduckgo-search`` library sends requests to DuckDuckGo's HTML search.
  2. It parses the response and returns structured results (title, URL, snippet).
  3. We wrap this in a LangChain ``@tool`` so LangGraph agents can call it.

RATE LIMITING:
  DuckDuckGo may rate-limit aggressive usage. We add a small delay between
  requests and handle rate-limit errors gracefully. For production use,
  consider caching results or switching to a paid API.

LEARNING RESOURCES:
  - duckduckgo-search library: https://github.com/deedy5/duckduckgo_search
  - LangChain Tools: https://python.langchain.com/docs/concepts/tools/
  - DuckDuckGo search operators: https://duckduckgo.com/duckduckgo-help-pages/results/syntax/
  - VIDEO: "Build AI Agents with Tools" -- https://www.youtube.com/watch?v=cN9S6CYjmi8
  - VIDEO: "LangChain Tools Deep Dive" -- https://www.youtube.com/watch?v=q-HNphrWsDE
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.tools import tool

from agentexplorr.core import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal search helpers
# ---------------------------------------------------------------------------

def _execute_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Execute a DuckDuckGo search and return parsed results.

    This is separated from the @tool function so it can be reused
    by other parts of the codebase (e.g., the multi-agent researcher).

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (1-10).

    Returns:
        List of dicts with keys: "title", "url", "snippet".

    Raises:
        RuntimeError: If the search fails after retries.
    """
    # Import here to make the tool optional -- if duckduckgo_search isn't
    # installed, the error message is clear rather than a cryptic ImportError
    # at module load time.
    try:
        from duckduckgo_search import DDGS
    except ImportError as exc:
        raise ImportError(
            "duckduckgo-search is required for web search. "
            "Install it with: uv sync --extra agents"
        ) from exc

    max_results = max(1, min(max_results, 10))

    # Retry logic: DuckDuckGo occasionally returns empty results or rate-limits.
    # We retry up to 3 times with exponential backoff.
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                # ddgs.text() returns an iterator of result dicts.
                # Each dict has: "title", "href", "body".
                raw_results: list[dict[str, str]] = list(
                    ddgs.text(
                        keywords=query,
                        max_results=max_results,
                        # "moderate" safe search is a reasonable default
                        safesearch="moderate",
                    )
                )

            if not raw_results:
                logger.warning("empty_search_results", query=query, attempt=attempt)
                # DuckDuckGo sometimes returns empty on first try
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return [{"title": "No results", "url": "", "snippet": f"No results found for: {query}"}]

            # Normalize the key names for consistency across the codebase.
            # DuckDuckGo returns "href" and "body"; we rename to "url" and "snippet".
            results: list[dict[str, str]] = []
            for r in raw_results:
                results.append({
                    "title": r.get("title", "No title"),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "No snippet available")),
                })

            logger.info(
                "search_completed",
                query=query,
                num_results=len(results),
            )
            return results

        except Exception as e:
            last_error = e
            logger.warning(
                "search_attempt_failed",
                query=query,
                attempt=attempt,
                error=str(e),
            )
            # Exponential backoff: 1s, 2s, 4s
            time.sleep(2**attempt)

    # All retries exhausted
    error_msg = f"Search failed after 3 attempts for query '{query}': {last_error}"
    logger.error("search_failed", query=query, error=str(last_error))
    raise RuntimeError(error_msg)


# ---------------------------------------------------------------------------
# LangChain Tool (used by agents)
# ---------------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return relevant results.

    Use this tool when you need to find current information, facts, or data
    from the internet. Provide a clear, specific search query for best results.

    Args:
        query: The search query. Be specific for better results.
              Example: "Python LangGraph tutorial 2024" rather than "langgraph".

    Returns:
        Formatted string of search results with titles, URLs, and snippets.
    """
    try:
        results = _execute_search(query, max_results=5)
    except (RuntimeError, ImportError) as e:
        return f"Search error: {e}"

    # Format results as a readable string for the LLM.
    # We include numbering so the agent can reference specific results
    # (e.g., "Based on result #3...").
    formatted_parts: list[str] = []
    for i, result in enumerate(results, start=1):
        formatted_parts.append(
            f"[{i}] {result['title']}\n"
            f"    URL: {result['url']}\n"
            f"    {result['snippet']}"
        )

    return "\n\n".join(formatted_parts)


@tool
def web_search_detailed(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web and return structured results as a list of dictionaries.

    Unlike ``web_search``, this returns structured data rather than formatted text.
    Use this when you need to process individual results programmatically.

    Args:
        query: The search query string.
        max_results: Number of results to return (1-10, default 5).

    Returns:
        List of result dicts, each with "title", "url", and "snippet" keys.
    """
    try:
        return _execute_search(query, max_results=max_results)
    except (RuntimeError, ImportError) as e:
        return [{"title": "Error", "url": "", "snippet": str(e)}]


# ---------------------------------------------------------------------------
# Convenience function for direct use (outside agent context)
# ---------------------------------------------------------------------------

def search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Public API for searching DuckDuckGo outside of an agent context.

    This is the same as ``_execute_search`` but with a friendlier name
    for direct use in scripts, notebooks, or other modules.

    Example:
        >>> from agentexplorr.agents.tools.search import search
        >>> results = search("what is ReAct prompting")
        >>> for r in results:
        ...     print(f"{r['title']}: {r['url']}")

    Args:
        query: The search query.
        max_results: Maximum number of results (1-10).

    Returns:
        List of result dicts with "title", "url", "snippet" keys.
    """
    return _execute_search(query, max_results=max_results)
