"""
Web Scraper Tool -- Extract Text Content from URLs
====================================================

WHY A WEB SCRAPER?
  Search tools return snippets, but agents often need the FULL content of a
  webpage. For example, after finding a relevant article via DuckDuckGo, the
  agent might need to read the entire article to answer a detailed question.

  This tool fetches a URL and extracts clean, readable text using
  BeautifulSoup. It strips out HTML tags, scripts, styles, and navigation
  elements, returning just the meaningful content.

HOW IT WORKS:
  1. ``httpx`` makes an async-capable HTTP GET request to the URL.
  2. ``BeautifulSoup`` parses the raw HTML into a navigable tree.
  3. We remove noise elements: <script>, <style>, <nav>, <footer>, etc.
  4. We extract text from the remaining elements and clean up whitespace.
  5. The result is truncated to a configurable max length (LLMs have context
     limits, and most webpages have A LOT of text).

WHY HTTPX OVER REQUESTS?
  - httpx supports both sync and async (future-proof for async agents)
  - httpx follows the same API as requests (easy migration)
  - httpx has HTTP/2 support and better timeout handling
  - httpx is actively maintained and modern

ETHICAL SCRAPING:
  - We set a descriptive User-Agent header (don't pretend to be a browser)
  - We respect reasonable timeouts (don't hang on slow servers)
  - We don't bypass robots.txt (though we don't check it either -- a
    production system should use robotparser)
  - We limit request frequency (one URL at a time)

LEARNING RESOURCES:
  - httpx docs: https://www.python-httpx.org/
  - BeautifulSoup docs: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
  - Web scraping best practices: https://scrapfly.io/blog/web-scraping-best-practices/
  - VIDEO: "Python Web Scraping with BeautifulSoup" -- https://www.youtube.com/watch?v=XVv6mJpFOb0
  - VIDEO: "httpx: The Modern Python HTTP Client" -- https://www.youtube.com/watch?v=OPyoXx0yA0I
  - PAPER: "WebGPT: Browser-assisted question-answering" -- https://arxiv.org/abs/2112.09332
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from langchain_core.tools import tool

from agentexplorr.core import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Maximum characters to return from a scraped page.
# GPT-family and Llama models typically have 4k-128k token context windows.
# 1 token ~= 4 chars, so 10,000 chars ~= 2,500 tokens -- leaves plenty of
# room for the agent's prompt and reasoning.
DEFAULT_MAX_CHARS: int = 10_000

# HTTP request timeout in seconds. 15s is generous enough for most sites
# but won't hang forever on unresponsive servers.
DEFAULT_TIMEOUT: float = 15.0

# User-Agent header: Be honest about what we are. Some sites block
# requests without a User-Agent, so we include one.
USER_AGENT: str = (
    "AgentExplorr/0.1 (AI Research Tool; +https://github.com/agentexplorr) "
    "httpx/0.27"
)

# HTML tags that contain noise rather than content.
# We remove these entirely before extracting text.
NOISE_TAGS: set[str] = {
    "script",   # JavaScript code
    "style",    # CSS rules
    "nav",      # Navigation menus
    "footer",   # Footer boilerplate
    "header",   # Header/banner areas (often just logos/nav)
    "aside",    # Sidebars, ads
    "form",     # Forms (login, search boxes)
    "noscript", # No-JS fallback content
    "iframe",   # Embedded frames
    "svg",      # SVG graphics
    "meta",     # Meta tags
    "link",     # Link tags (stylesheets etc.)
}


# ---------------------------------------------------------------------------
# Internal scraping logic
# ---------------------------------------------------------------------------

def _validate_url(url: str) -> str:
    """Validate and normalize a URL.

    We perform basic validation to prevent:
      - Non-HTTP(S) schemes (file://, ftp://, etc.)
      - Missing schemes (auto-prepend https://)
      - Localhost/private IPs (SSRF prevention -- though not exhaustive)

    Args:
        url: The URL to validate.

    Returns:
        The validated, normalized URL.

    Raises:
        ValueError: If the URL is invalid or disallowed.
    """
    url = url.strip()

    if not url:
        raise ValueError("URL cannot be empty")

    # Auto-prepend https:// if no scheme is provided
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)

    # Only allow HTTP and HTTPS
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Unsupported URL scheme: '{parsed.scheme}'. "
            "Only http:// and https:// are allowed."
        )

    # Must have a hostname
    if not parsed.hostname:
        raise ValueError(f"Invalid URL (no hostname): {url}")

    # Basic SSRF prevention: block common private/internal addresses.
    # NOTE: This is NOT exhaustive. A production system should use a
    # proper SSRF prevention library or network-level controls.
    blocked_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "[::1]"}
    if parsed.hostname.lower() in blocked_hosts:
        raise ValueError(
            f"Blocked URL: '{parsed.hostname}' is a local address. "
            "Scraping local addresses is not allowed for security reasons."
        )

    return url


def _extract_text(html: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Extract clean text content from raw HTML.

    This is where BeautifulSoup does the heavy lifting. We:
      1. Parse the HTML into a tree structure
      2. Remove noise elements (scripts, styles, nav, etc.)
      3. Extract text from what remains
      4. Clean up whitespace (collapse multiple newlines, strip edges)
      5. Truncate to max_chars

    WHY BEAUTIFULSOUP?
      BS4 is the de facto standard for HTML parsing in Python. It handles
      malformed HTML gracefully (most real-world HTML is messy), supports
      multiple parsers (html.parser, lxml, html5lib), and has an intuitive
      API. We use Python's built-in ``html.parser`` to avoid extra deps.

    Args:
        html: Raw HTML string.
        max_chars: Maximum characters to return.

    Returns:
        Cleaned text content.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "beautifulsoup4 is required for web scraping. "
            "Install it with: uv sync --extra agents"
        ) from exc

    # Parse HTML using Python's built-in parser (no extra dependencies).
    # "html.parser" is slower than "lxml" but doesn't require C libraries.
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements entirely from the tree.
    # decompose() removes the tag AND its contents from the tree.
    for tag_name in NOISE_TAGS:
        for element in soup.find_all(tag_name):
            element.decompose()

    # Also remove HTML comments (often contain templating artifacts)
    from bs4 import Comment
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Extract text. separator="\n" puts a newline between elements
    # rather than jamming everything together.
    text = soup.get_text(separator="\n", strip=True)

    # Clean up excessive whitespace:
    # - Collapse 3+ consecutive newlines into 2 (preserve paragraph breaks)
    # - Strip leading/trailing whitespace from each line
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    consecutive_empty = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            consecutive_empty += 1
            if consecutive_empty <= 2:
                cleaned_lines.append("")
        else:
            consecutive_empty = 0
            cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines).strip()

    # Truncate to max_chars, adding an indicator if truncated
    if len(text) > max_chars:
        # Try to truncate at a sentence boundary
        truncated = text[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.8:
            # Found a sentence end in the last 20% -- use it
            truncated = truncated[: last_period + 1]

        text = truncated + "\n\n[Content truncated -- full page is longer]"

    return text


def _fetch_url(url: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Fetch raw HTML content from a URL using httpx.

    Args:
        url: The URL to fetch (must be validated first).
        timeout: Request timeout in seconds.

    Returns:
        Raw HTML string.

    Raises:
        RuntimeError: If the HTTP request fails.
    """
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required for web scraping. "
            "Install it with: uv sync --extra agents"
        ) from exc

    try:
        # httpx.Client is a context manager -- ensures connections are
        # properly closed even if an exception occurs.
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,  # Follow HTTP 301/302 redirects
            max_redirects=5,        # But not too many (prevent redirect loops)
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        ) as client:
            response = client.get(url)
            # Raise an exception for HTTP 4xx/5xx status codes
            response.raise_for_status()
            return response.text

    except httpx.TimeoutException:
        raise RuntimeError(
            f"Request timed out after {timeout}s for URL: {url}. "
            "The server may be slow or unresponsive."
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"HTTP {e.response.status_code} error for URL: {url}. "
            f"Server responded with: {e.response.reason_phrase}"
        )
    except httpx.RequestError as e:
        raise RuntimeError(
            f"Failed to fetch URL: {url}. Error: {e}. "
            "Check your internet connection and that the URL is valid."
        )


def scrape_url(
    url: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, str]:
    """Scrape a URL and return structured results.

    This is the main internal function that combines URL validation,
    fetching, and text extraction.

    Args:
        url: The URL to scrape.
        max_chars: Maximum characters of content to return.
        timeout: HTTP request timeout in seconds.

    Returns:
        Dict with keys: "url", "title", "content", "status".

    Raises:
        ValueError: If the URL is invalid.
        RuntimeError: If fetching fails.
    """
    validated_url = _validate_url(url)

    logger.info("scraping_url", url=validated_url, max_chars=max_chars)

    html = _fetch_url(validated_url, timeout=timeout)

    # Extract the page title separately (useful metadata)
    title = ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = title_tag.string.strip()
    except Exception:
        pass  # Title extraction is best-effort

    content = _extract_text(html, max_chars=max_chars)

    logger.info(
        "scrape_completed",
        url=validated_url,
        title=title,
        content_length=len(content),
    )

    return {
        "url": validated_url,
        "title": title,
        "content": content,
        "status": "success",
    }


# ---------------------------------------------------------------------------
# LangChain Tool (used by agents)
# ---------------------------------------------------------------------------

@tool
def web_scrape(url: str) -> str:
    """Fetch and extract readable text content from a webpage.

    Use this tool when you need to read the full content of a specific webpage.
    For example, after finding a relevant URL via web search, use this tool
    to read the article.

    The tool removes HTML markup, scripts, styles, and navigation elements,
    returning only the main text content.

    Args:
        url: The URL of the webpage to scrape.
             Example: "https://en.wikipedia.org/wiki/Artificial_intelligence"

    Returns:
        The extracted text content from the page, or an error message.
    """
    try:
        result = scrape_url(url)
        # Format for the LLM: include title and URL as context
        header = f"Page: {result['title']}\nURL: {result['url']}\n"
        separator = "-" * 40
        return f"{header}{separator}\n{result['content']}"

    except ValueError as e:
        return f"Invalid URL: {e}"
    except ImportError as e:
        return f"Missing dependency: {e}"
    except RuntimeError as e:
        return f"Scraping error: {e}"
    except Exception as e:
        logger.error("web_scrape_unexpected_error", url=url, error=str(e))
        return f"Unexpected error while scraping {url}: {e}"
