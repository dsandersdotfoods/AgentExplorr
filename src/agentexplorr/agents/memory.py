"""
Agent Memory — Persistent Conversation History
=================================================

WHAT IS AGENT MEMORY?
  By default, agents are stateless: each ``run()`` call starts from scratch
  with no recollection of previous conversations. Memory adds persistence,
  enabling multi-turn conversations where the agent remembers context.

TYPES OF MEMORY:
  1. **Buffer Memory** (this module) — Stores the last N messages in a list.
     Simple, fast, bounded.  Good for chatbot-style conversations.

  2. **Summary Memory** — Periodically summarizes old messages into a compact
     form.  Good for very long conversations where full history is too large.

  3. **Entity Memory** — Extracts and tracks key entities (people, topics)
     mentioned across the conversation.  Good for personalization.

  4. **Vector Memory** — Embeds messages and retrieves relevant past
     context via similarity search.  Good for long-term recall.

  This module implements Buffer Memory as the foundational building block.
  The others can be built on top of it.

HOW IT INTEGRATES WITH AGENTS:
  Memory wraps around the agent, injecting conversation history into each
  query automatically:

    ┌───────────┐     ┌──────────┐     ┌───────────┐
    │   User    │ ──→ │  Memory  │ ──→ │   Agent   │
    │  "Hi!"    │     │ (inject  │     │  (run w/  │
    └───────────┘     │  history)│     │  context) │
                      └──────────┘     └───────────┘
                           ↑                │
                           │    store       │
                           └────────────────┘

LEARNING RESOURCES:
  - LangChain Memory: https://python.langchain.com/docs/concepts/memory/
  - LangGraph Persistence: https://langchain-ai.github.io/langgraph/concepts/persistence/
  - VIDEO: "AI Agent Memory Systems" — https://www.youtube.com/watch?v=lx-F6GxP0Yg
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentexplorr.core import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    """A single message in the conversation history.

    Attributes:
        role:      Who sent the message ("user", "assistant", "system").
        content:   The message text.
        timestamp: Unix timestamp of when the message was created.
        metadata:  Optional extra data (tool calls, model name, etc.).
    """

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", 0.0),
            metadata=data.get("metadata", {}),
        )


class ConversationMemory:
    """Buffer-based conversation memory with optional persistence.

    Stores the last ``max_messages`` messages and can save/load to disk
    for cross-session persistence.

    Args:
        max_messages: Maximum number of messages to retain. Oldest messages
                      are dropped when the limit is exceeded.
        system_prompt: Optional system prompt prepended to every context window.
        persist_path:  Optional file path for JSON persistence. If provided,
                       messages are saved after each addition and loaded
                       on initialization.

    Usage::

        from agentexplorr.agents.memory import ConversationMemory

        memory = ConversationMemory(max_messages=50)

        # Add messages
        memory.add_user_message("What is RAG?")
        memory.add_assistant_message("RAG stands for Retrieval Augmented Generation...")

        # Get context for the next agent call
        context = memory.get_context_string()

        # Persistence
        memory = ConversationMemory(persist_path="chat_history.json")
        memory.add_user_message("Hello!")
        # History is automatically saved to chat_history.json
    """

    def __init__(
        self,
        max_messages: int = 100,
        system_prompt: str | None = None,
        persist_path: str | Path | None = None,
    ) -> None:
        self._max_messages = max_messages
        self._system_prompt = system_prompt
        self._messages: list[Message] = []
        self._persist_path = Path(persist_path) if persist_path else None

        # Load existing history from disk if available
        if self._persist_path and self._persist_path.exists():
            self._load()

        logger.info(
            "conversation_memory_initialized",
            max_messages=max_messages,
            has_system_prompt=system_prompt is not None,
            persist_path=str(self._persist_path) if self._persist_path else None,
            loaded_messages=len(self._messages),
        )

    def add_user_message(self, content: str, **metadata: Any) -> None:
        """Record a user message."""
        self._add(Message(role="user", content=content, metadata=metadata))

    def add_assistant_message(self, content: str, **metadata: Any) -> None:
        """Record an assistant (agent) message."""
        self._add(Message(role="assistant", content=content, metadata=metadata))

    def add_system_message(self, content: str) -> None:
        """Record a system-level message."""
        self._add(Message(role="system", content=content))

    def _add(self, message: Message) -> None:
        """Add a message and enforce the rolling window."""
        self._messages.append(message)

        # Trim to max_messages (keep newest)
        if len(self._messages) > self._max_messages:
            overflow = len(self._messages) - self._max_messages
            self._messages = self._messages[overflow:]

        # Auto-persist if configured
        if self._persist_path:
            self._save()

    @property
    def messages(self) -> list[Message]:
        """All messages currently in memory."""
        return list(self._messages)

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def get_context_string(self, include_system: bool = True) -> str:
        """Format the conversation history as a string for LLM context injection.

        Args:
            include_system: Whether to prepend the system prompt.

        Returns:
            A formatted multi-line string of the conversation.
        """
        parts: list[str] = []

        if include_system and self._system_prompt:
            parts.append(f"System: {self._system_prompt}")

        for msg in self._messages:
            role_label = msg.role.capitalize()
            parts.append(f"{role_label}: {msg.content}")

        return "\n\n".join(parts)

    def get_messages_for_langchain(self) -> list[dict[str, str]]:
        """Export messages in the format LangChain chat models expect.

        Returns:
            List of dicts with "role" and "content" keys.
        """
        result: list[dict[str, str]] = []

        if self._system_prompt:
            result.append({"role": "system", "content": self._system_prompt})

        for msg in self._messages:
            result.append({"role": msg.role, "content": msg.content})

        return result

    def search(self, query: str, limit: int = 5) -> list[Message]:
        """Simple keyword search over message history.

        For production use, consider embedding-based search (vector memory).

        Args:
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            Messages containing the query (case-insensitive), newest first.
        """
        query_lower = query.lower()
        matches = [
            msg
            for msg in reversed(self._messages)
            if query_lower in msg.content.lower()
        ]
        return matches[:limit]

    def clear(self) -> None:
        """Clear all messages from memory."""
        self._messages.clear()
        if self._persist_path:
            self._save()
        logger.info("conversation_memory_cleared")

    def summarize(self) -> str:
        """Return a brief summary of the conversation state.

        Useful for debugging and monitoring.
        """
        if not self._messages:
            return "Empty conversation"

        user_msgs = sum(1 for m in self._messages if m.role == "user")
        assistant_msgs = sum(1 for m in self._messages if m.role == "assistant")

        first_msg = self._messages[0]
        last_msg = self._messages[-1]

        return (
            f"Conversation: {len(self._messages)} messages "
            f"({user_msgs} user, {assistant_msgs} assistant), "
            f"first: \"{first_msg.content[:50]}...\", "
            f"last: \"{last_msg.content[:50]}...\""
        )

    # --- Persistence ---

    def _save(self) -> None:
        """Save messages to disk as JSON."""
        if not self._persist_path:
            return

        data = {
            "system_prompt": self._system_prompt,
            "messages": [m.to_dict() for m in self._messages],
        }

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load(self) -> None:
        """Load messages from disk."""
        if not self._persist_path or not self._persist_path.exists():
            return

        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            self._messages = [Message.from_dict(m) for m in raw.get("messages", [])]

            # Restore system prompt from file if not set in constructor
            if self._system_prompt is None and raw.get("system_prompt"):
                self._system_prompt = raw["system_prompt"]

            logger.info("conversation_memory_loaded", count=len(self._messages))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("conversation_memory_load_failed", error=str(e))
            self._messages = []

    def __repr__(self) -> str:
        return (
            f"ConversationMemory(messages={len(self._messages)}, "
            f"max={self._max_messages})"
        )
