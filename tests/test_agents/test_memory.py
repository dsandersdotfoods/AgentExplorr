"""
Tests for Agent Memory System
===============================

Tests the ConversationMemory class: message storage, rolling window,
persistence, search, and LangChain export.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentexplorr.agents.memory import ConversationMemory, Message


class TestMessage:
    """Tests for the Message dataclass."""

    def test_create_message(self) -> None:
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp > 0

    def test_round_trip_dict(self) -> None:
        msg = Message(role="assistant", content="Hi!", metadata={"model": "llama3.2"})
        d = msg.to_dict()
        restored = Message.from_dict(d)
        assert restored.role == msg.role
        assert restored.content == msg.content
        assert restored.metadata == msg.metadata


class TestConversationMemory:
    """Tests for the ConversationMemory class."""

    def test_add_and_retrieve(self) -> None:
        mem = ConversationMemory(max_messages=10)
        mem.add_user_message("What is RAG?")
        mem.add_assistant_message("RAG stands for Retrieval Augmented Generation.")

        assert mem.message_count == 2
        assert mem.messages[0].role == "user"
        assert mem.messages[1].role == "assistant"

    def test_rolling_window(self) -> None:
        mem = ConversationMemory(max_messages=3)
        mem.add_user_message("msg1")
        mem.add_user_message("msg2")
        mem.add_user_message("msg3")
        mem.add_user_message("msg4")

        assert mem.message_count == 3
        assert mem.messages[0].content == "msg2"
        assert mem.messages[-1].content == "msg4"

    def test_context_string(self) -> None:
        mem = ConversationMemory(system_prompt="You are helpful.")
        mem.add_user_message("Hello")
        mem.add_assistant_message("Hi there!")

        ctx = mem.get_context_string()
        assert "System: You are helpful." in ctx
        assert "User: Hello" in ctx
        assert "Assistant: Hi there!" in ctx

    def test_context_string_no_system(self) -> None:
        mem = ConversationMemory(system_prompt="System prompt")
        mem.add_user_message("test")
        ctx = mem.get_context_string(include_system=False)
        assert "System:" not in ctx

    def test_langchain_export(self) -> None:
        mem = ConversationMemory(system_prompt="Be concise.")
        mem.add_user_message("Question")
        mem.add_assistant_message("Answer")

        messages = mem.get_messages_for_langchain()
        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "Be concise."}
        assert messages[1] == {"role": "user", "content": "Question"}
        assert messages[2] == {"role": "assistant", "content": "Answer"}

    def test_search(self) -> None:
        mem = ConversationMemory()
        mem.add_user_message("What is machine learning?")
        mem.add_assistant_message("ML is a subset of AI.")
        mem.add_user_message("What about deep learning?")
        mem.add_assistant_message("Deep learning uses neural networks.")

        results = mem.search("deep learning")
        assert len(results) >= 1
        assert any("deep learning" in r.content.lower() for r in results)

    def test_search_no_results(self) -> None:
        mem = ConversationMemory()
        mem.add_user_message("Hello")
        results = mem.search("nonexistent_term_xyz")
        assert len(results) == 0

    def test_clear(self) -> None:
        mem = ConversationMemory()
        mem.add_user_message("Hello")
        mem.add_assistant_message("Hi")
        mem.clear()
        assert mem.message_count == 0

    def test_summarize_empty(self) -> None:
        mem = ConversationMemory()
        assert mem.summarize() == "Empty conversation"

    def test_summarize(self) -> None:
        mem = ConversationMemory()
        mem.add_user_message("First message")
        mem.add_assistant_message("Response")
        summary = mem.summarize()
        assert "2 messages" in summary
        assert "1 user" in summary
        assert "1 assistant" in summary

    def test_persistence_save_and_load(self, tmp_dir: Path) -> None:
        filepath = tmp_dir / "history.json"

        # Save
        mem1 = ConversationMemory(
            system_prompt="Test prompt",
            persist_path=filepath,
        )
        mem1.add_user_message("Hello")
        mem1.add_assistant_message("Hi!")

        # Verify file was created
        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert len(data["messages"]) == 2

        # Load into a new instance
        mem2 = ConversationMemory(persist_path=filepath)
        assert mem2.message_count == 2
        assert mem2.messages[0].content == "Hello"
        assert mem2.messages[1].content == "Hi!"

    def test_persistence_clear(self, tmp_dir: Path) -> None:
        filepath = tmp_dir / "history.json"
        mem = ConversationMemory(persist_path=filepath)
        mem.add_user_message("test")
        mem.clear()

        data = json.loads(filepath.read_text())
        assert len(data["messages"]) == 0

    def test_repr(self) -> None:
        mem = ConversationMemory(max_messages=50)
        mem.add_user_message("test")
        assert "messages=1" in repr(mem)
        assert "max=50" in repr(mem)


@pytest.fixture
def tmp_dir() -> Path:
    """Provide a temporary directory (re-export from conftest for standalone use)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
