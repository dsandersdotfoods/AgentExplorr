"""
Tests for LLM Training Data Preparation
=========================================

These tests verify data formatting logic WITHOUT downloading
actual datasets (which would be slow and require network access).
"""

from __future__ import annotations

from agentexplorr.llm_training.data_preparation import DatasetPreparer


class TestDatasetPreparer:
    """Tests for DatasetPreparer formatting logic."""

    def test_format_alpaca_instruction(self) -> None:
        """Test formatting an Alpaca-style instruction example."""
        preparer = DatasetPreparer()

        example = {
            "instruction": "Explain what machine learning is.",
            "input": "",
            "output": "Machine learning is a subset of AI...",
        }

        formatted = preparer.format_instruction(example)
        assert "### Instruction:" in formatted
        assert "Explain what machine learning is." in formatted
        assert "### Response:" in formatted
        assert "Machine learning is a subset of AI..." in formatted

    def test_format_alpaca_with_input(self) -> None:
        """Test formatting with both instruction and input context."""
        preparer = DatasetPreparer()

        example = {
            "instruction": "Summarize the following text.",
            "input": "The quick brown fox jumps over the lazy dog.",
            "output": "A fox jumps over a dog.",
        }

        formatted = preparer.format_instruction(example)
        assert "### Input:" in formatted
        assert "quick brown fox" in formatted

    def test_format_instruction_no_input_field(self) -> None:
        """Test formatting when input field is missing entirely."""
        preparer = DatasetPreparer()

        example = {
            "instruction": "What is 2+2?",
            "output": "4",
        }

        formatted = preparer.format_instruction(example)
        assert "### Instruction:" in formatted
        assert "### Response:" in formatted
