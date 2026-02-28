"""
Tests for Structured Output Parsing
=====================================

Test JSON extraction from LLM output and Pydantic validation.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agentexplorr.prompt_engineering.structured_output import OutputParser


class SentimentResult(BaseModel):
    """Test model for structured output."""

    label: str
    confidence: float


class TestOutputParser:
    """Tests for OutputParser."""

    def setup_method(self) -> None:
        """Create parser instance."""
        self.parser = OutputParser()

    def test_extract_json_plain(self) -> None:
        """Test extracting plain JSON."""
        text = '{"label": "positive", "confidence": 0.95}'
        result = OutputParser.extract_json(text)
        assert '"positive"' in result

    def test_extract_json_from_markdown(self) -> None:
        """Test extracting JSON from markdown code blocks."""
        text = """Here's the result:
```json
{"label": "negative", "confidence": 0.8}
```
Hope that helps!"""
        result = OutputParser.extract_json(text)
        assert '"negative"' in result

    def test_extract_json_with_surrounding_text(self) -> None:
        """Test extracting JSON embedded in explanation text."""
        text = 'The sentiment is: {"label": "neutral", "confidence": 0.6} as you can see.'
        result = OutputParser.extract_json(text)
        assert '"neutral"' in result

    def test_extract_json_no_json(self) -> None:
        """Test that missing JSON raises ValueError."""
        with pytest.raises(ValueError, match="No valid JSON"):
            OutputParser.extract_json("No JSON here at all!")

    def test_parse_json(self) -> None:
        """Test parsing to dict."""
        text = '{"key": "value", "number": 42}'
        result = self.parser.parse_json(text)
        assert isinstance(result, dict)
        assert result["key"] == "value"

    def test_parse_json_to_model(self) -> None:
        """Test parsing to Pydantic model."""
        text = '{"label": "positive", "confidence": 0.95}'
        result = self.parser.parse_json_to_model(text, SentimentResult)
        assert result.label == "positive"
        assert result.confidence == 0.95

    def test_parse_with_retry_success(self) -> None:
        """Test parse_with_retry succeeds on valid input."""
        text = '{"label": "positive", "confidence": 0.9}'
        result = self.parser.parse_with_retry(text, SentimentResult)
        assert result is not None
        assert result.label == "positive"

    def test_parse_with_retry_failure(self) -> None:
        """Test parse_with_retry returns None on invalid input."""
        result = self.parser.parse_with_retry("not json at all", SentimentResult)
        assert result is None

    def test_format_instructions(self) -> None:
        """Test generating format instructions from a model."""
        instructions = OutputParser.get_format_instructions(SentimentResult)
        assert "label" in instructions
        assert "confidence" in instructions
        assert "JSON" in instructions
