"""
Tests for Prompt Templates
============================

Test Jinja2 template rendering, variable extraction, and pre-built templates.
"""

from __future__ import annotations

import pytest
from jinja2 import UndefinedError

from agentexplorr.prompt_engineering.templates import (
    CLASSIFICATION_TEMPLATE,
    EXTRACTION_TEMPLATE,
    SUMMARIZATION_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    PromptTemplate,
)


class TestPromptTemplate:
    """Tests for the PromptTemplate class."""

    def test_basic_render(self) -> None:
        """Test simple variable substitution."""
        template = PromptTemplate("Hello, {{ name }}!")
        result = template.render(name="World")
        assert result == "Hello, World!"

    def test_multiple_variables(self) -> None:
        """Test multiple variables in one template."""
        template = PromptTemplate("{{ role }} says: {{ message }}")
        result = template.render(role="Alice", message="Hi")
        assert result == "Alice says: Hi"

    def test_variable_extraction(self) -> None:
        """Test that template variables are correctly identified."""
        template = PromptTemplate("{{ a }} and {{ b }} and {{ c }}")
        assert template.variables == {"a", "b", "c"}

    def test_missing_variable_raises(self) -> None:
        """Test that missing variables raise an error (StrictUndefined)."""
        template = PromptTemplate("{{ required_var }}")
        with pytest.raises(UndefinedError):
            template.render()  # No variables provided

    def test_conditional_rendering(self) -> None:
        """Test Jinja2 conditionals in templates."""
        template = PromptTemplate(
            "{% if formal %}Dear Sir/Madam{% else %}Hey{% endif %}, {{ name }}!"
        )
        assert "Dear Sir/Madam" in template.render(name="Bob", formal=True)
        assert "Hey" in template.render(name="Bob", formal=False)

    def test_loop_rendering(self) -> None:
        """Test Jinja2 loops in templates."""
        template = PromptTemplate(
            "{% for item in items %}- {{ item }}\n{% endfor %}"
        )
        result = template.render(items=["apple", "banana"])
        assert "- apple" in result
        assert "- banana" in result

    def test_repr(self) -> None:
        """Test string representation."""
        template = PromptTemplate("{{ x }}", name="test")
        repr_str = repr(template)
        assert "test" in repr_str
        assert "x" in repr_str


class TestPrebuiltTemplates:
    """Tests for the pre-built prompt templates."""

    def test_system_prompt(self) -> None:
        """Test system prompt template rendering."""
        result = SYSTEM_PROMPT_TEMPLATE.render(
            role="data scientist",
            task="Analyze the dataset",
            constraints=["Be concise", "Use tables"],
            output_format="Markdown",
        )
        assert "data scientist" in result
        assert "Be concise" in result
        assert "Markdown" in result

    def test_summarization_template(self) -> None:
        """Test summarization template."""
        result = SUMMARIZATION_TEMPLATE.render(
            style="professional",
            max_words=50,
            text="Some long text here...",
        )
        assert "professional" in result
        assert "50" in result
        assert "Some long text" in result

    def test_extraction_template(self) -> None:
        """Test extraction template with fields."""
        result = EXTRACTION_TEMPLATE.render(
            fields=["name", "email", "phone"],
            text="Contact: John at john@example.com",
        )
        assert "name" in result
        assert "email" in result
        assert "JSON" in result

    def test_classification_template(self) -> None:
        """Test classification template with categories."""
        result = CLASSIFICATION_TEMPLATE.render(
            categories=["Positive", "Negative", "Neutral"],
            text="I love this product!",
        )
        assert "Positive" in result
        assert "I love this product" in result
