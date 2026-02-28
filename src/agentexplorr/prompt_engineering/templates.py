"""
Prompt Templates with Jinja2
=============================

WHAT ARE PROMPT TEMPLATES?
  A prompt template is a string with placeholders that get filled in at runtime.
  Instead of hardcoding prompts, templates let you:
  - Reuse the same prompt structure with different inputs
  - Version control your prompts alongside your code
  - Compose complex prompts from smaller building blocks

WHY JINJA2?
  Jinja2 is Python's most popular templating engine (also used by Flask,
  Ansible, dbt, etc.). It supports:
  - Variable substitution: {{ variable }}
  - Conditionals: {% if condition %} ... {% endif %}
  - Loops: {% for item in items %} ... {% endfor %}
  - Filters: {{ text | upper }}
  - Template inheritance and includes

LEARNING RESOURCES:
  - Jinja2 docs: https://jinja.palletsprojects.com/en/3.1.x/
  - LangChain PromptTemplate: https://python.langchain.com/docs/concepts/prompt_templates/
  - VIDEO: "Jinja2 Tutorial" — https://www.youtube.com/watch?v=bxhXQG1qJPM
"""

from __future__ import annotations

from typing import Any

from jinja2 import Environment, StrictUndefined, Template


class PromptTemplate:
    """A reusable prompt template powered by Jinja2.

    This class wraps Jinja2 templates with validation and a clean API
    for prompt engineering workflows.

    HOW IT WORKS:
      1. Create a template with {{ placeholders }}
      2. Call .render() with keyword arguments to fill them in
      3. The rendered string becomes your LLM prompt

    Example:
        >>> template = PromptTemplate(
        ...     "You are a {{ role }}. {{ task }}"
        ... )
        >>> prompt = template.render(role="data scientist", task="Explain PCA.")
        >>> print(prompt)
        "You are a data scientist. Explain PCA."

    Advanced Example (with conditionals):
        >>> template = PromptTemplate('''
        ... Analyze the following text:
        ... {{ text }}
        ...
        ... {% if format == "json" %}
        ... Respond in valid JSON format.
        ... {% else %}
        ... Respond in plain text.
        ... {% endif %}
        ... ''')
    """

    def __init__(self, template_string: str, name: str = "unnamed") -> None:
        """Initialize with a Jinja2 template string.

        Args:
            template_string: The template with {{ placeholders }}.
            name: Human-readable name for debugging and logging.
        """
        self.name = name
        self.template_string = template_string.strip()

        # StrictUndefined raises an error if a variable is missing,
        # rather than silently rendering as empty string. This catches
        # bugs where you forget to pass a required variable.
        self._env = Environment(undefined=StrictUndefined)
        self._template: Template = self._env.from_string(self.template_string)

        # Extract variable names from the template's AST (Abstract Syntax Tree).
        # This lets us validate inputs before rendering.
        ast = self._env.parse(self.template_string)
        from jinja2 import meta

        self.variables: set[str] = meta.find_undeclared_variables(ast)

    def render(self, **kwargs: Any) -> str:
        """Render the template with the given variables.

        Args:
            **kwargs: Variable names and their values.

        Returns:
            The rendered prompt string.

        Raises:
            jinja2.UndefinedError: If a required variable is missing.

        Example:
            >>> t = PromptTemplate("Summarize: {{ text }}")
            >>> t.render(text="The quick brown fox...")
            'Summarize: The quick brown fox...'
        """
        return self._template.render(**kwargs).strip()

    def __repr__(self) -> str:
        return f"PromptTemplate(name={self.name!r}, variables={self.variables})"


# =============================================================================
# Pre-built Templates — Ready-to-use prompt patterns
# =============================================================================
# These encode common prompt engineering patterns. Each one is documented
# with WHY it works and WHEN to use it.

SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
    template_string="""You are {{ role }}.

Your task: {{ task }}

{% if constraints %}
Constraints:
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}

{% if output_format %}
Output format: {{ output_format }}
{% endif %}""",
    name="system_prompt",
)
"""
The System Prompt Pattern
-------------------------
WHY: Setting a clear role + task + constraints drastically improves LLM output.
     The model "acts" as the specified role, which activates relevant knowledge.

WHEN: Use as the system message in any LLM conversation.

VIDEO: "System Prompts Explained" — https://www.youtube.com/watch?v=eTJD5GOz2ys
"""

SUMMARIZATION_TEMPLATE = PromptTemplate(
    template_string="""Summarize the following text in {{ style }} style.
Keep the summary under {{ max_words }} words.

Text:
{{ text }}

Summary:""",
    name="summarization",
)

EXTRACTION_TEMPLATE = PromptTemplate(
    template_string="""Extract the following information from the text below:
{% for field in fields %}
- {{ field }}
{% endfor %}

Text:
{{ text }}

Extracted information (as JSON):""",
    name="extraction",
)

CLASSIFICATION_TEMPLATE = PromptTemplate(
    template_string="""Classify the following text into one of these categories:
{% for category in categories %}
- {{ category }}
{% endfor %}

Text: {{ text }}

Category:""",
    name="classification",
)
