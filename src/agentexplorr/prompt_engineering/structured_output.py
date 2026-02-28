"""
Structured Output Parsing — From LLM Text to Python Objects
=============================================================

THE PROBLEM:
  LLMs output unstructured text. But your code needs structured data:
  dictionaries, Pydantic models, database rows. How do you bridge the gap?

THE SOLUTION:
  1. Instruct the LLM to output in a specific format (usually JSON)
  2. Parse the LLM's output into a Python object
  3. Validate the output matches your expected schema

WHY PYDANTIC?
  Pydantic models define both the SCHEMA (what fields exist, their types)
  and the VALIDATION (is the data correct?). If the LLM outputs invalid JSON
  or wrong types, Pydantic raises a clear error.

TECHNIQUES:
  1. **JSON mode** — Ask the LLM for JSON, parse with json.loads()
  2. **Pydantic parsing** — Parse JSON into validated Pydantic models
  3. **Regex extraction** — Extract structured data from free text
  4. **Function calling** — LLMs output structured tool calls (best approach)

LEARNING RESOURCES:
  - Pydantic docs: https://docs.pydantic.dev/latest/
  - LangChain Output Parsers: https://python.langchain.com/docs/concepts/output_parsers/
  - VIDEO: "Pydantic is All You Need" — https://www.youtube.com/watch?v=yj-wSRJwrrc
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class OutputParser:
    """Parse LLM text output into structured Python objects.

    This class provides multiple parsing strategies, from simple JSON
    extraction to full Pydantic model validation.

    Example:
        >>> from pydantic import BaseModel
        >>>
        >>> class Sentiment(BaseModel):
        ...     label: str
        ...     confidence: float
        ...     reasoning: str
        >>>
        >>> parser = OutputParser()
        >>> llm_output = '''
        ... {
        ...     "label": "positive",
        ...     "confidence": 0.95,
        ...     "reasoning": "The text contains enthusiastic language."
        ... }
        ... '''
        >>> result = parser.parse_json_to_model(llm_output, Sentiment)
        >>> print(result.label)  # "positive"
        >>> print(result.confidence)  # 0.95
    """

    @staticmethod
    def extract_json(text: str) -> str:
        """Extract JSON from LLM output that may contain markdown or extra text.

        THE PROBLEM:
          LLMs often wrap JSON in markdown code blocks:
            ```json
            {"key": "value"}
            ```
          Or include explanatory text before/after the JSON.

        THE SOLUTION:
          1. Try to find JSON in markdown code blocks
          2. Fall back to finding any {...} or [...] pattern
          3. Validate it's actually valid JSON

        Args:
            text: Raw LLM output text.

        Returns:
            Extracted JSON string.

        Raises:
            ValueError: If no valid JSON found in the text.
        """
        # Strategy 1: Look for JSON in markdown code blocks
        # Matches: ```json\n{...}\n``` or ```\n{...}\n```
        code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```"
        code_blocks = re.findall(code_block_pattern, text)
        for block in code_blocks:
            try:
                json.loads(block.strip())
                return block.strip()
            except json.JSONDecodeError:
                continue

        # Strategy 2: Find the outermost { } or [ ] pair
        # This handles cases where the LLM adds text around the JSON
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                candidate = text[start : end + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

        msg = f"No valid JSON found in text: {text[:200]}..."
        raise ValueError(msg)

    def parse_json(self, text: str) -> dict[str, object] | list[object]:
        """Parse LLM output into a Python dict or list.

        Args:
            text: Raw LLM output containing JSON.

        Returns:
            Parsed Python dictionary or list.
        """
        json_str = self.extract_json(text)
        result: dict[str, object] | list[object] = json.loads(json_str)
        return result

    def parse_json_to_model(self, text: str, model_class: type[T]) -> T:
        """Parse LLM output into a validated Pydantic model.

        WHY PYDANTIC VALIDATION?
          Without validation, you might get:
            {"label": "positive", "confidence": "very high"}
          But confidence should be a float! Pydantic catches this and
          raises a clear error, so you can retry or handle gracefully.

        Args:
            text: Raw LLM output containing JSON.
            model_class: The Pydantic model class to parse into.

        Returns:
            Validated Pydantic model instance.

        Raises:
            ValueError: If JSON extraction fails.
            ValidationError: If the data doesn't match the schema.
        """
        json_str = self.extract_json(text)
        return model_class.model_validate_json(json_str)

    def parse_with_retry(
        self,
        text: str,
        model_class: type[T],
        max_retries: int = 3,
    ) -> T | None:
        """Try to parse LLM output, returning None on failure.

        IN PRODUCTION:
          LLMs sometimes produce malformed output. Rather than crashing,
          this method retries parsing with progressively more lenient
          extraction strategies.

        Args:
            text: Raw LLM output.
            model_class: Pydantic model class.
            max_retries: Number of parsing attempts.

        Returns:
            Validated model or None if all attempts fail.
        """
        for attempt in range(max_retries):
            try:
                return self.parse_json_to_model(text, model_class)
            except (ValueError, ValidationError):
                # On retry, try cleaning the text more aggressively
                text = text.strip()
                if attempt > 0:
                    # Remove common LLM artifacts
                    text = re.sub(r"^(Here's|Here is|Output:)\s*", "", text)
        return None

    @staticmethod
    def get_format_instructions(model_class: type[BaseModel]) -> str:
        """Generate format instructions from a Pydantic model.

        HOW THIS HELPS:
          Include these instructions in your prompt to tell the LLM
          exactly what JSON structure you expect. This dramatically
          improves parsing success rates.

        Args:
            model_class: The Pydantic model to generate instructions for.

        Returns:
            A string describing the expected JSON format.

        Example:
            >>> class Sentiment(BaseModel):
            ...     label: str
            ...     confidence: float
            >>> print(OutputParser.get_format_instructions(Sentiment))
            Respond with valid JSON matching this schema:
            {
              "label": "string",
              "confidence": "number"
            }
        """
        schema = model_class.model_json_schema()
        properties = schema.get("properties", {})

        fields = {}
        for name, info in properties.items():
            json_type = info.get("type", "string")
            fields[name] = json_type

        schema_str = json.dumps(fields, indent=2)
        return f"Respond with valid JSON matching this schema:\n{schema_str}"
