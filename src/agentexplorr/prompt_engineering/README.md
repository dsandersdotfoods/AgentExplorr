# Prompt Engineering

> "Prompt engineering is the most accessible and impactful AI skill. A well-crafted prompt can turn a mediocre response into an expert-level one."

## What You'll Learn

| File | Concept | Difficulty |
|------|---------|-----------|
| `templates.py` | Jinja2 prompt templates with variable injection | Beginner |
| `few_shot.py` | Teaching LLMs by example (in-context learning) | Beginner |
| `chain_of_thought.py` | Step-by-step reasoning (CoT, self-consistency) | Intermediate |
| `structured_output.py` | Parsing LLM output into Pydantic models | Intermediate |

## Key Concepts

### 1. Prompt Templates
Reusable prompts with `{{ placeholders }}` — never hardcode prompts!

```python
from agentexplorr.prompt_engineering import PromptTemplate

template = PromptTemplate("You are a {{ role }}. {{ task }}")
prompt = template.render(role="data scientist", task="Explain PCA in simple terms.")
```

### 2. Few-Shot Learning
Show the model examples → it learns the pattern.

```python
from agentexplorr.prompt_engineering import FewShotSelector, Example

selector = FewShotSelector()
selector.add_example(Example("I love this!", "Positive"))
selector.add_example(Example("Terrible.", "Negative"))

prompt = selector.build_prompt(
    instruction="Classify sentiment:",
    input_text="Pretty good overall",
)
```

### 3. Chain of Thought
"Let's think step by step" → accuracy goes from 18% to 79% on math.

### 4. Structured Output
LLM text → validated Pydantic models. Production-ready parsing.

## Learning Resources

### Papers
- [Chain of Thought Prompting](https://arxiv.org/abs/2201.11903) — The original CoT paper
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 paper
- [Self-Consistency Improves CoT](https://arxiv.org/abs/2203.11171) — Voting over multiple reasoning paths

### Videos
- [Prompt Engineering Full Course](https://www.youtube.com/watch?v=_ZvnD96hOZo) — 1hr comprehensive overview
- [Chain of Thought Explained](https://www.youtube.com/watch?v=H4gZd4BCrDQ)

### Guides
- [Prompting Guide](https://www.promptingguide.ai/) — Community-maintained reference
- [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering)
