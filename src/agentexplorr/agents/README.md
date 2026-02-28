# AI Agents Module -- Learning Guide

> **"An agent is an LLM with tools and a plan."**

This module teaches you how to build AI agents from scratch using open-source tools.
No API keys required. Everything runs locally.

---

## Table of Contents

1. [What Are AI Agents?](#what-are-ai-agents)
2. [Architecture Overview](#architecture-overview)
3. [Technology Stack](#technology-stack)
4. [Agent Types](#agent-types)
5. [Tools](#tools)
6. [Quick Start](#quick-start)
7. [Code Examples](#code-examples)
8. [Evaluation & Benchmarking](#evaluation--benchmarking)
9. [Key Concepts Glossary](#key-concepts-glossary)
10. [Learning Path](#learning-path)
11. [Papers & Resources](#papers--resources)

---

## What Are AI Agents?

An **AI agent** is a system that uses a Large Language Model (LLM) as its reasoning
engine to autonomously:

1. **Perceive** -- Read and understand a user's request
2. **Plan** -- Decide what steps to take
3. **Act** -- Execute actions (call tools, search the web, compute)
4. **Observe** -- Read the results of actions
5. **Iterate** -- Repeat until the task is complete

### The Agent Spectrum

```
Simple Chat       Tool-Calling       ReAct Agent       Multi-Agent
(no tools)        (1-2 calls)        (loop)            (team)

"What is          "Calculate         "Research          "Research X,
 Python?"          sqrt(144)"         topic X,           analyze data,
                                      verify it,         write report"
                                      summarize"

<----------- Increasing Complexity + Capability ----------->
```

### Why Agents Matter

LLMs alone have critical limitations:
- **No current information** -- Training data has a cutoff date
- **Bad at math** -- LLMs hallucinate calculations
- **No external access** -- Can't browse the web, query databases, or call APIs
- **No memory across sessions** -- Forget everything between conversations

Agents solve all of these by giving the LLM **tools** and a **reasoning loop**.

---

## Architecture Overview

### System Architecture

```
+------------------------------------------------------------------+
|                        AgentExplorr                                |
|                                                                    |
|  +--------------------+    +----------------------------------+   |
|  |    User Query       |    |          Ollama Server           |   |
|  +--------+-----------+    |   (local LLM: llama3.2, etc.)    |   |
|           |                 +----------------+-----------------+   |
|           v                                  |                     |
|  +--------+-----------+                      |                     |
|  |   Agent (LangGraph) | <---  LLM calls --->+                    |
|  |                     |                                           |
|  |  State Machine:     |    +----------------------------------+  |
|  |  Think -> Act ->    +--->|            Tools                  | |
|  |  Observe -> Think   |    |  +----------+ +--------+ +-----+ | |
|  |  ...                |    |  | DuckDuck | | Calcul | | Web  | | |
|  +---------------------+    |  | Go Search| | ator   | |Scrape| | |
|                              |  +----------+ +--------+ +-----+ | |
|                              +----------------------------------+ |
+------------------------------------------------------------------+
```

### ReAct Agent Flow

```
                    User Query
                        |
                        v
                +---------------+
                |    THINK      |   "I need to search for this..."
                |  (LLM call)   |
                +-------+-------+
                        |
            +-----------+-----------+
            |                       |
            v                       v
    +---------------+       +---------------+
    |     ACT       |       |    FINISH     |
    | (execute tool)|       | (return answer)|
    +-------+-------+       +---------------+
            |
            v
    +---------------+
    |   OBSERVE     |
    | (read result) |
    +-------+-------+
            |
            v
        (back to THINK)
```

### Multi-Agent Supervisor Flow

```
                    User Query
                        |
                        v
            +-----------------------+
            |     SUPERVISOR        |
            | "Who should handle    |
            |  this part?"          |
            +---+-------+------+---+
                |       |      |
                v       v      v
          +--------+ +------+ +------+
          |RESEARCH| |ANALYST| |WRITER|
          | (search)| |(calc) | |(draft)|
          +---+----+ +--+---+ +--+---+
              |          |        |
              +----------+--------+
                         |
                         v
            +-----------------------+
            |     SUPERVISOR        |
            | "Next specialist or   |
            |  FINISH?"             |
            +-----------------------+
```

---

## Technology Stack

Everything is open-source. No commercial API keys needed.

| Component | Library | Purpose |
|-----------|---------|---------|
| Agent Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) | State machine for agent loops |
| LLM Inference | [langchain-ollama](https://python.langchain.com/docs/integrations/chat/ollama/) | ChatOllama wrapper for Ollama |
| Local LLM Server | [Ollama](https://ollama.com/) | Run LLMs locally (llama3.2, mistral, etc.) |
| Web Search | [duckduckgo-search](https://github.com/deedy5/duckduckgo_search) | Free search, no API key |
| Web Scraping | [httpx](https://www.python-httpx.org/) + [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) | Fetch and parse web pages |
| Tool Framework | [LangChain Core](https://python.langchain.com/docs/concepts/tools/) | @tool decorator, message types |

### Prerequisites

```bash
# 1. Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a model (llama3.2 is a great default -- fast and capable)
ollama pull llama3.2

# 3. Verify it's running
ollama list
# Should show: llama3.2:latest

# 4. Install Python dependencies
uv sync --extra agents
```

---

## Agent Types

### 1. ReAct Agent (`react_agent.py`)

**Best for:** Complex, multi-step research tasks

The ReAct (Reasoning + Acting) agent explicitly interleaves thinking with
action. It follows the loop from the seminal paper by Yao et al. (2022).

**Key insight:** By making the LLM "think out loud" before acting, it makes
fewer mistakes and can recover from errors.

```python
from agentexplorr.agents import ReActAgent

agent = ReActAgent(verbose=True)
result = agent.run("What is the population of Tokyo and what percentage of Japan's total is that?")

print(result.final_answer)
print(f"Iterations: {result.iterations}")
print(f"Tools used: {result.tools_used}")
```

### 2. Tool-Calling Agent (`tool_agent.py`)

**Best for:** Simple, direct queries that need one tool call

The simplest agent architecture. The LLM directly decides which tool to call
without explicit reasoning steps. Fast and efficient for straightforward tasks.

```python
from agentexplorr.agents import ToolAgent

agent = ToolAgent()
result = agent.run("What is sqrt(144) + 10?")
print(result.answer)

# See what tools are available
for tool_info in agent.list_tools():
    print(f"  {tool_info['name']}: {tool_info['description'][:60]}...")
```

### 3. Multi-Agent Supervisor (`multi_agent.py`)

**Best for:** Complex tasks requiring multiple skills (research + analysis + writing)

A supervisor agent routes tasks to specialist agents. Each specialist is
optimized for its role with a focused prompt and relevant tools.

```python
from agentexplorr.agents import MultiAgentSupervisor

system = MultiAgentSupervisor(verbose=True)
result = system.run(
    "Compare the GDP of the USA, China, and Japan. "
    "Which has grown the fastest in the last decade?"
)

print(result.final_answer)
print(f"Specialists used: {list(result.specialist_outputs.keys())}")
```

---

## Tools

Tools are the "hands" of an agent. They extend the LLM's capabilities.

### Web Search (`tools/search.py`)

```python
from agentexplorr.agents.tools import web_search, search

# As a LangChain tool (for agents)
result = web_search.invoke({"query": "LangGraph tutorial 2024"})

# As a plain function (for scripts)
results = search("Python AI agents")
for r in results:
    print(f"{r['title']}: {r['url']}")
```

### Calculator (`tools/calculator.py`)

```python
from agentexplorr.agents.tools import calculator, calculate

# As a tool
result = calculator.invoke({"expression": "sqrt(144) + pi * 2"})

# As a function
value = calculate("2 ** 10 + sin(pi/4)")
print(value)  # 1024.7071067811865
```

**Security:** The calculator uses AST parsing instead of `eval()`. It validates
every operation against a whitelist. Malicious expressions like
`__import__('os').system('rm -rf /')` are rejected immediately.

### Web Scraper (`tools/web_scraper.py`)

```python
from agentexplorr.agents.tools import web_scrape, scrape_url

# As a tool
content = web_scrape.invoke({"url": "https://en.wikipedia.org/wiki/Python_(programming_language)"})

# As a function
result = scrape_url("https://example.com")
print(result["title"])
print(result["content"][:500])
```

---

## Quick Start

### Minimal Example (5 lines)

```python
from agentexplorr.agents import ReActAgent

agent = ReActAgent()
result = agent.run("What is 25 * 17?")
print(result.final_answer)
```

### Full Example with Error Handling

```python
from agentexplorr.agents import ReActAgent, ReActResult

# Create agent with custom settings
agent = ReActAgent(
    model="llama3.2",       # Ollama model name
    max_iterations=5,        # Safety limit on reasoning loops
    verbose=True,            # Print the Think-Act-Observe trace
)

# Run a query
result: ReActResult = agent.run(
    "Search for the current weather in Tokyo and convert the temperature "
    "from Celsius to Fahrenheit."
)

# Check results
if result.success:
    print(f"Answer: {result.final_answer}")
    print(f"Steps taken: {result.iterations}")
    print(f"Tools used: {result.tools_used}")
else:
    print(f"Agent failed: {result.final_answer}")
```

### Comparing All Three Agent Types

```python
from agentexplorr.agents import ReActAgent, ToolAgent, MultiAgentSupervisor

query = "What is the square root of 256?"

# Method 1: ReAct (explicit reasoning loop)
react = ReActAgent()
r1 = react.run(query)
print(f"ReAct: {r1.final_answer} ({r1.iterations} iterations)")

# Method 2: Tool-calling (direct dispatch)
tool = ToolAgent()
r2 = tool.run(query)
print(f"Tool:  {r2.answer} ({len(r2.tools_called)} tool calls)")

# Method 3: Multi-agent (supervisor + specialists)
multi = MultiAgentSupervisor()
r3 = multi.run(query)
print(f"Multi: {r3.final_answer} ({r3.iterations} routing steps)")
```

---

## Evaluation & Benchmarking

### Running Built-in Benchmarks

```python
from agentexplorr.agents import ReActAgent
from agentexplorr.agents.evaluation import AgentBenchmark

agent = ReActAgent()
benchmark = AgentBenchmark(verbose=True)

# Run on math questions
summary = benchmark.run(agent, benchmark.math_questions())
print(f"Math accuracy: {summary.accuracy:.1%}")

# Run on search questions
summary = benchmark.run(agent, benchmark.search_questions())
print(f"Search accuracy: {summary.accuracy:.1%}")
```

### Custom Benchmark Questions

```python
from agentexplorr.agents.evaluation import AgentBenchmark, BenchmarkQuestion

custom_questions = [
    BenchmarkQuestion(
        question="What is the boiling point of water in Fahrenheit?",
        expected_answer="212",
        category="science",
        match_type="numeric",
    ),
    BenchmarkQuestion(
        question="Who painted the Mona Lisa?",
        expected_answer="Leonardo da Vinci",
        category="art",
        match_type="contains",
    ),
    BenchmarkQuestion(
        question="What are the primary colors?",
        expected_answer="red, blue, yellow",
        keywords=["red", "blue", "yellow"],
        category="art",
        match_type="keywords",
    ),
]

benchmark = AgentBenchmark()
summary = benchmark.run(agent, custom_questions)
```

### Comparing Agents Head-to-Head

```python
from agentexplorr.agents import ReActAgent, ToolAgent
from agentexplorr.agents.evaluation import AgentBenchmark

questions = AgentBenchmark.mixed_questions()
benchmark = AgentBenchmark(verbose=False)

react_summary = benchmark.run(ReActAgent(), questions)
tool_summary = benchmark.run(ToolAgent(), questions)

print(f"ReAct:  {react_summary.accuracy:.1%} accuracy, {react_summary.avg_latency:.1f}s avg")
print(f"Tool:   {tool_summary.accuracy:.1%} accuracy, {tool_summary.avg_latency:.1f}s avg")
```

---

## Key Concepts Glossary

| Concept | Definition |
|---------|-----------|
| **Agent** | LLM + tools + reasoning loop |
| **Tool** | A function the agent can call (search, calculate, etc.) |
| **ReAct** | Reasoning + Acting -- think before you act, observe after |
| **State Graph** | LangGraph's way of defining agent behavior as nodes + edges |
| **Tool Calling** | LLM's ability to output structured tool invocations |
| **bind_tools()** | LangChain method to attach tools to an LLM |
| **Supervisor** | An agent that routes tasks to specialist agents |
| **Observation** | The result returned by a tool call |
| **Iteration** | One complete Think-Act-Observe cycle |
| **Grounding** | Using external data to prevent LLM hallucination |
| **AST** | Abstract Syntax Tree -- used for safe expression parsing |
| **SSRF** | Server-Side Request Forgery -- a web security vulnerability |

---

## Learning Path

### Beginner (Start Here)

1. **Watch:** [What are AI Agents?](https://www.youtube.com/watch?v=F8NKVhkZZWI) (15 min)
2. **Read:** [LangGraph Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
3. **Code:** Run the [Quick Start](#quick-start) example above
4. **Read:** `tools/search.py` -- understand how tools work
5. **Read:** `tools/calculator.py` -- understand safe evaluation

### Intermediate

1. **Watch:** [Build AI Agents with LangGraph](https://www.youtube.com/watch?v=v9fkbTxPzs0) (45 min)
2. **Read:** The ReAct paper: [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
3. **Code:** Read through `react_agent.py` -- understand the state graph
4. **Code:** Modify `tool_agent.py` -- add your own custom tools
5. **Read:** `evaluation/benchmarks.py` -- understand how agents are evaluated

### Advanced

1. **Watch:** [Multi-Agent Systems with LangGraph](https://www.youtube.com/watch?v=hvAPnpSfSGo) (60 min)
2. **Read:** [A Survey on LLM-based Agents](https://arxiv.org/abs/2308.11432)
3. **Code:** Extend `multi_agent.py` -- add a new specialist agent
4. **Code:** Create custom benchmark questions for your domain
5. **Experiment:** Compare different Ollama models (llama3.2 vs mistral vs command-r)

---

## Papers & Resources

### Foundational Papers

| Paper | Year | Key Idea | Link |
|-------|------|----------|------|
| ReAct | 2022 | Interleave reasoning with acting | [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629) |
| Toolformer | 2023 | LLMs that learn to use tools | [arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761) |
| Chain of Thought | 2022 | Step-by-step reasoning | [arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903) |
| AutoGen | 2023 | Multi-agent conversations | [arxiv.org/abs/2308.08155](https://arxiv.org/abs/2308.08155) |
| AgentBench | 2023 | Evaluating LLMs as agents | [arxiv.org/abs/2308.03688](https://arxiv.org/abs/2308.03688) |
| LLM Agent Survey | 2023 | Comprehensive survey | [arxiv.org/abs/2308.11432](https://arxiv.org/abs/2308.11432) |
| WebGPT | 2021 | Browser-assisted QA | [arxiv.org/abs/2112.09332](https://arxiv.org/abs/2112.09332) |

### Video Tutorials

| Title | Duration | Link |
|-------|----------|------|
| What are AI Agents? | 15 min | [youtube.com/watch?v=F8NKVhkZZWI](https://www.youtube.com/watch?v=F8NKVhkZZWI) |
| LangGraph Crash Course | 30 min | [youtube.com/watch?v=R-o_a6F-SOQ](https://www.youtube.com/watch?v=R-o_a6F-SOQ) |
| Build AI Agents with LangGraph | 45 min | [youtube.com/watch?v=v9fkbTxPzs0](https://www.youtube.com/watch?v=v9fkbTxPzs0) |
| ReAct Agents Explained | 20 min | [youtube.com/watch?v=Eug2clsLtFs](https://www.youtube.com/watch?v=Eug2clsLtFs) |
| Multi-Agent Systems | 60 min | [youtube.com/watch?v=hvAPnpSfSGo](https://www.youtube.com/watch?v=hvAPnpSfSGo) |
| Build AI Agents from Scratch | 90 min | [youtube.com/watch?v=AxnL5GtWnGE](https://www.youtube.com/watch?v=AxnL5GtWnGE) |
| LangChain Tools Deep Dive | 30 min | [youtube.com/watch?v=q-HNphrWsDE](https://www.youtube.com/watch?v=q-HNphrWsDE) |
| Function Calling with LangChain | 25 min | [youtube.com/watch?v=p9v2fHLmQCU](https://www.youtube.com/watch?v=p9v2fHLmQCU) |
| How to Evaluate AI Agents | 35 min | [youtube.com/watch?v=2e_7VCnAzCQ](https://www.youtube.com/watch?v=2e_7VCnAzCQ) |

### Documentation

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangChain Tools](https://python.langchain.com/docs/concepts/tools/)
- [ChatOllama](https://python.langchain.com/docs/integrations/chat/ollama/)
- [Ollama Model Library](https://ollama.com/library)
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search)
- [Python ast Module](https://docs.python.org/3/library/ast.html)

---

## Module File Structure

```
agents/
  __init__.py            # Module entry point, exports main classes
  react_agent.py         # ReAct (Reasoning + Acting) agent
  tool_agent.py          # Simple tool-calling agent
  multi_agent.py         # Multi-agent supervisor system
  README.md              # This learning guide
  tools/
    __init__.py          # Tool exports
    search.py            # DuckDuckGo web search (no API key)
    calculator.py        # Safe math calculator (AST-based, no eval)
    web_scraper.py       # URL content extraction (httpx + BS4)
  evaluation/
    __init__.py          # Evaluation exports
    benchmarks.py        # AgentBenchmark class for systematic testing
```

---

## Troubleshooting

### "Connection refused" error
Ollama isn't running. Start it with:
```bash
ollama serve
```

### "Model not found" error
Pull the model first:
```bash
ollama pull llama3.2
```

### "Tool calls not working"
Not all Ollama models support tool calling. Use models that support it:
- `llama3.2` (recommended)
- `mistral`
- `command-r`

### Empty search results
DuckDuckGo may rate-limit aggressive usage. The search tool has built-in
retry logic with exponential backoff. If issues persist, wait a minute.

### Agent loops forever
All agents have built-in iteration limits (max 10 for ReAct, 5 for ToolAgent,
8 for MultiAgent). If hit, the agent stops and returns what it has.
